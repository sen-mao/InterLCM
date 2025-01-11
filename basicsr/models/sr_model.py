import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel

# CILP
import clip
import torchvision.transforms as transforms

from basicsr.utils.clip_util import VisionTransformer
clip.model.VisionTransformer = VisionTransformer

# LCM
from diffusers import DiffusionPipeline, UNet2DConditionModel, ControlNetModel
from basicsr.utils.lcm_utils import register_lcm_forward, register_lcmschedule_step

@MODEL_REGISTRY.register()
class SRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # ------------------ CLIPImageEncoder ------------------- #
        self.clip_model, clip_preprocess = clip.load('ViT-B/16')
        self.clip_model = self.model_to_device(self.clip_model)
        if self.opt['dist']:
            self.clip_model.encode_image = self.clip_model.module.encode_image

        self.clip_preprocess = clip_preprocess
        self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] +  # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
                                             clip_preprocess.transforms[:2] +  # to match CLIP input scale assumptions
                                             clip_preprocess.transforms[4:])  # + skip convert PIL to tensor
        # self.print_network(self.clip_model)
        # ------------------ CLIPImageEncoder ------------------- #

        # ------------------ Visual Encoder (VE) ------------------- #
        self.visual_encoder = build_network(opt['visual_encoder'])
        self.visual_encoder = self.model_to_device(self.visual_encoder)
        # self.print_network(self.visual_encoder)
        # ------------------ Visual Encoder (VE) ------------------- #

        # ------------------ frozen LCM ------------------- #
        self.num_inference_steps = opt['lcm']['num_inference_steps']

        self.lcm = DiffusionPipeline.from_pretrained(opt['lcm']['pretrained_model'])
        self.lcm.to(torch_dtype=torch.float32)
        self.lcm.vae = self.model_to_device(self.lcm.vae)
        self.lcm.text_encoder = self.model_to_device(self.lcm.text_encoder)
        self.lcm.unet = self.model_to_device(self.lcm.unet)

        if isinstance(self.lcm.vae, torch.nn.parallel.DataParallel) \
                or isinstance(self.lcm.vae, torch.nn.parallel.DistributedDataParallel):
            self.lcm.vae = self.lcm.vae.module
            self.lcm.text_encoder = self.lcm.text_encoder.module
            self.lcm.unet = self.lcm.unet.module
        # self.print_network(self.lcm.vae)

        self.lcm.vae.requires_grad_(False)
        self.lcm.text_encoder.requires_grad_(False)
        self.lcm.unet.requires_grad_(False)

        # disables safety checks: https://github.com/CompVis/stable-diffusion/issues/331#issuecomment-1562198856
        def disabled_safety_checker(images, clip_input):
            if len(images.shape) == 4:
                num_images = images.shape[0]
                return images, [False] * num_images
            else:
                return images, False
        self.lcm.safety_checker = disabled_safety_checker

        self.lcm.set_progress_bar_config(disable=True)  # if one wants to disable `tqdm`
        # ------------------ frozen LCM ------------------- #

        # ------------------ Spatial Encoder (SE) ------------------- #
        unet = UNet2DConditionModel.from_pretrained(opt['spatial_encoder']['pretrained_model'], subfolder="unet")
        self.spatial_encoder = ControlNetModel.from_unet(unet)
        self.spatial_encoder = self.model_to_device(self.spatial_encoder)
        del unet
        # ------------------ Spatial Encoder (SE) ------------------- #

        register_lcm_forward(self.lcm, self.spatial_encoder)
        register_lcmschedule_step(self.lcm.scheduler)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'ffhq_gt' in data:
            self.gt = data['ffhq_gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'ema_decay'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            if 'ffhq_gt' in visuals:
                gt_img = tensor2img([visuals['ffhq_gt']])
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    metric_data = dict(img1=sr_img, img2=gt_img)
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'ffhq_gt'):
            out_dict['ffhq_gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'ema_decay'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
