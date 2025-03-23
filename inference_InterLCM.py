import os
import cv2
import argparse
import glob
import re
import torch
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.misc import gpu_is_available, get_device
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray

from basicsr.utils.registry import ARCH_REGISTRY

# CILP
import clip
import torchvision.transforms as transforms

from basicsr.utils.clip_util import VisionTransformer
clip.model.VisionTransformer = VisionTransformer

# LCM
from diffusers import DiffusionPipeline, UNet2DConditionModel, ControlNetModel
from basicsr.utils.lcm_utils import register_lcm_forward, register_lcmschedule_step

from scripts.wavelet_color_fix import wavelet_reconstruction, adaptive_instance_normalization

def set_realesrgan(args):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.realesrgan_utils import RealESRGANer

    use_half = False
    if torch.cuda.is_available(): # set False in CPU/MPS mode
        no_half_gpu_list = ['1650', '1660'] # set False for GPUs that don't support f16
        if not True in [gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list]:
            use_half = True

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2,
    )
    upsampler = RealESRGANer(
        scale=2,
        model_path="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
        model=model,
        tile=args.bg_tile,
        tile_pad=40,
        pre_pad=0,
        half=use_half
    )

    if not gpu_is_available():  # CPU
        import warnings
        warnings.warn('Running on CPU now! Make sure your PyTorch version matches your CUDA.'
                        'The unoptimized RealESRGAN is slow on CPU. '
                        'If you want to disable it, please remove `--bg_upsampler` and `--face_upsample` in command.',
                        category=RuntimeWarning)
    return upsampler

@torch.no_grad()
def main():
    device = get_device()
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path', type=str, default='./inputs/whole_imgs', 
            help='Input image, video or folder. Default: inputs/whole_imgs')
    parser.add_argument('-o', '--output_path', type=str, default="results",
            help='Output folder. Default: results/<input_name>')
    parser.add_argument('-s', '--upscale', type=int, default=2, 
            help='The final upsampling scale of the image. Default: 2')
    parser.add_argument('--has_aligned', action='store_true', help='Input are cropped and aligned faces. Default: False')
    parser.add_argument('--only_center_face', action='store_true', help='Only restore the center face. Default: False')
    parser.add_argument('--draw_box', action='store_true', help='Draw the bounding box for the detected faces. Default: False')
    # large det_model: 'YOLOv5l', 'retinaface_resnet50'
    # small det_model: 'YOLOv5n', 'retinaface_mobile0.25'
    parser.add_argument('--detection_model', type=str, default='retinaface_resnet50', 
            help='Face detector. Optional: retinaface_resnet50, retinaface_mobile0.25, YOLOv5l, YOLOv5n, dlib. \
                Default: retinaface_resnet50')
    parser.add_argument('--bg_upsampler', type=str, default='None', help='Background upsampler. Optional: realesrgan')
    parser.add_argument('--face_upsample', action='store_true', help='Face upsampler after enhancement. Default: False')
    parser.add_argument('--bg_tile', type=int, default=400, help='Tile size for background sampler. Default: 400')
    parser.add_argument('--suffix', type=str, default=None, help='Suffix of the restored faces. Default: None')
    parser.add_argument('--save_video_fps', type=float, default=None, help='Frame rate for saving video. Default: None')
    # LCM
    parser.add_argument('--num_inference_steps', type=int, default=4, help='T for lcm')
    parser.add_argument('--visual_encoder_path', type=str,
                        default='weights/InterLCM/visual_encoder_3step.pth',
                        help='visual_encoder checkpoint')
    parser.add_argument('--spatial_encoder_path', type=str,
                        default='weights/InterLCM/spatial_encoder_3step.pth',
                        help='spatial_encoder checkpoint')
    parser.add_argument('--sd_path', type=str,
                        default='runwayml/stable-diffusion-v1-5',
                        help='sd pre-trined model')
    parser.add_argument('--lcm_path', type=str,
                        default='SimianLuo/LCM_Dreamshaper_v7',
                        help='lcm pre-trined model')

    parser.add_argument(
        "--colorfix_type",
        type=str,
        default="wavelet",
        help="Color fix type to adjust the color of reconstructed HR result according to LR input: "
             "adain; wavelet (used in paper); nofix",
    )

    args = parser.parse_args()
    print(args)

    interlcm_step = int(re.findall(r'\d+', args.visual_encoder_path)[0])
    args.output_path = args.output_path.replace(args.output_path.split('/')[0],
                                                f"{args.output_path.split('/')[0]}[{args.colorfix_type}]/interlcm_{interlcm_step}step")

    assert args.num_inference_steps - 1 == interlcm_step

    # ------------------------ input & output ------------------------
    input_video = False
    if args.input_path.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')): # input single img path
        input_img_list = [args.input_path]
        result_root = f'results/test_img'
    elif args.input_path.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')): # input video path
        from basicsr.utils.video_util import VideoReader, VideoWriter
        input_img_list = []
        vidreader = VideoReader(args.input_path)
        image = vidreader.get_frame()
        while image is not None:
            input_img_list.append(image)
            image = vidreader.get_frame()
        audio = vidreader.get_audio()
        fps = vidreader.get_fps() if args.save_video_fps is None else args.save_video_fps   
        video_name = os.path.basename(args.input_path)[:-4]
        result_root = f'results/{video_name}'
        input_video = True
        vidreader.close()
    else: # input img folder
        if args.input_path.endswith('/'):  # solve when path ends with /
            args.input_path = args.input_path[:-1]
        # scan all the jpg and png images
        input_img_list = sorted(glob.glob(os.path.join(args.input_path, '*.[jpJP][pnPN]*[gG]')))
        result_root = f'results/{os.path.basename(args.input_path)}'

    if not args.output_path is None: # set output path
        result_root = args.output_path

    test_img_num = len(input_img_list)
    if test_img_num == 0:
        raise FileNotFoundError('No input image/video is found...\n' 
            '\tNote that --input_path for video should end with .mp4|.mov|.avi')

    # ------------------ set up background upsampler ------------------
    if args.bg_upsampler == 'realesrgan':
        bg_upsampler = set_realesrgan(args)
    else:
        bg_upsampler = None

    # ------------------ set up face upsampler ------------------
    if args.face_upsample:
        if bg_upsampler is not None:
            face_upsampler = bg_upsampler
        else:
            face_upsampler = set_realesrgan(args)
    else:
        face_upsampler = None

    # ------------------ set up InterLCM restorer -------------------

    # CLIPImageEncoder
    clip_model, clip_preprocess = clip.load('ViT-B/16', device=device)
    preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] +  # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
                                    clip_preprocess.transforms[:2] +  # to match CLIP input scale assumptions
                                    clip_preprocess.transforms[4:])  # + skip convert PIL to tensor

    # Visual Encoder
    visual_encoder = ARCH_REGISTRY.get('VisualEncoder')(nf=64, emb_dim=197, ch_mult=[2,4,8], res_blocks=2, img_size=512).to(device)
    checkpoint_ve = torch.load(args.visual_encoder_path)['params_ema']
    visual_encoder.load_state_dict(checkpoint_ve)
    visual_encoder.eval()

    # Spatial Encoder
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path=args.sd_path, subfolder="unet")
    spatial_encoder = ControlNetModel.from_unet(unet).to(device)
    checkpoint_c = torch.load(args.spatial_encoder_path)['params_ema']
    spatial_encoder.load_state_dict(checkpoint_c)
    spatial_encoder.eval()

    # lcm
    lcm = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path=args.lcm_path).to(device)

    register_lcm_forward(lcm, spatial_encoder)
    register_lcmschedule_step(lcm.scheduler)


    # ------------------ set up FaceRestoreHelper -------------------
    # large det_model: 'YOLOv5l', 'retinaface_resnet50'
    # small det_model: 'YOLOv5n', 'retinaface_mobile0.25'
    if not args.has_aligned: 
        print(f'Face detection model: {args.detection_model}')
    if bg_upsampler is not None: 
        print(f'Background upsampling: True, Face upsampling: {args.face_upsample}')
    else:
        print(f'Background upsampling: False, Face upsampling: {args.face_upsample}')

    face_helper = FaceRestoreHelper(
        args.upscale,
        face_size=512,
        crop_ratio=(1, 1),
        det_model = args.detection_model,
        save_ext='png',
        use_parse=True,
        device=device)

    # -------------------- start to processing ---------------------
    for i, img_path in enumerate(input_img_list):
        # clean all the intermediate results to process the next image
        face_helper.clean_all()
        
        if isinstance(img_path, str):
            img_name = os.path.basename(img_path)
            basename, ext = os.path.splitext(img_name)
            print(f'[{i+1}/{test_img_num}] Processing: {img_name}')
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        else: # for video processing
            basename = str(i).zfill(6)
            img_name = f'{video_name}_{basename}' if input_video else basename
            print(f'[{i+1}/{test_img_num}] Processing: {img_name}')
            img = img_path

        if args.has_aligned: 
            # the input faces are already cropped and aligned
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            face_helper.is_gray = is_gray(img, threshold=10)
            if face_helper.is_gray:
                print('Grayscale input: True')
                if 'colorization' in args.visual_encoder_path:
                    face_helper.is_gray = False
            face_helper.cropped_faces = [img]
        else:
            face_helper.read_image(img)
            # get face landmarks for each face
            num_det_faces = face_helper.get_face_landmarks_5(
                only_center_face=args.only_center_face, resize=640, eye_dist_threshold=5)
            print(f'\tdetect {num_det_faces} faces')
            # align and warp each face
            face_helper.align_warp_face()

        # face restoration for each cropped face
        for idx, cropped_face in enumerate(face_helper.cropped_faces):
            # prepare data
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

            try:
                with torch.no_grad():
                    input = preprocess(cropped_face_t)
                    img_emb = clip_model.encode_image(input)
                    img_emb = img_emb.to(torch.float)

                    visual_feat = visual_encoder(img_emb)

                    latent_code = lcm.vae.encode(cropped_face_t)['latent_dist'].mean
                    latent_code = latent_code * 0.18215
                    output = lcm.forward(height=512, width=512, num_inference_steps=args.num_inference_steps, guidance_scale=8.0, latents=latent_code,
                                         prompt_embeds=visual_feat, output_type="pil", lcm_origin_steps=50, lq_input=cropped_face_t).images

                    # colorfix from StableSR
                    if args.colorfix_type == 'adain':
                        output = adaptive_instance_normalization(output, cropped_face_t)
                    elif args.colorfix_type == 'wavelet':
                        output = wavelet_reconstruction(output, cropped_face_t)

                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except Exception as error:
                print(f'\tFailed inference for CodeFormer: {error}')
                restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

            restored_face = restored_face.astype('uint8')
            face_helper.add_restored_face(restored_face, cropped_face)

        # paste_back
        if not args.has_aligned:
            # upsample the background
            if bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                bg_img = bg_upsampler.enhance(img, outscale=args.upscale)[0]
            else:
                bg_img = None
            face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            if args.face_upsample and face_upsampler is not None: 
                restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=args.draw_box, face_upsampler=face_upsampler)
            else:
                restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=args.draw_box)

        # save faces
        for idx, (cropped_face, restored_face) in enumerate(zip(face_helper.cropped_faces, face_helper.restored_faces)):
            # save cropped face
            if not args.has_aligned: 
                save_crop_path = os.path.join(result_root, 'cropped_faces', f'{basename}_{idx:02d}.png')
                imwrite(cropped_face, save_crop_path)
            # save restored face
            if args.has_aligned:
                save_face_name = f'{basename}.png'
            else:
                save_face_name = f'{basename}_{idx:02d}.png'
            if args.suffix is not None:
                save_face_name = f'{save_face_name[:-4]}_{args.suffix}.png'
            save_restore_path = os.path.join(result_root, 'restored_faces', save_face_name)
            imwrite(restored_face, save_restore_path)

        # save restored img
        if not args.has_aligned and restored_img is not None:
            if args.suffix is not None:
                basename = f'{basename}_{args.suffix}'
            save_restore_path = os.path.join(result_root, 'final_results', f'{basename}.png')
            imwrite(restored_img, save_restore_path)

    # save enhanced video
    if input_video:
        print('Video Saving...')
        # load images
        video_frames = []
        img_list = sorted(glob.glob(os.path.join(result_root, 'final_results', '*.[jp][pn]g')))
        for img_path in img_list:
            img = cv2.imread(img_path)
            video_frames.append(img)
        # write images to video
        height, width = video_frames[0].shape[:2]
        if args.suffix is not None:
            video_name = f'{video_name}_{args.suffix}.png'
        save_restore_path = os.path.join(result_root, f'{video_name}.mp4')
        vidwriter = VideoWriter(save_restore_path, height, width, fps, audio)
         
        for f in video_frames:
            vidwriter.write_frame(f)
        vidwriter.close()

    print(f'\nAll results are saved in {result_root}')

if __name__ == '__main__':
    main()