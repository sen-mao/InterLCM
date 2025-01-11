# Copyright 2024 Nankai University

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import torch

from diffusers.image_processor import PipelineImageInput
from diffusers.utils import (
    deprecate,
    replace_example_docstring,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers import DiffusionPipeline
        >>> import torch

        >>> pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
        >>> # To save GPU memory, torch.float16 can be used, but it may compromise image quality.
        >>> pipe.to(torch_device="cuda", torch_dtype=torch.float32)

        >>> prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"

        >>> # Can be set to 1~50 steps. LCM support fast inference even <= 4 steps. Recommend: 1~8 steps.
        >>> num_inference_steps = 4
        >>> images = pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=8.0).images
        >>> images[0].save("image.png")
        ```
"""

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
        scheduler,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

# diffusers 0.28.2
# if using other versions of diffusers, make slight adjustments to register_lcm_forward based on
# __call__ function of diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipline

def register_lcm_forward(model, spatial_encoder):
    def lcm_forward(self, spatial_encoder):
        # @torch.no_grad()
        @replace_example_docstring(EXAMPLE_DOC_STRING)
        def forward(
                prompt: Union[str, List[str]] = None,
                height: Optional[int] = None,
                width: Optional[int] = None,
                num_inference_steps: int = 4,
                original_inference_steps: int = None,
                timesteps: List[int] = None,
                guidance_scale: float = 8.5,
                num_images_per_prompt: Optional[int] = 1,
                generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
                latents: Optional[torch.Tensor] = None,
                prompt_embeds: Optional[torch.Tensor] = None,
                ip_adapter_image: Optional[PipelineImageInput] = None,
                ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
                output_type: Optional[str] = "pil",
                return_dict: bool = True,
                cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                clip_skip: Optional[int] = None,
                callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
                callback_on_step_end_tensor_inputs: List[str] = ["latents"],
                **kwargs,
        ):
            r"""
            The call function to the pipeline for generation.

            Args:
                prompt (`str` or `List[str]`, *optional*):
                    The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
                height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                    The height in pixels of the generated image.
                width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                    The width in pixels of the generated image.
                num_inference_steps (`int`, *optional*, defaults to 50):
                    The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                    expense of slower inference.
                original_inference_steps (`int`, *optional*):
                    The original number of inference steps use to generate a linearly-spaced timestep schedule, from which
                    we will draw `num_inference_steps` evenly spaced timesteps from as our final timestep schedule,
                    following the Skipping-Step method in the paper (see Section 4.3). If not set this will default to the
                    scheduler's `original_inference_steps` attribute.
                timesteps (`List[int]`, *optional*):
                    Custom timesteps to use for the denoising process. If not defined, equal spaced `num_inference_steps`
                    timesteps on the original LCM training/distillation timestep schedule are used. Must be in descending
                    order.
                guidance_scale (`float`, *optional*, defaults to 7.5):
                    A higher guidance scale value encourages the model to generate images closely linked to the text
                    `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
                    Note that the original latent consistency models paper uses a different CFG formulation where the
                    guidance scales are decreased by 1 (so in the paper formulation CFG is enabled when `guidance_scale >
                    0`).
                num_images_per_prompt (`int`, *optional*, defaults to 1):
                    The number of images to generate per prompt.
                generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                    A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                    generation deterministic.
                latents (`torch.Tensor`, *optional*):
                    Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                    generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                    tensor is generated by sampling using the supplied random `generator`.
                prompt_embeds (`torch.Tensor`, *optional*):
                    Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                    provided, text embeddings are generated from the `prompt` input argument.
                ip_adapter_image: (`PipelineImageInput`, *optional*):
                    Optional image input to work with IP Adapters.
                ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                    Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                    IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
                    contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
                    provided, embeddings are computed from the `ip_adapter_image` input argument.
                output_type (`str`, *optional*, defaults to `"pil"`):
                    The output format of the generated image. Choose between `PIL.Image` or `np.array`.
                return_dict (`bool`, *optional*, defaults to `True`):
                    Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                    plain tuple.
                cross_attention_kwargs (`dict`, *optional*):
                    A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                    [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
                clip_skip (`int`, *optional*):
                    Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                    the output of the pre-final layer will be used for computing the prompt embeddings.
                callback_on_step_end (`Callable`, *optional*):
                    A function that calls at the end of each denoising steps during the inference. The function is called
                    with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                    callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                    `callback_on_step_end_tensor_inputs`.
                callback_on_step_end_tensor_inputs (`List`, *optional*):
                    The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                    will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                    `._callback_tensor_inputs` attribute of your pipeline class.

            Examples:

            Returns:
                [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                    If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                    otherwise a `tuple` is returned where the first element is a list with the generated images and the
                    second element is a list of `bool`s indicating whether the corresponding generated image contains
                    "not-safe-for-work" (nsfw) content.
            """

            callback = kwargs.pop("callback", None)
            callback_steps = kwargs.pop("callback_steps", None)

            if callback is not None:
                deprecate(
                    "callback",
                    "1.0.0",
                    "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
                )
            if callback_steps is not None:
                deprecate(
                    "callback_steps",
                    "1.0.0",
                    "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
                )

            # 0. Default height and width to unet
            height = height or self.unet.config.sample_size * self.vae_scale_factor
            width = width or self.unet.config.sample_size * self.vae_scale_factor

            # 1. Check inputs. Raise error if not correct
            self.check_inputs(
                prompt,
                height,
                width,
                callback_steps,
                prompt_embeds,
                ip_adapter_image,
                ip_adapter_image_embeds,
                callback_on_step_end_tensor_inputs,
            )
            self._guidance_scale = guidance_scale
            self._clip_skip = clip_skip
            self._cross_attention_kwargs = cross_attention_kwargs

            # 2. Define call parameters
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            device = self._execution_device

            if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                image_embeds = self.prepare_ip_adapter_image_embeds(
                    ip_adapter_image,
                    ip_adapter_image_embeds,
                    device,
                    batch_size * num_images_per_prompt,
                    self.do_classifier_free_guidance,
                    )

            # 3. Encode input prompt
            lora_scale = (
                self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
            )

            # NOTE: when a LCM is distilled from an LDM via latent consistency distillation (Algorithm 1) with guided
            # distillation, the forward pass of the LCM learns to approximate sampling from the LDM using CFG with the
            # unconditional prompt "" (the empty string). Due to this, LCMs currently do not support negative prompts.
            if prompt is not None:
                prompt_embeds, _ = self.encode_prompt(
                    prompt,
                    device,
                    num_images_per_prompt,
                    self.do_classifier_free_guidance,
                    negative_prompt=None,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=None,
                    lora_scale=lora_scale,
                    clip_skip=self.clip_skip,
                )

            # 4. Prepare timesteps
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler, num_inference_steps, device, timesteps, original_inference_steps=original_inference_steps
            )

            # 5. Prepare latent variable
            num_channels_latents = self.unet.config.in_channels
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
                )
            bs = batch_size * num_images_per_prompt

            # 6. Get Guidance Scale Embedding
            # NOTE: We use the Imagen CFG formulation that StableDiffusionPipeline uses rather than the original LCM paper
            # CFG formulation, so we need to subtract 1 from the input guidance_scale.
            # LCM CFG formulation:  cfg_noise = noise_cond + cfg_scale * (noise_cond - noise_uncond), (cfg_scale > 0.0 using CFG)
            w = torch.tensor(self.guidance_scale - 1).repeat(bs)
            w_embedding = self.get_guidance_scale_embedding(w, embedding_dim=self.unet.config.time_cond_proj_dim).to(
                device=device, dtype=latents.dtype
            )

            # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, None)

            # 7.1 Add image embeds for IP-Adapter
            added_cond_kwargs = (
                {"image_embeds": image_embeds}
                if ip_adapter_image is not None or ip_adapter_image_embeds is not None
                else None
            )

            weight_dtype = torch.float32

            # 8. LCM MultiStep Sampling Loop:
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            self._num_timesteps = len(timesteps)
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    latents = latents.to(prompt_embeds.dtype)

                    if i == 0:  ## consider LQ as a result of the first step for LCM
                        model_pred = torch.zeros_like(latents)
                        denoised = latents
                    else:
                        # spatial_encoder
                        down_block_res_samples, mid_block_res_sample = spatial_encoder(
                            latents,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            controlnet_cond=kwargs['lq_input'],
                            return_dict=False,
                        )

                        # model prediction (v-prediction, eps, x)
                        model_pred = self.unet(
                            latents,
                            t,
                            timestep_cond=w_embedding,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=self.cross_attention_kwargs,
                            down_block_additional_residuals=[
                                sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                            ],
                            mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]
                        denoised = None

                    # compute the previous noisy sample x_t -> x_t-1
                    ## latents is x_t-1 = x_0 + noise, denoised is predicted x_0
                    latents, denoised = self.scheduler.step(model_pred, t, latents, **extra_step_kwargs, return_dict=False,
                                                            denoised=denoised)
                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        w_embedding = callback_outputs.pop("w_embedding", w_embedding)
                        denoised = callback_outputs.pop("denoised", denoised)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            step_idx = i // getattr(self.scheduler, "order", 1)
                            callback(step_idx, t, latents)

            denoised = denoised.to(prompt_embeds.dtype)
            if not output_type == "latent":
                image = self.vae.decode(denoised / self.vae.config.scaling_factor, return_dict=False)[0]
                # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
            else:
                image = denoised
                # has_nsfw_concept = None

            has_nsfw_concept = None

            # if has_nsfw_concept is None:
            #     do_denormalize = [True] * image.shape[0]
            # else:
            #     do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

            # image_pil = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

            # Offload all models
            # self.maybe_free_model_hooks()

            if not return_dict:
                return (image, has_nsfw_concept)

            return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
        return forward
    if model.__class__.__name__ == 'LatentConsistencyModelPipeline':
        model.forward = lcm_forward(model, spatial_encoder)


from typing import Tuple
from diffusers.schedulers.scheduling_lcm import LCMSchedulerOutput
from diffusers.utils.torch_utils import randn_tensor

def register_lcmschedule_step(model):
    def lcmschedule_step(self):
        def step(
            model_output: torch.Tensor,
            timestep: int,
            sample: torch.Tensor,
            generator: Optional[torch.Generator] = None,
            return_dict: bool = True,
            denoised: torch.Tensor = None
        ) -> Union[LCMSchedulerOutput, Tuple]:
            """
            Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
            process from the learned model outputs (most often the predicted noise).

            Args:
                model_output (`torch.Tensor`):
                    The direct output from learned diffusion model.
                timestep (`float`):
                    The current discrete timestep in the diffusion chain.
                sample (`torch.Tensor`):
                    A current instance of a sample created by the diffusion process.
                generator (`torch.Generator`, *optional*):
                    A random number generator.
                return_dict (`bool`, *optional*, defaults to `True`):
                    Whether or not to return a [`~schedulers.scheduling_lcm.LCMSchedulerOutput`] or `tuple`.
            Returns:
                [`~schedulers.scheduling_utils.LCMSchedulerOutput`] or `tuple`:
                    If return_dict is `True`, [`~schedulers.scheduling_lcm.LCMSchedulerOutput`] is returned, otherwise a
                    tuple is returned where the first element is the sample tensor.
            """
            if self.num_inference_steps is None:
                raise ValueError(
                    "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
                )

            if self.step_index is None:
                self._init_step_index(timestep)

            # 1. get previous step value
            prev_step_index = self.step_index + 1
            if prev_step_index < len(self.timesteps):
                prev_timestep = self.timesteps[prev_step_index]
            else:
                prev_timestep = timestep

            # 2. compute alphas, betas
            alpha_prod_t = self.alphas_cumprod[timestep]
            alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev

            if denoised is None:
                # 3. Get scalings for boundary conditions
                c_skip, c_out = self.get_scalings_for_boundary_condition_discrete(timestep)

                # 4. Compute the predicted original sample x_0 based on the model parameterization
                if self.config.prediction_type == "epsilon":  # noise-prediction
                    predicted_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
                elif self.config.prediction_type == "sample":  # x-prediction
                    predicted_original_sample = model_output
                elif self.config.prediction_type == "v_prediction":  # v-prediction
                    predicted_original_sample = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
                else:
                    raise ValueError(
                        f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                        " `v_prediction` for `LCMScheduler`."
                    )

                # 5. Clip or threshold "predicted x_0"
                if self.config.thresholding:
                    predicted_original_sample = self._threshold_sample(predicted_original_sample)
                elif self.config.clip_sample:
                    predicted_original_sample = predicted_original_sample.clamp(
                        -self.config.clip_sample_range, self.config.clip_sample_range
                    )

                # 6. Denoise model output using boundary conditions
                denoised = c_out * predicted_original_sample + c_skip * sample

            # 7. Sample and inject noise z ~ N(0, I) for MultiStep Inference
            # Noise is not used on the final timestep of the timestep schedule.
            # This also means that noise is not used for one-step sampling.
            if self.step_index != self.num_inference_steps - 1:
                noise = randn_tensor(
                    model_output.shape, generator=generator, device=model_output.device, dtype=denoised.dtype
                )
                prev_sample = alpha_prod_t_prev.sqrt() * denoised + beta_prod_t_prev.sqrt() * noise
            else:
                prev_sample = denoised

            # upon completion increase step index by one
            self._step_index += 1

            if not return_dict:
                return (prev_sample, denoised)

            return LCMSchedulerOutput(prev_sample=prev_sample, denoised=denoised)
        return step
    if model.__class__.__name__ == 'LCMScheduler':
         model.step = lcmschedule_step(model)