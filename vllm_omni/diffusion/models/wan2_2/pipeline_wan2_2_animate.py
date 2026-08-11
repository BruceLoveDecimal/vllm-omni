# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Wan2.2-Animate pipeline for vLLM-Omni (animation mode).

Drives a reference character image with a pre-processed pose video and face
video. Long videos are produced segment by segment, each segment conditioned on
the decoded tail frames of the previous one.

Pose and face videos are expected to be pre-processed already (the upstream
Wan-Animate repo ships ``preprocess_data.py`` for skeleton extraction and face
retargeting); this pipeline does not run those models.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any, ClassVar

import numpy as np
import PIL.Image
import torch
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.forward_context import DenoiseProgressMixin
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch, prefetch_subfolders
from vllm_omni.diffusion.models.interface import (
    ReferenceVideoDecodeSpec,
    SupportImageInput,
    SupportsComponentDiscovery,
)
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.models.schedulers import FlowUniPCMultistepScheduler
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import (
    get_wan22_post_process_func as get_wan22_animate_post_process_func,  # noqa: F401
)
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import (
    load_transformer_config,
    retrieve_latents,
)
from vllm_omni.diffusion.models.wan2_2.wan2_2_animate_transformer import (
    create_animate_transformer_from_config,
)
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniTextPrompt
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

# Shared Wan negative prompt (Wan2.2 shared_config.sample_neg_prompt).
_ANIMATE_DEFAULT_NEG_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
    "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
    "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


# ---------------------------------------------------------------------------
# Media helpers
# ---------------------------------------------------------------------------


def resize_and_fill(image: PIL.Image.Image, height: int, width: int) -> PIL.Image.Image:
    """Scale to fit while preserving aspect ratio, then center on a black canvas."""
    ratio = width / height
    src_ratio = image.width / image.height

    src_w = width if ratio < src_ratio else image.width * height // image.height
    src_h = height if ratio >= src_ratio else image.height * width // image.width

    resized = image.resize((src_w, src_h), resample=PIL.Image.Resampling.BILINEAR)
    canvas = PIL.Image.new("RGB", (width, height), color=0)
    canvas.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
    return canvas


def pad_video_frames(frames: list[Any], num_target_frames: int) -> list[Any]:
    """Extend ``frames`` to ``num_target_frames`` by bouncing back and forth.

    ``pad_video_frames([1, 2, 3, 4, 5], 10) -> [1, 2, 3, 4, 5, 4, 3, 2, 1, 2]``
    """
    if len(frames) == 1:
        return [frames[0]] * num_target_frames

    idx = 0
    flip = False
    padded: list[Any] = []
    while len(padded) < num_target_frames:
        padded.append(frames[idx])
        idx += -1 if flip else 1
        if idx == 0 or idx == len(frames) - 1:
            flip = not flip
    return padded


def load_video_frames(video: str | list) -> list[PIL.Image.Image]:
    """Normalize a video path/URL or a frame sequence into a list of PIL images."""
    if isinstance(video, str):
        from diffusers.utils import load_video

        return load_video(video)
    if isinstance(video, (list, tuple)):
        from diffusers.utils import load_video

        frames: list[PIL.Image.Image] = []
        for item in video:
            if not isinstance(item, str):
                frames.append(item.convert("RGB"))
                continue
            # A string entry is either one frame or a whole video file: online
            # serving hands us ``[path_to.mp4]`` when it defers decoding to the
            # worker, while offline callers pass a list of frame images.
            try:
                frames.append(PIL.Image.open(item).convert("RGB"))
            except PIL.UnidentifiedImageError:
                frames.extend(frame.convert("RGB") for frame in load_video(item))
        return frames
    raise TypeError(f"Unsupported video input type {type(video)}; expected a path or a list of frames.")


def preprocess_video(frames: list[PIL.Image.Image], height: int, width: int) -> torch.Tensor:
    """Resize frames and stack them into a ``[1, 3, T, H, W]`` tensor in ``[-1, 1]``."""
    arrays = []
    for frame in frames:
        if frame.size != (width, height):
            frame = frame.resize((width, height), PIL.Image.Resampling.LANCZOS)
        arrays.append(np.asarray(frame.convert("RGB"), dtype=np.float32) / 127.5 - 1.0)
    video = torch.from_numpy(np.stack(arrays))  # [T, H, W, 3]
    return video.permute(3, 0, 1, 2).unsqueeze(0)


# ---------------------------------------------------------------------------
# Pre-process
# ---------------------------------------------------------------------------


def get_wan22_animate_pre_process_func(od_config: OmniDiffusionConfig):
    """Pre-process for Wan-Animate: validate inputs and fit the reference image.

    Expects ``multi_modal_data`` to contain:
      - ``"image"``: reference character image (PIL.Image or path)
      - ``"pose_video"``: pre-processed skeleton video (path or frame list);
        online serving delivers this as ``"video"``
      - ``"face_video"``: pre-processed face video (path or frame list); online
        serving delivers this via ``sampling_params.extra_args``

    Frame decoding is deferred to the worker so full-resolution videos are not
    carried through the request.
    """

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        prompt = request.prompt
        multi_modal_data = prompt.get("multi_modal_data", {}) if not isinstance(prompt, str) else {}
        if isinstance(prompt, str):
            prompt = OmniTextPrompt(prompt=prompt)
        if "additional_information" not in prompt:
            prompt["additional_information"] = {}
        extra_args = getattr(request.sampling_params, "extra_args", None) or {}

        raw_image = multi_modal_data.get("image")
        if raw_image is None:
            raise ValueError(
                "No reference image provided. Wan-Animate requires a character image. "
                'Set `"multi_modal_data": {"image": <path or PIL.Image>, ...}`'
            )
        image = PIL.Image.open(raw_image).convert("RGB") if isinstance(raw_image, str) else raw_image.convert("RGB")

        pose_video = multi_modal_data.get("pose_video") or multi_modal_data.get("video")
        if pose_video is None:
            raise ValueError(
                "No pose video provided. Wan-Animate requires a pre-processed skeleton video. "
                'Set `"multi_modal_data": {"pose_video": <path or frame list>, ...}`'
            )

        face_video = multi_modal_data.get("face_video") or extra_args.get("face_video")
        if face_video is None:
            raise ValueError(
                "No face video provided. Wan-Animate requires a pre-processed face video. "
                'Set `"multi_modal_data": {"face_video": <path or frame list>, ...}`'
            )

        # Wan-Animate patchifies 2x2 on top of the 8x VAE downsample.
        mod_value = 16
        if request.sampling_params.height is None or request.sampling_params.width is None:
            max_area = 720 * 1280
            aspect_ratio = image.height / image.width
            height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
            width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
            if request.sampling_params.height is None:
                request.sampling_params.height = height
            if request.sampling_params.width is None:
                request.sampling_params.width = width

        height, width = request.sampling_params.height, request.sampling_params.width
        if height % mod_value != 0 or width % mod_value != 0:
            raise ValueError(f"`height` and `width` must be divisible by {mod_value}, got {height} and {width}.")

        prompt["multi_modal_data"]["image"] = resize_and_fill(image, height, width)
        prompt["additional_information"]["pose_video"] = pose_video
        prompt["additional_information"]["face_video"] = face_video

        request.prompt = prompt
        return request

    return pre_process_func


def _is_diffusers_format(model_path: str) -> bool:
    """Check whether a model directory uses the diffusers subfolder layout."""
    return os.path.isdir(os.path.join(model_path, "tokenizer")) and os.path.isdir(os.path.join(model_path, "vae"))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Wan22AnimatePipeline(
    nn.Module,
    SupportImageInput,
    CFGParallelMixin,
    DenoiseProgressMixin,
    ProgressBarMixin,
    DiffusionPipelineProfilerMixin,
    SupportsComponentDiscovery,
):
    """Wan2.2-Animate pipeline (animation mode).

    Each segment packs three things into the transformer input: noisy latents,
    a 4-channel conditioning mask, and the conditioning latents (reference image
    in the leading frame slot, previous-segment tail frames after it). Pose
    latents are added after patch embedding and the face video drives the
    motion branch.
    """

    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder", "image_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]

    _DEFAULT_SEGMENT_FRAMES = 77
    _DEFAULT_PREV_FRAMES = 1
    _DEFAULT_FPS = 30
    # Animate takes a reference image plus two videos; the warmup run cannot
    # synthesize those, so skip frame-shaped dummy input.
    dummy_run_num_frames: ClassVar[int] = 0

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)

        model_path = od_config.model
        local_files_only = os.path.exists(model_path)
        if local_files_only and not _is_diffusers_format(model_path):
            raise ValueError(
                f"{model_path} is not a diffusers-format Wan-Animate checkpoint. Use the "
                f"`Wan-AI/Wan2.2-Animate-14B-Diffusers` layout (tokenizer/, vae/, transformer/, ...)."
            )

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model_path,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
        ]

        # See ``hub_prefetch.py`` for the transformers v5 subfolder race.
        component_subfolders = ["tokenizer", "text_encoder", "image_encoder", "image_processor", "vae"]
        prefetch_subfolders(model_path, component_subfolders, local_files_only=local_files_only)

        self.tokenizer = from_pretrained_with_prefetch(
            AutoTokenizer.from_pretrained,
            model_path,
            subfolder="tokenizer",
            prefetch_list=component_subfolders,
            local_files_only=local_files_only,
        )
        self.text_encoder = from_pretrained_with_prefetch(
            UMT5EncoderModel.from_pretrained,
            model_path,
            subfolder="text_encoder",
            prefetch_list=component_subfolders,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        ).to(self.device)

        self.image_processor = from_pretrained_with_prefetch(
            CLIPImageProcessor.from_pretrained,
            model_path,
            subfolder="image_processor",
            prefetch_list=component_subfolders,
            local_files_only=local_files_only,
        )
        self.image_encoder = from_pretrained_with_prefetch(
            CLIPVisionModel.from_pretrained,
            model_path,
            subfolder="image_encoder",
            prefetch_list=component_subfolders,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        ).to(self.device)

        self.vae = from_pretrained_with_prefetch(
            DistributedAutoencoderKLWan.from_pretrained,
            model_path,
            subfolder="vae",
            prefetch_list=component_subfolders,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        ).to(self.device)

        transformer_config = load_transformer_config(model_path, "transformer", local_files_only)
        quant_config = getattr(od_config, "quantization_config", None)
        self.transformer = create_animate_transformer_from_config(transformer_config, quant_config=quant_config)
        self.motion_encoder_size = transformer_config.get("motion_encoder_size", 512)

        flow_shift = od_config.flow_shift if od_config.flow_shift is not None else 5.0
        self.scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=flow_shift,
            prediction_type="flow_prediction",
        )

        self.vae_scale_factor_temporal = getattr(self.vae.config, "scale_factor_temporal", 4)
        self.vae_scale_factor_spatial = getattr(self.vae.config, "scale_factor_spatial", 8)
        self.resolution_divisor = self.vae_scale_factor_spatial * 2

        self._guidance_scale = None
        self._num_timesteps = None
        self._current_timestep = None
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    @classmethod
    def reference_video_decode_spec(
        cls,
        *,
        num_frames: int | None = None,
        extra_args: dict[str, Any] | None = None,
    ) -> ReferenceVideoDecodeSpec:
        """Decode the pose video from its start, up to the requested frame count."""
        return ReferenceVideoDecodeSpec(max_frames=num_frames, keep="first")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale is not None and self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    # ------------------------------------------------------------------
    # Latent normalization
    # ------------------------------------------------------------------

    def _latent_stats(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(reference.device, reference.dtype)
        )
        inv_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            reference.device, reference.dtype
        )
        return mean, inv_std

    def _normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        mean, inv_std = self._latent_stats(latents)
        return (latents - mean) * inv_std

    def _denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        mean, inv_std = self._latent_stats(latents)
        return latents / inv_std + mean

    def _encode_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        """VAE-encode pixels deterministically and standardize the result."""
        pixels = pixels.to(device=self.device, dtype=self.vae.dtype)
        latents = retrieve_latents(self.vae.encode(pixels), sample_mode="argmax")
        return self._normalize_latents(latents)

    # ------------------------------------------------------------------
    # Encoders
    # ------------------------------------------------------------------

    @staticmethod
    def _prompt_clean(text: str) -> str:
        return " ".join(text.strip().split())

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        max_sequence_length: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode prompts with the umT5 text encoder."""
        device = device or self.device
        dtype = dtype or self.text_encoder.dtype

        def _encode(texts: list[str]) -> torch.Tensor:
            inputs = self.tokenizer(
                [self._prompt_clean(t) for t in texts],
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            ids, mask = inputs.input_ids, inputs.attention_mask
            seq_lens = mask.gt(0).sum(dim=1).long()
            embeds = self.text_encoder(ids.to(device), mask.to(device)).last_hidden_state.to(dtype=dtype, device=device)
            trimmed = [u[:v] for u, v in zip(embeds, seq_lens)]
            return torch.stack(
                [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in trimmed], dim=0
            )

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_embeds = _encode(prompt)

        negative_prompt_embeds = None
        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt = [negative_prompt] * len(prompt) if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_embeds = _encode(negative_prompt)

        return prompt_embeds, negative_prompt_embeds

    def encode_image(self, image: PIL.Image.Image, device: torch.device | None = None) -> torch.Tensor:
        """Extract CLIP visual features of the reference image."""
        device = device or self.device
        pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device=device, dtype=self.image_encoder.dtype)
        return self.image_encoder(pixel_values, output_hidden_states=True).hidden_states[-2]

    # ------------------------------------------------------------------
    # Latent preparation
    # ------------------------------------------------------------------

    def get_i2v_mask(
        self,
        latent_frames: int,
        latent_height: int,
        latent_width: int,
        mask_len: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build the 4-channel conditioning mask marking which frames are given.

        The pixel-space mask is folded into the channel dimension the same way
        the Wan VAE folds 4 pixel frames into 1 latent frame.
        """
        pixel_frames = (latent_frames - 1) * self.vae_scale_factor_temporal + 1
        mask = torch.zeros(1, 1, pixel_frames, latent_height, latent_width, dtype=dtype, device=self.device)
        mask[:, :, :mask_len] = 1

        first_frame = torch.repeat_interleave(mask[:, :, 0:1], dim=2, repeats=self.vae_scale_factor_temporal)
        mask = torch.cat([first_frame, mask[:, :, 1:]], dim=2)
        mask = mask.view(1, -1, self.vae_scale_factor_temporal, latent_height, latent_width)
        return mask.transpose(1, 2)

    def prepare_reference_image_latents(self, image_pixels: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Encode the reference image into a fully-conditioned latent frame."""
        if image_pixels.ndim == 4:
            image_pixels = image_pixels.unsqueeze(2)
        latents = self._encode_pixels(image_pixels)
        latent_height, latent_width = latents.shape[-2:]
        mask = self.get_i2v_mask(1, latent_height, latent_width, mask_len=1, dtype=latents.dtype)
        return torch.cat([mask, latents], dim=1).to(dtype)

    def prepare_prev_segment_cond_latents(
        self,
        prev_segment_pixels: torch.Tensor | None,
        segment_frame_length: int,
        height: int,
        width: int,
        prev_segment_cond_frames: int,
        is_first_segment: bool,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Encode the temporal hand-off from the previous segment.

        The leading ``prev_segment_cond_frames`` carry the previous segment's
        decoded tail (zeros for the first segment); the rest is zero-filled and
        left for the model to generate.
        """
        vae_dtype = self.vae.dtype
        if prev_segment_pixels is None:
            prev_segment_pixels = torch.zeros(
                1, 3, prev_segment_cond_frames, height, width, dtype=vae_dtype, device=self.device
            )

        remaining = torch.zeros(
            1,
            3,
            segment_frame_length - prev_segment_cond_frames,
            height,
            width,
            dtype=vae_dtype,
            device=self.device,
        )
        segment = torch.cat([prev_segment_pixels.to(dtype=vae_dtype, device=self.device), remaining], dim=2)

        latents = self._encode_pixels(segment)
        latent_frames = (segment_frame_length - 1) // self.vae_scale_factor_temporal + 1
        mask = self.get_i2v_mask(
            latent_frames,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
            mask_len=0 if is_first_segment else prev_segment_cond_frames,
            dtype=latents.dtype,
        )
        return torch.cat([mask, latents], dim=1).to(dtype)

    def prepare_latents(
        self,
        segment_frame_length: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample noise for one segment, with one extra frame for the reference slot."""
        latent_frames = (segment_frame_length - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            1,
            self.vae.config.z_dim,
            latent_frames + 1,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        return randn_tensor(shape, generator=generator, device=self.device, dtype=dtype)

    # ------------------------------------------------------------------
    # Denoising
    # ------------------------------------------------------------------

    def predict_noise(self, current_model: nn.Module | None = None, **kwargs: Any) -> torch.Tensor:
        if current_model is None:
            current_model = self.transformer
        param_dtype = next(current_model.parameters()).dtype
        with torch.amp.autocast(self.device.type, dtype=param_dtype):
            result = current_model(**kwargs)
        if isinstance(result, tuple):
            return result[0]
        return result.sample

    def diffuse(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        reference_latents: torch.Tensor,
        pose_latents: torch.Tensor,
        motion_vec: torch.Tensor,
        negative_motion_vec: torch.Tensor | None,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        image_embeds: torch.Tensor,
        guidance_scale: float,
        transformer_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Denoise one segment."""
        do_true_cfg = self.do_classifier_free_guidance and negative_prompt_embeds is not None

        with self.progress_bar(total=len(timesteps)) as pbar:
            for step_idx, t in enumerate(timesteps):
                self._current_timestep = t
                self.record_denoise_step(step_idx, t)

                latent_model_input = torch.cat([latents, reference_latents], dim=1).to(transformer_dtype)
                timestep = t.expand(latents.shape[0])

                positive_kwargs = {
                    "hidden_states": latent_model_input,
                    "timestep": timestep,
                    "encoder_hidden_states": prompt_embeds,
                    "encoder_hidden_states_image": image_embeds,
                    "pose_hidden_states": pose_latents,
                    "motion_vec": motion_vec,
                    "return_dict": False,
                }
                negative_kwargs = None
                if do_true_cfg:
                    negative_kwargs = {
                        **positive_kwargs,
                        "encoder_hidden_states": negative_prompt_embeds,
                        "motion_vec": negative_motion_vec,
                    }

                noise_pred = self.predict_noise_maybe_with_cfg(
                    do_true_cfg=do_true_cfg,
                    true_cfg_scale=guidance_scale,
                    positive_kwargs=positive_kwargs,
                    negative_kwargs=negative_kwargs,
                    cfg_normalize=False,
                )

                latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, do_true_cfg)
                pbar.update()

        self._current_timestep = None
        return latents

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        """Generate the animated video, one segment at a time."""
        if len(req.prompts) != 1:
            raise ValueError("Wan-Animate supports a single prompt per request.")

        prompt_data = req.prompts[0]
        if isinstance(prompt_data, str):
            raise ValueError("Wan-Animate requires multi-modal input; got a bare text prompt.")

        prompt = prompt_data.get("prompt")
        negative_prompt = prompt_data.get("negative_prompt") or _ANIMATE_DEFAULT_NEG_PROMPT
        multi_modal_data = prompt_data.get("multi_modal_data", {})
        additional_info = prompt_data.get("additional_information", {})

        image = multi_modal_data.get("image")
        if isinstance(image, str):
            image = PIL.Image.open(image).convert("RGB")
        if image is None:
            raise ValueError("Wan-Animate requires a reference character image.")

        pose_video = additional_info.get("pose_video") or multi_modal_data.get("pose_video")
        face_video = additional_info.get("face_video") or multi_modal_data.get("face_video")
        if pose_video is None or face_video is None:
            raise ValueError("Wan-Animate requires both a pose video and a face video.")

        sampling = req.sampling_params
        height = sampling.height or 720
        width = sampling.width or 1280
        if height % self.resolution_divisor != 0 or width % self.resolution_divisor != 0:
            raise ValueError(
                f"`height` and `width` must be divisible by {self.resolution_divisor}, got {height} and {width}."
            )

        extra_args = getattr(sampling, "extra_args", None) or {}
        segment_frame_length = int(extra_args.get("segment_frame_length", self._DEFAULT_SEGMENT_FRAMES))
        if segment_frame_length % self.vae_scale_factor_temporal != 1:
            segment_frame_length = (
                segment_frame_length // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
            )
            logger.warning(
                "segment_frame_length must satisfy (n * %d + 1); rounded to %d.",
                self.vae_scale_factor_temporal,
                segment_frame_length,
            )
        segment_frame_length = max(segment_frame_length, 1)

        prev_frames = int(extra_args.get("prev_segment_conditioning_frames", self._DEFAULT_PREV_FRAMES))
        if prev_frames not in (1, 5):
            raise ValueError(f"`prev_segment_conditioning_frames` must be 1 or 5, got {prev_frames}.")

        num_steps = sampling.num_inference_steps or 20
        guidance_scale = sampling.guidance_scale if sampling.guidance_scale_provided else 1.0
        self._guidance_scale = guidance_scale

        generator = sampling.generator
        seed = sampling.seed
        if generator is None and seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        device = self.device
        transformer_dtype = self.transformer.dtype

        # ---- 1. Load and pad the driving videos ----
        pose_frames = load_video_frames(pose_video)
        face_frames = load_video_frames(face_video)
        requested_frames = sampling.num_frames if sampling.num_frames and sampling.num_frames > 1 else None
        cond_video_frames = min(len(pose_frames), len(face_frames))
        if requested_frames is not None:
            cond_video_frames = min(cond_video_frames, requested_frames)
        if cond_video_frames < 1:
            raise ValueError("The pose and face videos must contain at least one frame.")
        pose_frames = pose_frames[:cond_video_frames]
        face_frames = face_frames[:cond_video_frames]

        effective_segment_length = segment_frame_length - prev_frames
        trailing = (cond_video_frames - prev_frames) % effective_segment_length
        num_padding_frames = 0 if trailing == 0 else effective_segment_length - trailing
        num_target_frames = cond_video_frames + num_padding_frames
        num_segments = max(1, num_target_frames // effective_segment_length)

        pose_pixels = preprocess_video(pad_video_frames(pose_frames, num_target_frames), height, width).to(device)
        face_pixels = preprocess_video(
            pad_video_frames(face_frames, num_target_frames), self.motion_encoder_size, self.motion_encoder_size
        ).to(device)

        # ---- 2. Text and image conditioning ----
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            device=device,
            dtype=transformer_dtype,
        )
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        image_embeds = self.encode_image(image, device).to(transformer_dtype)

        # ---- 3. Reference image latents (shared by every segment) ----
        # Idempotent when pre-processing already fitted the image, but forward
        # may also be called directly.
        image_pixels = preprocess_video([resize_and_fill(image, height, width)], height, width).to(device)
        reference_image_latents = self.prepare_reference_image_latents(image_pixels, transformer_dtype)

        # ---- 4. Segment loop ----
        clips: list[torch.Tensor] = []
        prev_segment_pixels: torch.Tensor | None = None
        start = 0

        for segment_idx in range(num_segments):
            end = start + segment_frame_length

            latents = self.prepare_latents(
                segment_frame_length, height, width, dtype=torch.float32, generator=generator
            )

            pose_latents = self._encode_pixels(pose_pixels[:, :, start:end]).to(transformer_dtype)
            face_segment = face_pixels[:, :, start:end].to(transformer_dtype)

            prev_cond_latents = self.prepare_prev_segment_cond_latents(
                prev_segment_pixels,
                segment_frame_length=segment_frame_length,
                height=height,
                width=width,
                prev_segment_cond_frames=prev_frames,
                is_first_segment=segment_idx == 0,
                dtype=transformer_dtype,
            )
            reference_latents = torch.cat([reference_image_latents, prev_cond_latents], dim=2)

            # The motion signal is timestep-independent, so encode it once here
            # rather than at every denoising step.
            param_dtype = next(self.transformer.parameters()).dtype
            with torch.amp.autocast(device.type, dtype=param_dtype):
                motion_vec = self.transformer.encode_face(face_segment)
                negative_motion_vec = None
                if self.do_classifier_free_guidance and negative_prompt_embeds is not None:
                    # CFG drops the face signal by feeding a blank (all -1) face video.
                    negative_motion_vec = self.transformer.encode_face(face_segment * 0 - 1)

            self.scheduler.set_timesteps(num_steps, device=device)
            timesteps = self.scheduler.timesteps
            self._num_timesteps = len(timesteps)

            latents = self.diffuse(
                latents=latents,
                timesteps=timesteps,
                reference_latents=reference_latents,
                pose_latents=pose_latents,
                motion_vec=motion_vec,
                negative_motion_vec=negative_motion_vec,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                image_embeds=image_embeds,
                guidance_scale=guidance_scale,
                transformer_dtype=transformer_dtype,
            )

            # Drop the reference frame slot before decoding.
            decode_latents = self._denormalize_latents(latents[:, :, 1:]).to(self.vae.dtype)
            frames = self.vae.decode(decode_latents, return_dict=False)[0]
            frames = self._broadcast_if_vae_parallel(frames, segment_frame_length, height, width)

            prev_segment_pixels = frames[:, :, -prev_frames:].clone()
            if segment_idx > 0:
                frames = frames[:, :, prev_frames:]
            clips.append(frames.cpu())

            start += effective_segment_length
            if current_omni_platform.is_available():
                current_omni_platform.empty_cache()

        video = torch.cat(clips, dim=2)[:, :, :cond_video_frames]
        return DiffusionOutput(
            output=video,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    def _broadcast_if_vae_parallel(
        self, frames: torch.Tensor, num_frames: int, height: int, width: int
    ) -> torch.Tensor:
        """Share decoded frames with non-zero ranks under VAE patch parallelism.

        The segment loop needs the decoded tail on every rank to build the next
        segment's conditioning, but patch-parallel decode only returns it on
        rank 0.
        """
        if frames.numel() > 0:
            return frames

        import torch.distributed as dist

        buffer = torch.empty((1, 3, num_frames, height, width), device=self.device, dtype=self.vae.dtype)
        vae_pp_group = getattr(self.vae, "_vae_pp_group", None)
        if vae_pp_group is not None:
            dist.broadcast(buffer, src=0, group=vae_pp_group)
        return buffer

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights)
