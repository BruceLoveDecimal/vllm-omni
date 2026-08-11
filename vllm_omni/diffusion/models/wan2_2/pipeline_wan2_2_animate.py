# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Wan-Animate pipeline for character animation and replacement.

Like VACE, the mode is determined by which inputs are provided rather than by an
explicit flag:

- **animate**: reference character image + pose video + face video. The model
  animates the reference character following the driving motion, generating the
  background itself.
- **replace**: the above plus a background video and a mask video. The model
  swaps the person in the background video for the reference character, keeping
  the original background and lighting.

Both modes share one set of weights and one transformer; the extra ``video`` /
``mask`` conditioning is what selects the replacement behaviour.

Long videos are produced segment by segment: each segment is conditioned on the
last ``prev_segment_conditioning_frames`` decoded frames of the previous one,
which is the same temporal chaining S2V uses for its motion frames.

Ported from ``diffusers.pipelines.wan.pipeline_wan_animate``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from typing import Any, ClassVar

import PIL.Image
import torch
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.forward_context import set_forward_context_denoise_step_idx
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch, prefetch_subfolders
from vllm_omni.diffusion.models.interface import SupportImageInput, SupportsComponentDiscovery
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import (
    build_wan_scheduler,
    load_transformer_config,
    retrieve_latents,
)
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import (
    get_wan22_post_process_func as get_wan22_animate_post_process_func,  # noqa: F401
)
from vllm_omni.diffusion.models.wan2_2.wan2_2_animate_transformer import create_animate_transformer_from_config
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniTextPrompt

logger = logging.getLogger(__name__)

_ANIMATE_DEFAULT_NEG_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，"
    "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)

# Frames per generated segment, and how many frames of the previous segment are
# re-used as temporal guidance. Both should be of the form 4N + 1.
_DEFAULT_SEGMENT_FRAME_LENGTH = 77
_DEFAULT_PREV_SEGMENT_COND_FRAMES = 1


def _prompt_clean(text: str) -> str:
    # Matches the other Wan pipelines here. Diffusers additionally runs ftfy over
    # the prompt, but ftfy is not a vllm-omni dependency and only changes
    # mojibake input, which the tokenizer handles the same way either way.
    return " ".join(text.strip().split())


def _load_video_frames(source: Any, name: str, mode: str = "RGB") -> list[PIL.Image.Image] | None:
    """Normalize a video-ish input into a list of PIL frames in ``mode``.

    Accepts a path/URL to a video file, a list of frame paths, a list of PIL
    images, a single PIL image, or ``None``. Masks pass ``mode="L"``.
    """
    if source is None:
        return None
    if isinstance(source, str):
        from diffusers.utils import load_video

        return [frame.convert(mode) for frame in load_video(source)]
    if isinstance(source, PIL.Image.Image):
        return [source.convert(mode)]
    if isinstance(source, (list, tuple)):
        if len(source) == 0:
            raise ValueError(f"`{name}` was provided as an empty sequence.")
        frames = []
        for frame in source:
            if isinstance(frame, str):
                frames.append(PIL.Image.open(frame).convert(mode))
            elif isinstance(frame, PIL.Image.Image):
                frames.append(frame.convert(mode))
            else:
                raise TypeError(f"Unsupported frame type {type(frame)} in `{name}`.")
        return frames
    raise TypeError(f"Unsupported type {type(source)} for `{name}`.")


def _load_mask_frames(source: Any) -> list[PIL.Image.Image] | None:
    """Load a character mask video as single-channel frames."""
    return _load_video_frames(source, "mask", mode="L")


def get_wan22_animate_pre_process_func(od_config: OmniDiffusionConfig):
    """Pre-process for Wan-Animate: resolve conditioning media and target size.

    Reads from ``multi_modal_data``:

    - ``"image"``: reference character image (required)
    - ``"pose_video"``: pose/skeleton driving video (required)
    - ``"face_video"``: face crop driving video (required)
    - ``"video"``: background video (replacement mode only)
    - ``"mask"``: character mask video (replacement mode only)

    The pose and face videos are expected to be pre-extracted (e.g. by the
    Wan-Animate ``preprocess_data.py`` tooling); this pipeline does not run pose
    estimation or face retargeting itself.
    """

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        prompt = request.prompt
        multi_modal_data = prompt.get("multi_modal_data", {}) if not isinstance(prompt, str) else None
        if isinstance(prompt, str):
            prompt = OmniTextPrompt(prompt=prompt)
        if "additional_information" not in prompt:
            prompt["additional_information"] = {}
        if multi_modal_data is None:
            multi_modal_data = {}

        raw_image = multi_modal_data.get("image")
        if raw_image is None:
            raise ValueError(
                "No reference image provided. Wan-Animate requires a character image. "
                'Set `"multi_modal_data": {"image": <path or PIL.Image>, ...}`'
            )
        if isinstance(raw_image, str):
            image = PIL.Image.open(raw_image).convert("RGB")
        elif isinstance(raw_image, PIL.Image.Image):
            image = raw_image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type {type(raw_image)}")

        pose_video = _load_video_frames(multi_modal_data.get("pose_video"), "pose_video")
        face_video = _load_video_frames(multi_modal_data.get("face_video"), "face_video")
        if pose_video is None or face_video is None:
            raise ValueError(
                "Wan-Animate requires both a pose video and a face video. Set "
                '`"multi_modal_data": {"pose_video": <path or frames>, "face_video": <path or frames>, ...}`'
            )

        background_video = _load_video_frames(multi_modal_data.get("video"), "video")
        mask_video = _load_mask_frames(multi_modal_data.get("mask"))
        if (background_video is None) != (mask_video is None):
            raise ValueError(
                "Replacement mode needs both a background video (`video`) and a character mask (`mask`); "
                "only one of them was provided. Omit both to run animation mode."
            )

        # Default the output size to the pose video's, rounded to the model's grid.
        if request.sampling_params.height is None or request.sampling_params.width is None:
            pose_width, pose_height = pose_video[0].size
            mod_value = 16  # vae spatial scale (8) x transformer spatial patch size (2)
            if request.sampling_params.height is None:
                request.sampling_params.height = max(pose_height // mod_value * mod_value, mod_value)
            if request.sampling_params.width is None:
                request.sampling_params.width = max(pose_width // mod_value * mod_value, mod_value)

        prompt["multi_modal_data"]["image"] = image
        prompt["additional_information"]["pose_video"] = pose_video
        prompt["additional_information"]["face_video"] = face_video
        prompt["additional_information"]["background_video"] = background_video
        prompt["additional_information"]["mask_video"] = mask_video
        request.prompt = prompt

        return request

    return pre_process_func


class Wan22AnimatePipeline(
    nn.Module,
    SupportImageInput,
    CFGParallelMixin,
    ProgressBarMixin,
    DiffusionPipelineProfilerMixin,
    SupportsComponentDiscovery,
):
    """Wan2.2-Animate character animation / replacement pipeline."""

    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder", "image_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]

    # Wan-Animate cannot run the generic warmup: it needs a real pose and face
    # video, which the dummy request does not carry.
    dummy_run_num_frames: ClassVar[int] = 0

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config

        # Fail at load time rather than on the first request -- see the
        # parallelism notes on WanAnimateTransformer3DModel.
        parallel_config = getattr(od_config, "parallel_config", None)
        if parallel_config is not None and (parallel_config.sequence_parallel_size or 1) > 1:
            raise NotImplementedError(
                "Wan-Animate does not support sequence parallelism yet. The face adapter cross-attends per latent "
                "frame and requires frame-aligned token shards, which the generic token-axis sharding does not "
                "guarantee. Use tensor parallelism and/or CFG parallelism instead."
            )

        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)

        model = od_config.model
        local_files_only = os.path.exists(model)

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
        ]

        subfolders = ["tokenizer", "text_encoder", "vae", "image_processor", "image_encoder"]
        prefetch_subfolders(model, subfolders, local_files_only=local_files_only)

        self.tokenizer = from_pretrained_with_prefetch(
            AutoTokenizer.from_pretrained,
            model,
            subfolder="tokenizer",
            prefetch_list=subfolders,
            local_files_only=local_files_only,
        )
        self.text_encoder = from_pretrained_with_prefetch(
            UMT5EncoderModel.from_pretrained,
            model,
            subfolder="text_encoder",
            prefetch_list=subfolders,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        ).to(self.device)

        self.image_processor = from_pretrained_with_prefetch(
            CLIPImageProcessor.from_pretrained,
            model,
            subfolder="image_processor",
            prefetch_list=subfolders,
            local_files_only=local_files_only,
        )
        self.image_encoder = from_pretrained_with_prefetch(
            CLIPVisionModel.from_pretrained,
            model,
            subfolder="image_encoder",
            prefetch_list=subfolders,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        ).to(self.device)

        self.vae = from_pretrained_with_prefetch(
            DistributedAutoencoderKLWan.from_pretrained,
            model,
            subfolder="vae",
            prefetch_list=subfolders,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        ).to(self.device)

        transformer_config = load_transformer_config(model, "transformer", local_files_only)
        self.transformer = create_animate_transformer_from_config(
            transformer_config,
            quant_config=getattr(od_config, "quantization_config", None),
        )

        self._flow_shift = od_config.flow_shift if od_config.flow_shift is not None else 5.0
        self.scheduler = build_wan_scheduler("unipc", self._flow_shift)

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if hasattr(self.vae, "config") else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if hasattr(self.vae, "config") else 8

        from diffusers.video_processor import VideoProcessor

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        self.video_processor_for_mask = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial,
            do_normalize=False,
            do_convert_grayscale=True,
        )
        self.vae_image_processor = self._build_reference_image_processor()

        self._guidance_scale = None
        self._num_timesteps = None
        self._current_timestep = None
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    def _build_reference_image_processor(self):
        """Build the "fill"-resize processor Wan-Animate uses for the character image."""
        try:
            from diffusers.pipelines.wan.image_processor import WanAnimateImageProcessor
        except ImportError as exc:  # pragma: no cover - depends on installed diffusers
            raise ImportError(
                "Wan-Animate needs `WanAnimateImageProcessor` from "
                "`diffusers.pipelines.wan.image_processor`, which requires diffusers >= 0.38.0 "
                "(the version pinned in requirements/common.txt)."
            ) from exc

        spatial_patch_size = tuple(self.transformer.config.patch_size[-2:])
        return WanAnimateImageProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial,
            spatial_patch_size=spatial_patch_size,
            resample="bilinear",
            fill_color=0,
        )

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
    # VAE latent normalization
    # ------------------------------------------------------------------

    def _latent_stats(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(reference.device, reference.dtype)
        )
        recip_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            reference.device, reference.dtype
        )
        return mean, recip_std

    def _normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        mean, recip_std = self._latent_stats(latents)
        return (latents - mean) * recip_std

    def _denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        mean, recip_std = self._latent_stats(latents)
        return latents / recip_std + mean

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _get_t5_prompt_embeds(
        self,
        prompt: str | list[str],
        max_sequence_length: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        device = device or self.device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [_prompt_clean(p) for p in prompt]

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype or self.text_encoder.dtype, device=device)
        # Wan zero-pads past the real token length instead of relying on the mask.
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )
        return prompt_embeds

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        max_sequence_length: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        prompt_embeds = self._get_t5_prompt_embeds(
            prompt, max_sequence_length=max_sequence_length, device=device, dtype=dtype
        )
        negative_prompt_embeds = None
        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt_embeds = self._get_t5_prompt_embeds(
                negative_prompt, max_sequence_length=max_sequence_length, device=device, dtype=dtype
            )
        return prompt_embeds, negative_prompt_embeds

    def encode_image(
        self,
        image: PIL.Image.Image | list[PIL.Image.Image],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """CLIP visual features of the reference character image."""
        device = device or self.device
        pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device=device, dtype=self.image_encoder.dtype)
        image_embeds = self.image_encoder(pixel_values, output_hidden_states=True)
        return image_embeds.hidden_states[-2]

    # ------------------------------------------------------------------
    # Latent preparation
    # ------------------------------------------------------------------

    def get_i2v_mask(
        self,
        batch_size: int,
        latent_t: int,
        latent_h: int,
        latent_w: int,
        mask_len: int = 1,
        mask_pixel_values: torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Build the 4-channel I2V mask that marks which frames are conditioning.

        Returns ``[B, 4, latent_t, latent_h, latent_w]``: the per-pixel-frame mask
        is folded into the channel axis by the VAE's temporal compression factor.
        """
        if mask_pixel_values is None:
            mask_lat_size = torch.zeros(
                batch_size,
                1,
                (latent_t - 1) * self.vae_scale_factor_temporal + 1,
                latent_h,
                latent_w,
                dtype=dtype,
                device=device,
            )
        else:
            mask_lat_size = mask_pixel_values.clone().to(device=device, dtype=dtype)
        mask_lat_size[:, :, :mask_len] = 1
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal)
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:]], dim=2)
        return mask_lat_size.view(batch_size, -1, self.vae_scale_factor_temporal, latent_h, latent_w).transpose(1, 2)

    def _vae_encode(
        self,
        video: torch.Tensor,
        generator: torch.Generator | None,
        sample_mode: str = "argmax",
    ) -> torch.Tensor:
        return self._normalize_latents(retrieve_latents(self.vae.encode(video), generator, sample_mode))

    def prepare_reference_image_latents(
        self,
        image: torch.Tensor,
        batch_size: int = 1,
        generator: torch.Generator | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """VAE-encode the character image and prepend its I2V mask on the channel axis."""
        dtype = dtype or self.vae.dtype
        if image.ndim == 4:
            image = image.unsqueeze(2)

        _, _, _, height, width = image.shape
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        ref_image_latents = self._vae_encode(image.to(device=device, dtype=dtype), generator)
        if ref_image_latents.shape[0] == 1 and batch_size > 1:
            ref_image_latents = ref_image_latents.expand(batch_size, -1, -1, -1, -1)

        reference_image_mask = self.get_i2v_mask(batch_size, 1, latent_height, latent_width, 1, None, dtype, device)
        return torch.cat([reference_image_mask, ref_image_latents], dim=1)

    def prepare_pose_latents(
        self,
        pose_video: torch.Tensor,
        batch_size: int = 1,
        generator: torch.Generator | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        pose_video = pose_video.to(device=device, dtype=dtype or self.vae.dtype)
        pose_latents = self._vae_encode(pose_video, generator)
        if pose_latents.shape[0] == 1 and batch_size > 1:
            pose_latents = pose_latents.expand(batch_size, -1, -1, -1, -1)
        return pose_latents

    def prepare_prev_segment_cond_latents(
        self,
        prev_segment_cond_video: torch.Tensor | None = None,
        background_video: torch.Tensor | None = None,
        mask_video: torch.Tensor | None = None,
        batch_size: int = 1,
        segment_frame_length: int = _DEFAULT_SEGMENT_FRAME_LENGTH,
        start_frame: int = 0,
        height: int = 720,
        width: int = 1280,
        prev_segment_cond_frames: int = 1,
        mode: str = "animate",
        interpolation_mode: str = "bicubic",
        generator: torch.Generator | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Build the conditioning latents for one segment.

        In animation mode the segment is the previous segment's trailing frames
        followed by zeros; in replacement mode the remainder is the background
        video, and the character mask drives the I2V mask.
        """
        dtype = dtype or self.vae.dtype
        is_replace = mode == "replace"

        if prev_segment_cond_video is None:
            if is_replace:
                prev_segment_cond_video = background_video[:, :, :prev_segment_cond_frames].to(dtype)
            else:
                prev_segment_cond_video = torch.zeros(
                    (batch_size, 3, prev_segment_cond_frames, height, width), dtype=dtype, device=device
                )

        _, channels, _, segment_height, segment_width = prev_segment_cond_video.shape
        num_latent_frames = (segment_frame_length - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        if segment_height != height or segment_width != width:
            logger.info(
                "Interpolating prev segment cond video from (%d, %d) to (%d, %d)",
                segment_width,
                segment_height,
                width,
                height,
            )
            # 4D (spatial) rather than 5D (spatiotemporal) interpolation, following the original code.
            prev_segment_cond_video = prev_segment_cond_video.transpose(1, 2).flatten(0, 1)
            prev_segment_cond_video = F.interpolate(
                prev_segment_cond_video, size=(height, width), mode=interpolation_mode
            )
            prev_segment_cond_video = prev_segment_cond_video.unflatten(0, (batch_size, -1)).transpose(1, 2)

        if is_replace:
            remaining_segment = background_video[:, :, prev_segment_cond_frames:].to(dtype)
        else:
            remaining_segment = torch.zeros(
                batch_size,
                channels,
                segment_frame_length - prev_segment_cond_frames,
                height,
                width,
                dtype=dtype,
                device=device,
            )

        full_segment_cond_video = torch.cat([prev_segment_cond_video.to(dtype=dtype), remaining_segment], dim=2)
        prev_segment_cond_latents = self._vae_encode(full_segment_cond_video, generator)

        if is_replace:
            # The model's mask convention is the inverse of the input mask.
            inverted_mask = 1 - mask_video
            inverted_mask = inverted_mask.permute(0, 2, 1, 3, 4).flatten(0, 1)
            inverted_mask = F.interpolate(inverted_mask, size=(latent_height, latent_width), mode="nearest")
            mask_pixel_values = inverted_mask.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3, 4)
        else:
            mask_pixel_values = None

        prev_segment_cond_mask = self.get_i2v_mask(
            batch_size,
            num_latent_frames,
            latent_height,
            latent_width,
            # The first segment has no previous frames to condition on.
            mask_len=prev_segment_cond_frames if start_frame > 0 else 0,
            mask_pixel_values=mask_pixel_values,
            dtype=dtype,
            device=device,
        )
        return torch.cat([prev_segment_cond_mask, prev_segment_cond_latents], dim=1)

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 720,
        width: int = 1280,
        num_frames: int = _DEFAULT_SEGMENT_FRAME_LENGTH,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            # +1 for the leading reference-image frame.
            num_latent_frames + 1,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    @staticmethod
    def pad_video_frames(frames: list[Any], num_target_frames: int) -> list[Any]:
        """Extend ``frames`` to ``num_target_frames`` by bouncing back and forth.

        ``pad_video_frames([1, 2, 3, 4, 5], 10) -> [1, 2, 3, 4, 5, 4, 3, 2, 1, 2]``
        """
        if len(frames) == 1:
            return [frames[0]] * num_target_frames

        idx = 0
        flip = False
        target_frames: list[Any] = []
        while len(target_frames) < num_target_frames:
            target_frames.append(frames[idx])
            idx += -1 if flip else 1
            if idx == 0 or idx == len(frames) - 1:
                flip = not flip
        return target_frames

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def check_inputs(
        self,
        prompt: str | None,
        image: PIL.Image.Image | None,
        pose_video: list[PIL.Image.Image] | None,
        face_video: list[PIL.Image.Image] | None,
        background_video: list[PIL.Image.Image] | None,
        mask_video: list[PIL.Image.Image] | None,
        height: int,
        width: int,
        segment_frame_length: int,
        prev_segment_cond_frames: int,
    ) -> None:
        if prompt is None:
            raise ValueError("`prompt` must be provided.")
        if image is None:
            raise ValueError("Wan-Animate requires a reference character image.")
        if not pose_video:
            raise ValueError("Wan-Animate requires a pose video.")
        if not face_video:
            raise ValueError("Wan-Animate requires a face video.")

        mod_value = self.vae_scale_factor_spatial * self.transformer.config.patch_size[-1]
        if height % mod_value != 0 or width % mod_value != 0:
            raise ValueError(f"`height` and `width` have to be divisible by {mod_value} but are {height} and {width}.")

        if len(pose_video) != len(face_video):
            raise ValueError(
                f"`pose_video` ({len(pose_video)} frames) and `face_video` ({len(face_video)} frames) must have the "
                "same length."
            )

        if (background_video is None) != (mask_video is None):
            raise ValueError("Replacement mode requires both `video` (background) and `mask`.")
        if background_video is not None:
            if len(background_video) != len(mask_video):
                raise ValueError(
                    f"`video` ({len(background_video)} frames) and `mask` ({len(mask_video)} frames) must have the "
                    "same length."
                )
            if len(background_video) < len(pose_video):
                raise ValueError(
                    f"Background video ({len(background_video)} frames) must be at least as long as the pose video "
                    f"({len(pose_video)} frames)."
                )

        if prev_segment_cond_frames >= segment_frame_length:
            raise ValueError(
                f"`prev_segment_conditioning_frames` ({prev_segment_cond_frames}) must be smaller than "
                f"`segment_frame_length` ({segment_frame_length})."
            )
        if prev_segment_cond_frames >= len(pose_video):
            raise ValueError(
                f"`prev_segment_conditioning_frames` ({prev_segment_cond_frames}) must be smaller than the number of "
                f"driving frames ({len(pose_video)})."
            )

    # ------------------------------------------------------------------
    # Denoising
    # ------------------------------------------------------------------

    def predict_noise(self, current_model: nn.Module | None = None, **kwargs: Any) -> torch.Tensor:
        if current_model is None:
            current_model = self.transformer
        result = current_model(**kwargs)
        if isinstance(result, (tuple, list)):
            return result[0]
        return result.sample

    def diffuse(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        image_embeds: torch.Tensor,
        reference_latents: torch.Tensor,
        pose_latents: torch.Tensor,
        face_video_segment: torch.Tensor,
        motion_encode_batch_size: int | None,
        guidance_scale: float,
        transformer_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Denoise one segment."""
        do_true_cfg = self.do_classifier_free_guidance and negative_prompt_embeds is not None

        with self.progress_bar(total=len(timesteps)) as pbar:
            for step_idx, t in enumerate(timesteps):
                self._current_timestep = t

                latent_model_input = torch.cat([latents, reference_latents], dim=1).to(transformer_dtype)
                timestep = t.expand(latents.shape[0])

                positive_kwargs = {
                    "hidden_states": latent_model_input,
                    "timestep": timestep,
                    "encoder_hidden_states": prompt_embeds,
                    "encoder_hidden_states_image": image_embeds,
                    "pose_hidden_states": pose_latents,
                    "face_pixel_values": face_video_segment,
                    "motion_encode_batch_size": motion_encode_batch_size,
                    "return_dict": False,
                }

                if do_true_cfg:
                    negative_kwargs = dict(positive_kwargs)
                    negative_kwargs["encoder_hidden_states"] = negative_prompt_embeds
                    # The unconditional branch blanks the face signal rather than dropping it.
                    negative_kwargs["face_pixel_values"] = face_video_segment * 0 - 1
                else:
                    negative_kwargs = None

                set_forward_context_denoise_step_idx(step_idx)
                noise_pred = self.predict_noise_maybe_with_cfg(
                    do_true_cfg=do_true_cfg,
                    true_cfg_scale=guidance_scale,
                    positive_kwargs=positive_kwargs,
                    negative_kwargs=negative_kwargs,
                    cfg_normalize=False,
                )

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                pbar.update()

        self._current_timestep = None
        return latents

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        if len(req.prompts) > 1:
            raise ValueError("Wan-Animate only supports a single prompt per request.")
        if req.sampling_params.num_outputs_per_prompt > 1:
            raise ValueError(
                "Wan-Animate generates one video per request; `num_outputs_per_prompt` "
                f"({req.sampling_params.num_outputs_per_prompt}) is not supported."
            )

        prompt_data = req.prompts[0]
        if isinstance(prompt_data, str):
            prompt: str | None = prompt_data
            negative_prompt: str | None = None
            multi_modal_data: dict[str, Any] = {}
            additional_info: dict[str, Any] = {}
        else:
            prompt = prompt_data.get("prompt")
            negative_prompt = prompt_data.get("negative_prompt")
            multi_modal_data = prompt_data.get("multi_modal_data", {}) or {}
            additional_info = prompt_data.get("additional_information", {}) or {}

        if negative_prompt is None:
            negative_prompt = _ANIMATE_DEFAULT_NEG_PROMPT

        image = multi_modal_data.get("image")
        if isinstance(image, str):
            image = PIL.Image.open(image).convert("RGB")

        # The pre-process hook normally materializes these; fall back to the raw
        # multi_modal_data so the pipeline also works when called directly.
        pose_video = additional_info.get("pose_video") or _load_video_frames(
            multi_modal_data.get("pose_video"), "pose_video"
        )
        face_video = additional_info.get("face_video") or _load_video_frames(
            multi_modal_data.get("face_video"), "face_video"
        )
        background_video = additional_info.get("background_video") or _load_video_frames(
            multi_modal_data.get("video"), "video"
        )
        mask_video = additional_info.get("mask_video") or _load_mask_frames(multi_modal_data.get("mask"))

        sampling_params = req.sampling_params
        extra_args = getattr(sampling_params, "extra_args", {}) or {}

        height = sampling_params.height or 720
        width = sampling_params.width or 1280
        num_inference_steps = sampling_params.num_inference_steps or 20
        # Wan-Animate is trained to run without CFG; only enable it if asked.
        guidance_scale = sampling_params.guidance_scale if sampling_params.guidance_scale_provided else 1.0
        self._guidance_scale = guidance_scale

        segment_frame_length = int(extra_args.get("segment_frame_length", _DEFAULT_SEGMENT_FRAME_LENGTH))
        prev_segment_cond_frames = int(
            extra_args.get("prev_segment_conditioning_frames", _DEFAULT_PREV_SEGMENT_COND_FRAMES)
        )
        motion_encode_batch_size = extra_args.get("motion_encode_batch_size")
        if motion_encode_batch_size is not None:
            motion_encode_batch_size = int(motion_encode_batch_size)

        if segment_frame_length % self.vae_scale_factor_temporal != 1:
            rounded = segment_frame_length // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
            logger.warning(
                "`segment_frame_length - 1` must be divisible by %d; rounding %d down to %d.",
                self.vae_scale_factor_temporal,
                segment_frame_length,
                rounded,
            )
            segment_frame_length = max(rounded, 1)

        # `num_frames`, when explicitly requested, truncates the driving videos.
        # Anything malformed here is reported by check_inputs below, so only
        # truncate what is actually present.
        requested_num_frames = sampling_params.num_frames
        if requested_num_frames and requested_num_frames > 1 and pose_video and requested_num_frames < len(pose_video):
            pose_video = pose_video[:requested_num_frames]
            if face_video:
                face_video = face_video[:requested_num_frames]
            if background_video:
                background_video = background_video[:requested_num_frames]
            if mask_video:
                mask_video = mask_video[:requested_num_frames]

        self.check_inputs(
            prompt,
            image,
            pose_video,
            face_video,
            background_video,
            mask_video,
            height,
            width,
            segment_frame_length,
            prev_segment_cond_frames,
        )

        mode = "replace" if background_video is not None else "animate"
        logger.info("Wan-Animate running in %s mode (%d driving frames)", mode, len(pose_video))

        device = self.device
        transformer_dtype = self.transformer.dtype

        generator = sampling_params.generator
        if generator is None and sampling_params.seed is not None:
            generator = torch.Generator(device=device).manual_seed(sampling_params.seed)

        # Round the driving length up to a whole number of segments.
        cond_video_frames = len(pose_video)
        effective_segment_length = segment_frame_length - prev_segment_cond_frames
        last_segment_frames = (cond_video_frames - prev_segment_cond_frames) % effective_segment_length
        num_padding_frames = 0 if last_segment_frames == 0 else effective_segment_length - last_segment_frames
        num_target_frames = cond_video_frames + num_padding_frames
        num_segments = num_target_frames // effective_segment_length

        # ---- 1. Text ----
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            device=device,
            dtype=transformer_dtype,
        )

        # ---- 2. Reference character image ----
        image_pixels = self.vae_image_processor.preprocess(image, height=height, width=width, resize_mode="fill").to(
            device, dtype=torch.float32
        )
        image_embeds = self.encode_image(image, device).to(transformer_dtype)

        # ---- 3. Driving videos ----
        pose_video = self.pad_video_frames(pose_video, num_target_frames)
        face_video = self.pad_video_frames(face_video, num_target_frames)

        pose_video_t = self.video_processor.preprocess_video(pose_video, height=height, width=width).to(
            device, dtype=torch.float32
        )
        face_size = self.transformer.config.motion_encoder_size
        face_video_t = self.video_processor.preprocess_video(face_video, height=face_size, width=face_size).to(
            device, dtype=torch.float32
        )

        if mode == "replace":
            background_video = self.pad_video_frames(background_video, num_target_frames)
            mask_video = self.pad_video_frames(mask_video, num_target_frames)
            background_video_t = self.video_processor.preprocess_video(background_video, height=height, width=width).to(
                device, dtype=torch.float32
            )
            mask_video_t = self.video_processor_for_mask.preprocess_video(mask_video, height=height, width=width).to(
                device, dtype=torch.float32
            )
        else:
            background_video_t = None
            mask_video_t = None

        # ---- 4. Constant latents ----
        num_channels_latents = self.vae.config.z_dim
        reference_image_latents = self.prepare_reference_image_latents(
            image_pixels, batch_size=1, generator=generator, device=device
        )

        # ---- 5. Segment loop ----
        start = 0
        end = segment_frame_length
        all_out_frames: list[torch.Tensor] = []
        out_frames: torch.Tensor | None = None

        for segment_idx in range(num_segments):
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps
            self._num_timesteps = len(timesteps)

            latents = self.prepare_latents(
                1,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                num_frames=segment_frame_length,
                dtype=torch.float32,
                device=device,
                generator=generator,
            )

            pose_latents = self.prepare_pose_latents(
                pose_video_t[:, :, start:end], batch_size=1, generator=generator, device=device
            ).to(dtype=transformer_dtype)

            face_video_segment = face_video_t[:, :, start:end].to(dtype=transformer_dtype)

            prev_segment_cond_video = (
                out_frames[:, :, -prev_segment_cond_frames:].clone().detach() if start > 0 else None
            )

            prev_segment_cond_latents = self.prepare_prev_segment_cond_latents(
                prev_segment_cond_video,
                background_video=background_video_t[:, :, start:end] if mode == "replace" else None,
                mask_video=mask_video_t[:, :, start:end] if mode == "replace" else None,
                batch_size=1,
                segment_frame_length=segment_frame_length,
                start_frame=start,
                height=height,
                width=width,
                prev_segment_cond_frames=prev_segment_cond_frames,
                mode=mode,
                generator=generator,
                device=device,
            )

            # Reference frame first, then the segment's conditioning frames.
            reference_latents = torch.cat([reference_image_latents, prev_segment_cond_latents], dim=2)

            logger.info("Wan-Animate segment %d/%d", segment_idx + 1, num_segments)
            latents = self.diffuse(
                latents=latents,
                timesteps=timesteps,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                image_embeds=image_embeds,
                reference_latents=reference_latents,
                pose_latents=pose_latents,
                face_video_segment=face_video_segment,
                motion_encode_batch_size=motion_encode_batch_size,
                guidance_scale=guidance_scale,
                transformer_dtype=transformer_dtype,
            )

            latents = self._denormalize_latents(latents.to(self.vae.dtype))
            # Drop the leading reference frame before decoding.
            out_frames = self.vae.decode(latents[:, :, 1:], return_dict=False)[0]

            if start > 0:
                out_frames = out_frames[:, :, prev_segment_cond_frames:]
            all_out_frames.append(out_frames)

            start += effective_segment_length
            end += effective_segment_length

        video = torch.cat(all_out_frames, dim=2)[:, :, :cond_video_frames]
        return DiffusionOutput(output=video)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
