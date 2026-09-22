# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Wan2.2-Animate-2 character-animation pipeline for vLLM-Omni.

Ported from ``Wan-Video/Wan-Animate-2`` (``pipelines/wan_animate_2_pipeline.py``,
``inference_core``) and loaded from the Diffusers layout
(``Wan-AI/Wan2.2-Animate-2-14B-Diffusers`` / ``...-Distilled-Diffusers``).

Animate-2 retargets the motion and expression of a *driving video* onto the
person in a *reference image*.  Unlike the other Wan2.2 pipelines it needs no
pre-extracted pose skeleton or face crop: the raw driving video is the
conditioning signal, and it reaches the DiT as in-context attention rather than
as concatenated or injected features.

Per segment the pipeline runs

1. ``transformer.extract_reference()`` over the driving-video latents, filling a
   per-layer :class:`ReferenceKVCache`;
2. the ordinary denoising loop, where every block attends the frame-aligned
   slice of that cache;
3. a VAE decode whose trailing frame seeds the next segment.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import PIL.Image
import torch
from diffusers import DPMSolverMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel
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
from vllm_omni.diffusion.models.schedulers import FlowMatchEulerDiscreteScheduler
from vllm_omni.diffusion.models.utils import _load_json
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import (
    _WAN_TEXT_ENCODER_OFFLOAD_PLAN,
    load_transformer_config,
    retrieve_latents,
)
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_i2v import get_wan22_i2v_post_process_func
from vllm_omni.diffusion.models.wan_animate2.reference_kv_cache import ReferenceKVCache
from vllm_omni.diffusion.models.wan_animate2.wan_animate2_transformer import (
    WanAnimate2Transformer3DModel,
    WanAnimate2TransformerConfig,
)
from vllm_omni.diffusion.offloader.config import DIT_COMPONENT, selected_offload_components
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch, split_diffusion_output_by_request
from vllm_omni.inputs.data import OmniTextPrompt
from vllm_omni.platforms import current_omni_platform

logger = logging.getLogger(__name__)

# `infer/wan_animate_2.yaml: test_cfg.sample_neg_prompt`
ANIMATE2_DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)
# `infer/wan_animate_2.yaml: test_cfg.input_prompts`
ANIMATE2_DEFAULT_PROMPT = "static background."
# `infer/wan_animate_2_demo.py` argument defaults.
ANIMATE2_DEFAULT_PROMPT_REF = "人物动作的参考视频"
ANIMATE2_DEFAULT_SEGMENT_FRAMES = 81
ANIMATE2_DEFAULT_FPS = 24
ANIMATE2_DEFAULT_WIDTH = 720
ANIMATE2_DEFAULT_HEIGHT = 1280
ANIMATE2_DEFAULT_FLOW_SHIFT = 5.0
ANIMATE2_BASE_STEPS = 40
ANIMATE2_BASE_GUIDANCE_SCALE = 3.0
# `infer/wan_animate_2_distillation.yaml` differs only in `log_scale`; the demo
# runs the distilled DiT for 10 steps without CFG.
ANIMATE2_DISTILLED_LOG_SCALE = -1.3
ANIMATE2_DISTILLED_STEPS = 10
ANIMATE2_DISTILLED_GUIDANCE_SCALE = 1.0
# About 30 s of driving video at the default frame rate, i.e. nine segments.
ANIMATE2_DEFAULT_MAX_DRIVING_FRAMES = 30 * ANIMATE2_DEFAULT_FPS

# The previous segment's trailing frames re-encoded as the next segment's
# condition (`first_num`, hardcoded upstream).
_OVERLAP_FRAMES = 1
# `modular_model_index.json` blocks class that marks the distilled release.
_DISTILLED_BLOCKS_CLASS = "WanAnimate2DistilledBlocks"
_TRANSFORMER_LATENT_CHANNELS = 16
_MASK_CHANNELS = 4


# ---------------------------------------------------------------------------
# Frame / geometry helpers (ported from pipelines/utils)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LetterboxInfo:
    """Where the source content sits inside the letterboxed canvas."""

    pad_axis: str  # "width" or "height": which axis carries the black bars
    offset: int
    length: int

    def crop(self, video: torch.Tensor) -> torch.Tensor:
        """Undo the letterbox on a ``[B, C, T, H, W]`` video."""
        if self.pad_axis == "width":
            return video[:, :, :, :, self.offset : self.offset + self.length]
        return video[:, :, :, self.offset : self.offset + self.length, :]


def calculate_letterbox_size(width: int, height: int, target_area: int, divisor: int = 16) -> tuple[int, int]:
    """Largest ``divisor``-aligned box of about ``target_area`` with the source aspect.

    This reproduces what upstream *actually* computes, which is its fallback
    path rather than its search: ``human_video_omni_utils.calculate_new_size``
    defines ``check_valid(w, h)`` but calls it as ``check_valid(w, h, divisor)``,
    so it raises ``TypeError`` on the first candidate, and ``resize_by_area``'s
    bare ``except`` swallows that and takes the closed-form branch below every
    time.  Matching the generated resolution matters for output parity, so the
    closed form is what is ported.
    """
    aspect_ratio = width / height
    new_height = math.sqrt(target_area / aspect_ratio)
    new_width = target_area / new_height
    return int(new_width // divisor) * divisor, int(new_height // divisor) * divisor


def resize_by_area(image: PIL.Image.Image, target_area: int, divisor: int = 16) -> tuple[np.ndarray, LetterboxInfo]:
    """Fit ``image`` into a divisor-aligned box, letterboxing with black.

    Ported from ``pipelines/utils/human_video_omni_utils.py`` (``resize_by_area``
    + ``padding_resize``).  Upstream resizes with OpenCV (``INTER_AREA`` when
    shrinking, ``INTER_LINEAR`` when enlarging); Pillow's ``BOX`` and
    ``BILINEAR`` filters are the matching resamplers.

    Returns:
        ``(canvas, info)`` with ``canvas`` a ``[H, W, 3]`` uint8 RGB array.
    """
    source_width, source_height = image.size
    width, height = calculate_letterbox_size(source_width, source_height, target_area, divisor)

    shrinking = width * height < source_width * source_height
    resample = PIL.Image.Resampling.BOX if shrinking else PIL.Image.Resampling.BILINEAR
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    if (source_height / source_width) > (height / width):
        new_width = int(height / source_height * source_width)
        resized = np.asarray(image.resize((new_width, height), resample=resample))
        offset = (width - new_width) // 2
        canvas[:, offset : offset + new_width, :] = resized
        return canvas, LetterboxInfo("width", offset, new_width)

    new_height = int(width / source_width * source_height)
    resized = np.asarray(image.resize((width, new_height), resample=resample))
    offset = (height - new_height) // 2
    canvas[offset : offset + new_height, :, :] = resized
    return canvas, LetterboxInfo("height", offset, new_height)


def get_padding_len(input_len: int, segment_len: int, overlap: int = _OVERLAP_FRAMES) -> int:
    """Length the driving video is padded to so every segment is well formed.

    Ported verbatim from ``pipelines/utils/multiclip_utils.py``: the trailing
    remainder is pushed to at least 28 frames, otherwise rounded up to a
    multiple of 4 (the VAE temporal stride).
    """
    remaining = (input_len - overlap) % (segment_len - overlap)
    if remaining < 28:
        padding_needed = 28 - remaining
    else:
        padding_needed = 4 - remaining % 4
    return input_len + padding_needed


def zigzag_padding(frames: Sequence[np.ndarray], target_len: int) -> list[np.ndarray]:
    """Extend ``frames`` to ``target_len`` by bouncing back and forth.

    ``[0 1 2 3 4]`` becomes ``[0 1 2 3 4 | 3 2 1 ...]``, which avoids the hard
    cut a simple repeat or clamp would introduce at the seam.
    """
    if not frames:
        raise ValueError("cannot pad an empty frame sequence")
    if len(frames) == 1:
        return [frames[0]] * target_len

    padded: list[np.ndarray] = []
    index = 0
    step = 1
    while len(padded) < target_len:
        padded.append(frames[index])
        index += step
        if index == 0 or index == len(frames) - 1:
            step = -step
    return padded[:target_len]


def get_frame_indices(frame_count: int, video_fps: float, target_count: int, target_fps: float) -> list[int]:
    """Resample frame indices from ``video_fps`` to ``target_fps``."""
    times = np.arange(0, target_count) / target_fps
    indices = np.round(times * video_fps).astype(int)
    return np.clip(indices, 0, frame_count - 1).tolist()


def decode_video_file(path: str) -> tuple[list[np.ndarray], float]:
    """Decode every frame of ``path`` to uint8 RGB arrays, plus the source fps."""
    import av

    with av.open(path) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate
        if not rate:
            raise ValueError(f"could not determine the frame rate of {path!r}")
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]

    if not frames:
        raise ValueError(f"driving video {path!r} contains no frames")
    return frames, float(rate)


def resample_frames(frames: Sequence[np.ndarray], video_fps: float, target_fps: float) -> list[np.ndarray]:
    """Pick the frames of a ``video_fps`` clip that fall on a ``target_fps`` grid."""
    target_count = int(len(frames) / video_fps * target_fps)
    if target_count <= 0:
        raise ValueError(
            f"a {len(frames)}-frame {video_fps} fps driving video is too short for {target_fps} fps output"
        )
    indices = get_frame_indices(len(frames), video_fps, target_count, target_fps)
    return [frames[index] for index in indices]


def get_i2v_mask(
    lat_t: int,
    lat_h: int,
    lat_w: int,
    mask_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the 4-channel conditioning mask concatenated onto the latents.

    Ported from ``wanxiang/eval_i2v.py: get_i2v_mask``.  ``mask_len`` counts
    *pixel-space* frames that are conditioned; the first latent frame covers 4
    pixel frames, hence the leading repeat.  Returns ``[4, lat_t, lat_h, lat_w]``.
    """
    mask = torch.zeros(1, (lat_t - 1) * 4 + 1, lat_h, lat_w, device=device)
    mask[:, :mask_len] = 1
    mask = torch.cat([torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1), mask[:, 1:]], dim=1)
    mask = mask.view(1, mask.shape[1] // 4, 4, lat_h, lat_w)
    return mask.transpose(1, 2)[0].to(dtype)


# ---------------------------------------------------------------------------
# Pre / post processing
# ---------------------------------------------------------------------------


def get_wan_animate2_pre_process_func(od_config: OmniDiffusionConfig):
    """Normalise the reference image and driving video onto the request.

    Animate-2 needs no new media channel: the reference image arrives on
    ``multi_modal_data["image"]`` and the driving video on
    ``multi_modal_data["video"]``, both of which the serving layer already
    populates from ``image_reference`` / ``video_reference``.
    """

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        if isinstance(request.prompt, str):
            prompt = OmniTextPrompt(prompt=request.prompt)
        else:
            prompt = request.prompt
        multi_modal_data = prompt.get("multi_modal_data") or {}
        prompt.setdefault("additional_information", {})

        image = multi_modal_data.get("image")
        if isinstance(image, list):
            image = image[0] if image else None
        if image is None:
            raise ValueError(
                "Wan2.2-Animate-2 requires a reference image: "
                'set `"multi_modal_data": {"image": <path or PIL.Image>, "video": <driving video>}`.'
            )
        if isinstance(image, str):
            image = PIL.Image.open(image)
        if not isinstance(image, PIL.Image.Image):
            raise TypeError(f"unsupported reference image type {type(image)}")
        multi_modal_data["image"] = image.convert("RGB")

        if multi_modal_data.get("video") is None:
            raise ValueError(
                "Wan2.2-Animate-2 requires a driving video: "
                'set `"multi_modal_data": {"video": <path or frame list>, ...}`.'
            )

        extra_args = request.sampling_params.extra_args or {}
        prompt["multi_modal_data"] = multi_modal_data
        prompt["additional_information"]["prompt_ref"] = extra_args.get("prompt_ref")
        prompt["additional_information"]["segment_frame_length"] = extra_args.get("segment_frame_length")
        request.prompt = prompt
        return request

    return pre_process_func


# The decoded video is an ordinary Wan `[B, C, T, H, W]` tensor in [-1, 1], so
# the I2V post-processing applies unchanged.
get_wan_animate2_post_process_func = get_wan22_i2v_post_process_func


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Animate2Request:
    """The single request's fields, resolved against the checkpoint defaults."""

    prompt: str
    negative_prompt: str
    prompt_ref: str
    image: PIL.Image.Image
    video: str | os.PathLike[str] | list[str | os.PathLike[str]] | list[PIL.Image.Image]
    width: int
    height: int
    fps: float
    max_frames: int | None
    segment_frames: int
    num_inference_steps: int
    guidance_scale: float


class Wan22Animate2Pipeline(
    nn.Module,
    SupportImageInput,
    CFGParallelMixin,
    DenoiseProgressMixin,
    ProgressBarMixin,
    DiffusionPipelineProfilerMixin,
    SupportsComponentDiscovery,
):
    """Wan2.2-Animate-2 character animation.

    One request produces an arbitrarily long video by chaining fixed-length
    segments: each segment re-encodes the previous segment's last frame as its
    conditioning anchor, so the seam carries motion continuity.

    Requests are handled one at a time (``supports_request_batch = False``),
    matching upstream, whose tensors are per-sample lists with an effective
    batch of one.  Batching several requests would need one reference K/V cache
    per request, which is deferred.
    """

    supports_request_batch = False

    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder", "image_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    _offload_plan = _WAN_TEXT_ENCODER_OFFLOAD_PLAN
    dummy_run_num_frames: ClassVar[int] = 0

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
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

        subfolders = ["tokenizer", "text_encoder", "vae", "image_processor", "image_encoder", "scheduler"]
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

        self.is_distilled = _is_distilled_checkpoint(model, local_files_only)
        log_scale = ANIMATE2_DISTILLED_LOG_SCALE if self.is_distilled else 0.0
        transformer_config = WanAnimate2TransformerConfig.from_dict(
            load_transformer_config(model, "transformer", local_files_only), log_scale=log_scale
        )
        self.transformer = WanAnimate2Transformer3DModel(
            transformer_config, quant_config=getattr(od_config, "quantization_config", None)
        )

        self.flow_shift = ANIMATE2_DEFAULT_FLOW_SHIFT if od_config.flow_shift is None else od_config.flow_shift
        self.scheduler = _load_scheduler(model, local_files_only, self.flow_shift)

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial
        # VAE spatial stride x transformer patch stride.
        self.resolution_divisor = self.vae_scale_factor_spatial * transformer_config.patch_size[1]
        latent_shape = (1, self.vae.config.z_dim, 1, 1, 1)
        self.latents_mean = torch.tensor(self.vae.config.latents_mean).view(latent_shape)
        self.latents_std = torch.tensor(self.vae.config.latents_std).view(latent_shape)

        self._guidance_scale = None
        self._num_timesteps = None
        self._current_timestep = None
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=od_config.enable_diffusion_pipeline_profiler
        )

    # ------------------------------------------------------------------
    # Properties and request-level contracts
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

    @classmethod
    def reference_video_decode_spec(
        cls,
        *,
        num_frames: int | None = None,
        extra_args: dict[str, object] | None = None,
    ) -> ReferenceVideoDecodeSpec:
        """Cap how much driving video a single request may pull in.

        Segment count grows linearly with the driving video's length, and every
        segment costs a full extraction pass plus a denoising loop, so an
        unbounded video is an unbounded request.  ``num_frames`` is the
        requested output length and only ever tightens the cap.
        """
        max_frames = ANIMATE2_DEFAULT_MAX_DRIVING_FRAMES
        if extra_args and extra_args.get("max_driving_frames"):
            max_frames = int(extra_args["max_driving_frames"])
        if num_frames is not None and num_frames > 0:
            max_frames = min(max_frames, int(num_frames))
        return ReferenceVideoDecodeSpec(max_frames=max_frames, keep="first")

    # ------------------------------------------------------------------
    # Encoders
    # ------------------------------------------------------------------

    def _encode_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        """VAE-encode ``[B, 3, T, H, W]`` pixels in ``[-1, 1]`` to normalised latents."""
        pixels = pixels.to(dtype=self.vae.dtype, device=self.device)
        latents = retrieve_latents(self.vae.encode(pixels), sample_mode="argmax")
        mean = self.latents_mean.to(latents.device, latents.dtype)
        std = self.latents_std.to(latents.device, latents.dtype)
        return (latents - mean) / std

    def _decode_latents(self, latents: torch.Tensor, num_frames: int, height: int, width: int) -> torch.Tensor:
        """VAE-decode one segment, broadcasting under VAE patch parallelism.

        The autoregressive chain needs the decoded pixels on *every* rank, but
        patch-parallel decode returns them only on rank 0.
        """
        latents = latents.to(self.vae.dtype)
        mean = self.latents_mean.to(latents.device, latents.dtype)
        std = self.latents_std.to(latents.device, latents.dtype)
        video = self.vae.decode(latents * std + mean, return_dict=False)[0]

        if video.numel() == 0:
            import torch.distributed as dist

            video = torch.empty(
                (latents.shape[0], 3, num_frames, height, width), device=self.device, dtype=self.vae.dtype
            )
            vae_pp_group = getattr(self.vae, "_vae_pp_group", None)
            if vae_pp_group is not None:
                dist.broadcast(video, src=0, group=vae_pp_group)

        return video[:, :, :num_frames].to(dtype=self.transformer.dtype)

    def encode_prompt(self, prompt: str, max_sequence_length: int = 512) -> torch.Tensor:
        """Encode one prompt into ``[1, 512, 4096]`` UMT5 embeddings.

        Same contract as the Wan2.2-I2V encoder: truncate to the true token
        count and right-pad with zeros, which is what ``text_embedding`` sees.
        """
        text_inputs = self.tokenizer(
            [" ".join(prompt.strip().split())],
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids = text_inputs.input_ids.to(self.device)
        mask = text_inputs.attention_mask.to(self.device)
        seq_len = int(mask.gt(0).sum(dim=1)[0])

        embeds = self.text_encoder(ids, mask).last_hidden_state.to(dtype=self.transformer.dtype)
        padded = embeds.new_zeros(1, max_sequence_length, embeds.shape[-1])
        padded[:, :seq_len] = embeds[:, :seq_len]
        return padded

    def encode_image(self, image: np.ndarray) -> torch.Tensor:
        """CLIP-encode one ``[H, W, 3]`` uint8 frame (penultimate layer, as in Wan I2V)."""
        pixel_values = self.image_processor(images=PIL.Image.fromarray(image), return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device=self.device, dtype=self.image_encoder.dtype)
        image_embeds = self.image_encoder(pixel_values, output_hidden_states=True)
        return image_embeds.hidden_states[-2].to(dtype=self.transformer.dtype)

    # ------------------------------------------------------------------
    # Denoising
    # ------------------------------------------------------------------

    def predict_noise(self, current_model: nn.Module | None = None, **kwargs) -> torch.Tensor:
        if current_model is None:
            current_model = self.transformer
        param_dtype = next(current_model.parameters()).dtype
        with torch.amp.autocast(self.device.type, dtype=param_dtype):
            result = current_model(**kwargs)
        return result[0]

    def diffuse(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        condition_latents: torch.Tensor,
        kv_cache: ReferenceKVCache,
        reference_grid_sizes: tuple[int, int, int],
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        image_embeds: torch.Tensor,
        origin_len: int,
        origin_area: tuple[int, int],
        guidance_scale: float,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Denoise one segment against a pre-filled reference cache."""
        do_true_cfg = guidance_scale > 1.0 and negative_prompt_embeds is not None

        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for step_idx, t in enumerate(timesteps):
                self._current_timestep = t
                self.record_denoise_step(step_idx, timestep=t, scheduler=self.scheduler)

                positive_kwargs = {
                    "hidden_states": latents,
                    "timestep": t.unsqueeze(0).to(latents.device),
                    "encoder_hidden_states": prompt_embeds,
                    "encoder_hidden_states_image": image_embeds,
                    "condition_latents": condition_latents,
                    "kv_cache": kv_cache,
                    "reference_grid_sizes": reference_grid_sizes,
                    "origin_len": origin_len,
                    "origin_area": origin_area,
                    "is_uncondition": False,
                    "return_dict": False,
                }
                negative_kwargs = None
                if do_true_cfg:
                    negative_kwargs = {
                        **positive_kwargs,
                        "encoder_hidden_states": negative_prompt_embeds,
                        "is_uncondition": True,
                    }

                noise_pred = self.predict_noise_maybe_with_cfg(
                    do_true_cfg=do_true_cfg,
                    true_cfg_scale=guidance_scale,
                    positive_kwargs=positive_kwargs,
                    negative_kwargs=negative_kwargs,
                    cfg_normalize=False,
                )
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False, generator=generator)[0]
                progress_bar.update()

        self._current_timestep = None
        return latents

    # ------------------------------------------------------------------
    # Request entry point
    # ------------------------------------------------------------------

    def _resolve_request(self, req: DiffusionRequestBatch) -> Animate2Request:
        """Flatten the single request into the fields the segment loop needs."""
        if req.num_reqs != 1:
            raise ValueError(
                f"Wan2.2-Animate-2 handles one request at a time, got {req.num_reqs}; "
                "the engine should not batch this pipeline (supports_request_batch=False)"
            )
        prompt_data = req.prompts[0]
        params = req.sampling_params_list[0]
        if isinstance(prompt_data, str):
            raise ValueError("Wan2.2-Animate-2 needs multi_modal_data; a bare string prompt has none")

        multi_modal_data = prompt_data.get("multi_modal_data") or {}
        additional = prompt_data.get("additional_information") or {}

        width = params.width or ANIMATE2_DEFAULT_WIDTH
        height = params.height or ANIMATE2_DEFAULT_HEIGHT
        if height % self.resolution_divisor or width % self.resolution_divisor:
            raise ValueError(
                f"height and width must be divisible by {self.resolution_divisor}, got {height} and {width}"
            )

        segment_frames = additional.get("segment_frame_length") or ANIMATE2_DEFAULT_SEGMENT_FRAMES
        if segment_frames < 5 or (segment_frames - 1) % 4:
            raise ValueError(f"segment_frame_length must be 4k+1 and at least 5, got {segment_frames}")

        if self.is_distilled:
            default_steps = ANIMATE2_DISTILLED_STEPS
            default_guidance = ANIMATE2_DISTILLED_GUIDANCE_SCALE
        else:
            default_steps = ANIMATE2_BASE_STEPS
            default_guidance = ANIMATE2_BASE_GUIDANCE_SCALE
        guidance_scale = params.guidance_scale if params.guidance_scale_provided else default_guidance
        if self.is_distilled and guidance_scale > 1.0:
            logger.warning(
                "The distilled Animate-2 checkpoint is trained without CFG; "
                "guidance_scale=%.2f doubles the cost for no quality gain.",
                guidance_scale,
            )

        return Animate2Request(
            prompt=prompt_data.get("prompt") or ANIMATE2_DEFAULT_PROMPT,
            negative_prompt=prompt_data.get("negative_prompt") or ANIMATE2_DEFAULT_NEGATIVE_PROMPT,
            prompt_ref=additional.get("prompt_ref") or ANIMATE2_DEFAULT_PROMPT_REF,
            image=multi_modal_data["image"],
            video=multi_modal_data["video"],
            width=width,
            height=height,
            fps=params.resolved_frame_rate or ANIMATE2_DEFAULT_FPS,
            max_frames=params.num_frames,
            segment_frames=segment_frames,
            num_inference_steps=params.num_inference_steps or default_steps,
            guidance_scale=guidance_scale,
        )

    def _load_driving_frames(self, request: Animate2Request) -> list[np.ndarray]:
        """Driving frames at the output frame rate, letterboxed like the reference.

        A path is decoded here and resampled to the request fps.  A frame list
        (what ``/v1/videos`` hands over after ``reference_video_decode_spec``)
        is taken as already being at the request fps: the serving layer keeps
        no source frame rate.
        """
        video = request.video
        if isinstance(video, (str, os.PathLike)):
            video_path = os.fspath(video)
        elif len(video) == 1 and isinstance(video[0], (str, os.PathLike)):
            # Multipart ``input_references`` are persisted by the online API
            # and passed to native pipelines as a list of paths, even when the
            # request contains exactly one driving video.
            video_path = os.fspath(video[0])
        else:
            video_path = None

        if video_path is not None:
            frames, source_fps = decode_video_file(video_path)
            frames = resample_frames(frames, source_fps, request.fps)
        else:
            if any(isinstance(frame, (str, os.PathLike)) for frame in video):
                raise ValueError("Wan2.2-Animate-2 requires exactly one driving video")
            frames = [np.asarray(frame.convert("RGB")) for frame in video]
        if not frames:
            raise ValueError("driving video contains no frames")
        if request.max_frames is not None and request.max_frames > 0:
            frames = frames[: request.max_frames]

        target_area = request.width * request.height
        letterboxed = []
        for frame in frames:
            canvas, _ = resize_by_area(PIL.Image.fromarray(frame), target_area, self.resolution_divisor)
            letterboxed.append(canvas)
        return letterboxed

    def _pixels_to_tensor(self, frames: Sequence[np.ndarray]) -> torch.Tensor:
        """Stack uint8 RGB frames into a ``[1, 3, T, H, W]`` tensor in ``[-1, 1]``."""
        stacked = np.stack(frames).astype(np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(stacked).permute(3, 0, 1, 2).unsqueeze(0)
        return tensor.to(device=self.device, dtype=self.transformer.dtype)

    def forward(self, req: DiffusionRequestBatch) -> list[DiffusionOutput]:
        """Animate the reference image with the driving video's motion."""
        request = self._resolve_request(req)
        device = self.device
        dtype = self.transformer.dtype
        segment_frames = request.segment_frames

        # ---- 1. Pre-process reference image and driving video ----
        reference_image, letterbox = resize_by_area(
            request.image, request.width * request.height, self.resolution_divisor
        )
        driving_frames = self._load_driving_frames(request)
        source_frame_count = len(driving_frames)
        # Pad so the trailing segment is well formed; trimmed back after decode.
        driving_frames = zigzag_padding(driving_frames, get_padding_len(source_frame_count, segment_frames))

        frame_height, frame_width = reference_image.shape[:2]
        lat_h = frame_height // self.vae_scale_factor_spatial
        lat_w = frame_width // self.vae_scale_factor_spatial

        # ---- 2. Conditioning that is constant across segments ----
        generators = req.collate_request_generators(1, None)
        generator = generators[0] if isinstance(generators, list) else generators

        prompt_embeds = self.encode_prompt(request.prompt)
        reference_prompt_embeds = self.encode_prompt(request.prompt_ref)
        self._guidance_scale = request.guidance_scale
        negative_prompt_embeds = None
        if request.guidance_scale > 1.0:
            negative_prompt_embeds = self.encode_prompt(request.negative_prompt)

        reference_pixels = self._pixels_to_tensor([reference_image])  # [1, 3, 1, H, W]
        reference_latents = self._encode_pixels(reference_pixels)  # [1, 16, 1, lat_h, lat_w]
        reference_image_embeds = self.encode_image(reference_image)
        reference_mask = get_i2v_mask(1, lat_h, lat_w, 1, device, dtype)
        reference_condition = torch.cat([reference_mask.unsqueeze(0), reference_latents], dim=1)

        # ---- 3. Segment loop ----
        segments: list[torch.Tensor] = []
        previous_frames: torch.Tensor | None = None
        start = 0
        origin_area = (request.width, request.height)

        while start + _OVERLAP_FRAMES < len(driving_frames):
            clip_len = min(segment_frames, len(driving_frames) - start)
            conditioned_frames = 0 if start == 0 else _OVERLAP_FRAMES

            # One latent frame more than the driving clip: the reference-image slot.
            latent_frames = (clip_len + 1) // 4 + 2
            self.scheduler.set_timesteps(request.num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps
            self._num_timesteps = len(timesteps)

            latents = randn_tensor(
                (1, _TRANSFORMER_LATENT_CHANNELS, latent_frames, lat_h, lat_w),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )

            # Segment conditioning: reference image at latent frame 0, the
            # previous segment's tail (or zeros) across the rest.
            tail_pixels = torch.zeros((1, 3, clip_len, frame_height, frame_width), device=device, dtype=dtype)
            if conditioned_frames and previous_frames is not None:
                tail_pixels[:, :, :conditioned_frames] = previous_frames[:, :, -conditioned_frames:]
            tail_latents = self._encode_pixels(tail_pixels)
            tail_mask = get_i2v_mask(latent_frames - 1, lat_h, lat_w, conditioned_frames, device, dtype)
            tail_condition = torch.cat([tail_mask.unsqueeze(0), tail_latents], dim=1)
            condition_latents = torch.cat([reference_condition, tail_condition], dim=2)

            # Driving segment: encoded per segment because the Wan VAE is causal.
            driving_pixels = self._pixels_to_tensor(driving_frames[start : start + clip_len])
            driving_latents = self._encode_pixels(driving_pixels)
            driving_mask = get_i2v_mask(driving_latents.shape[2], lat_h, lat_w, clip_len, device, dtype)
            driving_condition = torch.cat([driving_mask.unsqueeze(0), driving_latents], dim=1)
            driving_image_embeds = self.encode_image(driving_frames[start])

            patch_h = lat_h // self.transformer.config.patch_size[1]
            patch_w = lat_w // self.transformer.config.patch_size[2]
            generation_grid = (latent_frames, patch_h, patch_w)
            reference_grid = (driving_latents.shape[2], patch_h, patch_w)

            # Phase 1: fill the per-layer reference K/V cache.
            kv_cache = self._extract_reference(
                driving_latents, driving_condition, reference_prompt_embeds, driving_image_embeds, generation_grid
            )

            # Phase 2: denoise against it.
            latents = self.diffuse(
                latents,
                timesteps,
                condition_latents=condition_latents,
                kv_cache=kv_cache,
                reference_grid_sizes=reference_grid,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                image_embeds=reference_image_embeds,
                origin_len=segment_frames,
                origin_area=origin_area,
                guidance_scale=request.guidance_scale,
                generator=generator,
            )
            kv_cache.release()
            current_omni_platform.empty_cache()

            # Decode, dropping the reference-image latent slot.
            if self._should_release_dit_before_decode():
                self.transformer.to("cpu")
                current_omni_platform.empty_cache()
            segment_video = self._decode_latents(latents[:, :, 1:], clip_len, frame_height, frame_width)
            previous_frames = segment_video
            segments.append(segment_video[:, :, conditioned_frames:].cpu())

            start += clip_len - _OVERLAP_FRAMES
            current_omni_platform.empty_cache()

        # ---- 4. Reassemble: undo zigzag padding, then the letterbox ----
        video = torch.cat(segments, dim=2)[:, :, :source_frame_count]
        video = letterbox.crop(video)

        return split_diffusion_output_by_request(
            DiffusionOutput(output=video, stage_durations=getattr(self, "stage_durations", None)),
            req,
            num_outputs_per_prompt=1,
        )

    def _should_release_dit_before_decode(self) -> bool:
        if getattr(self.od_config.parallel_config, "use_hsdp", False):
            return False
        return self.od_config.enable_cpu_offload and DIT_COMPONENT in selected_offload_components(self.od_config)

    def _extract_reference(
        self,
        driving_latents: torch.Tensor,
        driving_condition: torch.Tensor,
        reference_prompt_embeds: torch.Tensor,
        driving_image_embeds: torch.Tensor,
        generation_grid: tuple[int, int, int],
    ) -> ReferenceKVCache:
        """Run the segment-level extraction pass, honouring offload/FSDP state.

        ``extract_reference`` bypasses the transformer's ``__call__``, so the
        hooks that model-level CPU offload and FSDP rely on never fire; both
        are driven explicitly here, mirroring how S2V guards
        ``transformer.encode_audio``.
        """
        transformer = self.transformer
        moved_to_gpu = False
        if self.od_config.enable_cpu_offload and next(transformer.parameters()).device.type == "cpu":
            transformer.to(self.device)
            moved_to_gpu = True

        is_fsdp = hasattr(transformer, "unshard") and hasattr(transformer, "reshard")
        if is_fsdp:
            transformer.unshard()

        try:
            param_dtype = next(transformer.parameters()).dtype
            with torch.amp.autocast(self.device.type, dtype=param_dtype):
                return transformer.extract_reference(
                    driving_latents,
                    driving_condition,
                    reference_prompt_embeds,
                    driving_image_embeds,
                    generation_grid,
                )
        finally:
            if is_fsdp:
                transformer.reshard()
            if moved_to_gpu:
                transformer.to("cpu")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights)


def _is_distilled_checkpoint(model: str, local_files_only: bool) -> bool:
    """Tell the base and distilled Diffusers releases apart.

    Both ship the same ``model_index.json`` (``_class_name: WanAnimate2Pipeline``)
    and the same ``transformer/config.json``; only the modular index differs in
    its blocks class.  A checkpoint without a modular index is treated as base.
    """
    try:
        modular_index = _load_json(model, "modular_model_index.json", local_files_only)
    except (OSError, ValueError):
        logger.info("No modular_model_index.json under %s; assuming the base (non-distilled) DiT.", model)
        return False
    return modular_index.get("_blocks_class_name") == _DISTILLED_BLOCKS_CLASS


def _load_scheduler(
    model: str, local_files_only: bool, flow_shift: float
) -> DPMSolverMultistepScheduler | FlowMatchEulerDiscreteScheduler:
    """Instantiate the scheduler the checkpoint ships, with its shift overridden.

    The base release uses DPM-Solver++ over flow sigmas and the distilled
    release a shifted flow-match Euler; both take ``flow_shift=5.0`` from the
    checkpoint by default.
    """
    scheduler_class = _load_json(model, "scheduler/scheduler_config.json", local_files_only).get("_class_name")
    if scheduler_class == "DPMSolverMultistepScheduler":
        return DPMSolverMultistepScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only, flow_shift=flow_shift
        )
    if scheduler_class == "FlowMatchEulerDiscreteScheduler":
        return FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only, shift=flow_shift
        )
    raise ValueError(f"unsupported Wan2.2-Animate-2 scheduler {scheduler_class!r} in {model}")
