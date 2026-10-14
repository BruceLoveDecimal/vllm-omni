# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from Alaya-EVOKE (https://github.com/AlayaLab/Evoke @ 1d271be),
# evoke/pipelines/pipeline_evoke.py and scripts/inference/infer_post_distill.sh, Apache-2.0.
"""Alaya-EVOKE real-time world model (``stage3_post_distillation``) for vLLM-Omni.

Each chunk is 9 latent frames (36 frames, 1.5 s at 24 fps), sampled with a
3-stage pyramid at one UniPC step per stage and no CFG. Camera control enters
only through geometry: a point-cloud bank built from the frames generated so
far is rendered at the chunk's commanded poses, VAE-encoded, and placed in the
short history tier as ``[prefix | warp | prev_short]``. Warp tokens of
disoccluded regions are dropped inside the transformer. The warp conditions
the coarsest pyramid stage only.

Supported modes: i2v (image + camera) and t2v (text only, no camera). The
recipe is pinned to the shipped checkpoint; see ``EvokeRecipe``.
"""

from __future__ import annotations

import html
import logging
import math
import os
import weakref
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import PIL.Image
import regex as re
import torch
from diffusers.utils.torch_utils import randn_tensor
from typing_extensions import override

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.interaction.modality_handlers.camera import CameraSession
from vllm_omni.diffusion.interaction.types import ChunkMediaSpec
from vllm_omni.diffusion.models.evoke.camera import default_intrinsics, unity_to_opencv
from vllm_omni.diffusion.models.evoke.evoke_transformer import EvokeTransformer3DModel
from vllm_omni.diffusion.models.evoke.geometry.cloud import render_intrinsics, visibility_to_latent
from vllm_omni.diffusion.models.evoke.geometry.session import GeometryRecipe, GeometrySession
from vllm_omni.diffusion.models.evoke.geometry.vigeo_adapter import ViGeoDepthEstimator
from vllm_omni.diffusion.models.helios.pipeline_helios import HeliosPipeline, calculate_shift
from vllm_omni.diffusion.models.helios.scheduling_helios import HeliosScheduler
from vllm_omni.diffusion.models.interface import SupportImageInput
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.experimental.ar_diffusion.streaming_decode import StreamingDecodeState, WanStreamingDecoder
from vllm_omni.platforms import current_omni_platform

if TYPE_CHECKING:
    from vllm_omni.diffusion.worker.input_batch import InputBatch
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvokeRecipe:
    """Sampling recipe the post-distilled checkpoint was trained for (not request knobs)."""

    history_sizes: tuple[int, int, int] = (16, 2, 1)
    latent_window_size: int = 9
    pyramid_num_stages: int = 3
    pyramid_steps: tuple[int, ...] = (1, 1, 1)
    stage_range: tuple[float, ...] = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
    gamma: float = 1.0 / 3.0
    # mu = exp(min(shift_mu, log(exp_max))) of the exponential time shift
    time_shift_exp_max: float = 7.0
    max_sequence_length: int = 512
    image_noise_sigma: tuple[float, float] = (0.0, 0.0)
    fake_image_noise_sigma: tuple[float, float] = (0.111, 0.135)
    warp_noise_sigma: tuple[float, float] = (0.0, 0.135)
    warp_noise_sigma_invisible: float = 1.0
    # The latent visibility mask is binary per token, so any threshold in (0, 1] is equivalent.
    visible_token_threshold: float = 0.5
    # Real prior latents prepended to warm the causal decoder on chunk 0 (i2v).
    decode_warm_latents: int = 4
    fps: int = 24


RECIPE = EvokeRecipe()

# Sampling-recipe knobs of Helios-style pipelines. The distilled checkpoint only
# works with its trained recipe, so these are rejected instead of ignored.
_RECIPE_EXTRA_ARGS = frozenset(
    {
        "history_sizes",
        "num_latent_frames_per_chunk",
        "keep_first_frame",
        "is_enable_stage2",
        "pyramid_num_stages",
        "pyramid_num_inference_steps_list",
        "is_amplify_first_chunk",
        "use_cfg_zero_star",
        "use_dmd",
        "video",
    }
)


def get_evoke_post_process_func(od_config: OmniDiffusionConfig):
    from diffusers.video_processor import VideoProcessor

    video_processor = VideoProcessor(vae_scale_factor=8)

    def post_process_func(video: torch.Tensor, output_type: str = "np"):
        if output_type == "latent":
            return video
        return video_processor.postprocess_video(video, output_type=output_type)

    return post_process_func


def prompt_clean(text: str) -> str:
    """Upstream ``prompt_clean``: ftfy (when installed), double HTML unescape, collapse whitespace."""
    try:
        import ftfy

        text = ftfy.fix_text(text)
    except ImportError:
        pass
    text = html.unescape(html.unescape(text)).strip()
    return re.sub(r"\s+", " ", text).strip()


def num_chunks_for_frames(num_frames: int, latent_window_size: int = 9, vae_t: int = 4) -> int:
    """Upstream chunk count: ``ceil(num_frames / 33)`` (each chunk then decodes 36 frames)."""
    window = (latent_window_size - 1) * vae_t + 1
    return max(1, (max(int(num_frames), 1) + window - 1) // window)


def chunk_media_frames(chunk_index: int, *, has_prior: bool, latent_window_size: int = 9, vae_t: int = 4) -> int:
    """Frames a chunk decodes to before output trimming.

    A cold causal decoder expands the session's first latent to one frame; with
    a real prior frame (i2v) the decoder is warmed and every latent gives
    ``vae_t`` frames.
    """
    if chunk_index == 0 and not has_prior:
        return (latent_window_size - 1) * vae_t + 1
    return latent_window_size * vae_t


def _load_image(image: Any) -> PIL.Image.Image:
    if isinstance(image, (str, os.PathLike)):
        with PIL.Image.open(image) as opened:
            return opened.convert("RGB")
    if isinstance(image, PIL.Image.Image):
        return image.convert("RGB")
    raise ValueError("Evoke image input must be a PIL image or a file path.")


def _image_to_tensor(image: PIL.Image.Image, height: int, width: int) -> torch.Tensor:
    """LANCZOS resize to the output size, then ``[1, 3, H, W]`` in ``[-1, 1]`` (upstream infer_single)."""
    resized = image.resize((int(width), int(height)), PIL.Image.LANCZOS)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)[None] * 2.0 - 1.0


def _as_c2ws(value: Any) -> torch.Tensor:
    poses = torch.as_tensor(np.asarray(value, dtype=np.float64))
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or poses.shape[0] < 1:
        raise ValueError("camera_c2ws must be a non-empty [N, 4, 4] list of camera-to-world matrices")
    if not torch.isfinite(poses).all():
        raise ValueError("camera_c2ws must be finite")
    return poses


def _request_prompt(state: StepRequestState) -> tuple[str, Any]:
    prompt = state.prompt
    image = None
    if isinstance(prompt, dict):
        multi_modal_data = prompt.get("multi_modal_data") or {}
        if not isinstance(multi_modal_data, dict):
            raise ValueError("prompt.multi_modal_data must be a mapping")
        image = multi_modal_data.get("image")
        prompt = prompt.get("prompt")
    if isinstance(prompt, list):
        if len(prompt) != 1:
            raise ValueError("Evoke generates one session per request; got a batched prompt.")
        prompt = prompt[0]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Evoke requires a non-empty text prompt.")
    if isinstance(image, list):
        if len(image) != 1:
            raise ValueError("Evoke accepts exactly one conditioning image.")
        image = image[0]
    return prompt, image


@dataclass
class _EvokeSession:
    """Request-owned rollout state that must never cross requests.

    Lives in ``state.extra`` so it is released with the request state on
    completion, cancellation or error; ``close`` frees it eagerly on completion.
    """

    decode_state: StreamingDecodeState
    geometry: GeometrySession | None
    # OpenCV c2w of every 24 fps frame committed so far, [N, 4, 4] float64 on CPU.
    c2ws: torch.Tensor = field(default_factory=lambda: torch.zeros((0, 4, 4), dtype=torch.float64))
    frames_emitted: int = 0

    def close(self, decoder: WanStreamingDecoder) -> None:
        decoder.release(self.decode_state)
        if self.geometry is not None:
            self.geometry.close()
            self.geometry = None


class EvokePipeline(HeliosPipeline, SupportImageInput):
    """Evoke real-time pipeline on the Helios step-execution skeleton."""

    supports_step_execution: ClassVar[bool] = True
    # A rollout needs a conditioning image, camera and ViGeo; a generic warmup
    # request exercises none of that and would fail the pinned-recipe checks.
    dummy_run_num_frames: ClassVar[int] = 0
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        model_config = getattr(od_config, "model_config", None) or {}
        self._streaming_decoder = WanStreamingDecoder(self.vae)
        self._geometry_recipe = GeometryRecipe(
            lag_chunks=int(model_config.get("evoke_warp_lag_chunks", 0)),
            max_bank_frames=int(model_config.get("evoke_geometry_max_bank_frames", 0)),
        )
        vigeo_path = model_config.get("evoke_vigeo_path")
        if vigeo_path is None and os.path.isdir(od_config.model):
            candidate = os.path.join(od_config.model, "vigeo", "vigeo.pt")
            vigeo_path = candidate if os.path.isfile(candidate) else None
        # Built eagerly (cheap), loaded on first use so the 5 GB model stays off
        # replicas that only serve t2v.
        self._depth_estimator = (
            ViGeoDepthEstimator(vigeo_path, self.device, depth_median_target=5.0) if vigeo_path else None
        )
        if bool(model_config.get("evoke_preload_vigeo", False)):
            if self._depth_estimator is None:
                raise ValueError("evoke_preload_vigeo requires model_config.evoke_vigeo_path")
            self._depth_estimator.load()
        # The ViGeo stream (kv-cache, locked scale) can serve one live session.
        self._geometry_owner: weakref.ref[_EvokeSession] | None = None

    @override
    def _create_transformer(self, config: dict) -> EvokeTransformer3DModel:
        return EvokeTransformer3DModel(quant_config=self.od_config.quantization_config, **config)

    @override
    def _create_scheduler(self, sched_cfg: dict) -> HeliosScheduler:
        """Pyramid scheduler of the shipped recipe.

        ``evoke-base/scheduler`` is the single-stage Stage-1 scheduler; upstream
        builds the 3-stage pyramid scheduler explicitly for the distilled
        checkpoints and applies the resolution-dependent time shift per stage.
        """
        del sched_cfg
        return HeliosScheduler(
            num_train_timesteps=1000,
            shift=1.0,
            stages=RECIPE.pyramid_num_stages,
            stage_range=list(RECIPE.stage_range),
            gamma=RECIPE.gamma,
            scheduler_type="unipc",
            use_dynamic_shifting=True,
            time_shift_type="exponential",
        )

    # ------------------------------------------------------------------
    # Request setup
    # ------------------------------------------------------------------

    def _encode_text(self, prompt: str) -> torch.Tensor:
        tokens = self.tokenizer(
            [prompt_clean(prompt)],
            padding="max_length",
            max_length=RECIPE.max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        mask = tokens.attention_mask
        seq_len = int(mask.gt(0).sum())
        embeds = self.text_encoder(tokens.input_ids.to(self.device), mask.to(self.device)).last_hidden_state
        embeds = embeds.to(dtype=self.transformer.dtype, device=self.device)
        embeds[:, seq_len:] = 0
        return embeds

    @override
    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Interaction-handler entry point: CFG-free, so no negative embedding."""
        del negative_prompt, do_classifier_free_guidance, num_videos_per_prompt, max_sequence_length, device, dtype
        text = prompt[0] if isinstance(prompt, list) else prompt
        return self._encode_text(text), None

    def _latent_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        z_dim = self.vae.config.z_dim
        mean = torch.tensor(self.vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to(self.vae.device, self.vae.dtype)
        inv_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, z_dim, 1, 1, 1).to(
            self.vae.device, self.vae.dtype
        )
        return mean, inv_std

    def _encode_video_window(self, video: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        """Upstream ``prepare_video_latents`` for a single 33-frame window.

        The first frame is encoded too (and discarded) so the generator is
        consumed in the same order as upstream.
        """
        mean, inv_std = self._latent_stats()
        video = video.to(device=self.device, dtype=self.vae.dtype)
        window = (RECIPE.latent_window_size - 1) * self.vae_scale_factor_temporal + 1
        if video.shape[2] < window:
            raise ValueError(f"warp video needs {window} frames, got {video.shape[2]}")
        start = video.shape[2] - (video.shape[2] // window) * window
        self.vae.encode(video[:, :, 0:1]).latent_dist.sample(generator=generator)
        latents = self.vae.encode(video[:, :, start : start + window]).latent_dist.sample(generator=generator)
        return ((latents - mean) * inv_std).to(device=self.device, dtype=torch.float32)

    def _validate_request(self, state: StepRequestState, extra: dict[str, Any]) -> None:
        pinned = sorted(set(extra) & _RECIPE_EXTRA_ARGS)
        if pinned:
            raise ValueError(f"Evoke extra_args {pinned} are pinned by the checkpoint recipe (or unsupported).")
        if state.sampling.guidance_scale_provided and float(state.sampling.guidance_scale) > 1.0:
            raise ValueError("The Evoke real-time checkpoint is distilled CFG-free; guidance_scale must be 1.0.")
        steps = state.sampling.num_inference_steps
        if steps is not None and int(steps) != sum(RECIPE.pyramid_steps):
            raise ValueError(
                f"The Evoke real-time checkpoint samples exactly {sum(RECIPE.pyramid_steps)} steps per chunk "
                f"(one per pyramid stage); got num_inference_steps={steps}."
            )
        fps = state.sampling.fps
        if fps is not None and int(fps) != RECIPE.fps:
            raise ValueError(f"Evoke generates {RECIPE.fps} fps video; got fps={fps}.")

    @override
    def prepare_encode(self, state: StepRequestState, **kwargs: Any) -> StepRequestState:
        del kwargs
        extra_args = dict(getattr(state.sampling, "extra_args", None) or {})
        self._validate_request(state, extra_args)
        prompt, image = _request_prompt(state)
        image = extra_args.get("image", image)

        height = int(state.sampling.height or 384)
        width = int(state.sampling.width or 640)
        if height % 16 or width % 16:
            raise ValueError(f"Evoke height and width must be multiples of 16, got {height}x{width}")
        num_frames = int(state.sampling.num_frames or 33)
        if state.sampling.fps is None:
            state.sampling.fps = RECIPE.fps
        generator = state.sampling.generator
        if generator is None and state.sampling.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(state.sampling.seed)
            state.sampling.generator = generator

        self._guidance_scale = 1.0
        prompt_embeds = self._encode_text(prompt)
        mean, inv_std = self._latent_stats()
        latent_h = height // self.vae_scale_factor_spatial
        latent_w = width // self.vae_scale_factor_spatial
        channels = self.transformer.config.in_channels

        image_pixels = None
        image_latents = None
        warm_latents = None
        history_latents = torch.zeros(
            1, channels, sum(RECIPE.history_sizes), latent_h, latent_w, device=self.device, dtype=torch.float32
        )
        if image is not None:
            image_pixels = _image_to_tensor(_load_image(image), height, width).to(self.device)
            image_latents, fake_latents = self.prepare_image_latents(
                image_pixels,
                latents_mean=mean,
                latents_std=inv_std,
                num_latent_frames_per_chunk=RECIPE.latent_window_size,
                dtype=torch.float32,
                device=self.device,
                generator=generator,
            )
            warm_latents = image_latents.clone()
            image_latents = self._noise(image_latents, RECIPE.image_noise_sigma, generator)
            fake_latents = self._noise(fake_latents, RECIPE.fake_image_noise_sigma, generator)
            history_latents = torch.cat([history_latents[:, :, 1:], fake_latents], dim=2)

        geometry_on = bool(extra_args.get("geometry", image is not None))
        if geometry_on and image is None:
            raise ValueError("Evoke camera geometry needs a conditioning image (i2v); t2v has no camera control.")
        explicit_c2ws = extra_args.get("camera_c2ws")
        if explicit_c2ws is not None and not geometry_on:
            raise ValueError("camera_c2ws needs geometry; Evoke has no other camera channel.")

        geometry = None
        if geometry_on:
            owner = self._geometry_owner() if self._geometry_owner is not None else None
            if owner is not None and owner.geometry is not None:
                raise RuntimeError("Evoke geometry serves one session at a time; deploy with max_num_seqs=1.")
            if self._depth_estimator is None:
                raise ValueError("Evoke geometry needs ViGeo weights: set model_config.evoke_vigeo_path.")
            intrinsics = extra_args.get("camera_intrinsics")
            if intrinsics is None:
                intrinsics = default_intrinsics(height, width, float(extra_args.get("camera_hfov_degrees", 70.0)))
            if len(intrinsics) != 4 or not all(math.isfinite(float(v)) and float(v) > 0 for v in intrinsics):
                raise ValueError("camera_intrinsics must be four positive numbers [fx, fy, cx, cy]")
            k_pix = render_intrinsics(*(float(v) for v in intrinsics), height=height, width=width)
            # i2v: baseline-free depth_median scale (anchor collapses on a near-static first chunk).
            self._depth_estimator.configure_session(scale_mode="depth_median")
            geometry = GeometrySession(
                self._depth_estimator,
                k_pix,
                height,
                width,
                self.device,
                self._geometry_recipe,
                reference_frame=image_pixels[0],
            )

        session = _EvokeSession(
            decode_state=self._streaming_decoder.new_decode_state(state.request_id),
            geometry=geometry,
        )
        if geometry is not None:
            self._geometry_owner = weakref.ref(session)

        state.prompt_embeds = prompt_embeds
        state.negative_prompt_embeds = None
        state.do_true_cfg = False
        state.chunk_index = 0
        state.step_in_chunk = 0
        state.total_chunks = num_chunks_for_frames(
            num_frames, RECIPE.latent_window_size, self.vae_scale_factor_temporal
        )
        state.extra.update(
            {
                "batch_size": 1,
                "dtype": self.transformer.dtype,
                "generator": generator,
                "height": height,
                "width": width,
                "history_latents": history_latents,
                "image_latents": image_latents,
                "warm_latents": warm_latents,
                "has_prior": image is not None,
                "explicit_c2ws": None if explicit_c2ws is None else _as_c2ws(explicit_c2ws),
                "output_type": extra_args.get("output_type", "np"),
                "num_channels_latents": channels,
                "num_latent_frames_per_chunk": RECIPE.latent_window_size,
                "window_num_frames": (RECIPE.latent_window_size - 1) * self.vae_scale_factor_temporal + 1,
                "pyramid_num_stages": RECIPE.pyramid_num_stages,
                "pyramid_num_inference_steps_list": list(RECIPE.pyramid_steps),
                "is_amplify_first_chunk": False,
                "is_enable_stage2": True,
                "is_skip_first_chunk": False,
                "keep_first_frame": True,
                "attention_kwargs": {},
                "evoke_session": session,
            }
        )
        return state

    @staticmethod
    def _noise(latents: torch.Tensor, sigma_range: tuple[float, float], generator) -> torch.Tensor:
        lo, hi = sigma_range
        sigma = torch.rand(1, device=latents.device, generator=generator) * (hi - lo) + lo
        return sigma * randn_tensor(latents.shape, generator=generator, device=latents.device) + (1 - sigma) * latents

    # ------------------------------------------------------------------
    # Chunk setup
    # ------------------------------------------------------------------

    @override
    def peek_chunk_media(self, state: StepRequestState) -> ChunkMediaSpec:
        has_prior = bool(state.extra["has_prior"])
        # Camera poses always advance one chunk stride (36 frames), even when the
        # cold first t2v chunk decodes fewer frames.
        frames = (
            RECIPE.latent_window_size * self.vae_scale_factor_temporal
            if has_prior
            else chunk_media_frames(state.chunk_index, has_prior=False)
        )
        return ChunkMediaSpec(
            num_media_frames=frames,
            fps=float(state.sampling.fps or RECIPE.fps),
            num_latent_frames=RECIPE.latent_window_size,
        )

    def _commit_chunk_poses(self, state: StepRequestState, session: _EvokeSession) -> torch.Tensor:
        """Append this chunk's 36 OpenCV c2w poses to the session and return them."""
        stride = RECIPE.latent_window_size * self.vae_scale_factor_temporal
        start = state.chunk_index * stride
        if session.c2ws.shape[0] != start:
            raise RuntimeError(f"Evoke pose track has {session.c2ws.shape[0]} frames, expected {start}.")
        explicit = state.extra["explicit_c2ws"]
        camera = state.interaction_sessions.get("camera")
        if session.geometry is None and isinstance(camera, CameraSession) and camera.has_received_input:
            raise ValueError("This Evoke session has no geometry (t2v), so it cannot follow camera input.")
        if explicit is not None:
            if isinstance(camera, CameraSession) and camera.has_received_input:
                raise ValueError("Cannot combine camera_c2ws with mid-generation camera interaction.")
            index = torch.arange(start, start + stride).clamp(max=explicit.shape[0] - 1)
            poses = explicit[index]
        elif isinstance(camera, CameraSession) and camera.last_absolute_poses is not None:
            poses = unity_to_opencv(camera.last_absolute_poses.double())
            if poses.shape[0] != stride:
                raise ValueError(f"camera interaction produced {poses.shape[0]} poses, expected {stride}.")
        else:
            last = session.c2ws[-1] if session.c2ws.shape[0] else torch.eye(4, dtype=torch.float64)
            poses = last[None].expand(stride, 4, 4).clone()
        session.c2ws = torch.cat([session.c2ws, poses.to(torch.float64)], dim=0)
        return poses

    @override
    def prepare_next_chunk(self, state: StepRequestState) -> None:
        extra = state.extra
        session: _EvokeSession = state.extra["evoke_session"]
        k = state.chunk_index
        generator = extra["generator"]
        history_sizes = list(RECIPE.history_sizes)
        window = RECIPE.latent_window_size
        latent_h = extra["height"] // self.vae_scale_factor_spatial
        latent_w = extra["width"] // self.vae_scale_factor_spatial

        indices = torch.arange(0, sum([1, *history_sizes, window]))
        idx_prefix, idx_long, idx_mid, idx_1x, idx_hidden = indices.split([1, *history_sizes, window], dim=0)
        long_lat, mid_lat, lat_1x = extra["history_latents"][:, :, -sum(history_sizes) :].split(history_sizes, dim=2)
        prefix = extra["image_latents"]
        if prefix is None:
            prefix = torch.zeros_like(lat_1x)

        poses = self._commit_chunk_poses(state, session)
        attention_kwargs: dict[str, Any] = {}
        if session.geometry is not None:
            target = poses[: extra["window_num_frames"]].float()
            warp_video, visibility = session.geometry.render_chunk(k, target)
            warp_latents = self._encode_video_window(warp_video, generator)
            frame_sigma = torch.rand(window, device=self.device, generator=generator)
            lo, hi = RECIPE.warp_noise_sigma
            frame_sigma = (frame_sigma * (hi - lo) + lo).view(1, 1, window, 1, 1)
            mask = visibility_to_latent(
                visibility, window, latent_h, latent_w, vae_t_stride=self.vae_scale_factor_temporal
            )
            sigma = mask * frame_sigma + (1.0 - mask) * RECIPE.warp_noise_sigma_invisible
            noise = randn_tensor(warp_latents.shape, generator=generator, device=self.device)
            warp_latents = sigma * noise + (1.0 - sigma) * warp_latents
            prev_short = lat_1x[:, :, -1:] if k > 0 else prefix[:, :, -1:]
            short_lat = torch.cat([prefix, warp_latents, prev_short], dim=2)
            short_idx = torch.cat([idx_prefix, idx_hidden, idx_1x[-1:]])
            ones = torch.ones(1, 1, 1, latent_h, latent_w, device=self.device)
            attention_kwargs = {
                "history_visible_token_threshold": RECIPE.visible_token_threshold,
                "history_visible_mask_short": torch.cat([ones, mask, ones], dim=2),
            }
            # Coarser pyramid stages see [prefix | prev_short] only (warp at stage 0).
            extra["short_without_warp"] = (
                torch.cat([prefix, prev_short], dim=2),
                torch.cat([idx_prefix, idx_1x[-1:]]).unsqueeze(0),
            )
        else:
            short_lat = torch.cat([prefix, lat_1x], dim=2)
            short_idx = torch.cat([idx_prefix, idx_1x])
            extra["short_without_warp"] = None

        state.latents = self.prepare_latents(
            1,
            extra["num_channels_latents"],
            extra["height"],
            extra["width"],
            extra["window_num_frames"],
            dtype=torch.float32,
            device=self.device,
            generator=generator,
        )
        extra.update(
            {
                "is_first_chunk": k == 0,
                "is_second_chunk": k == 1,
                "latents_history_long": long_lat,
                "latents_history_mid": mid_lat,
                "latents_history_short": short_lat,
                "indices_hidden_states": idx_hidden.unsqueeze(0),
                "indices_latents_history_long": idx_long.unsqueeze(0),
                "indices_latents_history_mid": idx_mid.unsqueeze(0),
                "indices_latents_history_short": short_idx.unsqueeze(0),
                "warp_attention_kwargs": attention_kwargs,
            }
        )
        self._prepare_stage2_chunk(state)
        state.step_in_chunk = 0
        self._num_timesteps = state.chunk_num_steps

    @override
    def _set_stage2_timesteps(self, state: StepRequestState) -> None:
        """Per-stage timesteps with upstream's capped exponential time shift."""
        extra = state.extra
        patch_size = self.transformer.config.patch_size
        image_seq_len = (state.latents.shape[-1] * state.latents.shape[-2] * state.latents.shape[-3]) // (
            patch_size[0] * patch_size[1] * patch_size[2]
        )
        mu = min(calculate_shift(image_seq_len), math.log(RECIPE.time_shift_exp_max))
        stage_index = extra["stage_index"]
        self.scheduler.set_timesteps(
            extra["pyramid_num_inference_steps_list"][stage_index],
            stage_index=stage_index,
            device=self.device,
            mu=mu,
        )
        state.timesteps = self.scheduler.timesteps
        state.step_index = extra["stage_step_index"]

    # ------------------------------------------------------------------
    # Denoising
    # ------------------------------------------------------------------

    @override
    def denoise_step(
        self,
        input_batch: InputBatch,
        states: Sequence[StepRequestState],
        **kwargs: Any,
    ) -> torch.Tensor | None:
        del input_batch, kwargs
        if len(states) != 1:
            raise ValueError("Evoke step execution runs one session per replica.")
        state = states[0]
        extra = state.extra
        t = state.current_timestep
        assert t is not None and state.latents is not None
        self._current_timestep = t
        dtype = extra["dtype"]
        short_lat = extra["latents_history_short"]
        short_idx = extra["indices_latents_history_short"]
        attention_kwargs = extra["warp_attention_kwargs"]
        if extra["stage_index"] > 0 and extra["short_without_warp"] is not None:
            short_lat, short_idx = extra["short_without_warp"]
            attention_kwargs = {}
        return self.transformer(
            hidden_states=state.latents.to(dtype),
            timestep=t.expand(1).to(torch.int64),
            encoder_hidden_states=state.prompt_embeds,
            indices_hidden_states=extra["indices_hidden_states"],
            indices_latents_history_short=short_idx,
            indices_latents_history_mid=extra["indices_latents_history_mid"],
            indices_latents_history_long=extra["indices_latents_history_long"],
            latents_history_short=short_lat.to(dtype),
            latents_history_mid=extra["latents_history_mid"].to(dtype),
            latents_history_long=extra["latents_history_long"].to(dtype),
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]

    @override
    def step_scheduler(self, state: StepRequestState, noise_pred: torch.Tensor, **kwargs: Any) -> None:
        del kwargs
        self._step_scheduler_stage2(state, noise_pred.float())

    # ------------------------------------------------------------------
    # Decode and geometry update
    # ------------------------------------------------------------------

    def _decode_chunk(self, state: StepRequestState, session: _EvokeSession, latents: torch.Tensor) -> torch.Tensor:
        mean, inv_std = self._latent_stats()
        warm = 0
        warm_latents = state.extra["warm_latents"]
        if state.chunk_index == 0 and warm_latents is not None:
            warm = min(RECIPE.decode_warm_latents, int(warm_latents.shape[2]))
            latents = torch.cat([warm_latents[:, :, -warm:].to(latents), latents], dim=2)
        z = latents.to(self.vae.dtype) / inv_std + mean
        video = self._streaming_decoder.decode_chunk(z, session.decode_state)
        if warm:
            video = video[:, :, 1 + self.vae_scale_factor_temporal * (warm - 1) :]
        return video

    @override
    def post_decode(self, state: StepRequestState, **kwargs: Any) -> DiffusionOutput:
        del kwargs
        extra = state.extra
        session: _EvokeSession = state.extra["evoke_session"]
        k = state.chunk_index
        latents = state.latents
        if extra["image_latents"] is None:
            extra["image_latents"] = latents[:, :, 0:1]
        history_frames = sum(RECIPE.history_sizes)
        extra["history_latents"] = torch.cat([extra["history_latents"], latents], dim=2)[:, :, -history_frames:]

        decoded = self._decode_chunk(state, session, latents)
        if session.geometry is not None:
            stride = RECIPE.latent_window_size * self.vae_scale_factor_temporal
            session.geometry.ingest(decoded[0], k * stride, session.c2ws)

        output = decoded
        if k == 0 and not extra["has_prior"] and output.shape[2] > 1:
            # A cold decoder's first frame is the VAE's I-frame slot; upstream drops it for t2v.
            output = output[:, :, 1:]
        state.chunk_index += 1
        finished = state.request_denoise_completed
        if finished:
            # Upstream trims the clip to 4n + 1 frames.
            total = session.frames_emitted + int(output.shape[2]) + (0 if extra["has_prior"] else 1)
            keep = (total - 1) // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
            excess = total - keep
            if excess > 0:
                output = output[:, :, : output.shape[2] - excess]
        session.frames_emitted += int(output.shape[2])
        if finished:
            self._current_timestep = None
            session.close(self._streaming_decoder)
            if current_omni_platform.is_available():
                current_omni_platform.empty_cache()
        if extra["output_type"] == "latent":
            output = latents
        return DiffusionOutput(
            output=output,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else {},
            chunk_index=k,
            total_chunks=state.total_chunks,
            finished=finished,
        )

    # ------------------------------------------------------------------
    # Request-level execution
    # ------------------------------------------------------------------

    @override
    def forward(self, req: DiffusionRequestBatch, **kwargs: Any) -> DiffusionOutput:
        """Offline generation: run the step-execution path to completion on the host."""
        del kwargs
        if len(req.requests) != 1:
            raise ValueError("Evoke generates one session per request.")
        request: OmniDiffusionRequest = req.requests[0]
        state = StepRequestState(
            request_id=request.request_id or "evoke-offline",
            sampling=request.sampling_params,
            prompt=request.prompt,
        )
        self.prepare_encode(state)
        chunks: list[torch.Tensor] = []
        try:
            while not state.request_denoise_completed:
                self.prepare_next_chunk(state)
                for _ in range(state.chunk_num_steps):
                    noise_pred = self.denoise_step(None, [state])
                    self.step_scheduler(state, noise_pred)
                chunks.append(self.post_decode(state).output.cpu())
        finally:
            state.extra["evoke_session"].close(self._streaming_decoder)
        return DiffusionOutput(output=torch.cat(chunks, dim=2))
