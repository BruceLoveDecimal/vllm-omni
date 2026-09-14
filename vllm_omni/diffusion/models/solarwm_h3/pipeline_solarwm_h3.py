# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SolarWM-H3: camera-controlled first-frame-to-video with the MiniMax-H3 Stage-2 student.

A request carries the first frame, a caption and one absolute camera-to-world
pose per output frame. The pipeline reuses MiniMax-H3's Qwen3-VL text encoder
and video VAE, merges the SolarWM LoRA into the DiT at load time and runs the
self-forcing rollout at the released 768x1344 geometry.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import Qwen2TokenizerFast, Qwen3VLProcessor
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import SupportImageInput, SupportsComponentDiscovery
from vllm_omni.diffusion.models.minimax_h3.encoder import MiniMaxH3Qwen3VLEncoder
from vllm_omni.diffusion.models.minimax_h3.packed_tokens import minimax_h3_pack_audio_latent
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _SingleRankEncoderGroup
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.offloader import OffloadPlan
from vllm_omni.diffusion.offloader.config import TEXT_ENCODER_COMPONENT
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.errors import OmniClientError
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific
from vllm_omni.model_executor.models.minimax_h3.preprocessing import minimax_h3_multi_image_presentation
from vllm_omni.quantization import resolve_component_quant_config

from .camera import first_frame_relative_w2c
from .layout import (
    SOLARWM_H3_AUDIO_LATENTS,
    SOLARWM_H3_LATENT_CHANNELS,
    SOLARWM_H3_LATENT_HEIGHT,
    SOLARWM_H3_LATENT_WIDTH,
    SOLARWM_H3_PIXEL_HEIGHT,
    SOLARWM_H3_PIXEL_WIDTH,
    build_prefix_layout,
    rollout_geometry,
)
from .rollout import (
    SOLARWM_H3_ANCHOR_TIMESTEP,
    SOLARWM_H3_AUDIO_SHIFT,
    RolloutConditions,
    SolarWMRollout,
    patchify_latents,
    shift_sigma,
)
from .solarwm_h3_transformer import SolarWMH3DiTModel
from .weights import SolarWMWeightAdapter, load_solarwm_lora, rope_inverse_frequencies

logger = init_logger(__name__)

SOLARWM_H3_HUB_ID = "junchaoh-cs/SolarWM-H3-33B"
SOLARWM_H3_BASE_SUBFOLDER = "SolarWM-h3-33B-base"
SOLARWM_H3_ADAPTER_SUBFOLDER = "SolarWM-h3-33B-sgf-stage2-158f"
SOLARWM_H3_DEFAULT_NUM_FRAMES = 158
SOLARWM_H3_DEFAULT_FPS = 24.0
# The audio VAE runs at 32 kHz with an 800-sample hop, 40 latents per second.
SOLARWM_H3_AUDIO_HOP_LENGTH = 800
SOLARWM_H3_KEYFRAME_ENCODE_SEED = 42
_PIXEL_MEAN = (0.485, 0.456, 0.406)
_PIXEL_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class SolarWMH3Paths:
    base: Path
    adapter: Path


def resolve_solarwm_h3_paths(model: str, revision: str | None) -> SolarWMH3Paths:
    """Locate the base checkpoint and the Stage-2 package under a SolarWM-H3 repository root."""
    root = Path(model)
    if not root.is_dir():
        root = Path(
            download_weights_from_hf_specific(
                model_name_or_path=model,
                cache_dir=None,
                allow_patterns=[f"{SOLARWM_H3_BASE_SUBFOLDER}/**", f"{SOLARWM_H3_ADAPTER_SUBFOLDER}/**"],
                revision=revision,
                require_all=True,
            )
        )
    return SolarWMH3Paths(base=root / SOLARWM_H3_BASE_SUBFOLDER, adapter=root / SOLARWM_H3_ADAPTER_SUBFOLDER)


def camera_c2w_from_request(value: Any) -> torch.Tensor:
    """Read absolute camera-to-world poses ``[T, 4, 4]`` from a nested list or a file path."""
    if isinstance(value, str):
        path = Path(value)
        if path.suffix == ".npy":
            poses = np.load(path)
        elif path.suffix == ".npz":
            poses = np.load(path)["c2w"]
        else:
            poses = json.loads(path.read_text())
        value = np.asarray(poses, dtype=np.float32)
    c2w = torch.as_tensor(value, dtype=torch.float32)
    if c2w.ndim != 3 or tuple(c2w.shape[1:]) != (4, 4):
        raise OmniClientError(f"camera_c2w must be [num_frames, 4, 4], got {list(c2w.shape)}")
    return c2w


def get_solarwm_h3_post_process_func(od_config: OmniDiffusionConfig) -> Callable[..., Any]:
    del od_config
    from diffusers.video_processor import VideoProcessor

    video_processor = VideoProcessor(vae_scale_factor=16)

    def post_process_func(video: torch.Tensor, output_type: str = "np", sampling_params: Any = None) -> Any:
        if sampling_params is not None and sampling_params.output_type is not None:
            output_type = sampling_params.output_type
        if output_type == "latent":
            return video
        return {
            "payload": {"video": video_processor.postprocess_video(video, output_type=output_type)},
            "metadata": {},
        }

    return post_process_func


class SolarWMH3Pipeline(nn.Module, ProgressBarMixin, SupportImageInput, SupportsComponentDiscovery):
    """Camera-controlled image-to-video with the SolarWM-H3 Stage-2 student."""

    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["video_vae"]
    _offload_plan: ClassVar[OffloadPlan] = OffloadPlan(
        encoder_component_types={"text_encoder": TEXT_ENCODER_COMPONENT},
        encoder_block_attrs={"text_encoder": ("vision.blocks", "text_model.layers")},
    )
    # The generic warmup request has no camera trajectory; requests validate
    # their own inputs instead.
    dummy_run_num_frames: ClassVar[int] = 0

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        del prefix
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        parallel = od_config.parallel_config
        if parallel.ulysses_degree != 1 or parallel.ring_degree != 1 or parallel.cfg_parallel_size != 1:
            raise ValueError("SolarWM-H3 runs its own windowed attention; use tensor parallelism only")

        paths = resolve_solarwm_h3_paths(str(od_config.model), od_config.revision)
        base = paths.base
        self.transformer = SolarWMH3DiTModel(
            od_config,
            quant_config=resolve_component_quant_config(od_config.quantization_config, "transformer"),
        )
        arch = self.transformer.arch
        transformer_config = json.loads((base / "transformer" / "config.json").read_text())
        checkpoint_arch = (
            transformer_config["num_layers"],
            transformer_config["num_attention_heads"],
            transformer_config["attention_head_dim"],
        )
        if checkpoint_arch != (arch.num_layers, arch.num_attention_heads, arch.attention_head_dim):
            raise ValueError(f"{base} is not the MiniMax-H3 33B transformer: {checkpoint_arch}")
        self._weight_adapter = SolarWMWeightAdapter(
            load_solarwm_lora(paths.adapter),
            head_dim=arch.attention_head_dim,
            rope_inv_freq=rope_inverse_frequencies(
                transformer_config["rope_freq_dim"], transformer_config["rope_theta"]
            ),
        )

        self.tokenizer = Qwen2TokenizerFast.from_pretrained(str(base), subfolder="tokenizer", local_files_only=True)
        self.processor = Qwen3VLProcessor.from_pretrained(str(base), subfolder="processor", local_files_only=True)
        # The 50-layer Qwen3-VL encoder does not fit next to the 33B DiT on one
        # 96 GiB accelerator. Build it on the host; it moves to the device for
        # each encode, whole or block by block when layerwise offload is on.
        with torch.device("cpu"):
            self.text_encoder = MiniMaxH3Qwen3VLEncoder(
                str(base / "text_encoder"),
                device=self.device,
                load_model=True,
                encoder_group=_SingleRankEncoderGroup(rank=0),
            )
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=str(base),
                subfolder="transformer",
                revision=od_config.revision,
                prefix="transformer.",
                fall_back_to_pt=False,
            ),
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=str(base),
                subfolder="text_encoder",
                revision=od_config.revision,
                prefix="text_encoder.",
                fall_back_to_pt=False,
            ),
        ]

        from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio

        self.video_vae = AutoencoderKLMiniMaxH3.from_pretrained(str(base), subfolder="vae", torch_dtype=torch.float32)
        self.video_vae.set_attention_backend("native")
        self.video_vae.eval().requires_grad_(False).to(self.device)
        self.vae = self.video_vae

        # The audio branch is a fixed condition: the encoded 158-frame stereo
        # silence. It depends only on the audio VAE, so it is produced once here.
        audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(
            str(base), subfolder="audio_vae", torch_dtype=torch.bfloat16
        )
        audio_vae.set_attention_backend("native")
        audio_vae.eval().requires_grad_(False).to(self.device)
        self.silence_rows = self._encode_silence(audio_vae)
        del audio_vae

    @torch.inference_mode()
    def _encode_silence(self, audio_vae: nn.Module) -> torch.Tensor:
        waveform = torch.zeros(2, 1, SOLARWM_H3_AUDIO_LATENTS * SOLARWM_H3_AUDIO_HOP_LENGTH, device=self.device)
        latent = audio_vae.encode(waveform, return_dict=False)[0].mode().float().cpu()
        mean = torch.tensor(audio_vae.config.latents_mean).view(1, -1, 1)
        std = torch.tensor(audio_vae.config.latents_std).view(1, -1, 1)
        normalized = ((latent - mean) / std).to(torch.bfloat16)
        return minimax_h3_pack_audio_latent(normalized.float()).to(self.device)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded: set[str] = set()
        for prefix, grouped in groupby(weights, key=lambda item: item[0].partition(".")[0] + "."):
            stream = ((name[len(prefix) :], tensor) for name, tensor in grouped)
            if prefix == "transformer.":
                names = self.transformer.load_weights(self._weight_adapter.apply(stream))
                self._weight_adapter.validate_fully_applied()
                self.transformer.post_load_weights()
            elif prefix == "text_encoder.":
                names = self.text_encoder.load_weights(stream)
            else:
                raise ValueError(f"unexpected SolarWM-H3 weight prefix {prefix!r}")
            loaded.update(prefix + name for name in names)
        return loaded

    # ------------------------------------------------------------------
    # Conditioning
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _encode_prompt(self, prompt: str, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        vision = self.processor.image_processor(images=[image], return_tensors="pt")
        merge = int(self.processor.image_processor.merge_size) ** 2
        image_tokens = int(vision["image_grid_thw"][0].prod().item()) // merge
        ids, tags = minimax_h3_multi_image_presentation(
            self.tokenizer, prompt=prompt, image_token_counts=[image_tokens]
        )
        self.text_encoder.load_to_device()
        hidden = self.text_encoder.encode_ids(
            ids, pixel_values=vision["pixel_values"], image_grid_thw=vision["image_grid_thw"]
        )
        if getattr(self.text_encoder, "_omni_layerwise_enabled", False):
            self.text_encoder.offload_to_cpu()
        return hidden.to(self.device), tags

    @torch.inference_mode()
    def _encode_anchor(self, image: Image.Image) -> torch.Tensor:
        """Encode the first frame like the released keyframe recipe; returns ``[1008, 96]`` rows."""
        pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)[None, :, None].to(self.device)
        pixel_mean = torch.tensor(_PIXEL_MEAN, device=self.device).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor(_PIXEL_STD, device=self.device).view(1, -1, 1, 1, 1)
        normalized = (pixels.float() / 255.0 - pixel_mean) / pixel_std
        posterior = self.video_vae.encode(normalized, return_dict=False)[0]
        latent = posterior.sample(generator=torch.Generator().manual_seed(SOLARWM_H3_KEYFRAME_ENCODE_SEED))
        # The reference rounds the sampled latent to float16 before normalizing.
        latent = latent.to(torch.float16).float().cpu()
        latents_mean = torch.tensor(self.video_vae.config.latents_mean).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(self.video_vae.config.latents_std).view(1, -1, 1, 1, 1)
        return patchify_latents((latent - latents_mean) / latents_std).to(self.device)

    @torch.inference_mode()
    def _decode(self, latents: torch.Tensor, num_frames: int) -> torch.Tensor:
        """``[1, 24, T, 48, 84]`` normalized latents -> ``[1, 3, num_frames, H, W]`` in ``[-1, 1]``."""
        latents_mean = torch.tensor(self.video_vae.config.latents_mean, device=self.device).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(self.video_vae.config.latents_std, device=self.device).view(1, -1, 1, 1, 1)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            decoded = self.video_vae.decode(latents.float() * latents_std + latents_mean, return_dict=False)[0]
        pixel_mean = torch.tensor(_PIXEL_MEAN, device=self.device).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor(_PIXEL_STD, device=self.device).view(1, -1, 1, 1, 1)
        frames = (decoded.float() * pixel_std + pixel_mean).clamp(0.0, 1.0)[:, :, :num_frames]
        return frames * 2.0 - 1.0

    # ------------------------------------------------------------------
    # Request execution
    # ------------------------------------------------------------------

    @staticmethod
    def _request_image(raw_prompt: Any) -> tuple[str, Image.Image]:
        if isinstance(raw_prompt, str):
            raise OmniClientError("SolarWM-H3 requires a first-frame image in multi_modal_data")
        prompt = str(raw_prompt.get("prompt") or "")
        image = (raw_prompt.get("multi_modal_data") or {}).get("image")
        if isinstance(image, list):
            if len(image) != 1:
                raise OmniClientError("SolarWM-H3 accepts exactly one first-frame image")
            image = image[0]
        if isinstance(image, str):
            image = Image.open(image)
        if not isinstance(image, Image.Image):
            raise OmniClientError("SolarWM-H3 requires a first-frame image in multi_modal_data")
        if not prompt:
            raise OmniClientError("SolarWM-H3 requires a non-empty prompt")
        return prompt, image.convert("RGB")

    @torch.no_grad()
    def forward(self, request: DiffusionRequestBatch) -> DiffusionOutput:
        if len(request.prompts) != 1:
            raise OmniClientError("SolarWM-H3 generates one video per forward")
        sampling = request.sampling_params
        extra = sampling.extra_args or {}
        prompt, image = self._request_image(request.prompts[0])
        if "camera_c2w" not in extra:
            raise OmniClientError(
                "SolarWM-H3 requires extra_args['camera_c2w']: one 4x4 camera-to-world pose per frame"
            )
        camera_c2w = camera_c2w_from_request(extra["camera_c2w"])
        num_frames = int(sampling.num_frames) if sampling.num_frames > 1 else SOLARWM_H3_DEFAULT_NUM_FRAMES
        if int(camera_c2w.shape[0]) != num_frames:
            raise OmniClientError(f"camera_c2w has {camera_c2w.shape[0]} poses but num_frames is {num_frames}")
        for name, value, expected in (
            ("height", sampling.height, SOLARWM_H3_PIXEL_HEIGHT),
            ("width", sampling.width, SOLARWM_H3_PIXEL_WIDTH),
        ):
            if value is not None and int(value) != expected:
                raise OmniClientError(
                    f"SolarWM-H3 generates {SOLARWM_H3_PIXEL_WIDTH}x{SOLARWM_H3_PIXEL_HEIGHT}; got {name}={value}"
                )
        seed = 0 if sampling.seed is None else int(sampling.seed)
        cache_device = self.device if extra.get("kv_cache_on_device", False) else torch.device("cpu")

        geometry = rollout_geometry(num_frames)
        image = image.resize((SOLARWM_H3_PIXEL_WIDTH, SOLARWM_H3_PIXEL_HEIGHT), Image.Resampling.LANCZOS)
        prompt_embeds, text_tags = self._encode_prompt(prompt, image)
        anchor_rows = self._encode_anchor(image)
        relative_w2c = first_frame_relative_w2c(camera_c2w)
        rollout_views = relative_w2c[list(geometry.camera_frame_indices)]
        viewmats = torch.cat((rollout_views[:1], rollout_views))

        # The generator lives on the accelerator and the draws follow the
        # reference order and shapes (anchor noise in latent layout, audio
        # level, audio noise, rollout noise, then per-step re-noising), so a
        # seed reproduces the reference noise stream.
        generator = torch.Generator(device=self.device).manual_seed(seed)
        anchor_shape = (1, SOLARWM_H3_LATENT_CHANNELS, 1, SOLARWM_H3_LATENT_HEIGHT, SOLARWM_H3_LATENT_WIDTH)
        anchor_noise = patchify_latents(torch.randn(anchor_shape, generator=generator, device=self.device))
        anchor_rows = SOLARWM_H3_ANCHOR_TIMESTEP * anchor_rows + (1.0 - SOLARWM_H3_ANCHOR_TIMESTEP) * anchor_noise
        audio_sigma = shift_sigma(torch.rand((1,), generator=generator, device=self.device), SOLARWM_H3_AUDIO_SHIFT)
        audio_timestep = float(1.0 - audio_sigma)
        audio_noise = torch.randn(self.silence_rows.shape, generator=generator, device=self.device)
        audio_rows = audio_timestep * self.silence_rows + (1.0 - audio_timestep) * audio_noise
        noise = torch.randn(
            (
                1,
                SOLARWM_H3_LATENT_CHANNELS,
                geometry.rollout_latents,
                SOLARWM_H3_LATENT_HEIGHT,
                SOLARWM_H3_LATENT_WIDTH,
            ),
            generator=generator,
            device=self.device,
        )

        conditions = RolloutConditions(
            prompt_embeds=prompt_embeds,
            layout=build_prefix_layout(text_tags),
            anchor_rows=anchor_rows,
            audio_rows=audio_rows,
            audio_timestep=audio_timestep,
            viewmats=viewmats,
            geometry=geometry,
        )
        rollout = SolarWMRollout(self.transformer, conditions, cache_device=cache_device, generator=generator)
        with self.progress_bar(total=geometry.num_chunks) as progress:
            latents = rollout.run(noise, on_chunk=lambda _chunk: progress.update())
        latents = latents[:, :, : geometry.decode_latents]
        post_process_func = get_solarwm_h3_post_process_func(self.od_config)
        if sampling.output_type == "latent":
            return DiffusionOutput(output=latents, post_process_func=post_process_func)
        video = self._decode(latents, num_frames)
        return DiffusionOutput(output=video, post_process_func=post_process_func)


__all__ = [
    "SOLARWM_H3_ADAPTER_SUBFOLDER",
    "SOLARWM_H3_BASE_SUBFOLDER",
    "SOLARWM_H3_DEFAULT_FPS",
    "SOLARWM_H3_DEFAULT_NUM_FRAMES",
    "SOLARWM_H3_HUB_ID",
    "SolarWMH3Paths",
    "SolarWMH3Pipeline",
    "camera_c2w_from_request",
    "get_solarwm_h3_post_process_func",
    "resolve_solarwm_h3_paths",
]
