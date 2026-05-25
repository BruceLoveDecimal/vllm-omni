# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sana-WM pipeline integration.

This module wires the registry-visible surface, release-layout validation, and
an in-process native reference backend for GPU e2e testing. The backend executes
the public NVlabs/Sana Stage-1 DiT/Gated-DeltaNet/refiner Python modules without
shelling out to the CLI; a future optimization pass can port those large modules
into vLLM-Omni-native layers incrementally.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Any, ClassVar, Iterable

import torch
import torch.nn.functional as F
from torch import nn

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import SupportImageInput, SupportsComponentDiscovery
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.models.sana_wm.official_backend import (
    SANA_WM_OFFICIAL_BACKEND_ERROR,
    get_sana_wm_official_repo_path,
    is_sana_wm_official_backend_requested,
    run_sana_wm_official_backend,
)
from vllm_omni.diffusion.models.sana_wm.config import SanaWmConfig
from vllm_omni.diffusion.models.sana_wm.camera_control import (
    SanaWmCameraCondition,
    build_plucker_condition,
)
from vllm_omni.diffusion.models.sana_wm.native_backend import (
    run_sana_wm_native_backend,
    should_force_sana_wm_cli_backend,
)
from vllm_omni.diffusion.models.sana_wm.sana_wm_transformer import (
    SANA_WM_STAGE1_PROMPT_CHANNELS,
    SanaWmCameraEmbedder,
    SanaWmTransformer3DModel,
)
from vllm_omni.diffusion.models.sana_wm.scheduling_sana_wm import SanaWmFlowDpmScheduler
from vllm_omni.diffusion.models.sana_wm.weight_mapping import normalize_sana_wm_stage1_weight_name
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import (
    DiffusionPipelineProfilerMixin,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific
from vllm_omni.model_executor.stage_input_processors.sana_wm import normalize_sana_wm_payload

SANA_WM_MODEL_ID = "Efficient-Large-Model/SANA-WM_bidirectional"
SANA_WM_SCAFFOLD_ERROR = (
    "Sana-WM native Stage-1 DiT fully internal vLLM-Omni layers, Gated DeltaNet, "
    "and denoising layers are not implemented yet. The in-process native "
    "reference backend requires an NVlabs/Sana checkout. "
    f"{SANA_WM_OFFICIAL_BACKEND_ERROR}"
)

SANA_WM_STAGE1_DIT_FILE = "dit/sana_wm_1600m_720p.safetensors"
SANA_WM_CONFIG_FILE = "config.yaml"
SANA_WM_VAE_CONFIG_FILE = "vae/config.json"
SANA_WM_VAE_WEIGHT_FILE = "vae/diffusion_pytorch_model.safetensors"
SANA_WM_REFINER_TRANSFORMER_CONFIG_FILE = "refiner/transformer/config.json"
SANA_WM_REFINER_TRANSFORMER_WEIGHT_FILE = "refiner/transformer/diffusion_pytorch_model.safetensors"
SANA_WM_REFINER_CONNECTORS_CONFIG_FILE = "refiner/connectors/config.json"
SANA_WM_REFINER_CONNECTORS_WEIGHT_FILE = "refiner/connectors/diffusion_pytorch_model.safetensors"
SANA_WM_REFINER_TEXT_ENCODER_DIR = "refiner/text_encoder"
SANA_WM_REFINER_TEXT_ENCODER_INDEX_FILE = "refiner/text_encoder/model.safetensors.index.json"
SANA_WM_STAGE1_TEXT_ENCODER_ID = "google/gemma-2-2b-it"
SANA_WM_STAGE1_TEXT_ENCODER_FALLBACK_ID = "Efficient-Large-Model/gemma-2-2b-it"
SANA_WM_STAGE1_TEXT_ENCODER_ENV = "VLLM_OMNI_SANA_WM_STAGE1_TEXT_ENCODER"
SANA_WM_REFINER_ROOT_ENV = "VLLM_OMNI_SANA_WM_REFINER_ROOT"
SANA_WM_OUTPUT_HEIGHT = 704
SANA_WM_OUTPUT_WIDTH = 1280
SANA_WM_DEFAULT_NUM_FRAMES = 321
SANA_WM_NATIVE_SMOKE_ENV = "VLLM_OMNI_SANA_WM_NATIVE_SMOKE"
SANA_WM_NATIVE_SMOKE_MAX_TOKENS = 4096

SANA_WM_STAGE1_PATTERNS = (
    SANA_WM_CONFIG_FILE,
    SANA_WM_STAGE1_DIT_FILE,
    SANA_WM_VAE_CONFIG_FILE,
    SANA_WM_VAE_WEIGHT_FILE,
)
SANA_WM_STAGE1_DIT_SUBFOLDER = "dit"
SANA_WM_STAGE1_DIT_BASENAME = Path(SANA_WM_STAGE1_DIT_FILE).name
SANA_WM_REFINER_PATTERNS = (
    SANA_WM_REFINER_TRANSFORMER_CONFIG_FILE,
    SANA_WM_REFINER_TRANSFORMER_WEIGHT_FILE,
    SANA_WM_REFINER_CONNECTORS_CONFIG_FILE,
    SANA_WM_REFINER_CONNECTORS_WEIGHT_FILE,
    f"{SANA_WM_REFINER_TEXT_ENCODER_DIR}/*",
)


@dataclass(frozen=True)
class SanaWmLocalPaths:
    """Resolved local file paths for a SANA-WM snapshot."""

    root: Path
    config: Path
    stage1_dit: Path
    vae_config: Path
    vae_weights: Path
    refiner_root: Path
    refiner_transformer_config: Path
    refiner_transformer_weights: Path
    refiner_connectors_config: Path
    refiner_connectors_weights: Path
    refiner_text_encoder_dir: Path


@dataclass(frozen=True)
class SanaWmNativeSmokeParams:
    """Small native fallback generation settings.

    This is not the production SANA-WM sampler; it exists to exercise the
    vLLM-Omni-native transformer/camera/scheduler stack at bounded sizes while
    the exact GDN path is being ported.
    """

    height: int
    width: int
    num_frames: int
    num_inference_steps: int
    seed: int


def build_sana_wm_download_patterns(*, include_refiner: bool = True) -> tuple[str, ...]:
    """Return the minimal HF allow-patterns needed for SANA-WM."""

    patterns = list(SANA_WM_STAGE1_PATTERNS)
    if include_refiner:
        patterns.extend(SANA_WM_REFINER_PATTERNS)
    return tuple(patterns)


def resolve_sana_wm_local_paths(
    snapshot_dir: str | Path,
    *,
    refiner_root: str | Path | None = None,
) -> SanaWmLocalPaths:
    root = Path(snapshot_dir)
    resolved_refiner_root = Path(refiner_root) if refiner_root is not None else root / "refiner"
    return SanaWmLocalPaths(
        root=root,
        config=root / SANA_WM_CONFIG_FILE,
        stage1_dit=root / SANA_WM_STAGE1_DIT_FILE,
        vae_config=root / SANA_WM_VAE_CONFIG_FILE,
        vae_weights=root / SANA_WM_VAE_WEIGHT_FILE,
        refiner_root=resolved_refiner_root,
        refiner_transformer_config=resolved_refiner_root / "transformer/config.json",
        refiner_transformer_weights=resolved_refiner_root / "transformer/diffusion_pytorch_model.safetensors",
        refiner_connectors_config=resolved_refiner_root / "connectors/config.json",
        refiner_connectors_weights=resolved_refiner_root / "connectors/diffusion_pytorch_model.safetensors",
        refiner_text_encoder_dir=resolved_refiner_root / "text_encoder",
    )


def validate_sana_wm_local_paths(paths: SanaWmLocalPaths, *, include_refiner: bool = True) -> None:
    required = [
        paths.config,
        paths.stage1_dit,
        paths.vae_config,
        paths.vae_weights,
    ]
    if include_refiner:
        required.extend(
            [
                paths.refiner_transformer_config,
                paths.refiner_transformer_weights,
                paths.refiner_connectors_config,
                paths.refiner_connectors_weights,
                paths.refiner_text_encoder_dir / "config.json",
                paths.refiner_text_encoder_dir / "model.safetensors.index.json",
            ]
        )
    missing = [path for path in required if not path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing SANA-WM checkpoint files: {joined}")


def resolve_or_download_sana_wm_checkpoint(
    model: str = SANA_WM_MODEL_ID,
    *,
    include_refiner: bool = True,
    revision: str | None = None,
    cache_dir: str | None = None,
) -> SanaWmLocalPaths:
    """Resolve a local SANA-WM tree or download the required HF files."""

    root = Path(model)
    if root.is_dir():
        snapshot_dir = root
    else:
        snapshot_dir = Path(
            download_weights_from_hf_specific(
                model,
                cache_dir,
                list(build_sana_wm_download_patterns(include_refiner=include_refiner)),
                revision=revision,
                require_all=True,
            )
        )
    refiner_root = os.environ.get(SANA_WM_REFINER_ROOT_ENV, "").strip() or None
    paths = resolve_sana_wm_local_paths(snapshot_dir, refiner_root=refiner_root)
    validate_sana_wm_local_paths(paths, include_refiner=include_refiner)
    return paths


def get_sana_wm_post_process_func(od_config: OmniDiffusionConfig):
    del od_config

    def post_process_func(output: Any) -> Any:
        return output

    return post_process_func


def get_sana_wm_pre_process_func(od_config: OmniDiffusionConfig):
    del od_config

    def pre_process_func(request: Any) -> Any:
        from vllm_omni.inputs.data import OmniTextPrompt

        sampling_params = getattr(request, "sampling_params", None)
        for idx, prompt in enumerate(request.prompts):
            prompt_mapping = OmniTextPrompt(prompt=prompt) if isinstance(prompt, str) else prompt
            if sampling_params is not None:
                prompt_mapping = dict(prompt_mapping)
                if getattr(sampling_params, "num_frames", None) not in (None, 1):
                    prompt_mapping.setdefault("num_frames", sampling_params.num_frames)
                if getattr(sampling_params, "height", None) is not None:
                    prompt_mapping.setdefault("height", sampling_params.height)
                if getattr(sampling_params, "width", None) is not None:
                    prompt_mapping.setdefault("width", sampling_params.width)
            request.prompts[idx] = normalize_sana_wm_payload(prompt_mapping)

        if sampling_params is not None and request.prompts:
            payload = request.prompts[0]["additional_information"]["sana_wm"]
            sampling_params.num_frames = payload["num_frames"]
            sampling_params.height = payload["height"]
            sampling_params.width = payload["width"]
        return request

    return pre_process_func


class SanaWmPipeline(
    nn.Module,
    CFGParallelMixin,
    SupportImageInput,
    SupportsComponentDiscovery,
    ProgressBarMixin,
    DiffusionPipelineProfilerMixin,
):
    """Stage-1 SANA-WM image-to-video pipeline placeholder."""

    support_image_input: ClassVar[bool] = True
    color_format: ClassVar[str] = "RGB"
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder", "camera_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    _resident_modules: ClassVar[list[str]] = []
    include_refiner: ClassVar[bool] = False

    def __init__(self, *, od_config: OmniDiffusionConfig | None = None, prefix: str = "") -> None:
        super().__init__()
        self.od_config = od_config
        self.prefix = prefix
        self.use_official_backend = is_sana_wm_official_backend_requested(od_config)
        self.output_height = SANA_WM_OUTPUT_HEIGHT
        self.output_width = SANA_WM_OUTPUT_WIDTH
        self.default_num_frames = SANA_WM_DEFAULT_NUM_FRAMES

        self.sana_wm_config = SanaWmConfig()
        self.transformer = SanaWmTransformer3DModel(config=self.sana_wm_config)
        self.camera_encoder: SanaWmCameraEmbedder | None = None

        # Filled by the real implementation.
        self.tokenizer: Any | None = None
        self.text_encoder: nn.Module | None = None
        self.vae: nn.Module | None = None
        self.release_paths: SanaWmLocalPaths | None = None
        self.weights_sources = []
        if od_config is not None and od_config.model is not None and not self.use_official_backend:
            self.weights_sources = [
                DiffusersPipelineLoader.ComponentSource(
                    model_or_path=od_config.model,
                    subfolder=SANA_WM_STAGE1_DIT_SUBFOLDER,
                    revision=od_config.revision,
                    prefix="",
                    fall_back_to_pt=False,
                    allow_patterns_overrides=[SANA_WM_STAGE1_DIT_BASENAME],
                )
            ]

    def resolve_checkpoint(self, *, include_refiner: bool | None = None) -> SanaWmLocalPaths:
        if self.od_config is None or self.od_config.model is None:
            raise ValueError("Sana-WM checkpoint resolution requires od_config.model.")
        include = self.include_refiner if include_refiner is None else include_refiner
        self.release_paths = resolve_or_download_sana_wm_checkpoint(
            self.od_config.model,
            include_refiner=include,
            revision=self.od_config.revision,
        )
        self.sana_wm_config = SanaWmConfig.from_yaml(self.release_paths.config)
        self.transformer.config = self.sana_wm_config
        self.camera_encoder = None
        return self.release_paths

    @staticmethod
    def _extra_args(sampling_params: Any | None) -> dict[str, Any]:
        extra_args = getattr(sampling_params, "extra_args", None) if sampling_params is not None else None
        return dict(extra_args or {})

    @classmethod
    def _native_smoke_requested(cls, sampling_params: Any | None) -> bool:
        env_value = os.environ.get(SANA_WM_NATIVE_SMOKE_ENV, "").strip().lower()
        if env_value in {"1", "true", "yes", "on"}:
            return True
        extra_args = cls._extra_args(sampling_params)
        return str(extra_args.get("sana_wm_native_smoke", "")).strip().lower() in {"1", "true", "yes", "on"}

    def _native_smoke_params(self, payload: dict[str, Any], sampling_params: Any | None) -> SanaWmNativeSmokeParams:
        extra_args = self._extra_args(sampling_params)
        height = int(getattr(sampling_params, "height", None) or payload["height"])
        width = int(getattr(sampling_params, "width", None) or payload["width"])
        num_frames = int(getattr(sampling_params, "num_frames", None) or payload["num_frames"])
        steps = int(getattr(sampling_params, "num_inference_steps", None) or extra_args.get("num_inference_steps", 1))
        seed = int(getattr(sampling_params, "seed", None) or extra_args.get("seed", 0))
        height = int(extra_args.get("sana_wm_native_smoke_height", height))
        width = int(extra_args.get("sana_wm_native_smoke_width", width))
        num_frames = int(extra_args.get("sana_wm_native_smoke_num_frames", num_frames))
        steps = int(extra_args.get("sana_wm_native_smoke_steps", steps))
        if min(height, width, num_frames, steps) <= 0:
            raise ValueError("Sana-WM native smoke height, width, num_frames, and steps must be positive.")
        return SanaWmNativeSmokeParams(
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=steps,
            seed=seed,
        )

    def _native_smoke_dtype(self, device: torch.device) -> torch.dtype:
        dtype = getattr(self.od_config, "dtype", None) if self.od_config is not None else None
        if isinstance(dtype, torch.dtype):
            return dtype
        return torch.bfloat16 if device.type == "cuda" else torch.float32

    def _runtime_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device, self._native_smoke_dtype(device)

    def _checkpoint_local_files_only(self) -> bool:
        if self.release_paths is not None:
            return True
        if self.od_config is None or self.od_config.model is None:
            return False
        return Path(str(self.od_config.model)).expanduser().exists()

    def _ensure_stage1_text_encoder(self, *, device: torch.device, dtype: torch.dtype) -> None:
        if self.text_encoder is not None:
            self.text_encoder.to(device=device, dtype=dtype)
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        local_files_only = self._checkpoint_local_files_only()
        model_ids = [os.environ.get(SANA_WM_STAGE1_TEXT_ENCODER_ENV, "").strip()]
        model_ids.extend([SANA_WM_STAGE1_TEXT_ENCODER_ID, SANA_WM_STAGE1_TEXT_ENCODER_FALLBACK_ID])
        errors: list[str] = []
        for model_id in dict.fromkeys(model_id for model_id in model_ids if model_id):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    local_files_only=local_files_only,
                )
                text_encoder_model_id = model_id
                break
            except OSError as exc:
                errors.append(f"{model_id}: {exc}")
        else:
            raise OSError("Could not load SANA-WM Stage-1 text encoder. Tried: " + "; ".join(errors))
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        with torch.device("cpu"):
            self.text_encoder = AutoModelForCausalLM.from_pretrained(
                text_encoder_model_id,
                torch_dtype=dtype,
                local_files_only=local_files_only,
            ).to(device)

    def _ensure_vae(self, *, device: torch.device, dtype: torch.dtype) -> None:
        if self.vae is not None:
            self.vae.to(device=device, dtype=dtype)
            return
        if self.release_paths is None and self.od_config is not None and self.od_config.model is not None:
            self.resolve_checkpoint(include_refiner=False)
        if self.release_paths is None:
            raise ValueError("Sana-WM VAE loading requires a resolved checkpoint.")

        from diffusers import AutoencoderKLLTX2Video

        with torch.device("cpu"):
            self.vae = AutoencoderKLLTX2Video.from_pretrained(
                str(self.release_paths.root),
                subfolder="vae",
                torch_dtype=dtype,
                local_files_only=True,
            ).to(device)

    def _ensure_camera_encoder(self, *, device: torch.device, dtype: torch.dtype) -> SanaWmCameraEmbedder:
        if self.camera_encoder is None or self.camera_encoder.hidden_size != self.sana_wm_config.hidden_size:
            self.camera_encoder = SanaWmCameraEmbedder(config=self.sana_wm_config)
        self.camera_encoder.to(device=device, dtype=dtype)
        return self.camera_encoder

    def _stage1_prompt_text(self, prompt: dict[str, Any]) -> str:
        parts = [part for part in self.sana_wm_config.chi_prompt if part]
        user_prompt = str(prompt.get("prompt") or "")
        if user_prompt:
            parts.append(user_prompt)
        return "\n".join(parts)

    def _hash_smoke_prompt_embeds(
        self,
        prompt_text: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, str]:
        shape = (1, self.sana_wm_config.model_max_length, SANA_WM_STAGE1_PROMPT_CHANNELS)
        if not prompt_text:
            return torch.zeros(shape, device=device, dtype=dtype), "empty"

        import hashlib

        digest = hashlib.sha256(prompt_text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)
        generator = torch.Generator()
        generator.manual_seed(seed)
        prompt_embeds = torch.randn(shape, generator=generator, dtype=torch.float32)
        prompt_embeds = prompt_embeds * float(self.sana_wm_config.y_norm_scale_factor)
        return prompt_embeds.to(device=device, dtype=dtype), "hash_smoke"

    def _native_smoke_prompt_embeds(
        self,
        prompt: dict[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
        allow_hash_fallback: bool = False,
    ) -> tuple[torch.Tensor, str]:
        prompt_text = self._stage1_prompt_text(prompt)
        if allow_hash_fallback:
            return self._hash_smoke_prompt_embeds(prompt_text, device=device, dtype=dtype)

        self._ensure_stage1_text_encoder(device=device, dtype=dtype)
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None or self.text_encoder is None:
            raise RuntimeError("Sana-WM Stage-1 text encoder did not initialize.")
        encoded = tokenizer(
            [prompt_text],
            padding="max_length",
            max_length=self.sana_wm_config.model_max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(device)
        outputs = self.text_encoder(**encoded, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        if hidden_states.shape[-1] != SANA_WM_STAGE1_PROMPT_CHANNELS:
            raise ValueError(
                "Sana-WM Stage-1 Gemma hidden size mismatch: expected "
                f"{SANA_WM_STAGE1_PROMPT_CHANNELS}, got {hidden_states.shape[-1]}."
            )
        hidden_states = F.normalize(hidden_states.float(), dim=-1)
        hidden_states = hidden_states * float(self.sana_wm_config.y_norm_scale_factor)
        return hidden_states.to(device=device, dtype=dtype), "gemma2"

    def _decode_native_smoke_latents(
        self,
        latents: torch.Tensor,
        *,
        output_type: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Any:
        if output_type == "latent":
            return latents
        self._ensure_vae(device=device, dtype=dtype)
        if self.vae is None:
            raise RuntimeError("Sana-WM VAE did not initialize.")
        timestep = None
        if getattr(getattr(self.vae, "config", None), "timestep_conditioning", False):
            timestep = torch.zeros(latents.shape[0], device=device, dtype=latents.dtype)
        video = self.vae.decode(latents.to(getattr(self.vae, "dtype", dtype)), timestep, return_dict=False)[0]
        try:
            from diffusers.video_processor import VideoProcessor

            processor = VideoProcessor(vae_scale_factor=getattr(self.vae, "spatial_compression_ratio", 32))
            return processor.postprocess_video(video, output_type=output_type)
        except Exception:
            return video

    def _run_native_smoke_backend(
        self,
        *,
        prompt: dict[str, Any],
        payload: dict[str, Any],
        sampling_params: Any | None,
    ) -> DiffusionOutput:
        params = self._native_smoke_params(payload, sampling_params)
        latent_frames = (params.num_frames - 1) // 8 + 1
        latent_height = max(params.height // 32, 1)
        latent_width = max(params.width // 32, 1)
        token_count = latent_frames * latent_height * latent_width
        extra_args = self._extra_args(sampling_params)
        max_tokens = int(extra_args.get("sana_wm_native_smoke_max_tokens", SANA_WM_NATIVE_SMOKE_MAX_TOKENS))
        if token_count > max_tokens:
            raise ValueError(
                "Sana-WM native smoke path is intentionally size-capped while the fused GDN path is being ported. "
                f"Requested latent tokens={token_count}, max={max_tokens}. Set smaller "
                "`sana_wm_native_smoke_height/width/num_frames` or use the official backend."
            )

        device, dtype = self._runtime_device_dtype()
        generator = torch.Generator(device=device)
        generator.manual_seed(params.seed)
        latents = torch.randn(
            (1, 128, latent_frames, latent_height, latent_width),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        allow_hash_fallback = bool(extra_args.get("sana_wm_hash_prompt_smoke", False))
        prompt_embeds, prompt_source = self._native_smoke_prompt_embeds(
            prompt,
            device=device,
            dtype=dtype,
            allow_hash_fallback=allow_hash_fallback,
        )

        camera = payload.get("camera") or {}
        condition = SanaWmCameraCondition(
            poses=camera.get("poses") if isinstance(camera, dict) else None,
            intrinsics=payload.get("intrinsics"),
            action=payload.get("action"),
            num_frames=params.num_frames,
            height=params.height,
            width=params.width,
            translation_speed=float(payload.get("translation_speed", 0.05)),
            rotation_speed_deg=float(payload.get("rotation_speed_deg", 1.2)),
        )
        camera_tensors = build_plucker_condition(condition)
        plucker = camera_tensors["chunk_plucker"].to(device=device, dtype=dtype)
        raymap = camera_tensors["raymap"].to(device=device, dtype=dtype)

        self.transformer.config = self.sana_wm_config
        camera_encoder = self._ensure_camera_encoder(device=device, dtype=dtype)
        scheduler = SanaWmFlowDpmScheduler(params.num_inference_steps, shift=self.sana_wm_config.inference_flow_shift)
        for timestep, delta in zip(scheduler.timesteps(device=device), scheduler.deltas(device=device), strict=True):
            noise_pred = self.transformer(
                latents,
                timestep.expand(1),
                encoder_hidden_states=prompt_embeds,
                camera_encoder=camera_encoder,
                plucker=plucker,
                raymap=raymap,
            )
            latents = scheduler.step(latents, noise_pred, delta)

        output_type = str(extra_args.get("sana_wm_output_type", "latent"))
        output = self._decode_native_smoke_latents(latents, output_type=output_type, device=device, dtype=dtype)
        return DiffusionOutput(
            output=output,
            custom_output={
                "sana_wm_backend": "native_gdn_smoke",
                "sana_wm_output_space": output_type,
                "sana_wm_prompt_source": prompt_source,
                "sana_wm_chi_prompt_applied": bool(self.sana_wm_config.chi_prompt),
                "sana_wm_num_frames": params.num_frames,
                "sana_wm_height": params.height,
                "sana_wm_width": params.width,
                "sana_wm_latent_tokens": token_count,
                "sana_wm_sampling_steps": params.num_inference_steps,
            },
        )

    def forward(self, req: OmniDiffusionRequest, *args: Any, **kwargs: Any) -> DiffusionOutput:
        del args, kwargs
        if len(req.prompts) != 1:
            raise ValueError("Sana-WM native backend currently supports exactly one prompt per request.")

        prompt = req.prompts[0]
        if isinstance(prompt, str):
            raise ValueError("Sana-WM requires a mapping prompt with first-frame image and camera/action metadata.")
        prompt = normalize_sana_wm_payload(prompt)
        payload = prompt["additional_information"]["sana_wm"]
        repo_path = get_sana_wm_official_repo_path(
            explicit_repo=payload.get("official_repo_path"),
            od_config=self.od_config,
        )
        if repo_path is None:
            if self._native_smoke_requested(req.sampling_params):
                return self._run_native_smoke_backend(
                    prompt=prompt,
                    payload=payload,
                    sampling_params=req.sampling_params,
                )
            raise NotImplementedError(SANA_WM_SCAFFOLD_ERROR)

        if (
            getattr(req.sampling_params, "num_frames", None) not in (None, 1)
            and int(req.sampling_params.num_frames) != int(payload["num_frames"])
        ):
            payload = dict(payload)
            payload["num_frames"] = int(req.sampling_params.num_frames)
            prompt = dict(prompt)
            additional = dict(prompt["additional_information"])
            additional["sana_wm"] = payload
            prompt["additional_information"] = additional

        start = time.perf_counter()
        paths = self.resolve_checkpoint()
        if should_force_sana_wm_cli_backend():
            result = run_sana_wm_official_backend(
                prompt=prompt,
                release_paths=paths,
                include_refiner=self.include_refiner,
                repo_path=repo_path,
            )
            return DiffusionOutput(
                output=result.frames,
                custom_output={
                    "sana_wm_backend": "official_cli",
                    "sana_wm_official_command": list(result.command),
                    "sana_wm_official_stdout_tail": result.stdout[-2000:],
                    "sana_wm_official_stderr_tail": result.stderr[-2000:],
                    "sana_wm_official_video_path": result.video_path,
                },
                stage_durations={"sana_wm_official_backend_s": time.perf_counter() - start},
            )

        result = run_sana_wm_native_backend(
            prompt=prompt,
            release_paths=paths,
            include_refiner=self.include_refiner,
            repo_path=repo_path,
            sampling_params=req.sampling_params,
        )
        return DiffusionOutput(
            output=result.frames,
            custom_output={
                "sana_wm_backend": result.backend,
                "sana_wm_repo_path": result.repo_path,
                "sana_wm_model_path": result.model_path,
                "sana_wm_include_refiner": result.include_refiner,
                "sana_wm_num_frames": result.num_frames,
                "sana_wm_fps": result.fps,
                "sana_wm_sampling_steps": result.sampling_steps,
                "sana_wm_cfg_scale": result.cfg_scale,
                "sana_wm_used_default_intrinsics": result.used_default_intrinsics,
            },
            stage_durations={"sana_wm_native_backend_s": time.perf_counter() - start},
        )

    def load_weights(self, weights: Iterable[tuple[str, Any]]) -> set[str]:
        if self.use_official_backend:
            return set()
        materialized_camera_params = dict(self.camera_encoder.named_parameters()) if self.camera_encoder else {}
        cached_weights = list(weights)
        loaded = self.transformer.load_weights(cached_weights)
        for source_name, tensor in cached_weights:
            remapped_name = normalize_sana_wm_stage1_weight_name(source_name)
            if remapped_name is None or not remapped_name.startswith("camera_encoder."):
                continue
            local_name = remapped_name.removeprefix("camera_encoder.")
            param = materialized_camera_params.get(local_name)
            if param is None:
                continue
            if tuple(param.shape) != tuple(tensor.shape):
                # Keep the transformer-side audit record even when this smoke
                # camera branch is not an exact official-module match yet.
                continue
            with torch.no_grad():
                param.copy_(tensor.to(device=param.device, dtype=param.dtype))
        return loaded
