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
from vllm_omni.diffusion.models.sana_wm.scheduling_sana_wm import (
    SanaWmFlowDpmScheduler,  # noqa: F401 – kept for backward-compat imports
    SanaWmFlowMatchScheduler,
)
from vllm_omni.diffusion.models.sana_wm.cuda_graph import (
    SanaWmCudaGraphDenoiser,
    parse_sana_wm_cudagraph_buckets,
    sana_wm_cudagraph_requested,
)
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
        # The official CLI bridge is only enabled when the user explicitly
        # opts in via ``VLLM_OMNI_SANA_WM_USE_OFFICIAL_CLI=1`` AND the
        # NVlabs/Sana repo path is provided. Setting only the repo path is
        # required to let the reference-alignment harness invoke the CLI
        # bridge for the *reference* run, but must not disable native
        # Stage-1 weight loading on the prediction-path pipeline instance —
        # otherwise ``load_weights`` early-returns ``set()`` and the
        # native model runs with zero-initialised parameters. See audit
        # §6.11 for the on-GPU trace evidence that produced MAE=98.31.
        self.use_official_backend = (
            is_sana_wm_official_backend_requested(od_config)
            and should_force_sana_wm_cli_backend()
        )
        self.output_height = SANA_WM_OUTPUT_HEIGHT
        self.output_width = SANA_WM_OUTPUT_WIDTH
        self.default_num_frames = SANA_WM_DEFAULT_NUM_FRAMES

        self.sana_wm_config = SanaWmConfig()
        self.quant_config = getattr(od_config, "quantization_config", None) if od_config is not None else None
        self.transformer = SanaWmTransformer3DModel(
            config=self.sana_wm_config,
            quant_config=self.quant_config,
            prefix=f"{prefix}.transformer" if prefix else "transformer",
        )
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
        self.tokenizer.padding_side = "right"
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
        # Match NVlabs' inference_video_scripts/inference_sana_wm.py VAE
        # construction. Long videos (e.g. 321 frames -> 41 latent frames)
        # otherwise decode as one large 3D volume and OOM on a 98GB RTX 6000.
        if hasattr(self.vae, "enable_tiling"):
            self.vae.enable_tiling()
        if hasattr(self.vae, "use_framewise_encoding"):
            self.vae.use_framewise_encoding = True
            self.vae.use_framewise_decoding = True
            vae_config = getattr(self.vae, "config", None)
            self.vae.tile_sample_stride_num_frames = int(
                getattr(vae_config, "tile_sample_stride_num_frames", 64)
            )
            self.vae.tile_sample_min_num_frames = int(
                getattr(vae_config, "tile_sample_min_num_frames", 96)
            )

    def _ensure_camera_encoder(self, *, device: torch.device, dtype: torch.dtype) -> SanaWmCameraEmbedder:
        if self.camera_encoder is None or self.camera_encoder.hidden_size != self.sana_wm_config.hidden_size:
            self.camera_encoder = SanaWmCameraEmbedder(config=self.sana_wm_config)
        self.camera_encoder.to(device=device, dtype=dtype)
        return self.camera_encoder

    @staticmethod
    def _preprocess_first_frame(
        image: Any,
        *,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the first-frame image as ``[1, 3, 1, H, W]`` in ``[-1, 1]``."""
        import numpy as np

        if hasattr(image, "convert"):
            image = image.convert("RGB").resize((width, height))
            arr = np.array(image, dtype=np.float32) / 127.5 - 1.0
        elif isinstance(image, np.ndarray):
            if image.shape[:2] != (height, width):
                from PIL import Image as _PIL

                image = _PIL.fromarray(image).resize((width, height))
                arr = np.array(image, dtype=np.float32) / 127.5 - 1.0
            else:
                arr = image.astype(np.float32) / 127.5 - 1.0
        elif isinstance(image, torch.Tensor):
            t = image.float()
            if t.ndim == 3:
                t = t.unsqueeze(0)
            if t.shape[1] != 3:
                t = t.permute(0, 3, 1, 2)
            t = F.interpolate(t, size=(height, width), mode="bilinear", align_corners=False)
            t = t * 2.0 - 1.0 if t.max() <= 1.0 + 1e-3 else t / 127.5 - 1.0
            return t.unsqueeze(2).to(device=device, dtype=dtype)
        else:
            raise TypeError(f"Sana-WM first-frame image must be PIL/ndarray/Tensor, got {type(image).__name__}.")

        # arr: [H, W, 3] → [1, 3, 1, H, W]
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)[np.newaxis, :, np.newaxis])
        return tensor.to(device=device, dtype=dtype)

    def _vae_normalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """LTX-2 VAE per-channel latent normalisation matching NVlabs.

        ``z_norm = (z_raw - latents_mean) * scaling_factor / latents_std``

        The LTX-2 VAE (``AutoencoderKLLTX2Video`` in diffusers) ships with
        per-channel ``latents_mean`` and ``latents_std`` tensors that must be
        applied alongside ``scaling_factor`` for the round-trip to be
        identity. The legacy `* scaling_factor` only path produced
        decoded videos with a systematic G/B colour shift (audit §6.13e
        diagnosis: pred RGB means ~(110, 220, 208) vs ref ~(94, 114, 138)).

        Returns the normalised latent in the same dtype as the input.
        """
        if self.vae is None:
            raise RuntimeError("Sana-WM VAE did not initialize.")
        latents_mean = self.vae.latents_mean.view(1, -1, 1, 1, 1).to(
            device=latent.device, dtype=latent.dtype
        )
        latents_std = self.vae.latents_std.view(1, -1, 1, 1, 1).to(
            device=latent.device, dtype=latent.dtype
        )
        scaling = float(getattr(getattr(self.vae, "config", None), "scaling_factor", 1.0))
        return (latent - latents_mean) * scaling / latents_std

    def _vae_denormalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`_vae_normalize_latent` — match NVlabs decode."""
        if self.vae is None:
            raise RuntimeError("Sana-WM VAE did not initialize.")
        latents_mean = self.vae.latents_mean.view(1, -1, 1, 1, 1).to(
            device=latent.device, dtype=latent.dtype
        )
        latents_std = self.vae.latents_std.view(1, -1, 1, 1, 1).to(
            device=latent.device, dtype=latent.dtype
        )
        scaling = float(getattr(getattr(self.vae, "config", None), "scaling_factor", 1.0))
        return latent * latents_std / scaling + latents_mean

    def _vae_encode_first_frame(
        self,
        image: Any,
        *,
        height: int,
        width: int,
        latent_height: int,
        latent_width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Encode the first-frame image to latent space via the LTX-2 VAE.

        Returns shape ``[1, 128, 1, latent_height, latent_width]``.
        """
        self._ensure_vae(device=device, dtype=dtype)
        if self.vae is None:
            raise RuntimeError("Sana-WM VAE did not initialize.")
        frame = self._preprocess_first_frame(image, height=height, width=width, device=device, dtype=dtype)
        with torch.inference_mode():
            encoded = self.vae.encode(frame)
            dist = getattr(encoded, "latent_dist", None)
            first_latent = dist.mean if dist is not None else encoded.latents
            first_latent = self._vae_normalize_latent(first_latent.to(dtype=dtype))
        # VAE latent may differ from expected spatial size; resize if needed.
        if first_latent.shape[-2:] != (latent_height, latent_width):
            first_latent = F.interpolate(
                first_latent.squeeze(2),
                size=(latent_height, latent_width),
                mode="bilinear",
                align_corners=False,
            ).unsqueeze(2)
        return first_latent  # [1, 128, 1, lh, lw]

    def _stage1_prompt_text(self, prompt: dict[str, Any]) -> str:
        user_prompt = str(prompt.get("prompt") or "")
        chi_prompt = "\n".join(part for part in self.sana_wm_config.chi_prompt if part)
        if chi_prompt:
            return chi_prompt + user_prompt
        return user_prompt

    def _hash_smoke_prompt_embeds(
        self,
        prompt_text: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, str]:
        shape = (1, self.sana_wm_config.model_max_length, SANA_WM_STAGE1_PROMPT_CHANNELS)
        if not prompt_text:
            self._last_prompt_attention_mask = torch.zeros(shape[:2], device=device, dtype=torch.float32)
            return torch.zeros(shape, device=device, dtype=dtype), "empty"

        import hashlib

        digest = hashlib.sha256(prompt_text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)
        generator = torch.Generator()
        generator.manual_seed(seed)
        prompt_embeds = torch.randn(shape, generator=generator, dtype=torch.float32)
        prompt_embeds = prompt_embeds * float(self.sana_wm_config.y_norm_scale_factor)
        self._last_prompt_attention_mask = torch.ones(shape[:2], device=device, dtype=torch.float32)
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
        chi_prompt = "\n".join(part for part in self.sana_wm_config.chi_prompt if part)
        model_max_length = int(self.sana_wm_config.model_max_length)
        max_length_all = model_max_length
        if chi_prompt:
            # NVlabs encodes the chi-prompt-prefixed text with extra room for
            # the chi tokens, then keeps BOS plus the final model window.
            max_length_all = len(tokenizer.encode(chi_prompt)) + model_max_length - 2
        encoded = tokenizer(
            [prompt_text],
            padding="max_length",
            max_length=max_length_all,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(device)
        outputs = self.text_encoder(**encoded, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        attention_mask = encoded.attention_mask
        if chi_prompt:
            select_index = [0] + list(range(-model_max_length + 1, 0))
            hidden_states = hidden_states[:, select_index]
            attention_mask = attention_mask[:, select_index]
        if hidden_states.shape[-1] != SANA_WM_STAGE1_PROMPT_CHANNELS:
            raise ValueError(
                "Sana-WM Stage-1 Gemma hidden size mismatch: expected "
                f"{SANA_WM_STAGE1_PROMPT_CHANNELS}, got {hidden_states.shape[-1]}."
            )
        # Audit §6.13f: pass RAW Gemma hidden states; the model's
        # internal ``attention_y_norm = RMSNorm(hidden_size,
        # scale_factor=y_norm_scale_factor)`` (set in
        # SanaWmTransformer3DModel) handles normalisation. The prior
        # ``F.normalize(...) * y_norm_scale_factor`` step double-
        # normalised and used the scale_factor as a multiplicative
        # post-normalise gain rather than as the RMSNorm scale, which
        # collapsed the prompt embed L2 norm to ~0.17 (vs the expected
        # ~450 at this shape) and left the model effectively
        # unconditioned on the prompt. Probed via
        # tools/scripts/probe_step0.py.
        self._last_prompt_attention_mask = attention_mask.to(device=device, dtype=torch.float32)
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
        # LTX-2 VAE denormalisation: see :meth:`_vae_denormalize_latent`.
        # NVlabs always passes ``temb=None`` to ``vae.decode`` for this VAE
        # (see ``builder.py::vae_decode -> LTX2VAE_diffusers``); we match
        # that unconditionally rather than threading a zero timestep
        # through ``timestep_conditioning`` like the older path did.
        denorm = self._vae_denormalize_latent(latents.to(getattr(self.vae, "dtype", dtype)))
        video = self.vae.decode(denorm, temb=None, return_dict=False)[0]
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
        noise = torch.randn(
            (1, 128, latent_frames, latent_height, latent_width),
            device=device,
            dtype=dtype,
            generator=generator,
        )

        # A.2: production flow-DPM-Solver++ scheduler (replaces shifted-Euler smoke)
        scheduler = SanaWmFlowMatchScheduler(
            params.num_inference_steps,
            shift=self.sana_wm_config.inference_flow_shift,
        )
        timesteps = scheduler.timesteps(device=device)

        # A.1: first-frame VAE encode — initialize latents from the request image.
        # Only attempt encoding for PIL/ndarray/Tensor; other types (e.g. test
        # placeholder objects) fall through to pure-noise initialization.
        import numpy as _np

        first_frame_image = (prompt.get("multi_modal_data") or {}).get("image")
        _is_image = first_frame_image is not None and (
            hasattr(first_frame_image, "convert")
            or isinstance(first_frame_image, (_np.ndarray, torch.Tensor))
        )
        # Per-frame timestep contract (audit §6.13a) is enabled by default
        # to match NVlabs ``LTXFlowEuler.sample``: frame 0 is the clean VAE-
        # encoded conditioning latent and its timestep is held at 0 while
        # the sampling loop drives the other frames from noise. Falls back
        # to the legacy "noise first frame + scalar timestep" path when
        # explicitly disabled, e.g. for back-compat smoke tests.
        per_frame_timestep_disabled = os.environ.get(
            "VLLM_OMNI_SANA_WM_DISABLE_PER_FRAME_TIMESTEP", ""
        ).lower() in {"1", "true", "yes", "on"}
        if _is_image:
            first_latent = self._vae_encode_first_frame(
                first_frame_image,
                height=params.height,
                width=params.width,
                latent_height=latent_height,
                latent_width=latent_width,
                device=device,
                dtype=dtype,
            )
            if per_frame_timestep_disabled:
                # Legacy contract: noise the first-frame latent to the
                # highest timestep before the denoising loop.
                first_noised = scheduler.add_noise(first_latent, noise[:, :, :1], timesteps[0])
                latents = torch.cat([first_noised, noise[:, :, 1:]], dim=2)
            else:
                # NVlabs contract: place the CLEAN VAE-encoded first frame
                # at frame 0 and rely on per-frame timesteps + the
                # `condition_mask` torch.where restore (below) to keep it
                # invariant across denoising steps.
                latents = torch.cat([first_latent, noise[:, :, 1:]], dim=2)
        else:
            first_latent = None
            latents = noise

        # Audit §6.13f probe hook: load the initial latent from a file
        # (typically an NVlabs step-0 dump) so both pipelines start the
        # sampling loop from byte-identical input. Lets us cleanly
        # isolate `noise_pred` divergence due to model forward vs the
        # RNG/init differences that previously caused cosine=+0.43 on
        # latent_in between the two pipelines.
        _load_latent_path = os.environ.get("SANA_WM_LOAD_LATENT_FROM", "")
        if _load_latent_path and os.path.exists(_load_latent_path):
            blob = torch.load(_load_latent_path, map_location=device, weights_only=True)
            loaded = blob["latent_in"] if isinstance(blob, dict) and "latent_in" in blob else blob
            if loaded.shape != latents.shape:
                raise ValueError(
                    f"SANA_WM_LOAD_LATENT_FROM shape mismatch: file={tuple(loaded.shape)}, "
                    f"pipeline={tuple(latents.shape)}."
                )
            latents = loaded.to(device=device, dtype=dtype)
            print(
                f"[sana-wm probe] overrode initial latent from {_load_latent_path}; "
                f"shape={tuple(latents.shape)} std={latents.float().std().item():.4f}",
                flush=True,
            )

        allow_hash_fallback = bool(extra_args.get("sana_wm_hash_prompt_smoke", False))
        prompt_embeds, prompt_source = self._native_smoke_prompt_embeds(
            prompt,
            device=device,
            dtype=dtype,
            allow_hash_fallback=allow_hash_fallback,
        )
        # Audit §6.13f probe hook: override prompt_embeds from a file
        # (typically the NVlabs step-0 dump's `prompt_embeds` field) so
        # the model sees byte-identical text conditioning.
        _load_prompt_path = os.environ.get("SANA_WM_LOAD_PROMPT_FROM", "")
        if _load_prompt_path and os.path.exists(_load_prompt_path):
            _blob = torch.load(_load_prompt_path, map_location=device, weights_only=True)
            _loaded_prompt = _blob.get("prompt_embeds") if isinstance(_blob, dict) else _blob
            if _loaded_prompt is None:
                raise ValueError(f"SANA_WM_LOAD_PROMPT_FROM file missing 'prompt_embeds': {_load_prompt_path}")
            _loaded_prompt = _loaded_prompt.to(device=device, dtype=dtype)
            # NVlabs CFG-doubled dump (B=2): drop uncond/cond duplication and
            # take the cond branch. If single-batch already, no-op.
            if _loaded_prompt.shape[0] == 2 and prompt_embeds.shape[0] == 1:
                _loaded_prompt = _loaded_prompt[1:]
            # Match dim layout: NVlabs uses (B, 1, N, D), ours (B, N, D).
            if _loaded_prompt.ndim == 4 and _loaded_prompt.shape[1] == 1 and prompt_embeds.ndim == 3:
                _loaded_prompt = _loaded_prompt.squeeze(1)
            if _loaded_prompt.shape != prompt_embeds.shape:
                print(
                    f"[sana-wm probe] WARNING prompt shape mismatch: "
                    f"loaded={tuple(_loaded_prompt.shape)} vs pipeline={tuple(prompt_embeds.shape)}; "
                    f"using loaded as-is, model must handle.",
                    flush=True,
                )
            prompt_embeds = _loaded_prompt
            prompt_source = f"loaded:{_load_prompt_path}"
            print(
                f"[sana-wm probe] overrode prompt_embeds from {_load_prompt_path}; "
                f"shape={tuple(prompt_embeds.shape)} norm={prompt_embeds.float().norm().item():.3f}",
                flush=True,
            )

        prompt_attention_mask = getattr(self, "_last_prompt_attention_mask", None)

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
        spatial_raymap = camera_tensors.get("spatial_raymap")
        if spatial_raymap is not None:
            spatial_raymap = spatial_raymap.to(device=device, dtype=dtype)

        self.transformer.config = self.sana_wm_config
        use_cudagraph = sana_wm_cudagraph_requested(extra_args)
        cudagraph_denoiser = SanaWmCudaGraphDenoiser() if use_cudagraph else None
        cudagraph_buckets = parse_sana_wm_cudagraph_buckets(extra_args) if use_cudagraph else ()

        # Build a per-frame condition mask matching NVlabs ``LTXFlowEuler``:
        # frame 0 is the conditioning frame at sigma=0 (preserved after
        # each ``scheduler.step``), all other frames carry the current
        # sampling sigma. We omit the legacy `add_noise_to_image_conditioning_latents`
        # motion-continuity term because the public SANA-WM config sets
        # ``condition_frame_info={0: 0.0}`` (image_cond_noise_scale=0).
        use_per_frame_timestep = first_latent is not None and not per_frame_timestep_disabled
        if use_per_frame_timestep:
            cam_batch = latents.shape[0]
            cam_frames = latents.shape[2]
            condition_mask = torch.zeros(
                cam_batch, 1, cam_frames, 1, 1, device=latents.device, dtype=latents.dtype
            )
            condition_mask[:, :, 0] = 1.0
        else:
            condition_mask = None
            cam_batch = latents.shape[0]
            cam_frames = latents.shape[2]
        # Audit §6.13c: the per-frame contract pairs with a per-token
        # flow-matching Euler step that consumes per-token sigmas. We
        # keep the wrapped DPMSolver fallback behind an opt-out env
        # var purely as an ablation knob — by default we run the
        # NVlabs-style step.
        per_token_step_disabled = os.environ.get(
            "VLLM_OMNI_SANA_WM_DISABLE_PER_TOKEN_STEP", ""
        ).lower() in {"1", "true", "yes", "on"}
        use_per_token_step = use_per_frame_timestep and not per_token_step_disabled
        # ``condition_mask`` post-step ``torch.where`` is now a no-op
        # safety belt under the per-token step (conditioning sigma is
        # already 0, so the step never moves those tokens). It's
        # enabled by default. The legacy DPMSolver-step path still
        # hurts under mask=ON, so default-off in that mode.
        mask_default = use_per_token_step
        mask_override = os.environ.get("VLLM_OMNI_SANA_WM_ENABLE_COND_MASK", "")
        if mask_override:
            mask_enabled = mask_override.lower() in {"1", "true", "yes", "on"}
        else:
            mask_enabled = mask_default
        if condition_mask is not None:
            # Pre-flatten the per-token timestep to (B, F*H*W) once;
            # values come from the (B, 1, F) model timestep at each step.
            pass

        for _step_idx, timestep in enumerate(timesteps):
            if use_per_frame_timestep:
                # (B, 1, F) per-frame timestep, frame 0 forced to 0.
                # Keep fp32 to match NVlabs ``LTXFlowEuler.sample`` — the
                # sinusoidal-embedding inside ``SanaWmTimestepEmbedder``
                # uses ``timestep.float()`` internally; casting through
                # the latent dtype (bf16) first would quantise the
                # timestep before the embed and inject a hidden
                # precision variable into the contract.
                model_timestep = timestep.float().expand(
                    cam_batch, 1, cam_frames
                ).clone()
                model_timestep[:, :, 0] = 0.0
            else:
                model_timestep = timestep.expand(1)
            if cudagraph_denoiser is not None:
                noise_pred, _ = cudagraph_denoiser.run(
                    self.transformer,
                    latents,
                    model_timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    plucker=plucker,
                    spatial_raymap=spatial_raymap,
                    num_frames=params.num_frames,
                    buckets=cudagraph_buckets,
                )
            else:
                noise_pred = self.transformer(
                    latents,
                    model_timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    plucker=plucker,
                    raymap=raymap,
                    spatial_raymap=spatial_raymap,
                )
            # Audit §6.13f probe hook: opt-in dump of native step-0
            # (input latent, per-frame timestep, noise_pred) for the
            # noise_pred parity comparison vs the NVlabs LTXFlowEuler
            # step-0 dump (see flow_euler_sampler.py patch).
            _dump_path = os.environ.get("SANA_WM_DUMP_STEP0", "")
            _dump_steps_path = os.environ.get("SANA_WM_DUMP_STEPS_PREFIX", "")
            _dump_step_count = int(os.environ.get("SANA_WM_DUMP_STEP_COUNT", "1"))
            # Audit §6.13t engine-vs-standalone probe: opt-in full transformer
            # input dump so a separate harness can re-call transformer.forward
            # with byte-identical inputs and measure dispatch-level drift.
            _dump_engine_path = os.environ.get(
                "SANA_WM_DUMP_ENGINE_STEP0_INPUTS", ""
            )
            if _dump_engine_path and _step_idx == 0:
                def _to_cpu(t):
                    return t.detach().cpu() if hasattr(t, "detach") else t

                torch.save(
                    {
                        "latents": _to_cpu(latents),
                        "model_timestep": _to_cpu(model_timestep),
                        "prompt_embeds": _to_cpu(prompt_embeds),
                        "plucker": _to_cpu(plucker),
                        "raymap": _to_cpu(raymap),
                        "spatial_raymap": _to_cpu(spatial_raymap),
                        "noise_pred": _to_cpu(noise_pred),
                        "t_scalar": _to_cpu(timestep),
                        "num_frames": int(params.num_frames),
                        "use_cudagraph": cudagraph_denoiser is not None,
                        "condition_mask": (
                            _to_cpu(condition_mask) if condition_mask is not None else None
                        ),
                    },
                    _dump_engine_path,
                )
                print(
                    f"[sana-wm probe] saved engine step-0 transformer inputs "
                    f"latents={tuple(latents.shape)} "
                    f"prompt_embeds={tuple(prompt_embeds.shape)} "
                    f"raymap={tuple(raymap.shape) if hasattr(raymap, 'shape') else None} "
                    f"plucker={tuple(plucker.shape) if hasattr(plucker, 'shape') else None} "
                    f"spatial_raymap={tuple(spatial_raymap.shape) if hasattr(spatial_raymap, 'shape') else None} "
                    f"-> {_dump_engine_path}",
                    flush=True,
                )
            if _dump_path and _step_idx == 0:
                torch.save(
                    {
                        "latent_in": latents.detach().cpu(),
                        "timestep_per_frame": (
                            model_timestep.detach().cpu()
                            if hasattr(model_timestep, "detach")
                            else model_timestep
                        ),
                        "noise_pred": noise_pred.detach().cpu(),
                        "raymap": raymap.detach().cpu() if hasattr(raymap, "detach") else raymap,
                        "prompt_embeds_first_row": prompt_embeds[:, 0].detach().cpu(),
                        "prompt_embeds_norm": prompt_embeds.float().norm().item(),
                        "t_scalar": timestep.detach().cpu() if hasattr(timestep, "detach") else timestep,
                    },
                    _dump_path,
                )
                print(
                    f"[sana-wm probe] saved native step-0 dump latent_in={tuple(latents.shape)} "
                    f"noise_pred={tuple(noise_pred.shape)} to {_dump_path}",
                    flush=True,
                )
            if _dump_steps_path and _step_idx < _dump_step_count:
                _step_path = f"{_dump_steps_path}_step{_step_idx}.pt"
                torch.save(
                    {
                        "latent_in": latents.detach().cpu(),
                        "timestep_per_frame": (
                            model_timestep.detach().cpu()
                            if hasattr(model_timestep, "detach")
                            else model_timestep
                        ),
                        "noise_pred": noise_pred.detach().cpu(),
                        "t_scalar": timestep.detach().cpu() if hasattr(timestep, "detach") else timestep,
                    },
                    _step_path,
                )
                print(
                    f"[sana-wm probe] native step-{_step_idx} dump "
                    f"latent_norm={latents.float().norm().item():.2f} "
                    f"noise_pred_norm={noise_pred.float().norm().item():.2f} -> {_step_path}",
                    flush=True,
                )

            if use_per_token_step:
                # Build per-token timesteps from the (B, 1, F) model
                # timestep by broadcasting to (B, 1, F, H, W) and
                # flattening F*H*W. Conditioning tokens (frame 0) are
                # already at 0 in model_timestep.
                pt_t = (
                    model_timestep.unsqueeze(-1)
                    .unsqueeze(-1)
                    .expand(cam_batch, 1, cam_frames, latents.shape[3], latents.shape[4])
                    .reshape(cam_batch, -1)
                )
                stepped = scheduler.step_flow_euler_per_token(
                    noise_pred, timestep, latents, pt_t
                )
            else:
                stepped = scheduler.step(noise_pred, timestep, latents)

            if condition_mask is not None and mask_enabled:
                if use_per_token_step:
                    # Match NVlabs LTXFlowEuler exactly. This is stricter
                    # than a plain condition-frame restore: with bf16 latents,
                    # the `t=1000` comparison rounds `1 - 1e-6` to `1`, so
                    # the first full-noise step is discarded for generated
                    # tokens as well.
                    tokens_to_denoise_mask = (
                        timestep / float(scheduler.num_train_timesteps) - 1e-6
                        < (1.0 - condition_mask)
                    )
                    latents = torch.where(tokens_to_denoise_mask, stepped, latents)
                else:
                    latents = torch.where(condition_mask > 0.5, latents, stepped)
            else:
                latents = stepped

        # Audit §6.13o probe hook: opt-in dump of native Stage-1 final latent
        # (post per-token Euler loop, pre refiner+VAE). Mirrors NVlabs's
        # SANA_WM_DUMP_STAGE1_LATENT for direct stage-1 latent parity probes.
        _stage1_dump = os.environ.get("SANA_WM_DUMP_NATIVE_STAGE1_LATENT", "")
        if _stage1_dump:
            torch.save(latents.detach().cpu(), _stage1_dump)
            print(
                f"[sana-wm probe] saved native Stage-1 latent shape={tuple(latents.shape)} "
                f"norm={latents.float().norm().item():.2f} -> {_stage1_dump}",
                flush=True,
            )

        output_type = str(extra_args.get("sana_wm_output_type", "latent"))
        output = self._decode_native_smoke_latents(latents, output_type=output_type, device=device, dtype=dtype)
        return DiffusionOutput(
            output=output,
            custom_output={
                "sana_wm_backend": "native_gdn",
                "sana_wm_output_space": output_type,
                "sana_wm_prompt_source": prompt_source,
                "sana_wm_chi_prompt_applied": bool(self.sana_wm_config.chi_prompt),
                "sana_wm_first_frame_encoded": _is_image,
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
