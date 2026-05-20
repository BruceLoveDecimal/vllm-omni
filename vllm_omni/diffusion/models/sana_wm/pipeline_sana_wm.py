# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sana-WM pipeline integration.

This module wires the registry-visible surface, release-layout validation, and
an optional official CLI backend for GPU e2e smoke testing. The native Stage-1
DiT/Gated-DeltaNet path still needs a full architecture port.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, ClassVar, Iterable

from torch import nn

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import SupportImageInput, SupportsComponentDiscovery
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.models.sana_wm.official_backend import (
    SANA_WM_OFFICIAL_BACKEND_ERROR,
    get_sana_wm_official_repo_path,
    is_sana_wm_official_backend_requested,
    require_sana_wm_official_script,
    run_sana_wm_official_backend,
)
from vllm_omni.diffusion.models.sana_wm.config import SanaWmConfig
from vllm_omni.diffusion.models.sana_wm.sana_wm_transformer import SanaWmTransformer3DModel
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import (
    DiffusionPipelineProfilerMixin,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific
from vllm_omni.model_executor.stage_input_processors.sana_wm import normalize_sana_wm_payload

SANA_WM_MODEL_ID = "Efficient-Large-Model/SANA-WM_bidirectional"
SANA_WM_SCAFFOLD_ERROR = (
    "Sana-WM native Stage-1 DiT, Gated DeltaNet, Plucker camera injection, "
    "and denoising are not implemented yet. "
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
SANA_WM_OUTPUT_HEIGHT = 704
SANA_WM_OUTPUT_WIDTH = 1280
SANA_WM_DEFAULT_NUM_FRAMES = 321

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
    refiner_transformer_config: Path
    refiner_transformer_weights: Path
    refiner_connectors_config: Path
    refiner_connectors_weights: Path
    refiner_text_encoder_dir: Path


def build_sana_wm_download_patterns(*, include_refiner: bool = True) -> tuple[str, ...]:
    """Return the minimal HF allow-patterns needed for SANA-WM."""

    patterns = list(SANA_WM_STAGE1_PATTERNS)
    if include_refiner:
        patterns.extend(SANA_WM_REFINER_PATTERNS)
    return tuple(patterns)


def resolve_sana_wm_local_paths(snapshot_dir: str | Path) -> SanaWmLocalPaths:
    root = Path(snapshot_dir)
    return SanaWmLocalPaths(
        root=root,
        config=root / SANA_WM_CONFIG_FILE,
        stage1_dit=root / SANA_WM_STAGE1_DIT_FILE,
        vae_config=root / SANA_WM_VAE_CONFIG_FILE,
        vae_weights=root / SANA_WM_VAE_WEIGHT_FILE,
        refiner_transformer_config=root / SANA_WM_REFINER_TRANSFORMER_CONFIG_FILE,
        refiner_transformer_weights=root / SANA_WM_REFINER_TRANSFORMER_WEIGHT_FILE,
        refiner_connectors_config=root / SANA_WM_REFINER_CONNECTORS_CONFIG_FILE,
        refiner_connectors_weights=root / SANA_WM_REFINER_CONNECTORS_WEIGHT_FILE,
        refiner_text_encoder_dir=root / SANA_WM_REFINER_TEXT_ENCODER_DIR,
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
    paths = resolve_sana_wm_local_paths(snapshot_dir)
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
    SupportImageInput,
    SupportsComponentDiscovery,
    ProgressBarMixin,
    DiffusionPipelineProfilerMixin,
):
    """Stage-1 SANA-WM image-to-video pipeline placeholder."""

    support_image_input: ClassVar[bool] = True
    color_format: ClassVar[str] = "RGB"
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
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

        # Filled by the real implementation.
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
        return self.release_paths

    def forward(self, req: OmniDiffusionRequest, *args: Any, **kwargs: Any) -> DiffusionOutput:
        del args, kwargs
        if len(req.prompts) != 1:
            raise ValueError("Sana-WM official backend currently supports exactly one prompt per request.")

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
            raise NotImplementedError(SANA_WM_SCAFFOLD_ERROR)
        require_sana_wm_official_script(repo_path)

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

    def load_weights(self, weights: Iterable[tuple[str, Any]]) -> set[str]:
        if self.use_official_backend:
            return set()
        return self.transformer.load_weights(weights)
