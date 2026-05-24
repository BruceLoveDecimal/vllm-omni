# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-stage SANA-WM pipeline scaffold."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Iterable

import torch
from torch import nn

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.sana_wm.pipeline_sana_wm import (
    SanaWmPipeline,
    get_sana_wm_pre_process_func,
    get_sana_wm_post_process_func,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest


class SanaWmTwoStagesPipeline(SanaWmPipeline):
    """SANA-WM Stage 1 plus LTX-2 refiner placeholder."""

    include_refiner: ClassVar[bool] = True
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = [
        "text_encoder",
        "camera_encoder",
        "refiner_text_encoder",
        "refiner_connectors",
    ]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    _resident_modules: ClassVar[list[str]] = ["refiner_transformer"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.refiner_transformer: nn.Module | None = None
        self.refiner_text_encoder: nn.Module | None = None
        self.refiner_connectors: nn.Module | None = None
        self.refiner_tokenizer: Any | None = None

    def _local_files_only(self) -> bool:
        if self.release_paths is not None:
            return True
        if self.od_config is None or self.od_config.model is None:
            return False
        return Path(str(self.od_config.model)).expanduser().exists()

    def _ensure_refiner_text_encoder(self, *, device: torch.device, dtype: torch.dtype) -> None:
        if self.refiner_text_encoder is not None and self.refiner_tokenizer is not None:
            self.refiner_text_encoder.to(device=device, dtype=dtype)
            return
        if self.release_paths is None:
            self.resolve_checkpoint(include_refiner=True)
        if self.release_paths is None:
            raise ValueError("Sana-WM two-stage refiner text encoder requires a resolved checkpoint.")

        from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

        local_path = str(self.release_paths.refiner_text_encoder_dir)
        self.refiner_tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
        with torch.device("cpu"):
            self.refiner_text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
                local_path,
                torch_dtype=dtype,
                local_files_only=True,
            ).to(device)

    def _ensure_refiner_connectors(self, *, device: torch.device, dtype: torch.dtype) -> None:
        if self.refiner_connectors is not None:
            self.refiner_connectors.to(device=device, dtype=dtype)
            return
        if self.release_paths is None:
            self.resolve_checkpoint(include_refiner=True)
        if self.release_paths is None:
            raise ValueError("Sana-WM two-stage refiner connectors require a resolved checkpoint.")

        from diffusers.pipelines.ltx2 import LTX2TextConnectors

        self.refiner_connectors = LTX2TextConnectors.from_pretrained(
            str(self.release_paths.root),
            subfolder="refiner/connectors",
            torch_dtype=dtype,
            local_files_only=True,
        ).to(device)

    def _load_refiner_transformer_config(self) -> dict[str, Any]:
        if self.release_paths is None:
            self.resolve_checkpoint(include_refiner=True)
        if self.release_paths is None:
            raise ValueError("Sana-WM two-stage refiner transformer requires a resolved checkpoint.")
        with self.release_paths.refiner_transformer_config.open(encoding="utf-8") as f:
            return json.load(f)

    def _ensure_refiner_transformer(self, *, device: torch.device, dtype: torch.dtype) -> None:
        if self.refiner_transformer is not None:
            self.refiner_transformer.to(device=device, dtype=dtype)
            return
        if self.release_paths is None:
            self.resolve_checkpoint(include_refiner=True)
        if self.release_paths is None:
            raise ValueError("Sana-WM two-stage refiner transformer requires a resolved checkpoint.")

        from safetensors.torch import load_file
        from contextlib import nullcontext

        from vllm.config import (
            CompilationConfig,
            DeviceConfig,
            VllmConfig,
            get_current_vllm_config,
            set_current_vllm_config,
        )
        from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import create_transformer_from_config

        try:
            get_current_vllm_config()
            vllm_config_context = nullcontext()
        except AssertionError:
            vllm_config = VllmConfig(
                compilation_config=CompilationConfig(),
                device_config=DeviceConfig(device=device),
            )
            vllm_config_context = set_current_vllm_config(vllm_config)

        with vllm_config_context, torch.device("cpu"):
            self.refiner_transformer = create_transformer_from_config(self._load_refiner_transformer_config())
        state_dict = load_file(str(self.release_paths.refiner_transformer_weights), device="cpu")
        if hasattr(self.refiner_transformer, "load_weights"):
            self.refiner_transformer.load_weights(state_dict.items())
        else:
            self.refiner_transformer.load_state_dict(state_dict, strict=False)
        self.refiner_transformer.to(device=device, dtype=dtype)

    def ensure_refiner_components(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Load the LTX-2 19B refiner components from the SANA-WM release tree."""

        runtime_device, runtime_dtype = self._runtime_device_dtype()
        device = device or runtime_device
        dtype = dtype or runtime_dtype
        self._ensure_refiner_text_encoder(device=device, dtype=dtype)
        self._ensure_refiner_connectors(device=device, dtype=dtype)
        self._ensure_refiner_transformer(device=device, dtype=dtype)

    def forward(self, req: OmniDiffusionRequest, *args: Any, **kwargs: Any) -> DiffusionOutput:
        extra_args = self._extra_args(getattr(req, "sampling_params", None))
        if bool(extra_args.get("sana_wm_load_refiner_components", False)):
            device, dtype = self._runtime_device_dtype()
            self.ensure_refiner_components(device=device, dtype=dtype)
        return super().forward(req, *args, **kwargs)

    def load_weights(self, weights: Iterable[tuple[str, Any]]) -> set[str]:
        return super().load_weights(weights)


__all__ = ["SanaWmTwoStagesPipeline", "get_sana_wm_pre_process_func", "get_sana_wm_post_process_func"]
