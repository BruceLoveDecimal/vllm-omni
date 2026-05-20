# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SANA-WM Stage-1 transformer scaffold."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, ClassVar, Iterable

import torch
from torch import nn

from vllm_omni.diffusion.models.sana_wm.config import SanaWmConfig
from vllm_omni.diffusion.models.sana_wm.weight_mapping import normalize_sana_wm_stage1_weight_name

SANA_WM_TRANSFORMER_FORWARD_ERROR = (
    "Sana-WM Stage-1 transformer forward is not implemented yet. "
    "Weights can be loaded and audited, but Gated DeltaNet, Plucker injection, "
    "and denoising still need the official architecture port."
)


@dataclass(frozen=True)
class SanaWmStage1LoadReport:
    total_weights: int = 0
    loaded_weights: int = 0
    unmapped_weights: tuple[str, ...] = ()
    duplicate_weights: tuple[str, ...] = ()
    loaded_names: tuple[str, ...] = field(default_factory=tuple)


class SanaWmTransformer3DModel(nn.Module):
    """Placeholder for the 20-block SANA-WM DiT.

    The actual implementation must port the official
    ``SanaMSVideoCamCtrl_1600M_P1_D20`` architecture, including hybrid
    Bidirectional Gated DeltaNet attention and Plucker post-attention camera
    injection.
    """

    _repeated_blocks: ClassVar[list[str]] = ["blocks"]
    _layerwise_offload_blocks_attr: ClassVar[str] = "blocks"

    def __init__(self, config: SanaWmConfig | None = None, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.config = config or SanaWmConfig()
        self.args = args
        self.kwargs = kwargs
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)
        self._loaded_parameters = nn.ParameterDict()
        self._source_to_remapped_name: dict[str, str] = {}
        self._remapped_to_storage_name: dict[str, str] = {}
        self._storage_to_remapped_name: dict[str, str] = {}
        self.last_load_report = SanaWmStage1LoadReport()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(SANA_WM_TRANSFORMER_FORWARD_ERROR)

    @staticmethod
    def _storage_name(remapped_name: str, index: int) -> str:
        digest = hashlib.sha1(remapped_name.encode("utf-8")).hexdigest()[:16]
        return f"w_{index}_{digest}"

    def _store_tensor(self, remapped_name: str, tensor: torch.Tensor) -> None:
        storage_name = self._storage_name(remapped_name, len(self._remapped_to_storage_name))
        target_device = self._device_anchor.device
        if target_device.type != "meta" and tensor.device != target_device:
            tensor = tensor.to(target_device)
        if torch.is_floating_point(tensor) or torch.is_complex(tensor):
            self._loaded_parameters[storage_name] = nn.Parameter(tensor.detach(), requires_grad=False)
        else:
            self.register_buffer(storage_name, tensor.detach(), persistent=True)
        self._remapped_to_storage_name[remapped_name] = storage_name
        self._storage_to_remapped_name[storage_name] = remapped_name

    def get_loaded_tensor(self, remapped_name: str) -> torch.Tensor:
        storage_name = self._remapped_to_storage_name[remapped_name]
        if storage_name in self._loaded_parameters:
            return self._loaded_parameters[storage_name]
        return getattr(self, storage_name)

    def load_weights(self, weights: Iterable[tuple[str, Any]]) -> set[str]:
        loaded: set[str] = set()
        unmapped: list[str] = []
        duplicates: list[str] = []
        total = 0

        for source_name, tensor in weights:
            total += 1
            remapped_name = normalize_sana_wm_stage1_weight_name(source_name)
            if remapped_name is None:
                unmapped.append(source_name)
                continue
            if remapped_name in self._remapped_to_storage_name:
                duplicates.append(remapped_name)
                continue
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Sana-WM weight {source_name!r} must be a torch.Tensor, got {type(tensor).__name__}.")
            self._store_tensor(remapped_name, tensor)
            self._source_to_remapped_name[source_name] = remapped_name
            loaded.add(remapped_name)

        self.last_load_report = SanaWmStage1LoadReport(
            total_weights=total,
            loaded_weights=len(loaded),
            unmapped_weights=tuple(unmapped),
            duplicate_weights=tuple(duplicates),
            loaded_names=tuple(sorted(loaded)),
        )
        if unmapped or duplicates:
            details = []
            if unmapped:
                details.append(f"unmapped={unmapped[:10]}")
            if duplicates:
                details.append(f"duplicates={duplicates[:10]}")
            raise ValueError("Invalid SANA-WM Stage-1 checkpoint keys: " + "; ".join(details))
        return loaded
