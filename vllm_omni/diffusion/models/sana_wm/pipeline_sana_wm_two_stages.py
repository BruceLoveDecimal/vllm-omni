# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-stage SANA-WM pipeline scaffold."""

from __future__ import annotations

from typing import Any, ClassVar, Iterable

from torch import nn

from vllm_omni.diffusion.models.sana_wm.pipeline_sana_wm import (
    SanaWmPipeline,
    get_sana_wm_pre_process_func,
    get_sana_wm_post_process_func,
)


class SanaWmTwoStagesPipeline(SanaWmPipeline):
    """SANA-WM Stage 1 plus LTX-2 refiner placeholder."""

    include_refiner: ClassVar[bool] = True
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder", "refiner_text_encoder", "refiner_connectors"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    _resident_modules: ClassVar[list[str]] = ["refiner_transformer"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.refiner_transformer: nn.Module | None = None
        self.refiner_text_encoder: nn.Module | None = None
        self.refiner_connectors: nn.Module | None = None

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return super().forward(*args, **kwargs)

    def load_weights(self, weights: Iterable[tuple[str, Any]]) -> set[str]:
        return super().load_weights(weights)


__all__ = ["SanaWmTwoStagesPipeline", "get_sana_wm_pre_process_func", "get_sana_wm_post_process_func"]
