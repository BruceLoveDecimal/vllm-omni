# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class EasyCacheConfig:
    """
    Configuration for EasyCache applied to transformer models.

    EasyCache is a runtime-adaptive caching technique that skips the whole
    transformer block stack on a denoising step when the *predicted* relative
    change of the block-stack output, accumulated over the steps since the
    last full computation, stays below a threshold. The prediction is driven
    by an online estimate of the input-to-output transformation rate, so no
    model-specific calibration is required.

    Reference:
        Zhou et al., "Less is Enough: Training-Free Video Diffusion
        Acceleration via Runtime-Adaptive Caching", arXiv:2507.02860.
        https://github.com/H-EmbodVis/EasyCache

    Args:
        threshold: Accumulated predicted relative change above which the block
            stack is recomputed. Higher = more aggressive skipping (faster,
            lower quality). Default: 0.1
        warmup_steps: Number of initial denoising steps that always compute.
            The estimator needs at least two full steps before it can skip.
            Default: 5
        cooldown_steps: Number of final denoising steps that always compute.
            Default: 1
        max_skip_steps: Maximum consecutive skipped steps before a full
            computation is forced. 0 disables the cap. Default: 0
        num_inference_steps: Total denoising steps of the current generation.
            Required for the cooldown window and end-of-run state reset.
            Default: 50
        transformer_type: Transformer class name for logging. Default: ""
    """

    threshold: float = 0.1
    warmup_steps: int = 5
    cooldown_steps: int = 1
    max_skip_steps: int = 0
    num_inference_steps: int = 50
    transformer_type: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError(f"threshold must be finite and positive, got {self.threshold}")
        for name in ("warmup_steps", "cooldown_steps", "max_skip_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
        if isinstance(self.num_inference_steps, bool) or not isinstance(self.num_inference_steps, int):
            raise ValueError(f"num_inference_steps must be an integer, got {self.num_inference_steps!r}")
        if self.num_inference_steps <= 0:
            raise ValueError(f"num_inference_steps must be positive, got {self.num_inference_steps}")

    @property
    def first_cooldown_step(self) -> int:
        """First step index (inclusive) of the always-compute cooldown window."""
        return max(self.num_inference_steps - self.cooldown_steps, 0)
