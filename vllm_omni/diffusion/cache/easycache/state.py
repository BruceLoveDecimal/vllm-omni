# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
EasyCache state management.

This module contains the EasyCacheState class which tracks, per CFG branch,
the tensors and scalars needed by the EasyCache skip decision: the previous
step input, the last fully computed input/output pair, the cached block-stack
residual, the online transformation rate and the accumulated error.
"""

from __future__ import annotations

import torch


class EasyCacheState:
    """State management for EasyCache caching logic."""

    def __init__(self) -> None:
        """Initialize empty EasyCache state."""
        # Input of the head block at the previous step (computed or skipped).
        self.previous_step_input: torch.Tensor | None = None
        # Head input / tail output of the last fully computed step.
        self.last_full_input: torch.Tensor | None = None
        self.last_full_output: torch.Tensor | None = None
        # Cached residual of the whole block stack (tail output - head input).
        self.cached_residual: torch.Tensor | None = None
        # Online transformation rate k = mean|dy| / mean|dx| between full steps.
        self.transform_rate: float | None = None
        self.accumulated_error: float = 0.0
        self.consecutive_skips: int = 0
        self.step_index: int = 0
        # Per-forward scratch: head input of the current step and the decision.
        self.head_block_input: torch.Tensor | None = None
        self.should_compute: bool = True
        # Block output layout learned on the first computed step, used to
        # rebuild outputs on skipped steps. Either "tensor" or a tuple of
        # per-element tags ("hidden", "encoder").
        self.output_layout: str | tuple[str, ...] | None = None
        # Counters for the current run (reported at end of run).
        self.num_computed: int = 0
        self.num_skipped: int = 0

    def clear_cache(self) -> None:
        """Drop cached tensors and the estimator, keeping the step schedule."""
        self.previous_step_input = None
        self.last_full_input = None
        self.last_full_output = None
        self.cached_residual = None
        self.transform_rate = None
        self.accumulated_error = 0.0
        self.consecutive_skips = 0

    def reset(self) -> None:
        """Reset all state variables for a new inference run."""
        self.clear_cache()
        self.step_index = 0
        self.head_block_input = None
        self.should_compute = True
        self.output_layout = None
        self.num_computed = 0
        self.num_skipped = 0
