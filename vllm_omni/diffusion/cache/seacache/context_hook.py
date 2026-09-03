# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SeaCache driven by a TeaCache ``CacheContext`` extractor.

:class:`SeaCacheRootHook` needs the transformer to expose a block boundary it
can bypass. Models that already publish a TeaCache extractor describe that
boundary as ``run_transformer_blocks`` / ``postprocess`` callables, so this
hook reuses the extractor and adds only the spectral gate plus residual
extrapolation. No model code changes are required. Step metadata comes from
the forward context the denoise loop publishes; the cache tracks a single
(cfg-distilled) branch.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from vllm_omni.diffusion.cache.seacache.config import SeaCacheConfig
from vllm_omni.diffusion.cache.seacache.hook import (
    SeaCacheRootHook,
    _is_parameter_sharded,
    collect_offload_groups,
)
from vllm_omni.diffusion.cache.seacache.sea_filter import extrapolate_residual
from vllm_omni.diffusion.cache.seacache.state import SeaCacheState
from vllm_omni.diffusion.cache.teacache.extractors import get_extractor
from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available
from vllm_omni.diffusion.hooks import HookRegistry

IndicatorFn = Callable[..., "list[torch.Tensor] | None"]


class SeaCacheContextHook(SeaCacheRootHook):
    """Spectral-evolution-aware caching over an extractor-provided block boundary."""

    def __init__(self, config: SeaCacheConfig, *, transformer_type: str, indicator_fn: IndicatorFn) -> None:
        super().__init__(config)
        self.extractor_fn = get_extractor(transformer_type)
        self.indicator_fn = indicator_fn

    def initialize_hook(self, module: torch.nn.Module) -> torch.nn.Module:
        self._parameter_sharded = _is_parameter_sharded(module)
        self._collective_skip_groups = collect_offload_groups(getattr(module, "blocks", ()))
        self.state_manager.set_context(self._HOOK_NAME)
        return module

    # The root hook gates in pre/post_forward; here the whole forward is replaced.
    def pre_forward(self, module: torch.nn.Module, *args: Any, **kwargs: Any) -> tuple[tuple, dict]:
        return args, kwargs

    def post_forward(self, module: torch.nn.Module, output: Any) -> Any:
        return output

    @torch.compiler.disable
    def new_forward(self, module: torch.nn.Module, *args: Any, **kwargs: Any) -> Any:
        ctx = self.extractor_fn(module, *args, **kwargs)
        state: SeaCacheState = self.state_manager.get_state()
        context = get_forward_context() if is_forward_context_available() else None
        step, num_steps = getattr(context, "denoise_step_idx", None), getattr(context, "total_denoise_steps", None)
        if torch.is_grad_enabled() or step is None or num_steps is None or not 0 <= step < num_steps:
            self._warn_once("SeaCache needs an inference-mode call with denoise progress on the forward context.")
            return ctx.postprocess(ctx.run_transformer_blocks()[0])

        try:
            indicator = self.indicator_fn(module, self.config, **kwargs)
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            self._warn_once(f"SeaCache could not construct its indicator; running full: {error}")
            indicator = None
        local_compute = self._resolve_gate(state, indicator, int(step), int(num_steps))
        should_compute = self._synchronize_compute(local_compute, ctx.hidden_states.device)
        synchronize = (ctx.extra_states or {}).get("synchronize_cache_decision")
        if synchronize is not None:
            should_compute = bool(synchronize(should_compute))
        if should_compute and not local_compute:
            state.accumulated_distance = 0.0

        hidden = ctx.hidden_states
        if not should_compute:
            residual = extrapolate_residual(state.history, int(step), self.config.residual_order)
            if residual.shape == hidden.shape:
                state.consecutive_cached += 1
                self.skip_count += 1
                return ctx.postprocess(hidden + residual)
            state.history.clear()

        original = hidden.clone()
        hidden = ctx.run_transformer_blocks()[0]
        state.history.append((int(step), (hidden - original).detach()))
        state.history = state.history[-(self.config.residual_order + 1) :]
        state.consecutive_cached = 0
        self.full_count += 1
        return ctx.postprocess(hidden)

    def reset_state(self, module: torch.nn.Module) -> torch.nn.Module:
        self.state_manager.reset()
        self.full_count = 0
        self.skip_count = 0
        return module


def apply_sea_cache_context_hook(
    module: torch.nn.Module,
    config: SeaCacheConfig,
    *,
    transformer_type: str,
    indicator_fn: IndicatorFn,
) -> SeaCacheContextHook:
    """Install extractor-driven SeaCache on ``module`` under the shared hook name."""
    registry = HookRegistry.get_or_create(module)
    hook = SeaCacheContextHook(config, transformer_type=transformer_type, indicator_fn=indicator_fn)
    registry.register_hook(SeaCacheRootHook._HOOK_NAME, hook)
    return hook
