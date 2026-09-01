# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Hook-based EasyCache implementation for vLLM-Omni.

EasyCache accelerates diffusion inference by skipping the whole transformer
block stack on steps where the predicted relative change of the block-stack
output is small. Unlike TeaCache or MagCache it needs no offline coefficient
fitting: the transformation rate between input drift and output drift is
estimated online from consecutive fully computed steps.

Algorithm (per denoising step ``t``, per CFG branch):

1. ``x_t`` is the head block input. Warmup and cooldown steps always compute.
2. Otherwise predict the relative output change from the input drift::

       e_t = k * mean|x_t - x_{t-1}| / mean|y_last|
       acc += e_t

   where ``k = mean|y_last - y_prev| / mean|x_last - x_prev|`` is the
   transformation rate measured between the last two fully computed steps and
   ``y_last`` is the last fully computed block-stack output.
3. If ``acc < threshold`` reuse the cached residual: ``y_t = x_t + (y_last - x_last)``.
   Otherwise run the block stack, refresh ``k``, the residual and reset ``acc``.

All ``mean|.|`` statistics are reduced across the sequence-parallel group, so
every rank derives the same skip decision from the same global statistics.

Reference:
    Zhou et al., "Less is Enough: Training-Free Video Diffusion Acceleration
    via Runtime-Adaptive Caching", arXiv:2507.02860.
    https://github.com/H-EmbodVis/EasyCache

Architecture (mirrors MagCache):
- EasyCacheHeadHook: first block; measures input drift, decides compute/skip
  and applies the cached residual on skip.
- EasyCacheBlockHook: remaining blocks; pass-through on skip. The tail
  instance records the block-stack residual, updates ``k`` and advances the
  step schedule.
"""

from __future__ import annotations

import inspect
import weakref
from typing import Any

import torch

from vllm_omni.diffusion.cache.easycache.config import EasyCacheConfig
from vllm_omni.diffusion.cache.easycache.state import EasyCacheState
from vllm_omni.diffusion.hooks.base import HookRegistry, ModelHook, StateManager
from vllm_omni.logger import init_logger

logger = init_logger(__name__)

_EASY_CACHE_HEAD_HOOK = "easy_cache_head_hook"
_EASY_CACHE_BLOCK_HOOK = "easy_cache_block_hook"
_EASY_CACHE_HOOK_NAMES = (_EASY_CACHE_HEAD_HOOK, _EASY_CACHE_BLOCK_HOOK)

_HIDDEN_STATES_ARG = "hidden_states"
_ENCODER_HIDDEN_STATES_ARG = "encoder_hidden_states"
_EPS = 1e-8


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------


def _get_sequence_parallel_group() -> Any | None:
    """Return the SP group coordinator when SP is active, else ``None``."""
    try:
        from vllm_omni.diffusion.distributed.parallel_state import get_sp_group

        group = get_sp_group()
    except (AssertionError, ImportError, RuntimeError):
        return None
    return group if getattr(group, "world_size", 1) > 1 else None


@torch.no_grad()
def synchronized_mean_abs(tensors: list[torch.Tensor]) -> list[float]:
    """Mean absolute value of each tensor over the whole (SP-sharded) sequence.

    Local ``sum|.|`` and element counts are reduced with ``SUM`` across the
    sequence-parallel group before dividing, so the returned statistics are
    identical on every rank. This keeps the skip decision consistent across
    ranks; a divergent decision would otherwise desynchronize the collectives
    inside the transformer blocks.
    """
    if not tensors:
        return []
    sums = torch.stack([tensor.abs().sum(dtype=torch.float32) for tensor in tensors])
    counts = torch.tensor([tensor.numel() for tensor in tensors], dtype=torch.float32, device=sums.device)
    totals = torch.stack((sums, counts))
    group = _get_sequence_parallel_group()
    if group is not None:
        totals = group.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
    means = totals[0] / totals[1].clamp_min(1.0)
    return [float(value) for value in means.cpu().tolist()]


def _cfg_branch(forward_index: int) -> str:
    """Select the CFG branch when the transformer is called once per branch."""
    try:
        from vllm_omni.diffusion.distributed.parallel_state import (
            get_classifier_free_guidance_rank,
            get_classifier_free_guidance_world_size,
        )

        if get_classifier_free_guidance_world_size() > 1:
            return "negative" if get_classifier_free_guidance_rank() > 0 else "positive"
    except (AssertionError, ImportError, RuntimeError):
        pass
    # No CFG-parallel: positive and negative forwards alternate on one rank.
    return "negative" if forward_index % 2 == 1 else "positive"


# ---------------------------------------------------------------------------
# Block argument / output helpers
# ---------------------------------------------------------------------------


class _BlockIO:
    """Resolve hidden/encoder states from block arguments and outputs."""

    def __init__(self) -> None:
        self._signature: inspect.Signature | None = None

    def bind(self, module: torch.nn.Module, args: tuple, kwargs: dict) -> dict[str, Any]:
        if self._signature is None:
            self._signature = inspect.signature(type(module).forward)
        try:
            return self._signature.bind_partial(module, *args, **kwargs).arguments
        except TypeError:
            return {}

    def hidden_states(self, module: torch.nn.Module, args: tuple, kwargs: dict) -> torch.Tensor:
        bound = self.bind(module, args, kwargs)
        hidden = bound.get(_HIDDEN_STATES_ARG)
        if hidden is None and args:
            hidden = args[0]
        if not isinstance(hidden, torch.Tensor):
            raise TypeError(
                f"EasyCache could not resolve `{_HIDDEN_STATES_ARG}` from the arguments of "
                f"{type(module).__name__}.forward; got {type(hidden).__name__}"
            )
        return hidden

    def encoder_hidden_states(self, module: torch.nn.Module, args: tuple, kwargs: dict) -> torch.Tensor | None:
        value = self.bind(module, args, kwargs).get(_ENCODER_HIDDEN_STATES_ARG)
        return value if isinstance(value, torch.Tensor) else None

    @staticmethod
    def learn_layout(
        output: Any,
        hidden_in: torch.Tensor,
        encoder_in: torch.Tensor | None,
    ) -> str | tuple[str, ...]:
        """Classify a computed block output so it can be rebuilt on skip."""
        if isinstance(output, torch.Tensor):
            return "tensor"
        if not isinstance(output, tuple):
            raise TypeError(f"EasyCache supports blocks returning a tensor or a tuple, got {type(output).__name__}")
        tags: list[str] = []
        hidden_found = False
        for item in output:
            if isinstance(item, torch.Tensor) and not hidden_found and item.shape == hidden_in.shape:
                tags.append("hidden")
                hidden_found = True
            elif isinstance(item, torch.Tensor) and encoder_in is not None and item.shape == encoder_in.shape:
                tags.append("encoder")
            else:
                raise TypeError(
                    "EasyCache supports blocks returning hidden_states or a tuple of "
                    f"(hidden_states, encoder_hidden_states); got an unrecognized element at index {len(tags)}"
                )
        if not hidden_found:
            raise TypeError("EasyCache could not find hidden_states in the block output tuple")
        return tuple(tags)

    @staticmethod
    def output_hidden(output: Any, layout: str | tuple[str, ...]) -> torch.Tensor:
        if layout == "tensor":
            return output
        return output[layout.index("hidden")]

    @staticmethod
    def build_output(
        layout: str | tuple[str, ...],
        hidden: torch.Tensor,
        encoder: torch.Tensor | None,
    ) -> Any:
        if layout == "tensor":
            return hidden
        return tuple(hidden if tag == "hidden" else encoder for tag in layout)


# ---------------------------------------------------------------------------
# Shared step bookkeeping
# ---------------------------------------------------------------------------


class _EasyCacheHookBase(ModelHook):
    def __init__(self, state_manager: StateManager, config: EasyCacheConfig, is_tail: bool = False):
        super().__init__()
        self.state_manager = state_manager
        self.config = config
        self.is_tail = is_tail
        self._io = _BlockIO()

    def reset_state(self, module: torch.nn.Module) -> torch.nn.Module:
        self.state_manager.reset()
        return module

    def _record_full_step(self, state: EasyCacheState, out_hidden: torch.Tensor) -> None:
        """Refresh the residual and the transformation rate after a full compute."""
        in_hidden = state.head_block_input
        if in_hidden is None:
            logger.warning("EasyCache: tail block ran without a recorded head input; cache not updated")
            return
        if out_hidden.shape != in_hidden.shape:
            logger.warning(
                "EasyCache: block stack changed the hidden_states shape (%s -> %s); caching disabled for this step",
                tuple(in_hidden.shape),
                tuple(out_hidden.shape),
            )
            state.clear_cache()
            return

        out_hidden = out_hidden.detach()
        if (
            state.last_full_input is not None
            and state.last_full_output is not None
            and state.last_full_input.shape == in_hidden.shape
            and state.last_full_output.shape == out_hidden.shape
        ):
            input_change, output_change = synchronized_mean_abs(
                [in_hidden - state.last_full_input, out_hidden - state.last_full_output]
            )
            state.transform_rate = output_change / max(input_change, _EPS)

        state.last_full_input = in_hidden
        state.last_full_output = out_hidden
        state.cached_residual = out_hidden - in_hidden
        state.accumulated_error = 0.0
        state.consecutive_skips = 0
        state.num_computed += 1
        logger.debug(
            "[EasyCache][TAIL] STEP=%d: RESIDUAL_COMPUTED (k=%s)",
            state.step_index,
            f"{state.transform_rate:.6f}" if state.transform_rate is not None else "None",
        )

    def _advance_step(self, state: EasyCacheState) -> None:
        state.step_index += 1
        state.head_block_input = None
        if state.step_index >= self.config.num_inference_steps:
            logger.info(
                "EasyCache: run finished (%s): computed %d, skipped %d of %d steps",
                self.state_manager._context,
                state.num_computed,
                state.num_skipped,
                state.step_index,
            )
            state.reset()


class EasyCacheHeadHook(_EasyCacheHookBase):
    """Head block hook for EasyCache - decides whether to skip computation."""

    _HOOK_NAME = "easy_cache_head"

    def __init__(
        self,
        state_manager: StateManager,
        config: EasyCacheConfig,
        is_tail: bool = False,
        parent: torch.nn.Module | None = None,
    ):
        super().__init__(state_manager, config, is_tail=is_tail)
        self._parent_ref = weakref.ref(parent) if parent is not None else None
        self._forward_cnt = 0

    def reset_state(self, module: torch.nn.Module) -> torch.nn.Module:
        self._forward_cnt = 0
        return super().reset_state(module)

    def _select_context(self) -> EasyCacheState:
        parent = self._parent_ref() if self._parent_ref is not None else None
        branch = _cfg_branch(self._forward_cnt) if getattr(parent, "do_true_cfg", False) else "positive"
        self._forward_cnt += 1
        self.state_manager.set_context(f"easycache_{branch}")
        return self.state_manager.get_state()

    def _decide(self, state: EasyCacheState, hidden: torch.Tensor) -> tuple[bool, str]:
        step = state.step_index
        if step < self.config.warmup_steps:
            state.accumulated_error = 0.0
            return True, "warmup"
        if step >= self.config.first_cooldown_step:
            state.accumulated_error = 0.0
            return True, "cooldown"
        if (
            state.cached_residual is None
            or state.previous_step_input is None
            or state.last_full_output is None
            or state.transform_rate is None
        ):
            return True, "initialize"
        if state.cached_residual.shape != hidden.shape or state.previous_step_input.shape != hidden.shape:
            state.clear_cache()
            return True, "shape_change"
        if self.config.max_skip_steps > 0 and state.consecutive_skips >= self.config.max_skip_steps:
            return True, "skip_cap"

        input_change, output_norm = synchronized_mean_abs([hidden - state.previous_step_input, state.last_full_output])
        estimate = state.transform_rate * input_change / max(output_norm, _EPS)
        state.accumulated_error += estimate
        if state.accumulated_error >= self.config.threshold:
            state.accumulated_error = 0.0
            return True, "threshold"
        return False, "below_threshold"

    @torch.compiler.disable
    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        state = self._select_context()
        hidden = self._io.hidden_states(module, args, kwargs)

        should_compute, reason = self._decide(state, hidden)
        current_input = hidden.detach().clone()
        state.previous_step_input = current_input
        state.should_compute = should_compute
        logger.debug(
            "[EasyCache][HEAD] STEP=%d: %s (reason=%s, acc=%.6f, k=%s, consecutive_skips=%d)",
            state.step_index,
            "CACHE_MISS" if should_compute else "CACHE_HIT",
            reason,
            state.accumulated_error,
            f"{state.transform_rate:.6f}" if state.transform_rate is not None else "None",
            state.consecutive_skips,
        )

        if not should_compute:
            residual = state.cached_residual
            if residual.device != hidden.device:
                residual = residual.to(hidden.device)
            output_hidden = hidden + residual.to(hidden.dtype)
            state.consecutive_skips += 1
            state.num_skipped += 1
            encoder = self._io.encoder_hidden_states(module, args, kwargs)
            output = self._io.build_output(state.output_layout, output_hidden, encoder)
            if self.is_tail:
                # Single-block configuration: this hook is also the tail, so
                # the step schedule must advance on skipped steps too.
                self._advance_step(state)
            return output

        state.head_block_input = current_input
        output = self.fn_ref.original_forward(*args, **kwargs)
        if state.output_layout is None:
            encoder = self._io.encoder_hidden_states(module, args, kwargs)
            state.output_layout = self._io.learn_layout(output, hidden, encoder)

        if self.is_tail:
            self._record_full_step(state, self._io.output_hidden(output, state.output_layout))
            self._advance_step(state)
        return output


class EasyCacheBlockHook(_EasyCacheHookBase):
    """Block hook for EasyCache - pass-through on skip, records residual at tail."""

    _HOOK_NAME = "easy_cache_block"

    @torch.compiler.disable
    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        state: EasyCacheState = self.state_manager.get_state()

        if not state.should_compute:
            # The head hook already applied the cached residual of the whole
            # block stack; every other block passes its inputs through.
            hidden = self._io.hidden_states(module, args, kwargs)
            encoder = self._io.encoder_hidden_states(module, args, kwargs)
            output = self._io.build_output(state.output_layout, hidden, encoder)
            if self.is_tail:
                self._advance_step(state)
            return output

        output = self.fn_ref.original_forward(*args, **kwargs)
        if self.is_tail:
            if state.output_layout is None:
                hidden = self._io.hidden_states(module, args, kwargs)
                encoder = self._io.encoder_hidden_states(module, args, kwargs)
                state.output_layout = self._io.learn_layout(output, hidden, encoder)
            self._record_full_step(state, self._io.output_hidden(output, state.output_layout))
            self._advance_step(state)
        return output


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _collect_transformer_blocks(module: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    blocks: list[tuple[str, torch.nn.Module]] = []
    for name, submodule in module.named_children():
        if not isinstance(submodule, torch.nn.ModuleList):
            continue
        for index, block in enumerate(submodule):
            blocks.append((f"{name}.{index}", block))
    return blocks


def apply_easy_cache_hook(module: torch.nn.Module, config: EasyCacheConfig) -> None:
    """Apply EasyCache optimization to a transformer module.

    Every ``nn.ModuleList`` child of ``module`` is treated as (part of) the
    transformer block stack, in registration order. The first block receives
    the head hook, the last block the tail hook and the rest pass-through
    block hooks.

    Args:
        module: Transformer model to optimize (e.g., SanaVideoTransformer3DModel)
        config: EasyCacheConfig specifying caching parameters
    """
    HookRegistry.check_if_exists_or_initialize(module)

    blocks = _collect_transformer_blocks(module)
    if not blocks:
        logger.warning("EasyCache: No transformer blocks found to apply hooks.")
        return

    if config.warmup_steps + config.cooldown_steps >= config.num_inference_steps:
        logger.warning(
            "EasyCache: warmup_steps (%d) + cooldown_steps (%d) >= num_inference_steps (%d); no step can be skipped",
            config.warmup_steps,
            config.cooldown_steps,
            config.num_inference_steps,
        )

    state_manager = StateManager(EasyCacheState, (), {})
    remove_easy_cache_hook(module)

    if len(blocks) == 1:
        name, block = blocks[0]
        logger.info("EasyCache: Applying Head+Tail Hook to single block '%s'", name)
        _register(block, _EASY_CACHE_HEAD_HOOK, EasyCacheHeadHook(state_manager, config, is_tail=True, parent=module))
        return

    head_name, head_block = blocks[0]
    tail_name, tail_block = blocks[-1]
    logger.info("EasyCache: Applying Head Hook to %s and Tail Hook to %s", head_name, tail_name)
    _register(head_block, _EASY_CACHE_HEAD_HOOK, EasyCacheHeadHook(state_manager, config, parent=module))
    for _, block in blocks[1:-1]:
        _register(block, _EASY_CACHE_BLOCK_HOOK, EasyCacheBlockHook(state_manager, config))
    _register(tail_block, _EASY_CACHE_BLOCK_HOOK, EasyCacheBlockHook(state_manager, config, is_tail=True))


def remove_easy_cache_hook(module: torch.nn.Module) -> None:
    """Remove EasyCache hooks from every transformer block of ``module``."""
    for _, block in _collect_transformer_blocks(module):
        registry = getattr(block, "_hook_registry", None)
        if registry is None:
            continue
        for hook_name in _EASY_CACHE_HOOK_NAMES:
            if registry.get_hook(hook_name) is not None:
                registry.remove_hook(hook_name)


def iter_easy_cache_hooks(module: torch.nn.Module):
    """Yield ``(block, hook)`` for every EasyCache hook installed on ``module``."""
    for _, block in _collect_transformer_blocks(module):
        registry = getattr(block, "_hook_registry", None)
        if registry is None:
            continue
        for hook_name in _EASY_CACHE_HOOK_NAMES:
            hook = registry.get_hook(hook_name)
            if hook is not None:
                yield block, hook


def _register(block: torch.nn.Module, name: str, hook: ModelHook) -> None:
    registry = HookRegistry.check_if_exists_or_initialize(block)
    registry.register_hook(name, hook)
