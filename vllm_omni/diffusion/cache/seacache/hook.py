# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import inspect
import math
from collections.abc import Callable
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.cache.seacache.config import SeaCacheConfig
from vllm_omni.diffusion.cache.seacache.sea_filter import (
    apply_sea_filter,
    extrapolate_residual,
    indicator_distance,
)
from vllm_omni.diffusion.cache.seacache.state import DenoiseStepTracker, SeaCacheState
from vllm_omni.diffusion.hooks import HookRegistry, ModelHook, StateManager

logger = init_logger(__name__)


def _is_parameter_sharded(module: torch.nn.Module) -> bool:
    """Detect parameter-sharding runtimes whose collectives cannot be skipped."""
    for submodule in module.modules():
        module_type = type(submodule)
        if callable(getattr(submodule, "_get_fsdp_state", None)):
            return True
        if module_type.__name__ == "FullyShardedDataParallel" and module_type.__module__.startswith(
            "torch.distributed.fsdp"
        ):
            return True
        for parameter in submodule.parameters(recurse=False):
            parameter_type = type(parameter)
            if (
                parameter_type.__name__ == "FlatParameter"
                and parameter_type.__module__.startswith("torch.distributed.fsdp")
            ) or (
                parameter_type.__name__ == "DTensor"
                and parameter_type.__module__.startswith("torch.distributed.tensor")
            ):
                return True
    return False


class SeaCacheRootHook(ModelHook):
    """Drive the SeaCache gate and the Cosmos3 forward guard."""

    _HOOK_NAME = "sea_cache"

    def __init__(
        self,
        config: SeaCacheConfig,
        *,
        num_train_timesteps: int = 1000,
        num_inference_steps_callback: Callable[[], int] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_train_timesteps = int(num_train_timesteps)
        self.num_inference_steps = 0
        self.num_inference_steps_callback = num_inference_steps_callback
        self.state_manager = StateManager(SeaCacheState)
        self.tracker = DenoiseStepTracker()
        self._warned_messages: set[str] = set()
        self.full_count = 0
        self.skip_count = 0
        self._parameter_sharded = False
        self._collective_skip_groups: list[torch.distributed.ProcessGroup] = []

    def initialize_hook(self, module: torch.nn.Module) -> torch.nn.Module:
        if self.num_train_timesteps <= 0:
            raise ValueError(f"num_train_timesteps must be positive, got {self.num_train_timesteps}")
        self._parameter_sharded = _is_parameter_sharded(module)
        seen_groups: set[int] = set()
        for block in getattr(module, "gen_layers", ()):
            registry = getattr(block, "_hook_registry", None)
            dlo_hook = registry.get_hook("distributed_layerwise_offload") if registry is not None else None
            group = getattr(dlo_hook, "dp_group", None)
            if group is not None and int(getattr(dlo_hook, "dp_size", 1)) > 1 and id(group) not in seen_groups:
                seen_groups.add(id(group))
                self._collective_skip_groups.append(group)
        self._clear_forward_control(module)
        return module

    def _warn_once(self, message: str) -> None:
        if message not in self._warned_messages:
            logger.warning(message)
            self._warned_messages.add(message)

    @staticmethod
    def _clear_forward_control(module: torch.nn.Module) -> None:
        module._seacache_skip = False
        module._seacache_record = False
        module._seacache_residual = None
        module._seacache_last_residual = None

    @staticmethod
    def _bind_forward_arguments(
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        signature = inspect.signature(module.__class__.forward)
        return signature.bind_partial(module, *args, **kwargs).arguments

    def _build_indicator(
        self,
        hidden_states: torch.Tensor,
        control_latents: list[torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor | None,
        sigma: float,
    ) -> list[torch.Tensor] | None:
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 5:
            return None

        controls: list[torch.Tensor]
        if control_latents is None:
            controls = []
        elif isinstance(control_latents, torch.Tensor):
            controls = [control_latents]
        elif isinstance(control_latents, (list, tuple)):
            controls = list(control_latents)
        else:
            return None

        # Controls precede the denoised target in packed vision-token order,
        # and each vision item is filtered independently.
        vision_items = [*controls, hidden_states]
        if any(
            not isinstance(item, torch.Tensor)
            or item.ndim != 5
            or item.shape[0] != hidden_states.shape[0]
            or item.shape[1:] != hidden_states.shape[1:]
            for item in vision_items
        ):
            return None

        indicator = []
        for batch_index in range(hidden_states.shape[0]):
            for latent in vision_items:
                thwc = latent[batch_index].movedim(0, -1)
                indicator.append(
                    apply_sea_filter(
                        thwc,
                        sigma=sigma,
                        power_exp=self.config.power_exp,
                    ).detach()
                )
        return indicator or None

    def _resolve_gate(
        self,
        state: SeaCacheState,
        indicator: list[torch.Tensor] | None,
        step: int,
    ) -> bool:
        max_consecutive = bool(
            self.config.max_consecutive_cached and state.consecutive_cached >= self.config.max_consecutive_cached
        )
        forced_compute = (
            step < 1
            or step >= self.num_inference_steps - 1
            or max_consecutive
            or not state.history
            or indicator is None
            or state.previous_indicator is None
        )
        if forced_compute:
            state.accumulated_distance = 0.0
            state.previous_indicator = None if indicator is None else [value.detach() for value in indicator]
            return True

        assert indicator is not None
        assert state.previous_indicator is not None
        distance = indicator_distance(indicator, state.previous_indicator)
        state.previous_indicator = [value.detach() for value in indicator]
        if not math.isfinite(distance):
            state.accumulated_distance = 0.0
            self._warn_once("SeaCache indicator history changed shape, device, or dtype; running full.")
            return True

        state.accumulated_distance += distance
        if state.accumulated_distance < self.config.threshold:
            return False
        state.accumulated_distance = 0.0
        return True

    def _synchronize_compute(self, compute: bool, device: torch.device) -> bool:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return compute
        decision = torch.tensor(int(compute), dtype=torch.int32, device=device)
        for group in self._collective_skip_groups:
            torch.distributed.all_reduce(
                decision,
                op=torch.distributed.ReduceOp.MAX,
                group=group,
            )
        try:
            from vllm_omni.diffusion.distributed.parallel_state import (
                get_sequence_parallel_world_size,
                get_sp_group,
            )

            if get_sequence_parallel_world_size() <= 1:
                return bool(decision.item())
            get_sp_group().all_reduce(decision, op=torch.distributed.ReduceOp.MAX)
            return bool(decision.item())
        except (AssertionError, RuntimeError):
            return bool(decision.item())

    @torch.compiler.disable
    def pre_forward(
        self,
        module: torch.nn.Module,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[tuple, dict]:
        self._clear_forward_control(module)
        if torch.is_grad_enabled():
            self._warn_once("SeaCache is inference-only; autograd-enabled calls run in full.")
            return args, kwargs
        if self.num_inference_steps <= 0 and self.num_inference_steps_callback is not None:
            try:
                callback_steps = int(self.num_inference_steps_callback())
                if callback_steps > 0:
                    self.num_inference_steps = callback_steps
            except (TypeError, ValueError):
                pass
        if self.num_inference_steps <= 0:
            self._warn_once("SeaCache was not refreshed with a positive inference step count; running full.")
            return args, kwargs
        if self._parameter_sharded:
            self._warn_once("SeaCache cannot bypass parameter-sharded Cosmos3 GEN layers; running full.")
            return args, kwargs

        try:
            bound = self._bind_forward_arguments(module, args, kwargs)
            hidden_states = bound.get("hidden_states")
            timestep = bound.get("timestep")
            noisy_frame_mask = bound.get("noisy_frame_mask")
            control_latents = bound.get("control_latents")
            if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 5:
                raise ValueError("hidden_states must be a rank-5 tensor")
            if not isinstance(timestep, torch.Tensor) or timestep.numel() == 0:
                raise ValueError("timestep must be a non-empty tensor")
            if timestep.numel() != 1:
                raise ValueError("batched timesteps are not supported")
            if isinstance(noisy_frame_mask, torch.Tensor) and not bool(torch.any(noisy_frame_mask != 0).item()):
                self._warn_once("SeaCache requires noisy vision; conditioning-only calls run in full.")
                return args, kwargs

            timestep_key = float(timestep.flatten()[0].item())
            if not math.isfinite(timestep_key):
                raise ValueError("timestep must be finite")
            sigma = timestep_key / self.num_train_timesteps
            if not 0.0 <= sigma <= 1.0:
                raise ValueError(f"derived sigma must be in [0, 1], got {sigma}")
        except (TypeError, ValueError, RuntimeError) as error:
            self._warn_once(f"SeaCache metadata is invalid; running full: {error}")
            return args, kwargs

        is_new_step, is_new_sample = self.tracker.advance(timestep_key)
        if is_new_step and (is_new_sample or self.tracker.step >= self.num_inference_steps):
            self.state_manager.reset()
            self.tracker.reset_for_new_sample()

        self.state_manager.set_context(self.tracker.pass_name)
        state: SeaCacheState = self.state_manager.get_state()
        try:
            indicator = self._build_indicator(hidden_states, control_latents, sigma)
        except (TypeError, ValueError, RuntimeError) as error:
            self._warn_once(f"SeaCache could not construct its vision indicator; running full: {error}")
            indicator = None

        local_compute = self._resolve_gate(state, indicator, self.tracker.step)
        should_compute = self._synchronize_compute(local_compute, hidden_states.device)
        if should_compute and not local_compute:
            state.accumulated_distance = 0.0

        if should_compute:
            module._seacache_record = True
            self.full_count += 1
            return args, kwargs

        residual = extrapolate_residual(
            state.history,
            self.tracker.step,
            self.config.residual_order,
        )
        if residual.device != hidden_states.device:
            residual = residual.to(hidden_states.device)
        module._seacache_skip = True
        module._seacache_residual = residual
        state.consecutive_cached += 1
        self.skip_count += 1
        return args, kwargs

    def post_forward(self, module: torch.nn.Module, output: Any) -> Any:
        if getattr(module, "_seacache_record", False):
            state: SeaCacheState = self.state_manager.get_state()
            residual = getattr(module, "_seacache_last_residual", None)
            if isinstance(residual, torch.Tensor):
                state.history.append((self.tracker.step, residual.detach().clone()))
                state.history = state.history[-(self.config.residual_order + 1) :]
                state.consecutive_cached = 0
            else:
                state.history.clear()
                state.accumulated_distance = 0.0
                self._warn_once("SeaCache did not receive a GEN residual; clearing cache history.")
        self._clear_forward_control(module)
        return output

    def reset_state(self, module: torch.nn.Module) -> torch.nn.Module:
        self.state_manager.reset()
        self.tracker.reset()
        self.full_count = 0
        self.skip_count = 0
        self._clear_forward_control(module)
        return module

    def refresh(self, module: torch.nn.Module, num_inference_steps: int) -> None:
        if num_inference_steps < 0:
            raise ValueError(f"num_inference_steps must be non-negative, got {num_inference_steps}")
        self.num_inference_steps = int(num_inference_steps)
        self.reset_state(module)


def apply_sea_cache_hook(
    module: torch.nn.Module,
    config: SeaCacheConfig,
    *,
    num_train_timesteps: int = 1000,
    num_inference_steps_callback: Callable[[], int] | None = None,
) -> SeaCacheRootHook:
    registry = HookRegistry.get_or_create(module)
    hook = SeaCacheRootHook(
        config,
        num_train_timesteps=num_train_timesteps,
        num_inference_steps_callback=num_inference_steps_callback,
    )
    registry.register_hook(SeaCacheRootHook._HOOK_NAME, hook)
    return hook
