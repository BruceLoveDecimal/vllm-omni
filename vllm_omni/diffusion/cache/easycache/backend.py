# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
EasyCache backend implementation.

This module provides the EasyCache backend that implements the CacheBackend
interface using the hooks-based EasyCache system.
"""

from __future__ import annotations

from operator import attrgetter
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.cache.base import CacheBackend, dit_module_names
from vllm_omni.diffusion.cache.easycache.config import EasyCacheConfig
from vllm_omni.diffusion.cache.easycache.hook import (
    apply_easy_cache_hook,
    iter_easy_cache_hooks,
)
from vllm_omni.diffusion.data import DiffusionCacheConfig

logger = init_logger(__name__)

_DEFAULT_NUM_INFERENCE_STEPS = 50


class EasyCacheBackend(CacheBackend):
    """
    EasyCache implementation using hooks.

    EasyCache is a runtime-adaptive caching technique that skips the whole
    transformer block stack when the predicted relative output change,
    accumulated since the last full computation, stays below a threshold.
    It requires no model-specific calibration.

    The backend applies EasyCache hooks to the transformer which intercept the
    block forward passes and implement the caching logic transparently.

    Example:
        >>> from vllm_omni.diffusion.data import DiffusionCacheConfig
        >>> cache_config = DiffusionCacheConfig(
        ...     easy_threshold=0.1,
        ...     easy_warmup_steps=5,
        ...     easy_cooldown_steps=1,
        ... )
        >>> backend = EasyCacheBackend(cache_config)
        >>> backend.enable(pipeline)
        >>> backend.refresh(pipeline, num_inference_steps=50)
    """

    def __init__(self, config: DiffusionCacheConfig):
        super().__init__(config)
        self._registered = False
        self._easycache_config: EasyCacheConfig | None = None
        self._transformer_ids: tuple[int, ...] = ()

    def _build_config(self, transformer_type: str, num_inference_steps: int | None = None) -> EasyCacheConfig:
        steps = num_inference_steps or self.config.num_inference_steps or _DEFAULT_NUM_INFERENCE_STEPS
        return EasyCacheConfig(
            threshold=self.config.easy_threshold,
            warmup_steps=self.config.easy_warmup_steps,
            cooldown_steps=self.config.easy_cooldown_steps,
            max_skip_steps=self.config.easy_max_skip_steps,
            num_inference_steps=int(steps),
            transformer_type=transformer_type,
        )

    def _transformers(self, pipeline: Any) -> list[tuple[str, torch.nn.Module]]:
        """Return every DiT the pipeline denoises with, as ``(name, module)``.

        A pipeline that serves several tasks from separate DiT partitions
        (MiniMax H3 routes ``ref2va`` to ``transformers_ref``) must have hooks
        on all of them; hooking only ``transformer`` would silently leave the
        other partition uncached.
        """
        return [(name, attrgetter(name)(pipeline)) for name in dit_module_names(pipeline)]

    def enable(self, pipeline: Any, num_inference_steps: int | None = None) -> None:
        """Enable EasyCache on every pipeline DiT using hooks.

        Args:
            pipeline: Diffusion pipeline instance. Its DiT attributes are taken
                from ``_dit_modules``, defaulting to ``transformer``.
            num_inference_steps: Optional step count for the first generation.
                ``refresh()`` updates it per request.
        """
        transformers = self._transformers(pipeline)
        if not transformers:
            logger.warning(
                "EasyCache: pipeline %s exposes no DiT module; cache acceleration stays off",
                type(pipeline).__name__,
            )
            return

        transformer_type = transformers[0][1].__class__.__name__
        self._easycache_config = self._build_config(transformer_type, num_inference_steps)
        self._transformer_ids = tuple(id(transformer) for _, transformer in transformers)

        for _, transformer in transformers:
            apply_easy_cache_hook(transformer, self._easycache_config)
        logger.info(
            "EasyCache enabled for %s on %s (threshold=%s, warmup_steps=%d, cooldown_steps=%d, max_skip_steps=%d)",
            transformer_type,
            ", ".join(name for name, _ in transformers),
            self._easycache_config.threshold,
            self._easycache_config.warmup_steps,
            self._easycache_config.cooldown_steps,
            self._easycache_config.max_skip_steps,
        )

        self._registered = True
        self.enabled = True

    def refresh(self, pipeline: Any, num_inference_steps: int, verbose: bool = True) -> None:
        """Refresh EasyCache state for a new generation.

        Updates the step schedule for the incoming request and clears cached
        residuals, estimator and counters on every DiT. Re-registers the hooks
        when the set of transformer instances changed.

        Args:
            pipeline: Diffusion pipeline instance.
            num_inference_steps: Number of inference steps for the current generation.
            verbose: Whether to log refresh operations (default: True)
        """
        transformers = self._transformers(pipeline)
        current_ids = tuple(id(transformer) for _, transformer in transformers)

        if self._registered and current_ids != self._transformer_ids:
            logger.warning(
                "Transformer set was replaced (ids changed from %s to %s), re-registering hooks",
                self._transformer_ids,
                current_ids,
            )
            self._registered = False

        if not self._registered:
            self.enable(pipeline, num_inference_steps)
            return

        assert self._easycache_config is not None
        if num_inference_steps and num_inference_steps > 0:
            # Hooks share this config object, so updating it in place moves
            # the cooldown window and end-of-run reset for every block.
            self._easycache_config.num_inference_steps = int(num_inference_steps)

        for name, transformer in transformers:
            hooks = list(iter_easy_cache_hooks(transformer))
            if not hooks:
                logger.warning("No EasyCache hooks found on %s blocks, re-registering", name)
                apply_easy_cache_hook(transformer, self._easycache_config)
                continue
            for block, hook in hooks:
                hook.reset_state(block)

        if verbose:
            logger.debug(
                "EasyCache state refreshed (num_inference_steps=%d)",
                self._easycache_config.num_inference_steps,
            )

    def is_enabled(self) -> bool:
        """Check if EasyCache is enabled.

        Returns:
            True if enabled, False otherwise.
        """
        return self.enabled
