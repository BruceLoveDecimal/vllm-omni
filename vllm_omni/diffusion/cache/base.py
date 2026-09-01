# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Base cache backend interface for diffusion models.

This module defines the abstract base class that all cache backends must implement.
Cache backends provide a unified interface for applying different caching strategies
to transformer models.

Main cache backend implementations:
1. CacheDiTBackend: Implements cache-dit acceleration (DBCache, SCM, TaylorSeer) using
   the cache-dit library. Inherits from CacheBackend. Used via cache_backend="cache_dit".
2. TeaCacheBackend: Hook-based backend for TeaCache acceleration. Inherits from
   CacheBackend. Used via cache_backend="tea_cache".
3. StepCacheBackend: Velocity cosine step-skipping for DreamZero. Inherits from
   CacheBackend. Used via cache_backend="step_cache".

All backends implement the same interface:
- enable(pipeline): Enable cache on the pipeline
- refresh(pipeline, num_inference_steps, verbose): Refresh cache state
- is_enabled(): Check if cache is enabled
"""

from abc import ABC, abstractmethod
from operator import attrgetter
from typing import Any

import torch.nn as nn

from vllm_omni.diffusion.data import DiffusionCacheConfig


def dit_module_names(pipeline: Any) -> tuple[str, ...]:
    """Return the pipeline DiT attributes that are present at runtime.

    A pipeline that serves several tasks from separate DiT partitions (MiniMax
    H3) or runs a two-expert schedule (Wan 2.2) declares every denoising module
    in ``_dit_modules``. Backends must install and refresh their state on all of
    them, not only on ``transformer``.
    """
    names = getattr(pipeline, "_dit_modules", None)
    if not isinstance(names, (list, tuple)):
        names = ("transformer",)

    resolved_names = []
    for name in names:
        if not isinstance(name, str):
            continue
        try:
            module = attrgetter(name)(pipeline)
        except AttributeError:
            continue
        if module is not None:
            resolved_names.append(name)
    return tuple(resolved_names)


def resolve_denoise_steps(pipeline: Any, num_inference_steps: int) -> int:
    """Return how many transformer forwards ``num_inference_steps`` will run.

    Cache backends schedule warmup and cooldown against the number of times the
    block stack is invoked, but a request's ``num_inference_steps`` does not
    always equal that count: MiniMax H3 speaks in sigma points and denoises the
    intervals between them, so a 50-point request runs 49 forwards. A pipeline
    whose two counts differ reports the real one through ``cache_denoise_steps``;
    every other pipeline is left unchanged.
    """
    resolve = getattr(pipeline, "cache_denoise_steps", None)
    if not callable(resolve):
        return num_inference_steps
    resolved = int(resolve(num_inference_steps))
    if resolved <= 0:
        raise ValueError(
            f"{type(pipeline).__name__}.cache_denoise_steps returned {resolved} for "
            f"num_inference_steps={num_inference_steps}; it must be positive"
        )
    return resolved


class CacheBackend(ABC):
    """
    Abstract base class for cache backends.

    All cache backend implementations (CacheDiTBackend, TeaCacheBackend, etc.) inherit
    from this base class and implement the enable() and refresh() methods to manage
    cache lifecycle.

    Cache backends apply caching strategies to transformer models to accelerate
    inference. Different backends use different underlying mechanisms (e.g., cache-dit
    library for CacheDiTBackend, hooks for TeaCacheBackend), but all share the same
    unified interface.

    Attributes:
        config: DiffusionCacheConfig instance containing cache-specific configuration parameters
        enabled: Boolean flag indicating whether cache is enabled (set to True after enable() is called)
    """

    def __init__(self, config: DiffusionCacheConfig):
        """
        Initialize cache backend with configuration.

        Args:
            config: DiffusionCacheConfig instance with cache-specific parameters
        """
        self.config = config
        self.enabled = False

    @abstractmethod
    def enable(self, pipeline: Any) -> None:
        """
        Enable cache on the pipeline.

        This method applies the caching strategy to the transformer(s) in the pipeline.
        The specific implementation depends on the backend (e.g., hooks for TeaCacheBackend,
        cache-dit library for CacheDiTBackend). Called once during pipeline initialization.

        Args:
            pipeline: Diffusion pipeline instance. The backend can extract:
                     - transformer: via pipeline.transformer
                     - model_type: via pipeline.__class__.__name__
        """
        raise NotImplementedError("Subclasses must implement enable()")

    @abstractmethod
    def refresh(self, pipeline: Any, num_inference_steps: int, verbose: bool = True) -> None:
        """
        Refresh cache state for new generation.

        This method should clear any cached values and reset counters/accumulators.
        Called at the start of each generation to ensure clean state.

        Args:
            pipeline: Diffusion pipeline instance. The backend can extract:
                     - transformer: via pipeline.transformer
            num_inference_steps: Number of inference steps for the current generation.
                                May be used for cache context updates.
            verbose: Whether to log refresh operations (default: True)
        """
        raise NotImplementedError("Subclasses must implement refresh()")

    def is_enabled(self) -> bool:
        """
        Check if cache is enabled on this backend.

        Returns:
            True if cache is enabled, False otherwise.
        """
        return self.enabled

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"


class CachedTransformer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.do_true_cfg = False

    def __init_subclass__(cls, enable_separate_cfg: bool = True, **kwargs):
        cls.enable_separate_cfg = enable_separate_cfg
        super().__init_subclass__(**kwargs)
