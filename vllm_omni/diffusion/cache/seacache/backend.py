# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.cache.base import CacheBackend
from vllm_omni.diffusion.cache.seacache.config import SeaCacheConfig
from vllm_omni.diffusion.cache.seacache.hook import (
    SeaCacheRootHook,
    apply_sea_cache_hook,
)
from vllm_omni.diffusion.data import DiffusionCacheConfig

logger = init_logger(__name__)


def _num_train_timesteps(pipeline: Any) -> int:
    scheduler = getattr(pipeline, "scheduler", None)
    scheduler_config = getattr(scheduler, "config", None)
    value = getattr(scheduler_config, "num_train_timesteps", 1000)
    if isinstance(scheduler_config, dict):
        value = scheduler_config.get("num_train_timesteps", value)
    value = int(value)
    if value <= 0:
        raise ValueError(f"Scheduler num_train_timesteps must be positive, got {value}")
    return value


def enable_cosmos3_seacache(
    pipeline: Any,
    config: DiffusionCacheConfig,
) -> SeaCacheRootHook:
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        raise ValueError("Cosmos3 SeaCache requires pipeline.transformer")
    if not callable(getattr(transformer, "_run_gen_layers", None)):
        raise ValueError("Cosmos3 transformer does not expose the SeaCache GEN-stack boundary")

    sea_config = SeaCacheConfig(
        threshold=config.sea_threshold,
        residual_order=config.sea_residual_order,
        max_consecutive_cached=config.sea_max_consecutive_cached,
        power_exp=config.sea_power_exp,
    )
    hook = apply_sea_cache_hook(
        transformer,
        sea_config,
        num_train_timesteps=_num_train_timesteps(pipeline),
        num_inference_steps_callback=lambda: getattr(
            pipeline,
            "_num_timesteps",
            0,
        ),
    )
    logger.info(
        "SeaCache enabled for Cosmos3 (threshold=%s, residual_order=%d, max_consecutive_cached=%d, power_exp=%s)",
        sea_config.threshold,
        sea_config.residual_order,
        sea_config.max_consecutive_cached,
        sea_config.power_exp,
    )
    return hook


CUSTOM_SEACACHE_ENABLERS = {
    "Cosmos3OmniDiffusersPipeline": enable_cosmos3_seacache,
    "Cosmos3OmniPipeline": enable_cosmos3_seacache,
}


class SeaCacheBackend(CacheBackend):
    """SeaCache backend for vLLM-Omni's native Cosmos3 transformer."""

    def __init__(self, config: DiffusionCacheConfig):
        super().__init__(config)
        self._transformer_id: int | None = None

    def enable(self, pipeline: Any) -> None:
        pipeline_type = pipeline.__class__.__name__
        enabler = CUSTOM_SEACACHE_ENABLERS.get(pipeline_type)
        if enabler is None:
            raise ValueError(f"SeaCache currently supports Cosmos3 pipelines only, got {pipeline_type}")
        hook = enabler(pipeline, self.config)
        self._transformer_id = id(pipeline.transformer)
        self.enabled = True
        pipeline._sea_cache_hook = hook

    def refresh(
        self,
        pipeline: Any,
        num_inference_steps: int,
        verbose: bool = True,
    ) -> None:
        transformer = getattr(pipeline, "transformer", None)
        if transformer is None:
            raise ValueError("Cosmos3 SeaCache requires pipeline.transformer")
        if not self.enabled or self._transformer_id != id(transformer):
            self.enable(pipeline)

        registry = getattr(transformer, "_hook_registry", None)
        hook = registry.get_hook(SeaCacheRootHook._HOOK_NAME) if registry is not None else None
        if not isinstance(hook, SeaCacheRootHook):
            raise RuntimeError("SeaCache hook is not installed on the Cosmos3 transformer")
        hook.refresh(transformer, num_inference_steps)
        pipeline._sea_cache_hook = hook
        if verbose:
            logger.debug(
                "SeaCache state refreshed (num_inference_steps=%d)",
                num_inference_steps,
            )
