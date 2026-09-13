# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-shape CUDA graph for one AuK DiT denoise step."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm_omni.diffusion.models.auk.auk_transformer import AuKTransformer

logger = init_logger(__name__)


@dataclass
class _GraphEntry:
    graph: torch.cuda.CUDAGraph
    static_x: torch.Tensor
    static_text: torch.Tensor
    static_c_mask: torch.Tensor
    static_ref: torch.Tensor
    static_ref_mask: torch.Tensor
    static_timestep: torch.Tensor
    static_cfg: torch.Tensor | None
    static_out: torch.Tensor


class AuKCUDAGraphWrapper:
    """Replay one DiT denoise step and leave Euler scheduling to the caller.

    The graph is keyed by the target, text and reference sequence lengths plus
    whether the CFG branch is enabled. Timestep and the CFG strength are
    mutable scalar buffers, so all Euler steps and CFG values within one path
    reuse one graph.
    """

    def __init__(self, dit: AuKTransformer, *, enabled: bool = True, max_graphs: int = 32) -> None:
        self.dit = dit
        self.enabled = bool(enabled)
        self.max_graphs = max(1, int(max_graphs))
        self._cache: OrderedDict[tuple, _GraphEntry] = OrderedDict()
        self._failed_keys: set[tuple] = set()

    @staticmethod
    def _key(
        x: torch.Tensor,
        text: torch.Tensor,
        ref: torch.Tensor,
        uses_cfg: bool,
    ) -> tuple[int, int, int, bool]:
        return (x.shape[1], text.shape[1], ref.shape[1], uses_cfg)

    @torch.no_grad()
    def __call__(
        self,
        *,
        x: torch.Tensor,
        text: torch.Tensor,
        c_mask: torch.Tensor,
        ref: torch.Tensor,
        ref_mask: torch.Tensor,
        timestep: torch.Tensor,
        cfg_strength: float,
    ) -> torch.Tensor:
        inputs = (x, text, c_mask, ref, ref_mask, timestep)
        uses_cfg = cfg_strength >= 1e-5
        cfg_strength = torch.tensor(cfg_strength, device=x.device, dtype=torch.float32)
        if not self.enabled or x.device.type != "cuda" or torch.cuda.is_current_stream_capturing():
            return self._run_cfg(*inputs, cfg_strength=cfg_strength) if uses_cfg else self._run(*inputs)

        key = self._key(x, text, ref, uses_cfg)
        entry = self._cache.get(key)
        if entry is None:
            if key in self._failed_keys:
                return self._run_cfg(*inputs, cfg_strength=cfg_strength) if uses_cfg else self._run(*inputs)
            entry = self._capture(*inputs, cfg_strength=cfg_strength, uses_cfg=uses_cfg)
            if entry is None:
                self._failed_keys.add(key)
                return self._run_cfg(*inputs, cfg_strength=cfg_strength) if uses_cfg else self._run(*inputs)
            if len(self._cache) >= self.max_graphs:
                self._cache.popitem(last=False)
            self._cache[key] = entry
        else:
            self._cache.move_to_end(key)

        entry.static_x.copy_(x)
        entry.static_text.copy_(text)
        entry.static_c_mask.copy_(c_mask)
        entry.static_ref.copy_(ref)
        entry.static_ref_mask.copy_(ref_mask)
        entry.static_timestep.copy_(timestep)
        if entry.static_cfg is not None:
            entry.static_cfg.copy_(cfg_strength)
        entry.graph.replay()
        return entry.static_out.clone()

    def _run(
        self,
        x: torch.Tensor,
        text: torch.Tensor,
        c_mask: torch.Tensor,
        ref: torch.Tensor,
        ref_mask: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        return self.dit(x, text, timestep, c_mask=c_mask, ref=ref, ref_mask=ref_mask)

    def _run_cfg(
        self,
        x: torch.Tensor,
        text: torch.Tensor,
        c_mask: torch.Tensor,
        ref: torch.Tensor,
        ref_mask: torch.Tensor,
        timestep: torch.Tensor,
        *,
        cfg_strength: torch.Tensor,
    ) -> torch.Tensor:
        pred = self.dit(
            x,
            text,
            timestep,
            c_mask=c_mask,
            ref=ref,
            ref_mask=ref_mask,
            cfg_infer=True,
            cache=True,
        )
        conditional, unconditional = pred.chunk(2, dim=0)
        return conditional + (conditional - unconditional) * cfg_strength

    def _capture(
        self,
        *inputs: torch.Tensor,
        cfg_strength: torch.Tensor,
        uses_cfg: bool,
    ) -> _GraphEntry | None:
        static_inputs = tuple(value.clone() for value in inputs)
        static_cfg = cfg_strength.clone() if uses_cfg else None
        try:
            for _ in range(3):
                if uses_cfg:
                    assert static_cfg is not None
                    self._run_cfg(*static_inputs, cfg_strength=static_cfg)
                else:
                    self._run(*static_inputs)
            # CFG warm-up populates AuKTransformer's Python-side projected-text
            # cache. Clear it before capture so the graph includes
            # project_text(static_text), rather than closing over the first
            # request's projection and ignoring later static_text updates.
            self.dit.clear_cache()
            if static_cfg is not None:
                static_cfg.copy_(cfg_strength)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
                # Capture the actual CFG path. The scalar value remains a
                # mutable buffer, so cfg=2 and cfg=3 share this graph.
                if uses_cfg:
                    assert static_cfg is not None
                    static_out = self._run_cfg(*static_inputs, cfg_strength=static_cfg)
                else:
                    static_out = self._run(*static_inputs)
        except Exception:
            logger.warning(
                "AuK DiT single-step CUDA graph capture failed for one shape; using eager denoise steps for it.",
                exc_info=True,
            )
            return None
        finally:
            self.dit.clear_cache()

        logger.info(
            "Captured AuK DiT single-step CUDA graph: target_frames=%d text_tokens=%d ref_frames=%d cfg=%s",
            inputs[0].shape[1],
            inputs[1].shape[1],
            inputs[3].shape[1],
            cfg_strength,
        )
        return _GraphEntry(
            graph=graph,
            static_x=static_inputs[0],
            static_text=static_inputs[1],
            static_c_mask=static_inputs[2],
            static_ref=static_inputs[3],
            static_ref_mask=static_inputs[4],
            static_timestep=static_inputs[5],
            static_cfg=static_cfg,
            static_out=static_out,
        )


__all__ = ["AuKCUDAGraphWrapper"]
