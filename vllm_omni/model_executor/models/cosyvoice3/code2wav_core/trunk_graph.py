# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA-graph replay for the causal HiFT conv trunk.

The trunk is ~50 small convolutions, so at streaming window sizes (tens of mel
frames) its cost is dominated by kernel launches rather than compute: decoding
one 30-frame chunk was measured at ~2x the cost of decoding a whole 1400-frame
utterance in one call. Bounded-window streaming makes the trunk's input shapes
a small fixed set — precisely what CUDA graphs need (cf. the deploy yaml's
``enforce_eager`` note about *dynamic* conv shapes, and MiniCPM-o's
``HiFTGraphWrapper``, which plays the same trick on the CosyVoice2 HiFT).

Only the trunk (``conv_pre`` + upsample/resblock stack + ``conv_post``) is
captured. The f0 predictor runs on CPU in float64 and the per-row stft/istft
stay eager, so the graph body is pure convs and elementwise ops.

Graphs are captured lazily per ``(batch, frames)`` shape, up to ``max_graphs``;
uncaptured shapes and capture failures fall back to eager per key. Graphs are
never evicted — replacing a live graph invalidates shared library workspaces
(see the CFM cudagraph generation-flush fix), and the steady-state shape set is
small enough to never need it.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


class HiFTTrunkGraph:
    """Lazily captured CUDA graphs of ``fn(mel, s_stft) -> (magnitude, phase)``.

    The returned tensors are the graph's static output buffers: valid until the
    next ``run`` with the same shape key, so consume (or clone) them before
    invoking the wrapper again. ``inference_batch`` slices them straight into
    the per-row istft, which satisfies that.
    """

    def __init__(
        self,
        fn: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        max_graphs: int = 24,
    ):
        self._fn = fn
        self._max_graphs = max_graphs
        # (batch, mel_frames) -> (graph, static_mel, static_sstft, static_mag, static_phase)
        self._graphs: dict[tuple[int, int], tuple] = {}
        self._failed_keys: set[tuple[int, int]] = set()

    def run(self, mel: torch.Tensor, s_stft: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key = (int(mel.shape[0]), int(mel.shape[-1]))
        if not mel.is_cuda or key in self._failed_keys:
            return self._fn(mel, s_stft)

        record = self._graphs.get(key)
        if record is None:
            if len(self._graphs) >= self._max_graphs:
                return self._fn(mel, s_stft)
            try:
                record = self._capture(key, mel, s_stft)
            except Exception as exc:
                logger.warning("HiFT trunk graph capture failed for shape %s; staying eager there (%s)", key, exc)
                self._failed_keys.add(key)
                return self._fn(mel, s_stft)
            self._graphs[key] = record

        graph, static_mel, static_sstft, static_mag, static_phase = record
        static_mel.copy_(mel)
        static_sstft.copy_(s_stft)
        graph.replay()
        return static_mag, static_phase

    def _capture(self, key: tuple[int, int], mel: torch.Tensor, s_stft: torch.Tensor) -> tuple:
        static_mel = mel.clone()
        static_sstft = s_stft.clone()

        # Warm up on a side stream so lazy kernel/workspace initialisation
        # happens outside the capture (standard CUDA-graph protocol).
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self._fn(static_mel, static_sstft)
        torch.cuda.current_stream().wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_mag, static_phase = self._fn(static_mel, static_sstft)

        logger.info("HiFT trunk graph captured for shape %s (%d graphs live)", key, len(self._graphs) + 1)
        return graph, static_mel, static_sstft, static_mag, static_phase
