# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA graph replay for the AuK codec decode.

The BigVGAN-style decoder is launch-bound: six upsample stages, each with
three residual blocks of three dilations and two alias-free Snake
activations, come to roughly a thousand small kernels per clip while the
GPU sits mostly idle. Capturing one decode per latent length and replaying
it collapses those launches into one.

Graphs are keyed by the latent length rounded up to ``frame_alignment``.
With the default alignment of 1 every distinct length gets its own graph
and the replay is bit-identical to the eager decode; duration-driven
traffic has few distinct lengths, so the LRU cache stays small. A coarser
alignment right-pads the latents with zeros, which the non-causal
``conv_pre`` (three latent frames of lookahead) and the alias-free
upsamplers let bleed into the last frames of the clip; keep it at 1 unless
that tail has been checked for the workload.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
from vllm.logger import init_logger
from vllm.utils.math_utils import round_up

from vllm_omni.diffusion.models.auk.auk_vae import AuKVAE

logger = init_logger(__name__)


@dataclass
class _GraphEntry:
    graph: torch.cuda.CUDAGraph
    static_latents: torch.Tensor
    static_wav: torch.Tensor


class AuKVAEDecodeGraph:
    """Replay ``AuKVAE.decode`` for one clip per call, capturing lazily per latent length."""

    def __init__(
        self,
        vae: AuKVAE,
        *,
        enabled: bool = True,
        frame_alignment: int = 1,
        max_graphs: int = 32,
    ) -> None:
        self.vae = vae
        self.enabled = bool(enabled)
        self.frame_alignment = max(1, int(frame_alignment))
        self.max_graphs = max(1, int(max_graphs))
        self._cache: OrderedDict[int, _GraphEntry] = OrderedDict()
        self._pool_handle: int | None = None

    @torch.no_grad()
    def __call__(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode ``[1, frames, latent_dim]`` latents into a ``[1, frames * hop]`` waveform."""

        if (
            not self.enabled
            or latents.device.type != "cuda"
            or latents.ndim != 3
            or latents.shape[0] != 1
            or torch.cuda.is_current_stream_capturing()
        ):
            return self.vae.decode(latents)

        frames = int(latents.shape[1])
        bucket = round_up(frames, self.frame_alignment)
        entry = self._cache.get(bucket)
        if entry is None:
            entry = self._capture(bucket, latents)
            if len(self._cache) >= self.max_graphs:
                self._cache.popitem(last=False)
            self._cache[bucket] = entry
        else:
            self._cache.move_to_end(bucket)

        if bucket == frames:
            entry.static_latents.copy_(latents)
        else:
            entry.static_latents.zero_()
            entry.static_latents[:, :frames].copy_(latents)
        entry.graph.replay()
        return entry.static_wav[:, : frames * self.vae.hop_size].clone()

    def _capture(self, bucket: int, like: torch.Tensor) -> _GraphEntry:
        static_latents = torch.zeros(1, bucket, self.vae.latent_dim, device=like.device, dtype=torch.float32)
        # An eager run first, so cuDNN picks its algorithms and the Triton
        # activation kernel is compiled outside the capture.
        for _ in range(2):
            self.vae.decode(static_latents)
        torch.accelerator.synchronize(like.device)
        if self._pool_handle is None:
            self._pool_handle = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool_handle):
            static_wav = self.vae.decode(static_latents)
        logger.info("Captured AuK codec decode CUDA graph: latent_frames=%d", bucket)
        return _GraphEntry(graph=graph, static_latents=static_latents, static_wav=static_wav)


__all__ = ["AuKVAEDecodeGraph"]
