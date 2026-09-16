# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA graph replay for the AuK codec decode, with a torch.compile tier on top.

The BigVGAN-style decoder is launch-bound: six upsample stages, each with
three residual blocks of three dilations and two alias-free Snake
activations, come to roughly a thousand small kernels per clip while the
GPU sits mostly idle. Capturing one decode per latent length and replaying
it collapses those launches into one.

Two tiers of graphs, dispatched compiled > plain > eager, as the IndexTTS
BigVGAN wrapper does:

* **Compiled buckets** (``compile_shapes``). At startup :meth:`warmup`
  compiles ``AuKVAE.decode`` with ``torch.compile(mode="default")`` so
  Inductor fuses the pad / FIR / Snake elementwise chains, then captures one
  graph per bucket. Latents shorter than a bucket are right-padded with
  zeros, which the non-causal ``conv_pre`` (three latent frames of
  lookahead) and the alias-free upsamplers let bleed into the last frames
  of the clip, so a padded replay is close to but not identical with the
  eager decode. Inductor sees the bare Snake formula (the Triton kernel
  would only split the graph), so the compiled decode differs from eager
  by fusion order alone.
* **Plain graphs**, captured lazily per latent length (LRU) for anything
  longer than the largest bucket, or for everything when no bucket was
  compiled. With the default ``frame_alignment`` of 1 they replay the very
  same kernels as eager and are bit-identical to it.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from vllm.logger import init_logger
from vllm.utils.math_utils import round_up

from vllm_omni.diffusion.models.auk.auk_vae import AuKVAE, SnakeBeta

logger = init_logger(__name__)

# Latent-frame buckets for the compiled tier, at 50 Hz: 2.56 s, 5.12 s, 10.24 s.
# Each bucket costs one Inductor compilation (tens of seconds) at startup.
DEFAULT_COMPILE_SHAPES: tuple[int, ...] = (128, 256, 512)


@dataclass
class _GraphEntry:
    graph: torch.cuda.CUDAGraph
    static_latents: torch.Tensor
    static_wav: torch.Tensor


class AuKVAEDecodeGraph:
    """Replay ``AuKVAE.decode`` for one clip per call: compiled buckets first, lazy per-length graphs otherwise."""

    def __init__(
        self,
        vae: AuKVAE,
        *,
        enabled: bool = True,
        frame_alignment: int = 1,
        max_graphs: int = 32,
        compile_shapes: Sequence[int] = DEFAULT_COMPILE_SHAPES,
    ) -> None:
        self.vae = vae
        self.enabled = bool(enabled)
        self.frame_alignment = max(1, int(frame_alignment))
        self.max_graphs = max(1, int(max_graphs))
        self.compile_shapes = sorted({int(size) for size in compile_shapes if int(size) > 0})
        self._cache: OrderedDict[int, _GraphEntry] = OrderedDict()
        self._compiled: dict[int, _GraphEntry] = {}
        self._compiled_decode: Callable[[torch.Tensor], torch.Tensor] | None = None
        self._pool_handle: int | None = None
        # Which tier served the last call: "compiled", "graph" or "eager".
        self.last_mode: str | None = None

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
            self.last_mode = "eager"
            return self.vae.decode(latents)

        frames = int(latents.shape[1])
        bucket = self.compiled_bucket(frames)
        if bucket is not None:
            self.last_mode = "compiled"
            return self._replay(self._compiled[bucket], latents, frames, bucket)

        bucket = round_up(frames, self.frame_alignment)
        entry = self._cache.get(bucket)
        if entry is None:
            entry = self._capture(bucket, latents)
            if len(self._cache) >= self.max_graphs:
                self._cache.popitem(last=False)
            self._cache[bucket] = entry
        else:
            self._cache.move_to_end(bucket)
        self.last_mode = "graph"
        return self._replay(entry, latents, frames, bucket)

    def compiled_bucket(self, frames: int) -> int | None:
        """The smallest compiled bucket that holds ``frames`` latents, or None past the largest."""
        for size in self.compile_shapes:
            if frames <= size and size in self._compiled:
                return size
        return None

    def _replay(self, entry: _GraphEntry, latents: torch.Tensor, frames: int, bucket: int) -> torch.Tensor:
        if bucket == frames:
            entry.static_latents.copy_(latents)
        else:
            entry.static_latents.zero_()
            entry.static_latents[:, :frames].copy_(latents)
        entry.graph.replay()
        return entry.static_wav[:, : frames * self.vae.hop_size].clone()

    def _graph_pool(self) -> int:
        if self._pool_handle is None:
            self._pool_handle = torch.cuda.graph_pool_handle()
        return self._pool_handle

    def _capture(self, bucket: int, like: torch.Tensor) -> _GraphEntry:
        static_latents = torch.zeros(1, bucket, self.vae.latent_dim, device=like.device, dtype=torch.float32)
        # An eager run first, so cuDNN picks its algorithms and the Triton
        # activation kernel is compiled outside the capture.
        for _ in range(2):
            self.vae.decode(static_latents)
        torch.accelerator.synchronize(like.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._graph_pool()):
            static_wav = self.vae.decode(static_latents)
        logger.info("Captured AuK codec decode CUDA graph: latent_frames=%d", bucket)
        return _GraphEntry(graph=graph, static_latents=static_latents, static_wav=static_wav)

    # ------------------------------------------------------------------
    # Compiled tier
    # ------------------------------------------------------------------

    def warmup(self, device: torch.device | str) -> None:
        """Compile the decode and capture every bucket in ``compile_shapes``.

        Meant for service startup: each bucket costs one Inductor compilation.
        Failures only log a warning; the affected bucket then falls through to
        the plain per-length graph tier.
        """
        device = torch.device(device)
        if not self.enabled or not self.compile_shapes or device.type != "cuda" or self._compiled:
            return
        if torch.cuda.is_current_stream_capturing():
            return

        snakes = [module for module in self.vae.modules() if isinstance(module, SnakeBeta)]
        # The compiled graph must see exp(alpha) as a fixed buffer, not recompute it.
        for module in snakes:
            module.precompute_exp_cache()
        was_fused = any(module.fused for module in snakes)

        try:
            self._compiled_decode = torch.compile(self.vae.decode, mode="default", fullgraph=False, dynamic=False)
        except Exception:
            logger.warning("torch.compile of the AuK codec decode failed; using plain CUDA graphs", exc_info=True)
            self._compiled_decode = None
            return

        # Inductor fuses the bare Snake formula with its neighbours; the
        # Triton kernel would only split the graph around itself.
        self.vae.set_decode_fast_paths(fused_snake=False)
        try:
            for size in self.compile_shapes:
                try:
                    self._compiled[size] = self._capture_compiled(size, device)
                    logger.info("Compiled and captured AuK codec decode: latent_frames=%d", size)
                except Exception:
                    logger.warning(
                        "Compiled AuK codec decode failed for latent_frames=%d; falling back to plain CUDA graphs",
                        size,
                        exc_info=True,
                    )
        finally:
            self.vae.set_decode_fast_paths(fused_snake=was_fused)
        logger.info(
            "AuK codec decode compile warmup done: %d/%d buckets", len(self._compiled), len(self.compile_shapes)
        )

    def _capture_compiled(self, bucket: int, device: torch.device) -> _GraphEntry:
        assert self._compiled_decode is not None
        static_latents = torch.zeros(1, bucket, self.vae.latent_dim, device=device, dtype=torch.float32)
        with torch.inference_mode():
            # Eager once so cuDNN settles its algorithms and the filter caches
            # exist before tracing; then let Inductor trace and autotune.
            self.vae.decode(static_latents)
            for _ in range(5):
                self._compiled_decode(static_latents)
            torch.accelerator.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._graph_pool()):
                static_wav = self._compiled_decode(static_latents)
        return _GraphEntry(graph=graph, static_latents=static_latents, static_wav=static_wav)


__all__ = ["DEFAULT_COMPILE_SHAPES", "AuKVAEDecodeGraph"]
