# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA graph replay for the AuK codec decode.

The decoder launches about a thousand small kernels per clip, so it is
bound by launch overhead rather than by the GPU. This wrapper replays the
whole decode as one CUDA graph. It keeps two kinds of graphs:

Compiled bucket graphs are built at startup by warmup(): decode is passed
through torch.compile so Inductor fuses the elementwise chains, then one
graph is captured per bucket in compile_shapes. Shorter clips are
right-padded with zeros to their bucket. The padding leaks slightly into
the last frames through the non-causal conv_pre, and the fused kernels
differ from eager in rounding order, so these graphs are close to but not
bit-identical with the eager decode.

Plain graphs are captured on demand, one per exact latent length, for clips
longer than the largest bucket. They replay the same kernels as eager and
are bit-identical with it.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from vllm.logger import init_logger
from vllm.utils.math_utils import round_up

from vllm_omni.diffusion.models.auk.auk_vae import AuKVAE, SnakeBeta
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

# Latent-frame buckets for the compiled graphs, at 50 Hz: 2.56, 5.12, 10.24
# and 15.36 s. Each bucket costs one Inductor compilation (tens of seconds)
# at startup and one private CUDA graph pool. Deployments override them with
# ``model_config.auk_vae_compile_shapes`` on the diffusion stage.
DEFAULT_COMPILE_SHAPES: tuple[int, ...] = (128, 256, 512, 768)


@dataclass
class _GraphEntry:
    graph: torch.cuda.CUDAGraph
    static_latents: torch.Tensor
    static_wav: torch.Tensor


class AuKVAEDecodeGraph:
    """Replay AuKVAE.decode for one clip per call."""

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
        # Plain graphs keyed by latent length, least recently used first.
        self._cache: OrderedDict[int, _GraphEntry] = OrderedDict()
        # Compiled graphs keyed by bucket, filled by warmup().
        self._compiled: dict[int, _GraphEntry] = {}
        self._compiled_decode: Callable[[torch.Tensor], torch.Tensor] | None = None
        # Which path served the last call: "compiled", "graph" or "eager".
        self.last_mode: str | None = None

    @torch.no_grad()
    def __call__(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode [1, frames, latent_dim] latents into a [1, frames * hop] waveform."""

        # Graphs are captured for one clip at a time on the accelerator.
        # Anything else, or a call that is itself being captured into an
        # outer graph, runs the plain eager decode. The capture query is
        # asked last because it needs a CUDA runtime.
        single_clip = latents.ndim == 3 and latents.shape[0] == 1
        if not self.enabled or not single_clip or not latents.is_cuda or torch.cuda.is_current_stream_capturing():
            self.last_mode = "eager"
            return self.vae.decode(latents)

        frames = int(latents.shape[1])
        bucket = self.compiled_bucket(frames)
        if bucket is not None:
            entry = self._compiled[bucket]
            self.last_mode = "compiled"
        else:
            bucket = round_up(frames, self.frame_alignment)
            entry = self._cache.get(bucket)
            if entry is None:
                entry = self._capture(bucket, latents.device, self.vae.decode, warm_iters=2)
                if len(self._cache) >= self.max_graphs:
                    self._cache.popitem(last=False)
                self._cache[bucket] = entry
            else:
                self._cache.move_to_end(bucket)
            self.last_mode = "graph"

        if bucket == frames:
            entry.static_latents.copy_(latents)
        else:
            entry.static_latents.zero_()
            entry.static_latents[:, :frames].copy_(latents)
        entry.graph.replay()
        return entry.static_wav[:, : frames * self.vae.hop_size].clone()

    def compiled_bucket(self, frames: int) -> int | None:
        """The smallest compiled bucket that holds frames latents, or None past the largest."""
        for size in self.compile_shapes:
            if frames <= size and size in self._compiled:
                return size
        return None

    def warmup(self, device: torch.device | str) -> None:
        """Compile the decode and capture every bucket in compile_shapes.

        Meant for service startup, since each bucket costs one Inductor
        compilation. A failure only logs a warning and the bucket is served
        by a plain graph instead.
        """
        device = torch.device(device)
        if not self.enabled or not self.compile_shapes or self._compiled:
            return
        on_accelerator = current_omni_platform.is_cuda_alike() and device.type == current_omni_platform.device_type
        if not on_accelerator or torch.cuda.is_current_stream_capturing():
            return

        # The traced graph must read exp(alpha) from a buffer, not recompute it.
        for module in self.vae.modules():
            if isinstance(module, SnakeBeta):
                module.precompute_exp_cache()

        try:
            self._compiled_decode = torch.compile(self.vae.decode, mode="default", fullgraph=False, dynamic=False)
        except Exception:
            logger.warning("torch.compile of the AuK codec decode failed; using plain CUDA graphs", exc_info=True)
            self._compiled_decode = None
            return

        # Inductor fuses the plain Snake formula with the ops around it.
        for size in self.compile_shapes:
            try:
                self._compiled[size] = self._capture(size, device, self._compiled_decode, warm_iters=5)
                logger.info("Compiled and captured AuK codec decode: latent_frames=%d", size)
            except Exception:
                logger.warning(
                    "Compiled AuK codec decode failed for latent_frames=%d; falling back to plain CUDA graphs",
                    size,
                    exc_info=True,
                )
        logger.info(
            "AuK codec decode compile warmup done: %d/%d buckets", len(self._compiled), len(self.compile_shapes)
        )

    def _capture(
        self,
        bucket: int,
        device: torch.device,
        decode: Callable[[torch.Tensor], torch.Tensor],
        *,
        warm_iters: int,
    ) -> _GraphEntry:
        """Capture one graph of decode on zero latents of bucket frames.

        The warm iterations let cuDNN pick its algorithms and, for the
        compiled decode, Inductor finish tracing and autotuning before
        anything is recorded.
        """
        static_latents = torch.zeros(1, bucket, self.vae.latent_dim, device=device, dtype=torch.float32)
        with torch.inference_mode():
            self.vae.decode(static_latents)
            for _ in range(warm_iters):
                decode(static_latents)
            torch.accelerator.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=current_omni_platform.get_global_graph_pool()):
                static_wav = decode(static_latents)
        logger.info("Captured AuK codec decode CUDA graph: latent_frames=%d", bucket)
        return _GraphEntry(graph=graph, static_latents=static_latents, static_wav=static_wav)


__all__ = ["DEFAULT_COMPILE_SHAPES", "AuKVAEDecodeGraph"]
