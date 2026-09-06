# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-scoped chunk-causal cache for the SANA-WM streaming transformer.

One request is one streaming session, so every piece of causal state the
distilled Stage-1 model carries across chunks lives here and is owned by the
request's ``StepRequestState.extra``. It mirrors the 10-slot per-block cache
of the NVlabs ``SelfForcingFlowEulerCamCtrl`` sampler, with the slot table
replaced by named fields:

* GDN blocks keep the forward delta-rule states of the main and camera
  branches and the left context of the temporal short convolutions. These
  are fixed-size recurrent states (the recurrence already decays), so they
  are never evicted.
* Softmax blocks keep the post-RoPE main K/V and the post-UCPE camera K/V of
  every committed chunk. ``trim`` applies the NVlabs window rule
  (``num_cached_blocks`` with the optional chunk-0 sink anchor).
* Every block keeps the left context of the FFN temporal convolution.

Nothing here touches the shared paged KV or the scheduler's capacity planning.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class SanaWmSoftmaxChunk:
    """K/V of one committed chunk for a softmax hybrid block.

    Keys are stored after RoPE (main) / after the UCPE canonical-frame
    transform (camera) at their absolute positions, so replaying them for a
    later chunk needs no re-rotation. Layout is the attention layout
    ``[B, N, H, D]`` with ``N = frames * H_lat * W_lat``.
    """

    chunk_index: int
    frames: int
    k_main: torch.Tensor
    v_main: torch.Tensor
    k_cam: torch.Tensor | None = None
    v_cam: torch.Tensor | None = None

    def nbytes(self) -> int:
        total = 0
        for tensor in (self.k_main, self.v_main, self.k_cam, self.v_cam):
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
        return total


@dataclass
class SanaWmBlockCache:
    """Causal state of one transformer block (TP-local shapes)."""

    # GDN blocks: forward-scan states (fp32) and short-conv left context.
    gdn_state_kv: torch.Tensor | None = None  # [B, H, D, D]
    gdn_state_z: torch.Tensor | None = None  # [B, H, D, 1]
    cam_state_kv: torch.Tensor | None = None  # [B, H_cam, D_cam, D_cam]
    conv_k_tail: torch.Tensor | None = None  # [B*S, K-1, C]
    conv_k_cam_tail: torch.Tensor | None = None  # [B*S, K-1, C_cam]
    # Softmax blocks: one entry per committed chunk still inside the window.
    softmax_chunks: list[SanaWmSoftmaxChunk] = field(default_factory=list)
    # FFN temporal conv left context: [B, C, t_kernel_size // 2, S].
    ffn_tconv_tail: torch.Tensor | None = None

    def cached_main_kv(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [(chunk.k_main, chunk.v_main) for chunk in self.softmax_chunks]

    def cached_cam_kv(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            (chunk.k_cam, chunk.v_cam)
            for chunk in self.softmax_chunks
            if chunk.k_cam is not None and chunk.v_cam is not None
        ]

    def nbytes(self) -> int:
        total = 0
        for tensor in (
            self.gdn_state_kv,
            self.gdn_state_z,
            self.cam_state_kv,
            self.conv_k_tail,
            self.conv_k_cam_tail,
            self.ffn_tconv_tail,
        ):
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
        return total + sum(chunk.nbytes() for chunk in self.softmax_chunks)


@dataclass
class SanaWmStreamingCache:
    """All per-block caches of one streaming request plus chunk bookkeeping.

    ``chunks_committed`` / ``frames_committed`` count the chunks whose clean
    ``save_cache=True`` forward has run; ``frames_committed`` is the absolute
    latent frame index the next chunk starts at.
    """

    blocks: list[SanaWmBlockCache]
    chunk_size: int
    num_cached_blocks: int
    sink_token: bool
    chunks_committed: int = 0
    frames_committed: int = 0
    chunk_frames: list[int] = field(default_factory=list)

    @classmethod
    def new(
        cls,
        *,
        num_blocks: int,
        chunk_size: int,
        num_cached_blocks: int,
        sink_token: bool,
    ) -> SanaWmStreamingCache:
        if num_blocks <= 0:
            raise ValueError(f"Sana-WM streaming cache needs num_blocks > 0, got {num_blocks}.")
        if chunk_size <= 0:
            raise ValueError(f"Sana-WM streaming cache needs chunk_size > 0, got {chunk_size}.")
        return cls(
            blocks=[SanaWmBlockCache() for _ in range(num_blocks)],
            chunk_size=int(chunk_size),
            num_cached_blocks=int(num_cached_blocks),
            sink_token=bool(sink_token),
        )

    def kept_chunk_indices(self, next_chunk_index: int) -> list[int]:
        """Committed chunks whose softmax K/V ``next_chunk_index`` may attend to.

        Mirrors ``SelfForcingFlowEulerCamCtrl._accumulate_softmax_kv_cache``:
        ``num_cached_blocks <= 0`` keeps every chunk; otherwise the window is
        the last ``num_cached_blocks`` chunks, and with ``sink_token`` chunk 0
        is pinned at the front while the window shrinks to the last
        ``num_cached_blocks - 1`` chunks (chunk 0 takes the remaining slot).
        """
        if next_chunk_index <= 0:
            return []
        if self.num_cached_blocks <= 0:
            return list(range(next_chunk_index))
        if self.sink_token:
            window_start = max(next_chunk_index - self.num_cached_blocks + 1, 0)
            kept = {0, *range(window_start, next_chunk_index)}
            return sorted(kept)
        window_start = max(next_chunk_index - self.num_cached_blocks, 0)
        return list(range(window_start, next_chunk_index))

    def commit(self, frames: int) -> None:
        """Record that the current chunk's ``save_cache`` forward has run."""
        if frames <= 0:
            raise ValueError(f"Sana-WM streaming cache commit needs frames > 0, got {frames}.")
        self.chunk_frames.append(int(frames))
        self.chunks_committed += 1
        self.frames_committed += int(frames)
        self.trim()

    def trim(self) -> None:
        """Evict softmax K/V outside the window for the next chunk.

        GDN states and conv tails are fixed-size and always come from the most
        recently committed chunk, so only the softmax entries are touched.
        """
        kept = set(self.kept_chunk_indices(self.chunks_committed))
        for block in self.blocks:
            if not block.softmax_chunks:
                continue
            block.softmax_chunks = [chunk for chunk in block.softmax_chunks if chunk.chunk_index in kept]

    def cached_frames(self) -> int:
        """Latent frames the next chunk's softmax attention will see from the cache."""
        return sum(self.chunk_frames[index] for index in self.kept_chunk_indices(self.chunks_committed))

    def nbytes(self) -> int:
        return sum(block.nbytes() for block in self.blocks)

    def clear(self) -> None:
        for block in self.blocks:
            block.gdn_state_kv = None
            block.gdn_state_z = None
            block.cam_state_kv = None
            block.conv_k_tail = None
            block.conv_k_cam_tail = None
            block.softmax_chunks.clear()
            block.ffn_tconv_tail = None
