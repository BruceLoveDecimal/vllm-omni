# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Numerical core of Wan2.2-Animate-2 in-context reference attention.

Kept free of any vLLM import so the parity tests that guard it (the riskiest
part of the port) can run on a bare CPU box with nothing but PyTorch.

Upstream (``Wan-Video/Wan-Animate-2``) expresses the joint attention over
generated and reference tokens as one flex-attention call with a compiled
``BlockMask`` plus a ``score_mod`` bias.  That formulation materialises a
padded, 128-aligned key/value buffer and needs ``torch.compile`` to avoid
quadratic memory.  The decomposition here is mathematically identical: the key
axis is cut into disjoint chunks, each chunk is a dense attention, and the
chunks are recombined by log-sum-exp.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ReferenceGridInfo:
    """Frame/token bookkeeping shared by every layer's reference attention.

    ``origin_len`` and ``origin_area`` in the upstream forward describe the
    *nominal* request (configured segment length and ``width * height``), while
    the latent grids describe what is actually generated.  They diverge in two
    ways, both of which upstream absorbs as zero padding inside the mask
    window: the trailing segment of a video is shorter than the nominal one,
    and letterboxing rounds the frame down to a divisor-aligned box whose area
    is at most the requested one.

    Attributes:
        num_frames: real latent frames in this segment, including the
            reference-image slot at index 0.
        frame_tokens: real tokens per latent frame after patchification.
        num_reference_frames: real driving-video latent frames.
        padded_frame_tokens: per-frame stride of the mask window
            (``origin_area[0] * origin_area[1] // 256``).
        padded_gen_frames: generated frame slots the mask treats as valid.
        padded_reference_frames: reference frame slots the mask treats as valid.
    """

    num_frames: int
    frame_tokens: int
    num_reference_frames: int
    padded_frame_tokens: int
    padded_gen_frames: int
    padded_reference_frames: int

    @classmethod
    def from_grids(
        cls,
        generation_grid: tuple[int, int, int],
        reference_grid: tuple[int, int, int],
        origin_len: int,
        origin_area: tuple[int, int],
    ) -> ReferenceGridInfo:
        """Derive the mask window from the latent grids and the nominal request."""
        frame_tokens = generation_grid[1] * generation_grid[2]
        padded_frame_tokens = origin_area[0] * origin_area[1] // 256
        if reference_grid[1] * reference_grid[2] != frame_tokens:
            raise ValueError(
                f"reference grid {reference_grid} and generation grid {generation_grid} must share spatial extents"
            )
        if padded_frame_tokens < frame_tokens:
            raise ValueError(
                f"origin_area {tuple(origin_area)} yields {padded_frame_tokens} tokens per frame, fewer than the "
                f"{frame_tokens} actually generated; the mask window cannot be smaller than the latent grid"
            )
        padded_reference_frames = origin_len // 4 + 1
        return cls(
            num_frames=generation_grid[0],
            frame_tokens=frame_tokens,
            num_reference_frames=reference_grid[0],
            padded_frame_tokens=padded_frame_tokens,
            padded_gen_frames=padded_reference_frames + 1,
            padded_reference_frames=padded_reference_frames,
        )

    def validate(self, seq_len: int) -> None:
        if seq_len != self.num_frames * self.frame_tokens:
            raise ValueError(
                f"query length {seq_len} != num_frames*frame_tokens ({self.num_frames}*{self.frame_tokens})"
            )
        if self.padded_gen_frames < self.num_frames:
            raise ValueError(f"padded_gen_frames ({self.padded_gen_frames}) < num_frames ({self.num_frames})")
        if self.padded_reference_frames < self.num_reference_frames:
            raise ValueError(
                f"padded_reference_frames ({self.padded_reference_frames}) < "
                f"num_reference_frames ({self.num_reference_frames})"
            )
        if self.num_frames < 2:
            # Latent frame 0 is the reference-image slot, so a real segment
            # always has at least one generated frame after it.  A single-frame
            # segment would also put the distilled bias region entirely inside
            # padding, which the branch split does not model.
            raise ValueError(f"a segment needs at least 2 latent frames, got {self.num_frames}")


def attention_with_lse(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense attention returning both the output and its log-sum-exp.

    ``torch.nn.functional.scaled_dot_product_attention`` does not expose the
    log-sum-exp, so this calls the fused kernels behind it directly.  They are
    the same kernels SDPA dispatches to and are what makes the branch merge
    possible without materialising the score matrix.

    Args:
        query: ``[B, Sq, H, D]``.
        key/value: ``[B, Skv, H, D]``.
        softmax_scale: scaling applied to the raw logits.

    Returns:
        ``(out, lse)`` with ``out`` shaped ``[B, Sq, H, D]`` (dtype of
        ``query``) and ``lse`` shaped ``[B, Sq, H]`` in float32.  ``lse`` is the
        log-sum-exp of the *scaled* logits, which is what
        :func:`merge_attention_branches` consumes.
    """
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)

    if q.is_cuda:
        out, lse = torch.ops.aten._scaled_dot_product_efficient_attention(
            q, k, v, attn_bias=None, compute_log_sumexp=True, dropout_p=0.0, is_causal=False, scale=softmax_scale
        )[:2]
    else:
        out, lse = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(
            q, k, v, dropout_p=0.0, is_causal=False, scale=softmax_scale
        )[:2]

    out = out.transpose(1, 2).to(query.dtype)
    # `lse` is [B, H, Sq] (some kernels pad Sq); trim then match `out`'s layout.
    lse = lse[..., : q.shape[2]].transpose(1, 2).to(torch.float32)
    return out, lse


def merge_attention_branches(
    branches: list[tuple[torch.Tensor | None, torch.Tensor]],
) -> torch.Tensor:
    """Combine independent attention branches into one softmax.

    Splitting the key/value axis into disjoint chunks and merging the per-chunk
    results by log-sum-exp reproduces the single dense softmax over their
    concatenation exactly::

        out = sum_i out_i * exp(lse_i - lse_total),  lse_total = log sum_i exp(lse_i)

    The ring-attention helper ``update_out_and_lse`` accumulates one
    ``(out, lse)`` block at a time and needs an output tensor for every block;
    the zero-key branches here carry only a log-sum-exp, so the merge is
    written out directly.

    Args:
        branches: ``(out, lse)`` pairs.  ``out`` may be ``None`` for a branch
            whose keys and values are all zero: such a branch contributes
            nothing to the numerator but still widens the denominator, which is
            precisely how upstream's zero-padded key slots behave.  A branch
            that does not apply to some query rows carries ``-inf`` there.

    Returns:
        The merged output, in the dtype of the first non-``None`` branch.
    """
    if not branches:
        raise ValueError("merge_attention_branches() needs at least one branch")

    template = None
    for out, _ in branches:
        if out is not None:
            template = out
            break
    if template is None:
        raise ValueError("at least one branch must carry an output tensor")

    stacked_lse = torch.stack([lse for _, lse in branches], dim=0)
    max_lse = stacked_lse.amax(dim=0)
    weights = torch.exp(stacked_lse - max_lse.unsqueeze(0))
    denominator = weights.sum(dim=0)

    numerator = torch.zeros(template.shape, dtype=torch.float32, device=template.device)
    for idx, (out, _) in enumerate(branches):
        if out is None:
            continue
        numerator = numerator + out.to(torch.float32) * weights[idx].unsqueeze(-1)

    return (numerator / denominator.unsqueeze(-1)).to(template.dtype)


def reference_context_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    reference_key: torch.Tensor,
    reference_value: torch.Tensor,
    grid: ReferenceGridInfo,
    softmax_scale: float,
    log_scale: float = 0.0,
) -> torch.Tensor:
    """Joint attention over generated tokens and frame-aligned reference tokens.

    Reproduces upstream's flex-attention ``BlockMask`` semantics
    (``WanAnimate2Transformer.create_mask``):

    * every generated query attends **all** generated key slots inside the
      padded window;
    * a query in latent frame ``f`` additionally attends every key slot of
      reference frame ``f - 1`` (latent frame 0 is the reference-image slot and
      has no aligned reference frame);
    * the distilled checkpoints add a constant ``log_scale`` to the logits of
      the generated frame-1 key block (upstream ``_score_mod_impl``).

    Upstream scatters the real tokens into a padded ``[frames,
    padded_frame_tokens]`` grid sized from ``origin_area``, which is the
    request's nominal ``width * height`` rather than the letterboxed grid that
    is actually generated.  Both the spatial remainder
    (``padded_frame_tokens - frame_tokens``) and the trailing frames therefore
    stay zero, and zero keys are *not* masked out: they score 0 against every
    query and so widen the softmax denominator without contributing any value.
    Each such run is folded in analytically as a branch with no output and a
    log-sum-exp of ``log(count)``, which costs nothing to evaluate.

    Args:
        query/key/value: ``[B, num_frames * frame_tokens, H, D]``, RoPE already
            applied, in the *compact* layout (no padding slots).
        reference_key/reference_value: ``[B, num_reference_frames *
            frame_tokens, H, D]``; RoPE already applied to the key.
        grid: see :class:`ReferenceGridInfo`.
        softmax_scale: attention logit scale.
        log_scale: distilled-checkpoint bias on the generated frame-1 block.

    Returns:
        ``[B, num_frames * frame_tokens, H, D]``.
    """
    batch, seq_len, num_heads, _ = query.shape
    grid.validate(seq_len)
    frame_tokens = grid.frame_tokens
    spatial_padding = grid.padded_frame_tokens - frame_tokens
    lse_shape = (batch, seq_len, num_heads)

    def zero_key_lse(num_zero_keys: int) -> torch.Tensor:
        # A zero key yields a zero logit regardless of the query, so a run of
        # `num_zero_keys` such slots contributes log(num_zero_keys) to every
        # query row and head, and nothing to the value sum.
        return torch.full(lse_shape, math.log(num_zero_keys), dtype=torch.float32, device=query.device)

    branches: list[tuple[torch.Tensor | None, torch.Tensor]] = []

    # --- Branch group A: generated tokens ---------------------------------
    # The distilled bias covers the whole frame-1 slot block, real and padded
    # alike, so the block is split off only when that bias is active.
    if log_scale != 0.0:
        block_bounds = (
            (0, frame_tokens, 0.0),
            (frame_tokens, 2 * frame_tokens, log_scale),
            (2 * frame_tokens, seq_len, 0.0),
        )
        for start, stop, bias in block_bounds:
            if start >= stop:
                continue
            out, lse = attention_with_lse(query, key[:, start:stop], value[:, start:stop], softmax_scale)
            branches.append((out, lse + bias))
        if spatial_padding > 0:
            branches.append((None, zero_key_lse(spatial_padding) + log_scale))
        gen_zero_keys = grid.padded_gen_frames * grid.padded_frame_tokens - seq_len - spatial_padding
    else:
        branches.append(attention_with_lse(query, key, value, softmax_scale))
        gen_zero_keys = grid.padded_gen_frames * grid.padded_frame_tokens - seq_len

    if gen_zero_keys > 0:
        branches.append((None, zero_key_lse(gen_zero_keys)))

    # --- Branch B: frame-aligned reference tokens -------------------------
    # Query frame f attends reference frame f - 1, so query frames 1..N-1 map
    # onto reference frames 0..N-2.
    aligned_frames = min(grid.num_frames - 1, grid.num_reference_frames)
    aligned_rows = slice(frame_tokens, (aligned_frames + 1) * frame_tokens)
    ref_lse = torch.full(lse_shape, -math.inf, dtype=torch.float32, device=query.device)
    ref_out = torch.zeros_like(query)

    if aligned_frames > 0:
        # Fold the frame axis into the batch so every frame pair is one small
        # dense attention: [B, F*hw, H, D] -> [B*F, hw, H, D].
        q_aligned = query[:, aligned_rows].reshape(batch * aligned_frames, frame_tokens, num_heads, -1)
        k_aligned = reference_key[:, : aligned_frames * frame_tokens]
        k_aligned = k_aligned.reshape(batch * aligned_frames, frame_tokens, num_heads, -1)
        v_aligned = reference_value[:, : aligned_frames * frame_tokens]
        v_aligned = v_aligned.reshape(batch * aligned_frames, frame_tokens, num_heads, -1)
        out_aligned, lse_aligned = attention_with_lse(q_aligned, k_aligned, v_aligned, softmax_scale)
        ref_out[:, aligned_rows] = out_aligned.reshape(batch, aligned_frames * frame_tokens, num_heads, -1)
        ref_lse[:, aligned_rows] = lse_aligned.reshape(batch, aligned_frames * frame_tokens, num_heads)

    branches.append((ref_out, ref_lse))

    # Zero slots inside the aligned reference frame: the spatial remainder for
    # frames that exist, the whole block for frames past the real sequence.
    reference_zero_lse = torch.full(lse_shape, -math.inf, dtype=torch.float32, device=query.device)
    if aligned_frames > 0 and spatial_padding > 0:
        reference_zero_lse[:, aligned_rows] = math.log(spatial_padding)
    padded_start = aligned_frames + 1
    if padded_start < grid.num_frames and grid.num_reference_frames < grid.padded_reference_frames:
        reference_zero_lse[:, padded_start * frame_tokens :] = math.log(grid.padded_frame_tokens)
    if torch.isfinite(reference_zero_lse).any():
        branches.append((None, reference_zero_lse))

    return merge_attention_branches(branches)


class WanAnimate2RotaryPosEmbed(nn.Module):
    """3D rotary embeddings for the generation and reference grids.

    Upstream builds the frequency table per forward with ``torch.polar`` in
    float64 and applies it in a per-sample Python loop.  Here the table is
    built once per ``(grid, offsets, stride)`` combination and cached, and the
    application is delegated to the platform-optimised
    ``RotaryEmbeddingWanS2V`` kernel.

    The reference grid deliberately lives on positions disjoint from the
    generation grid (``refer_offset_*``), so reference tokens never collide
    with generated ones in position space.
    """

    def __init__(self, attention_head_dim: int, max_seq_len: int = 512, theta: float = 10000.0):
        super().__init__()
        if attention_head_dim % 2 != 0:
            raise ValueError(f"attention_head_dim must be even, got {attention_head_dim}")
        self.max_seq_len = max_seq_len
        self.theta = theta

        half = attention_head_dim // 2
        # Same (t, h, w) split as upstream `rope_apply`.
        self.dim_split = (half - 2 * (half // 3), half // 3, half // 3)
        self._cache: dict[tuple[tuple[int, int, int], tuple[int, int, int], int, str], torch.Tensor] = {}

    def _axis_freqs(self, dim: int, offset: int) -> torch.Tensor:
        # Built on CPU in float64 to match upstream's `rope_params` bit for bit,
        # then narrowed to complex64 once.
        positions = torch.arange(self.max_seq_len, dtype=torch.float64) + offset
        exponents = torch.arange(0, dim * 2, 2, dtype=torch.float64).div(dim * 2)
        inv_freq = 1.0 / torch.pow(torch.tensor(self.theta, dtype=torch.float64), exponents)
        return torch.outer(positions, inv_freq)

    def forward(
        self,
        grid_sizes: tuple[int, int, int],
        device: torch.device,
        offsets: tuple[int, int, int] = (0, 0, 0),
        time_stride: int = 1,
    ) -> torch.Tensor:
        """Return complex RoPE frequencies shaped ``[1, f*h*w, 1, head_dim//2]``."""
        cache_key = (grid_sizes, offsets, time_stride, str(device))
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        num_frames, height, width = grid_sizes
        dim_t, dim_h, dim_w = self.dim_split
        freqs_t = self._axis_freqs(dim_t, offsets[0])[: num_frames * time_stride : time_stride]
        freqs_h = self._axis_freqs(dim_h, offsets[1])[:height]
        freqs_w = self._axis_freqs(dim_w, offsets[2])[:width]

        freqs = torch.cat(
            [
                freqs_t.view(num_frames, 1, 1, -1).expand(num_frames, height, width, -1),
                freqs_h.view(1, height, 1, -1).expand(num_frames, height, width, -1),
                freqs_w.view(1, 1, width, -1).expand(num_frames, height, width, -1),
            ],
            dim=-1,
        ).reshape(1, num_frames * height * width, 1, -1)

        freqs = torch.polar(torch.ones_like(freqs), freqs).to(device=device, dtype=torch.complex64)
        self._cache[cache_key] = freqs
        return freqs

    def clear_cache(self) -> None:
        self._cache.clear()
