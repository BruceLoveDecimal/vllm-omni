# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""A deliberately naive transcription of VDN's linear branch.

Written from the algorithm rather than from the implementation: explicit loops
over frames and taps, ``linalg.inv`` instead of a Cholesky, a product of alphas
instead of a difference of log-prefix sums, and the complement of each window
assembled one frame at a time. It is slow and only usable at test shapes, which
is the point - the shipped branch is vectorised, batched and fused, and an
oracle that shared those choices would share their bugs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

TEXT_STATE_SCALE = 0.5
KERNEL = 5


def _short_conv(tokens: torch.Tensor, conv, proj: str, num_frames: int, frame_size: tuple[int, int]) -> torch.Tensor:
    """Depthwise 5x5 spatial then 5-tap temporal, one output element at a time."""
    if proj not in conv.targets:
        return tokens
    heads, head_dim = tokens.shape[-2], tokens.shape[-1]
    channels = heads * head_dim
    grid_h, grid_w = frame_size
    volume = tokens.reshape(num_frames, grid_h, grid_w, channels).float()
    spatial_weight = getattr(conv, f"{proj}_sp").weight.float()  # [C, 1, 5, 5]
    temporal_weight = getattr(conv, f"{proj}_tm").weight.float()  # [C, 1, 5]
    pad = KERNEL // 2

    spatial = torch.zeros_like(volume)
    for frame in range(num_frames):
        for row in range(grid_h):
            for col in range(grid_w):
                for tap_r in range(KERNEL):
                    for tap_c in range(KERNEL):
                        source_r, source_c = row + tap_r - pad, col + tap_c - pad
                        if 0 <= source_r < grid_h and 0 <= source_c < grid_w:
                            spatial[frame, row, col] += (
                                volume[frame, source_r, source_c] * spatial_weight[:, 0, tap_r, tap_c]
                            )

    rows = spatial.reshape(num_frames, grid_h * grid_w, channels)
    temporal = torch.zeros_like(rows)
    for frame in range(num_frames):
        for tap in range(KERNEL):
            source = frame + tap - pad
            if 0 <= source < num_frames:
                temporal[frame] += rows[source] * temporal_weight[:, 0, tap]
    return temporal.reshape(-1, heads, head_dim)


def _features(branch, tokens: torch.Tensor, proj: str, num_frames: int, frame_size, use_conv: bool) -> torch.Tensor:
    if use_conv and branch.short_conv is not None:
        tokens = _short_conv(tokens, branch.short_conv, proj, num_frames, frame_size)
    activated = F.silu(tokens.float())
    if proj == "v":
        return activated
    return activated / activated.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def _statistics(key: torch.Tensor, value: torch.Tensor, beta: torch.Tensor):
    """A = sum_s b_s k_s k_s^T, B = sum_s b_s v_s k_s^T, one token at a time."""
    frames, heads, tokens, dim = key.shape
    stats_a = torch.zeros(frames, heads, dim, dim, dtype=torch.float64)
    stats_b = torch.zeros(frames, heads, dim, dim, dtype=torch.float64)
    for frame in range(frames):
        for head in range(heads):
            for token in range(tokens):
                weight = beta[frame, head, token].double()
                k = key[frame, head, token].double()
                v = value[frame, head, token].double()
                stats_a[frame, head] += weight * torch.outer(k, k)
                stats_b[frame, head] += weight * torch.outer(v, k)
    return stats_a, stats_b


def _chunk_state(branch, text_x, text_qkv):
    """The prompt written into a zero state as one delta-rule chunk."""
    if text_qkv is None:
        return None
    _, key_raw, value_raw = text_qkv
    length = key_raw.shape[0]
    heads, dim = branch.num_heads, branch.head_dim
    key = _features(branch, key_raw, "k", 1, None, use_conv=False).view(1, length, heads, dim).permute(0, 2, 1, 3)
    value = _features(branch, value_raw, "v", 1, None, use_conv=False).view(1, length, heads, dim).permute(0, 2, 1, 3)
    beta = torch.sigmoid(branch.beta_proj(text_x).float()).view(1, length, heads).permute(0, 2, 1)
    stats_a, stats_b = _statistics(key, value, beta)
    eye = torch.eye(dim, dtype=torch.float64)
    state = torch.zeros(heads, dim, dim, dtype=torch.float64)
    for head in range(heads):
        symmetric = 0.5 * (stats_a[0, head] + stats_a[0, head].T)
        state[head] = stats_b[0, head] @ torch.linalg.inv(eye + symmetric)
    return TEXT_STATE_SCALE * state


def reference_forward(
    branch,
    *,
    video_x: torch.Tensor,
    video_qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    text_x: torch.Tensor | None,
    text_qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    num_frames: int,
    tokens_per_frame: int,
    frame_size: tuple[int, int],
    bounds,
) -> torch.Tensor:
    """The branch's whole contract: the readout for every video row."""
    heads, dim = branch.num_heads, branch.head_dim
    total_rows = num_frames * tokens_per_frame
    out = torch.zeros(total_rows, heads * dim, dtype=torch.float64)
    if num_frames <= 2:
        return out

    # The anchors are exact softmax in both directions, so the branch never
    # sees them and their readout rows stay zero.
    inner = slice(tokens_per_frame, (num_frames - 1) * tokens_per_frame)
    inner_frames = num_frames - 2
    inner_bounds = [(low - 1, high - 1) for low, high in bounds[1 : num_frames - 1]]
    x = video_x[inner].float()
    q_raw, k_raw, v_raw = (tensor[inner] for tensor in video_qkv)

    shape = (inner_frames, tokens_per_frame, heads, dim)
    query = _features(branch, q_raw, "q", inner_frames, frame_size, True).view(shape).permute(0, 2, 1, 3).double()
    key = _features(branch, k_raw, "k", inner_frames, frame_size, True).view(shape).permute(0, 2, 1, 3).double()
    value = _features(branch, v_raw, "v", inner_frames, frame_size, True).view(shape).permute(0, 2, 1, 3).double()

    beta = torch.sigmoid(branch.beta_proj(x).float()).view(inner_frames, tokens_per_frame, heads).permute(0, 2, 1)
    frame_mean = x.view(inner_frames, tokens_per_frame, -1).mean(dim=1)
    alpha = branch.alpha(frame_mean).double()  # [F, H, d]
    gate = torch.sigmoid(branch.output_gate["up"](branch.output_gate["down"](x)).float())
    gate = gate.view(-1, heads, dim).double()

    stats_a, stats_b = _statistics(key, value, beta)
    eye = torch.eye(dim, dtype=torch.float64)
    transitions = torch.zeros(inner_frames, heads, dim, dim, dtype=torch.float64)
    injections = torch.zeros(inner_frames, heads, dim, dim, dtype=torch.float64)
    for frame in range(inner_frames):
        for head in range(heads):
            symmetric = 0.5 * (stats_a[frame, head] + stats_a[frame, head].T)
            inverse = torch.linalg.inv(eye + symmetric)
            transitions[frame, head] = torch.diag(alpha[frame, head]) @ inverse
            injections[frame, head] = stats_b[frame, head] @ inverse

    text_state = _chunk_state(branch, text_x, text_qkv)
    start = torch.zeros(heads, dim, dim, dtype=torch.float64) if text_state is None else text_state
    prefix = [None] * inner_frames
    suffix = [None] * inner_frames
    state = start
    for frame in range(inner_frames):
        state = state @ transitions[frame] + injections[frame]
        prefix[frame] = state
    state = start
    for frame in range(inner_frames - 1, -1, -1):
        state = state @ transitions[frame] + injections[frame]
        suffix[frame] = state

    readout = torch.zeros(inner_frames, tokens_per_frame, heads, dim, dtype=torch.float64)
    for frame in range(inner_frames):
        low, high = inner_bounds[frame]
        before, after = low - 1, high + 1
        # Each side is decayed through the frames the window covers: the softmax
        # already saw them, so the recurrence advances without their injections.
        decay_before = torch.ones(heads, dim, dtype=torch.float64)
        for step in range(max(before + 1, 0), frame + 1):
            decay_before = decay_before * alpha[step]
        decay_after = torch.ones(heads, dim, dtype=torch.float64)
        for step in range(frame, min(after, inner_frames)):
            decay_after = decay_after * alpha[step]

        if before >= 0:
            state_before = prefix[before]
        elif text_state is not None:
            state_before = text_state
        else:
            state_before = torch.zeros(heads, dim, dim, dtype=torch.float64)
        if after < inner_frames:
            state_after = suffix[after]
        elif text_state is not None:
            state_after = text_state
        else:
            state_after = torch.zeros(heads, dim, dim, dtype=torch.float64)

        combined = state_before * decay_before.unsqueeze(1) + state_after * decay_after.unsqueeze(1)
        for head in range(heads):
            for token in range(tokens_per_frame):
                readout[frame, token, head] = combined[head] @ query[frame, head, token]

    flat = readout.reshape(-1, heads, dim)
    normed = flat / (flat.pow(2).mean(dim=-1, keepdim=True) + 1e-6).sqrt()
    normed = normed * branch.norm.weight.double()
    out[inner] = (normed * gate).reshape(-1, heads * dim)
    return out


__all__ = ["reference_forward"]
