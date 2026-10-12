# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""H3-World's per-latent language controls and directed attention.

Semantics follow https://github.com/Danzer1xxxxChan/H3-World (Apache-2.0),
code/abot/action_script.py and code/diffsynth_h3_action.patch. No DiffSynth
runtime is required. Controls and block masks belong to a request. Small
label buffers retain stable identities in the serial request-mode worker.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from typing import Any

import torch

from vllm_omni.errors import OmniClientError

H3_WORLD_ACTIONS_KEY = "h3_world_actions"
H3_WORLD_KEYS = frozenset("WASDIJKLF")
H3_WORLD_PRESETS = {
    "still": (),
    "forward": ("W",),
    "back": ("S",),
    "strafe-left": ("A",),
    "strafe-right": ("D",),
    "tilt-down": ("I",),
    "tilt-up": ("K",),
    "pan-left": ("J",),
    "pan-right": ("L",),
    "pan-left-fast": ("J", "F"),
    "pan-right-fast": ("L", "F"),
}


def action_sentence(keys: object) -> str:
    if not isinstance(keys, Sequence) or isinstance(keys, (str, bytes)):
        raise OmniClientError("H3-World keys must be a list of W/A/S/D/I/J/K/L/F")
    on: set[str] = set()
    for key in keys:
        if not isinstance(key, str) or key.upper() not in H3_WORLD_KEYS:
            raise OmniClientError(f"Unsupported H3-World key: {key!r}")
        on.add(key.upper())
    for a, b in (("W", "S"), ("A", "D"), ("J", "L"), ("I", "K")):
        if a in on and b in on:
            on.difference_update((a, b))
    motion = [
        word
        for key, word in (
            ("W", "walks forward"),
            ("S", "walks backward"),
            ("A", "strafes left"),
            ("D", "strafes right"),
        )
        if key in on
    ]
    speed = "sharply" if "F" in on else "slowly"
    camera = [f"pans {side} {speed}" for key, side in (("J", "left"), ("L", "right")) if key in on]
    camera.extend(word for key, word in (("I", "tilts down"), ("K", "tilts up")) if key in on)
    return (
        f"the man {' and '.join(motion) if motion else 'stands still'}, "
        f"camera {' and '.join(camera) if camera else 'follows him' if motion else 'holds steady'}"
    )


def action_script(value: object, latent_t: int) -> tuple[str, ...]:
    """Accept held keys or exactly one key list per video latent (not pixel)."""
    if not isinstance(value, Mapping) or set(value) not in ({"keys"}, {"frames"}):
        raise OmniClientError("h3_world_actions requires exactly one of 'keys' or 'frames'")
    if "keys" in value:
        return (action_sentence(value["keys"]),) * latent_t
    frames = value["frames"]
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)) or len(frames) != latent_t:
        raise OmniClientError(f"H3-World frames requires exactly {latent_t} per-latent key lists")
    return tuple(action_sentence(keys) for keys in frames)


def append_action_embeddings(
    head: torch.Tensor,
    tags: torch.Tensor,
    script: Sequence[str],
    encode: Callable[[str], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, tuple[tuple[int, int], ...]]:
    """Encode sentences independently, deduplicating only within this request."""
    cache: dict[str, torch.Tensor] = {}
    parts, spans, cursor = [head], [], head.shape[0]
    for sentence in script:
        if sentence not in cache:
            cache[sentence] = encode(sentence).to(device=head.device, dtype=head.dtype)
        rows = cache[sentence]
        if rows.ndim != 2 or rows.shape[1] != head.shape[1] or rows.shape[0] == 0:
            raise ValueError("H3-World action encoder returned invalid hidden states")
        spans.append((cursor, cursor + rows.shape[0]))
        cursor += rows.shape[0]
        parts.append(rows)
    return (
        torch.cat(parts),
        torch.cat((tags, torch.ones(cursor - head.shape[0], device=tags.device, dtype=tags.dtype))),
        tuple(spans),
    )


def bind_action_spans(packed: dict[str, Any], spans: Sequence[tuple[int, int]]) -> tuple[int, ...]:
    """Mirror action RoPE positions before video; return isolated refiner bounds."""
    latent_t = int(packed["latent_grid"][0])
    text_len = packed["text_pos"].numel()
    if len(spans) != latent_t or not spans or spans[0][0] <= 0 or spans[-1][1] != text_len:
        raise ValueError("H3-World action spans must tile the text tail with one span per video latent")
    bounds = [0, spans[0][0]]
    for lo, hi in spans:
        if lo != bounds[-1] or hi <= lo:
            raise ValueError("H3-World action spans must be contiguous and nonempty")
        bounds.append(hi)
    video_start = int(packed["video_row_start"])
    frame_rows = int(packed["latent_grid"][1] * packed["latent_grid"][2])
    positions = packed["img_position_ids"]
    times = positions[video_start : video_start + latent_t * frame_rows : frame_rows, 0] - text_len
    origin = text_len - float(times[-1]) - 1.0
    if origin < spans[0][0]:
        raise OmniClientError("H3-World mirrored action positions overlap the scene prompt; shorten the clip")
    for (lo, hi), time in zip(spans, times, strict=True):
        positions[lo:hi, 0] = origin + time
    return tuple(bounds)


def directed_visibility(ann: torch.Tensor, frame: torch.Tensor, used: int, q, kv):
    """Allow A_k -> V_k only; A_k reads scene/anchors/audio and its own V_k.

    Attention indices are query, key: information flows from key to query.
    Padding sees only itself, ensuring no empty softmax rows.
    """
    aq, ak, fq, fk = ann[q], ann[kv], frame[q], frame[kv]
    same = (aq >= 0) & (aq == ak)
    own = (fq >= 0) & (fq == ak)
    leak_out = (ak >= 0) & ~same & ~own
    leak_in = (aq >= 0) & (fk >= 0) & (aq != fk)
    return (q == kv) | ((q < used) & (kv < used) & ~(leak_out | leak_in))


@lru_cache(maxsize=8)
def _mask_label_buffers(device: torch.device, padded: int):
    # Match the official patch's stable captured tensor identities. Compiled
    # FlexAttention can otherwise retain an earlier closure's labels. The
    # request-mode worker executes whole forwards serially; step execution is
    # rejected, so these buffers cannot change during a branch's denoise run.
    ann = torch.full((padded,), -1, dtype=torch.int32, device=device)
    return ann, torch.full_like(ann, -1), torch.zeros((), dtype=torch.int32, device=device)


def build_action_block_mask(packed: Mapping[str, Any], spans, device: torch.device):
    from torch.nn.attention.flex_attention import create_block_mask

    seq_len, used = int(packed["seq_len"]), int(packed["cu_seqlens"][1])
    # FlexAttention rounds mask construction to block boundaries.
    padded = (seq_len + 127) // 128 * 128
    ann, frame, limit = _mask_label_buffers(device, padded)
    ann.fill_(-1)
    frame.fill_(-1)
    limit.fill_(used)
    for index, (lo, hi) in enumerate(spans):
        ann[lo:hi] = index
    start = int(packed["video_row_start"])
    frame_rows = int(packed["latent_grid"][1] * packed["latent_grid"][2])
    count = len(spans) * frame_rows
    frame[start : start + count] = torch.arange(count, device=device) // frame_rows

    def mask_mod(b, h, q, kv):
        return directed_visibility(ann, frame, limit, q, kv)

    return create_block_mask(mask_mod, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device, _compile=True)


@lru_cache(maxsize=1)
def _compiled_flex_attention():
    from torch.nn.attention.flex_attention import flex_attention

    return torch.compile(flex_attention, dynamic=False, fullgraph=True)


def action_attention(q, k, v, block_mask, scale):
    """Sparse CUDA attention: no quadratic dense video mask allocation."""
    out = _compiled_flex_attention()(
        q.transpose(0, 1)[None],
        k.transpose(0, 1)[None],
        v.transpose(0, 1)[None],
        block_mask=block_mask,
        scale=scale,
    )
    return out.squeeze(0).transpose(0, 1)
