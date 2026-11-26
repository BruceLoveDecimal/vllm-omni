# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""TaoMate-R1 session geometry, chunk plan and per-chunk packed layouts.

TaoMate-R1 (``TaoLiveAIGC/TaoMate-R1``) generates one fixed-length MiniMax-H3
Ref2VA session in causal chunks: a 39-frame opening chunk (12 video latents)
followed by 34-frame continuation chunks (10 latents), with a 17-frame
(5-latent) tail when the aligned frame count requires it. Every chunk gathers
its rows from the packed sequence of the *whole* session, so text, reference
and target RoPE positions never restart (the release's ``compact_layout``).

Everything here is host-side integer bookkeeping and tensor gathers, kept
free of model state so it can be tested on CPU.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch

VIDEO_FPS = 24
AUDIO_LATENT_RATE = 40
AUDIO_CHANNELS = 2
OPENING_FRAMES = 39
CONTINUATION_FRAMES = 34
# Native frame groups: 17 frames per five latents after a five-frame prefix.
_GROUP_FRAMES = 17
_PREFIX_FRAMES = 5
SEQ_ALIGN = 64

Bank = Literal["cold", "warm"]
BANKS: tuple[Bank, Bank] = ("cold", "warm")


def align_frame_count(frame_count: int) -> int:
    """Round up to MiniMax-H3's native ``17 j + 5`` frame grid (the release's ``align_frame_count``)."""
    frame_count = int(frame_count)
    if frame_count <= 0:
        return 1
    return frame_count + (_PREFIX_FRAMES - frame_count) % _GROUP_FRAMES


def video_latent_t(frame_count: int) -> int:
    frame_count = int(frame_count)
    if frame_count <= _PREFIX_FRAMES:
        return 2
    return ((frame_count - _PREFIX_FRAMES) // _GROUP_FRAMES) * 5 + 2


def chunk_audio_end(end_frame: int) -> int:
    """Per-channel audio latent boundary of a chunk ending at ``end_frame`` (release rounding)."""
    return round(int(end_frame) * AUDIO_LATENT_RATE / VIDEO_FPS)


def session_audio_latent_t(frame_count: int) -> int:
    """Per-channel audio latents of the whole session: ``round(duration * 40)``."""
    return int(round(float(int(frame_count) / VIDEO_FPS) * float(AUDIO_LATENT_RATE)))


def bank_for_chunk(index: int) -> Bank:
    """The opening (cold) adapter serves chunk 0; the continuation (warm) adapter every later chunk."""
    if index < 0:
        raise ValueError("chunk index must be non-negative")
    return "cold" if index == 0 else "warm"


@dataclass(frozen=True)
class R1Chunk:
    index: int
    frame_start: int
    frame_stop: int
    video_latent_start: int
    video_latent_stop: int
    audio_latent_start: int
    audio_latent_stop: int

    @property
    def frame_count(self) -> int:
        return self.frame_stop - self.frame_start

    @property
    def video_latent_count(self) -> int:
        return self.video_latent_stop - self.video_latent_start

    @property
    def audio_latent_count(self) -> int:
        return self.audio_latent_stop - self.audio_latent_start

    @property
    def bank(self) -> Bank:
        return bank_for_chunk(self.index)


@dataclass(frozen=True)
class R1SessionPlan:
    frame_count: int
    video_latent_t: int
    audio_latent_t: int
    chunks: tuple[R1Chunk, ...]

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    def commits_clean_history(self, index: int) -> bool:
        """Every chunk but the last writes its clean K/V; the terminal chunk has no successor to read it."""
        if not 0 <= index < self.total_chunks:
            raise IndexError(f"chunk {index} outside a {self.total_chunks}-chunk session")
        return index + 1 < self.total_chunks


def plan_session(num_frames: int) -> R1SessionPlan:
    """Chunk plan of a session that publishes at least ``num_frames`` frames."""
    if int(num_frames) <= 0:
        raise ValueError("num_frames must be positive")
    frames = align_frame_count(num_frames)
    chunks: list[R1Chunk] = []
    start_frame = start_video = start_audio = 0
    while start_frame < frames:
        end_frame = min(frames, OPENING_FRAMES if start_frame == 0 else start_frame + CONTINUATION_FRAMES)
        end_video = video_latent_t(end_frame)
        end_audio = chunk_audio_end(end_frame)
        chunks.append(
            R1Chunk(
                index=len(chunks),
                frame_start=start_frame,
                frame_stop=end_frame,
                video_latent_start=start_video,
                video_latent_stop=end_video,
                audio_latent_start=start_audio,
                audio_latent_stop=end_audio,
            )
        )
        start_frame, start_video, start_audio = end_frame, end_video, end_audio
    return R1SessionPlan(
        frame_count=frames,
        video_latent_t=video_latent_t(frames),
        audio_latent_t=session_audio_latent_t(frames),
        chunks=tuple(chunks),
    )


def chunk_audio_ids(chunk: R1Chunk, *, audio_latent_t: int, channels: int = AUDIO_CHANNELS) -> torch.Tensor:
    """Indices of a chunk's target audio rows in the session's channel-major audio rows."""
    return torch.cat(
        [
            torch.arange(
                channel * audio_latent_t + chunk.audio_latent_start, channel * audio_latent_t + chunk.audio_latent_stop
            )
            for channel in range(channels)
        ]
    )


def compact_chunk_layout(
    packed: Mapping[str, Any],
    *,
    reference_rows: int,
    chunk: R1Chunk,
    frame_rows: int,
    audio_latent_t: int,
) -> dict[str, Any]:
    """Gather ``[text | references | chunk audio | chunk video | pad]`` from the session document.

    ``packed`` is the whole-session Ref2VA layout (with the prompt's token tags
    written into its text rows); ``reference_rows`` counts its leading visual
    condition rows. Positions keep their global values; only the rows are
    renumbered, so the chunk sees the same RoPE as the full sequence.
    """
    if reference_rows < 0 or frame_rows <= 0:
        raise ValueError("reference_rows must be >= 0 and frame_rows > 0")
    text = packed["text_pos"].view(-1).long()
    img_pos = packed["img_pos"].view(-1).long()
    if reference_rows > img_pos.numel():
        raise ValueError(f"{reference_rows} reference rows exceed {img_pos.numel()} visual rows")
    references = img_pos[:reference_rows]
    audio = packed["audio_pos"].view(-1).long()[chunk_audio_ids(chunk, audio_latent_t=audio_latent_t)]
    video = img_pos[reference_rows:][chunk.video_latent_start * frame_rows : chunk.video_latent_stop * frame_rows]
    if video.numel() != chunk.video_latent_count * frame_rows:
        raise ValueError(f"chunk {chunk.index} reaches past the session's video rows")
    selected = torch.cat((text, references, audio, video))
    used = int(selected.numel())
    length = -(-used // SEQ_ALIGN) * SEQ_ALIGN
    source_grid = packed["img_position_ids"]
    grid = torch.zeros(length, 3, dtype=source_grid.dtype)
    grid[:used] = source_grid[selected]
    tags = torch.full((length,), -1, dtype=torch.long)
    tags[:used] = packed["token_tags"].view(-1).long()[selected]
    nt, nr, na = int(text.numel()), int(references.numel()), int(audio.numel())
    image_positions = torch.cat((torch.arange(nt, nt + nr), torch.arange(nt + nr + na, used)))
    latent_grid = packed.get("latent_grid")
    ph = int(latent_grid[1]) if latent_grid is not None else None
    pw = int(latent_grid[2]) if latent_grid is not None else None
    if ph is None or pw is None or ph * pw != frame_rows:
        raise ValueError("packed latent_grid does not match frame_rows")
    return {
        "seq_len": length,
        "text_pos": torch.arange(nt),
        "img_pos": image_positions,
        "audio_pos": torch.arange(nt + nr, nt + nr + na),
        "update_mask": torch.arange(int(image_positions.numel())) >= nr,
        "audio_update_mask": torch.ones(na, dtype=torch.bool),
        "img_position_ids": grid,
        "token_tags": tags,
        "cu_seqlens": torch.tensor([0, used, length], dtype=torch.int32),
        "video_spans": ({"start": nt + nr + na, "latent_grid": (chunk.video_latent_count, ph, pw), "role": "target"},),
    }


__all__ = [
    "BANKS",
    "OPENING_FRAMES",
    "CONTINUATION_FRAMES",
    "R1Chunk",
    "R1SessionPlan",
    "align_frame_count",
    "bank_for_chunk",
    "chunk_audio_end",
    "chunk_audio_ids",
    "compact_chunk_layout",
    "plan_session",
    "session_audio_latent_t",
    "video_latent_t",
]
