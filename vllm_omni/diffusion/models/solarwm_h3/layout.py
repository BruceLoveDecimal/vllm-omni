# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Geometry and packed-row layout of the SolarWM-H3 causal rollout.

The Stage-2 student runs one packed document per denoiser call::

    [Qwen image+text | VisualVAE anchor | audio silence | current 5-latent chunk]

The prefix (text, anchor, audio) is fixed for a request. The current chunk
attends to the prefix, to the cached keys of the previous five chunks and to
itself, with window-local temporal RoPE positions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# Released 158-frame, 768x1344 geometry: 24-channel latents at 48x84 with a
# [1, 2, 2] patch give 1008 rows per latent frame.
SOLARWM_H3_PIXEL_HEIGHT = 768
SOLARWM_H3_PIXEL_WIDTH = 1344
SOLARWM_H3_LATENT_CHANNELS = 24
SOLARWM_H3_LATENT_HEIGHT = 48
SOLARWM_H3_LATENT_WIDTH = 84
SOLARWM_H3_PATCH_SIZE = (1, 2, 2)
SOLARWM_H3_ROWS_PER_LATENT = (SOLARWM_H3_LATENT_HEIGHT // 2) * (SOLARWM_H3_LATENT_WIDTH // 2)
SOLARWM_H3_VIDEO_ROW_WIDTH = SOLARWM_H3_LATENT_CHANNELS * 4

SOLARWM_H3_CHUNK_LATENTS = 5
SOLARWM_H3_WINDOW_CHUNKS = 6
SOLARWM_H3_HISTORY_CHUNKS = SOLARWM_H3_WINDOW_CHUNKS - 1
SOLARWM_H3_CHUNK_ROWS = SOLARWM_H3_CHUNK_LATENTS * SOLARWM_H3_ROWS_PER_LATENT

# The audio branch is a fixed condition: the encoded 158-frame stereo silence,
# 263 latents per channel, packed channel-major.
SOLARWM_H3_AUDIO_LATENTS = 263
SOLARWM_H3_AUDIO_CHANNELS = 2
SOLARWM_H3_AUDIO_ROWS = SOLARWM_H3_AUDIO_LATENTS * SOLARWM_H3_AUDIO_CHANNELS
SOLARWM_H3_AUDIO_ROW_WIDTH = 32

# The VisualVAE covers 17 pixel frames per 5 latents (plus a 5-frame tail); the
# native RoPE time axis advances 5/3 * (1, 4, 4, 4, 4) per latent.
SOLARWM_H3_PIXEL_FRAMES_PER_CHUNK = 17
SOLARWM_H3_MIN_SOURCE_FRAMES = 22
_ROPE_FRAME_RESCALE = 5.0 / 3.0
_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_ROPE_SPATIAL_SCALE = 32

VIDEO_TAG = 0
TEXT_TAG = 1
AUDIO_TAG = 2


@dataclass(frozen=True)
class RolloutGeometry:
    """Latent counts and camera alignment for one generated video."""

    source_frames: int
    decoded_frames: int
    decode_latents: int
    rollout_latents: int
    camera_frame_indices: tuple[int, ...]

    @property
    def num_chunks(self) -> int:
        return self.rollout_latents // SOLARWM_H3_CHUNK_LATENTS


def rollout_geometry(source_frames: int) -> RolloutGeometry:
    """Map a requested frame count to the chunks the student generates.

    Every rollout latent is aligned to the source frame its camera comes from
    using the VAE cadence ``(1, 4, 4, 4, 4)``; latents past the last source
    frame repeat the last camera and are decoded but trimmed afterwards.
    """
    if source_frames < SOLARWM_H3_MIN_SOURCE_FRAMES:
        raise ValueError(f"SolarWM-H3 needs at least {SOLARWM_H3_MIN_SOURCE_FRAMES} frames, got {source_frames}")
    num_vae_chunks = (source_frames - 5 + 16) // SOLARWM_H3_PIXEL_FRAMES_PER_CHUNK
    decode_latents = 5 * num_vae_chunks + 2
    rollout_latents = math.ceil(decode_latents / SOLARWM_H3_CHUNK_LATENTS) * SOLARWM_H3_CHUNK_LATENTS
    indices = [0]
    source_index = 0
    for latent_index in range(rollout_latents - 1):
        source_index += _ROPE_FRAMES_PER_LATENT[latent_index % 5]
        if source_index < source_frames:
            indices.append(source_index)
        else:
            indices.append(indices[-1])
    return RolloutGeometry(
        source_frames=source_frames,
        decoded_frames=SOLARWM_H3_PIXEL_FRAMES_PER_CHUNK * num_vae_chunks + 5,
        decode_latents=decode_latents,
        rollout_latents=rollout_latents,
        camera_frame_indices=tuple(indices),
    )


def _spatial_axis(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    steps = dim // patch
    axis = left + torch.arange(steps, dtype=torch.float64) * (ratio / steps)
    return axis * _ROPE_SPATIAL_SCALE


def frame_position_grid() -> tuple[torch.Tensor, torch.Tensor]:
    """Return the flattened ``(h, w)`` coordinates of one latent frame and the width axis."""
    sqrt_area = math.sqrt(SOLARWM_H3_LATENT_HEIGHT * SOLARWM_H3_LATENT_WIDTH)
    height_axis = _spatial_axis(SOLARWM_H3_LATENT_HEIGHT, 2, sqrt_area)
    width_axis = _spatial_axis(SOLARWM_H3_LATENT_WIDTH, 2, sqrt_area)
    grid_h, grid_w = torch.meshgrid(height_axis, width_axis, indexing="ij")
    return torch.stack((grid_h.reshape(-1), grid_w.reshape(-1)), dim=-1), width_axis


def temporal_position_grid(num_latents: int, origin: float) -> torch.Tensor:
    spans = torch.tensor(
        [_ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[index % 5] for index in range(num_latents)],
        dtype=torch.float64,
    )
    offsets = torch.cat((torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)))
    return origin + offsets


def video_position_grid(num_latents: int, origin: float) -> torch.Tensor:
    """Return frame-major ``(t, h, w)`` RoPE positions for ``num_latents`` frames."""
    frame_grid, _ = frame_position_grid()
    times = temporal_position_grid(num_latents, origin)
    positions = torch.empty(num_latents, SOLARWM_H3_ROWS_PER_LATENT, 3, dtype=torch.float64)
    positions[:, :, 0] = times[:, None]
    positions[:, :, 1:] = frame_grid[None]
    return positions.reshape(-1, 3)


@dataclass(frozen=True)
class PrefixLayout:
    """Row bookkeeping for the request-constant prefix ``[text | anchor | audio]``."""

    text_len: int
    position_ids: torch.Tensor
    token_tags: torch.Tensor

    @property
    def condition_len(self) -> int:
        return self.text_len + SOLARWM_H3_ROWS_PER_LATENT

    @property
    def prefix_len(self) -> int:
        return self.condition_len + SOLARWM_H3_AUDIO_ROWS

    @property
    def sequence_len(self) -> int:
        return self.prefix_len + SOLARWM_H3_CHUNK_ROWS


def build_prefix_layout(text_tags: torch.Tensor) -> PrefixLayout:
    """Lay out the prefix rows given the Qwen presentation's per-token tags."""
    text_len = int(text_tags.shape[0])
    condition_len = text_len + SOLARWM_H3_ROWS_PER_LATENT
    prefix_len = condition_len + SOLARWM_H3_AUDIO_ROWS
    frame_grid, width_axis = frame_position_grid()

    position_ids = torch.zeros(prefix_len, 3, dtype=torch.float64)
    position_ids[:text_len, 0] = torch.arange(text_len, dtype=torch.float64)
    position_ids[text_len:condition_len, 0] = float(text_len)
    position_ids[text_len:condition_len, 1:] = frame_grid
    audio_time = float(text_len) + torch.arange(SOLARWM_H3_AUDIO_LATENTS, dtype=torch.float64)
    position_ids[condition_len:prefix_len, 0] = audio_time.repeat(SOLARWM_H3_AUDIO_CHANNELS)
    position_ids[condition_len : condition_len + SOLARWM_H3_AUDIO_LATENTS, 2] = width_axis[0]
    position_ids[condition_len + SOLARWM_H3_AUDIO_LATENTS : prefix_len, 2] = width_axis[-1]

    token_tags = torch.empty(prefix_len, dtype=torch.long)
    token_tags[:text_len] = text_tags.to(torch.long)
    token_tags[text_len:condition_len] = VIDEO_TAG
    token_tags[condition_len:prefix_len] = AUDIO_TAG
    return PrefixLayout(text_len=text_len, position_ids=position_ids, token_tags=token_tags)


@dataclass(frozen=True)
class ChunkWindow:
    """Which cached chunks the current chunk may attend to."""

    chunk_index: int

    @property
    def first_chunk(self) -> int:
        return max(0, self.chunk_index - SOLARWM_H3_HISTORY_CHUNKS)

    @property
    def history_chunks(self) -> range:
        return range(self.first_chunk, self.chunk_index)

    @property
    def local_latents(self) -> int:
        """Latent frames covered by the window, history first, current chunk last."""
        return (self.chunk_index - self.first_chunk + 1) * SOLARWM_H3_CHUNK_LATENTS

    @property
    def first_latent(self) -> int:
        return self.first_chunk * SOLARWM_H3_CHUNK_LATENTS

    @property
    def current_latent(self) -> int:
        return self.chunk_index * SOLARWM_H3_CHUNK_LATENTS

    @property
    def stop_latent(self) -> int:
        return self.current_latent + SOLARWM_H3_CHUNK_LATENTS


__all__ = [
    "AUDIO_TAG",
    "SOLARWM_H3_AUDIO_CHANNELS",
    "SOLARWM_H3_AUDIO_LATENTS",
    "SOLARWM_H3_AUDIO_ROW_WIDTH",
    "SOLARWM_H3_AUDIO_ROWS",
    "SOLARWM_H3_CHUNK_LATENTS",
    "SOLARWM_H3_CHUNK_ROWS",
    "SOLARWM_H3_HISTORY_CHUNKS",
    "SOLARWM_H3_LATENT_CHANNELS",
    "SOLARWM_H3_LATENT_HEIGHT",
    "SOLARWM_H3_LATENT_WIDTH",
    "SOLARWM_H3_MIN_SOURCE_FRAMES",
    "SOLARWM_H3_PATCH_SIZE",
    "SOLARWM_H3_PIXEL_HEIGHT",
    "SOLARWM_H3_PIXEL_WIDTH",
    "SOLARWM_H3_ROWS_PER_LATENT",
    "SOLARWM_H3_VIDEO_ROW_WIDTH",
    "SOLARWM_H3_WINDOW_CHUNKS",
    "TEXT_TAG",
    "VIDEO_TAG",
    "ChunkWindow",
    "PrefixLayout",
    "RolloutGeometry",
    "build_prefix_layout",
    "frame_position_grid",
    "rollout_geometry",
    "temporal_position_grid",
    "video_position_grid",
]
