# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.models.solarwm_h3.layout import (
    AUDIO_TAG,
    SOLARWM_H3_AUDIO_ROWS,
    SOLARWM_H3_ROWS_PER_LATENT,
    TEXT_TAG,
    VIDEO_TAG,
    ChunkWindow,
    build_prefix_layout,
    frame_position_grid,
    rollout_geometry,
    video_position_grid,
)
from vllm_omni.diffusion.models.solarwm_h3.rollout import denoise_timesteps

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_rollout_geometry_for_the_released_158_frame_profile():
    geometry = rollout_geometry(158)
    assert geometry.decode_latents == 47
    assert geometry.rollout_latents == 50
    assert geometry.decoded_frames == 158
    assert geometry.num_chunks == 10
    # Latent i sits on source frame 17 * (i // 5) + (0, 1, 5, 9, 13)[i % 5];
    # the three latents past frame 157 repeat the last in-range frame.
    expected = [min(17 * (index // 5) + (0, 1, 5, 9, 13)[index % 5], 154) for index in range(50)]
    assert list(geometry.camera_frame_indices) == expected


def test_rollout_geometry_repeats_the_last_camera_past_the_source():
    geometry = rollout_geometry(22)
    assert geometry.decode_latents == 7
    assert geometry.rollout_latents == 10
    assert geometry.camera_frame_indices == (0, 1, 5, 9, 13, 17, 18, 18, 18, 18)
    with pytest.raises(ValueError):
        rollout_geometry(21)


def test_prefix_layout_positions_and_tags():
    text_tags = torch.tensor([TEXT_TAG, TEXT_TAG, VIDEO_TAG, VIDEO_TAG, TEXT_TAG])
    layout = build_prefix_layout(text_tags)
    text_len = 5
    assert layout.condition_len == text_len + SOLARWM_H3_ROWS_PER_LATENT
    assert layout.prefix_len == layout.condition_len + SOLARWM_H3_AUDIO_ROWS
    assert layout.sequence_len == layout.prefix_len + 5 * SOLARWM_H3_ROWS_PER_LATENT

    positions = layout.position_ids
    torch.testing.assert_close(positions[:text_len, 0], torch.arange(text_len, dtype=torch.float64))
    frame_grid, width_axis = frame_position_grid()
    anchor = positions[text_len : layout.condition_len]
    assert torch.all(anchor[:, 0] == text_len)
    torch.testing.assert_close(anchor[:, 1:], frame_grid)
    audio = positions[layout.condition_len :]
    torch.testing.assert_close(audio[:263, 0], text_len + torch.arange(263, dtype=torch.float64))
    torch.testing.assert_close(audio[263:, 0], audio[:263, 0])
    assert torch.all(audio[:263, 2] == width_axis[0])
    assert torch.all(audio[263:, 2] == width_axis[-1])

    tags = layout.token_tags
    torch.testing.assert_close(tags[:text_len], text_tags)
    assert torch.all(tags[text_len : layout.condition_len] == VIDEO_TAG)
    assert torch.all(tags[layout.condition_len :] == AUDIO_TAG)


def test_video_position_grid_uses_the_native_temporal_cadence():
    positions = video_position_grid(6, origin=10.0)
    assert positions.shape == (6 * SOLARWM_H3_ROWS_PER_LATENT, 3)
    times = positions[::SOLARWM_H3_ROWS_PER_LATENT, 0]
    scale = 5.0 / 3.0
    expected = 10.0 + scale * torch.tensor([0.0, 1.0, 5.0, 9.0, 13.0, 17.0], dtype=torch.float64)
    torch.testing.assert_close(times, expected)


def test_chunk_window_covers_at_most_five_previous_chunks():
    first = ChunkWindow(0)
    assert first.first_chunk == 0
    assert list(first.history_chunks) == []
    assert first.local_latents == 5

    late = ChunkWindow(7)
    assert late.first_chunk == 2
    assert list(late.history_chunks) == [2, 3, 4, 5, 6]
    assert late.local_latents == 30
    assert (late.first_latent, late.current_latent, late.stop_latent) == (10, 35, 40)


def test_denoise_timesteps_follow_the_shift_12_grid():
    timesteps = denoise_timesteps()
    expected = [1.0 - 12.0 * b / (1.0 + 11.0 * b) for b in (1.0, 0.75, 0.5, 0.25)]
    assert timesteps == pytest.approx(expected, abs=1e-6)
    assert timesteps[0] == 0.0
