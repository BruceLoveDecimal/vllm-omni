# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import math

import pytest
import torch

from vllm_omni.diffusion.models.solarwm_h3.camera import (
    SOLARWM_H3_PROPE_DIM_START,
    apply_camera_projection,
    camera_projection,
    first_frame_relative_w2c,
    logd4_translation,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _poses(translations: list[tuple[float, float, float]]) -> torch.Tensor:
    poses = torch.eye(4).repeat(len(translations), 1, 1)
    poses[:, :3, 3] = torch.tensor(translations)
    return poses


def test_first_frame_relative_w2c_anchors_at_identity_and_keeps_scale():
    relative = first_frame_relative_w2c(_poses([(10, 0, 0), (12, 0, 0), (12, 3, 0)]))
    torch.testing.assert_close(relative[0], torch.eye(4))
    torch.testing.assert_close(relative[1, :3, 3], torch.tensor([-2.0, 0.0, 0.0]))
    torch.testing.assert_close(relative[2, :3, 3], torch.tensor([-2.0, -3.0, 0.0]))


def test_logd4_translation_is_zero_safe():
    compressed = logd4_translation(_poses([(0, 0, 0), (3, 4, 0)]))
    torch.testing.assert_close(compressed[0, :3, 3], torch.zeros(3))
    scale = math.log1p(5.0) / (4.0 * 5.0)
    torch.testing.assert_close(compressed[1, :3, 3], torch.tensor([3.0, 4.0, 0.0]) * scale)
    torch.testing.assert_close(compressed[:, :3, :3], torch.eye(3).repeat(2, 1, 1))


def test_camera_projection_preserves_native_prefix_and_projective_inner_products():
    generator = torch.Generator().manual_seed(7)
    rows, heads = 3, 2
    q = torch.randn(rows, heads, 128, generator=generator)
    k = torch.randn(rows, heads, 128, generator=generator)
    views = first_frame_relative_w2c(_poses([(0, 0, 0), (1, 0, 0), (1, 2, 0)]))
    projection = camera_projection(views, torch.float32)

    q_camera = apply_camera_projection(q, projection.query)
    k_camera = apply_camera_projection(k, projection.key_value)
    start = SOLARWM_H3_PROPE_DIM_START
    torch.testing.assert_close(q_camera[..., :start], q[..., :start])
    torch.testing.assert_close(k_camera[..., :start], k[..., :start])

    # P^T q against P^-1 k keeps every 4-D projective inner product of a row with itself.
    before = (q[..., start:].reshape(rows, heads, 8, 4) * k[..., start:].reshape(rows, heads, 8, 4)).sum(-1)
    after = (q_camera[..., start:].reshape(rows, heads, 8, 4) * k_camera[..., start:].reshape(rows, heads, 8, 4)).sum(
        -1
    )
    torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-5)

    # The output transform undoes the key/value transform of the same row.
    restored = apply_camera_projection(apply_camera_projection(k, projection.key_value), projection.output)
    torch.testing.assert_close(restored, k, rtol=1e-5, atol=1e-5)


def test_camera_projection_runs_in_attention_dtype():
    views = first_frame_relative_w2c(_poses([(0, 0, 0), (0.5, 0.25, 0)]))
    projection = camera_projection(views, torch.bfloat16)
    assert projection.query.dtype == torch.bfloat16
    assert projection.key_value.dtype == torch.bfloat16
    assert projection.output.dtype == torch.bfloat16
    features = torch.randn(2, 1, 128, dtype=torch.bfloat16)
    assert apply_camera_projection(features, projection.query).dtype == torch.bfloat16
