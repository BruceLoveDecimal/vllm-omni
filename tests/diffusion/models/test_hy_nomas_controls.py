# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Camera conventions and structured action masking for Hy-Nomas."""

import json

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from vllm_omni.diffusion.models.hy_nomas.camera import normalize_camera
from vllm_omni.diffusion.models.hy_nomas.prompt import _action_variants, _action_windows

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_camera_basis_framewise_motion_and_rotation_gain():
    poses = torch.eye(4, dtype=torch.float64).repeat(4, 1, 1)
    poses[:, 1, 3] = torch.arange(4, dtype=torch.float64)
    poses[:, :3, :3] = torch.from_numpy(Rotation.from_euler("x", np.array([0, 0.01, 0.02, 0.03])[:, None]).as_matrix())
    actual = normalize_camera(poses, 4)[0].numpy()
    np.testing.assert_allclose(actual[0], np.eye(4), atol=1e-7)
    # A Unity upwards displacement becomes negative OpenCV Y. Relative motion
    # remains camera-local; its length is normalized by the native p99 rule.
    assert np.all(actual[1:, 1, 3] < 0)
    np.testing.assert_allclose(np.linalg.norm(actual[1:, :3, 3], axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(Rotation.from_matrix(actual[1:, :3, :3]).as_rotvec()[:, 0], -0.025, atol=1e-7)
    np.testing.assert_allclose(np.linalg.det(actual[:, :3, :3]), 1, atol=1e-6)


def test_camera_hold_and_resampling():
    poses = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
    poses[1, 2, 3] = 2
    actual = normalize_camera(poses, 8)
    assert actual.shape == (1, 8, 4, 4)
    torch.testing.assert_close(actual[0, 1:, 2, 3], torch.ones(7))
    torch.testing.assert_close(normalize_camera(poses[:1], 8)[0], torch.eye(4).repeat(8, 1, 1))
    poses[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        normalize_camera(poses, 8)


def test_action_windows_keep_gaps_and_reject_overlap():
    prompt = {
        "segments": [
            {
                "contained_actions": [
                    {"start_time": "00:00.5", "end_time": "00:01", "action_description": "turn"},
                    {"start_time": "00:01.5", "end_time": "00:02", "action_description": "walk"},
                ]
            }
        ]
    }
    text = json.dumps(prompt)
    assert _action_windows(text, 15, 24) == [(0, 3, -1), (3, 6, 0), (6, 9, -1), (9, 12, 1), (12, 15, -1)]
    variants = [json.loads(v) for v in _action_variants(text)]
    assert "contained_actions" not in variants[0]["segments"][0]
    assert len(variants[1]["segments"][0]["contained_actions"]) == 1
    prompt["segments"][0]["contained_actions"][1]["start_time"] = "00:00.75"
    with pytest.raises(ValueError, match="overlap"):
        _action_windows(json.dumps(prompt), 15, 24)
    assert _action_windows("plain text", 30, 24) is None


def test_camera_native_float32_boundary_before_rotation_gain():
    # A fixed smooth-trajectory increment near a BF16 rounding midpoint.
    # Native prepare_turn rounds normalized rotations to FP32, then the gain
    # helper promotes those rounded values to FP64 for its axis-angle math.
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, :3, :3] = [
        [0.99969368636290168, 0.00091448768836366215, -0.024732512173662912],
        [-0.0010052773004971339, 0.99999280172546212, -0.0036586766473755104],
        [0.024728988327500596, 0.0036824189778971472, 0.99968740960700808],
    ]
    basis = np.diag([1.0, -1.0, 1.0])
    # With an identity first pose, first-frame and framewise normalization
    # leave this single local increment unchanged after the basis change.
    normalized = (basis @ poses[1, :3, :3] @ basis).astype(np.float32)
    expected = (
        Rotation.from_rotvec(Rotation.from_matrix(normalized.astype(np.float64)).as_rotvec() * 2.5)
        .as_matrix()
        .astype(np.float32)
    )
    actual = normalize_camera(torch.from_numpy(poses), 2)[0, 1, :3, :3]
    torch.testing.assert_close(actual.to(torch.bfloat16), torch.from_numpy(expected).to(torch.bfloat16), rtol=0, atol=0)
    assert actual[2, 2].to(torch.bfloat16).item() == 1.0  # Late FP32 cast gives 0.99609375.
