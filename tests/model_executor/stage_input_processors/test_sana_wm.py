# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest

from vllm_omni.model_executor.stage_input_processors.sana_wm import (
    SANA_WM_CANONICAL_KEY,
    SANA_WM_DEFAULT_HEIGHT,
    SANA_WM_DEFAULT_NUM_FRAMES,
    SANA_WM_DEFAULT_WIDTH,
    ar2diffusion,
    normalize_sana_wm_payload,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _identity_pose() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _poses(num_frames: int) -> list[list[list[float]]]:
    return [_identity_pose() for _ in range(num_frames)]


def _intrinsics_matrix() -> list[list[float]]:
    return [
        [500.0, 0.0, 640.0],
        [0.0, 500.0, 352.0],
        [0.0, 0.0, 1.0],
    ]


def test_normalize_explicit_camera_payload() -> None:
    image = object()
    prompt = {
        "prompt": "drive forward",
        "multi_modal_data": {"image": image},
        "sana_wm": {
            "num_frames": 2,
            "camera": {"poses": _poses(2)},
            "intrinsics": [500.0, 500.0, 640.0, 352.0],
        },
    }

    result = normalize_sana_wm_payload(prompt)

    assert result["multi_modal_data"]["image"] is image
    assert SANA_WM_CANONICAL_KEY not in result
    payload = result["additional_information"][SANA_WM_CANONICAL_KEY]
    assert payload["num_frames"] == 2
    assert payload["height"] == SANA_WM_DEFAULT_HEIGHT
    assert payload["width"] == SANA_WM_DEFAULT_WIDTH
    assert payload["camera"]["format"] == "c2w_4x4"
    assert payload["camera"]["coordinate_system"] == "official"
    assert payload["camera"]["poses"] == _poses(2)
    assert payload["action"] is None
    assert payload["intrinsics"] == [500.0, 500.0, 640.0, 352.0]


def test_normalize_per_frame_intrinsics_list() -> None:
    prompt = {
        "image": object(),
        "sana_wm": {
            "num_frames": 2,
            "camera": {"poses": _poses(2)},
            "intrinsics": [_intrinsics_matrix(), _intrinsics_matrix()],
        },
    }

    result = normalize_sana_wm_payload(prompt)
    payload = result["additional_information"][SANA_WM_CANONICAL_KEY]

    assert payload["intrinsics"] == [_intrinsics_matrix(), _intrinsics_matrix()]


def test_normalize_legacy_camera_trajectory() -> None:
    image = object()
    prompt = {
        "image": image,
        "num_frames": 1,
        "camera_trajectory": {
            "format": "extrinsics_4x4",
            "coordinate_system": "opencv",
            "poses": _poses(1),
        },
    }

    result = normalize_sana_wm_payload(prompt)
    payload = result["additional_information"][SANA_WM_CANONICAL_KEY]

    assert result["multi_modal_data"]["image"] is image
    assert payload["camera"]["format"] == "extrinsics_4x4"
    assert payload["camera"]["coordinate_system"] == "opencv"
    assert payload["camera"]["poses"] == _poses(1)


def test_normalize_action_payload_does_not_require_camera_pose_count() -> None:
    image = object()
    prompt = {
        "multi_modal_data": {"images": [image]},
        "extra_args": {
            "sana_wm": {
                "num_frames": 321,
                "action": "w-80,jw-40,w-40,lw-60,w-100",
            }
        },
    }

    result = normalize_sana_wm_payload(prompt)
    payload = result["additional_information"][SANA_WM_CANONICAL_KEY]

    assert payload["num_frames"] == 321
    assert payload["camera"] is None
    assert payload["action"] == "w-80,jw-40,w-40,lw-60,w-100"


def test_normalize_preserves_official_cli_motion_options() -> None:
    result = normalize_sana_wm_payload(
        {
            "image": object(),
            "sana_wm": {
                "action": "w-1",
                "translation_speed": 0.055,
                "rotation_speed_deg": 1.2,
                "official_repo_path": "/tmp/Sana",
            },
        }
    )

    payload = result["additional_information"][SANA_WM_CANONICAL_KEY]
    assert payload["translation_speed"] == 0.055
    assert payload["rotation_speed_deg"] == 1.2
    assert payload["official_repo_path"] == "/tmp/Sana"


def test_normalize_uses_defaults() -> None:
    result = normalize_sana_wm_payload(
        {
            "image": object(),
            "action": "w-1",
        }
    )

    payload = result["additional_information"][SANA_WM_CANONICAL_KEY]
    assert payload["num_frames"] == SANA_WM_DEFAULT_NUM_FRAMES
    assert payload["height"] == SANA_WM_DEFAULT_HEIGHT
    assert payload["width"] == SANA_WM_DEFAULT_WIDTH


def test_requires_first_frame_image() -> None:
    with pytest.raises(ValueError, match="first-frame image"):
        normalize_sana_wm_payload({"sana_wm": {"action": "w-1"}})


def test_rejects_missing_camera_and_action() -> None:
    with pytest.raises(ValueError, match="requires either explicit camera poses or an action"):
        normalize_sana_wm_payload({"image": object()})


def test_rejects_both_camera_and_action() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        normalize_sana_wm_payload(
            {
                "image": object(),
                "sana_wm": {
                    "num_frames": 1,
                    "camera": {"poses": _poses(1)},
                    "action": "w-1",
                },
            }
        )


def test_rejects_bad_camera_shape() -> None:
    with pytest.raises(ValueError, match="4x4"):
        normalize_sana_wm_payload(
            {
                "image": object(),
                "sana_wm": {
                    "num_frames": 1,
                    "camera": {"poses": [[[1.0, 0.0], [0.0, 1.0]]]},
                },
            }
        )


def test_rejects_camera_length_mismatch() -> None:
    with pytest.raises(ValueError, match="must equal num_frames"):
        normalize_sana_wm_payload(
            {
                "image": object(),
                "sana_wm": {
                    "num_frames": 2,
                    "camera": {"poses": _poses(1)},
                },
            }
        )


def test_rejects_bad_intrinsics() -> None:
    with pytest.raises(ValueError, match="intrinsics"):
        normalize_sana_wm_payload(
            {
                "image": object(),
                "sana_wm": {
                    "action": "w-1",
                    "intrinsics": {"fx": 1.0, "fy": 1.0},
                },
            }
        )


def test_ar2diffusion_normalizes_prompt_list() -> None:
    image = object()
    result = ar2diffusion(
        [],
        prompt=[
            {
                "multi_modal_data": {"img2img": image},
                "sana_wm": {"num_frames": 1, "camera": {"poses": _poses(1)}},
            }
        ],
    )

    assert len(result) == 1
    payload = result[0]["additional_information"][SANA_WM_CANONICAL_KEY]
    assert payload["num_frames"] == 1
    assert result[0]["multi_modal_data"]["image"] is image
