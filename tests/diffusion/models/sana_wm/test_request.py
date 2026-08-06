# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SANA-WM request normalization."""

import pytest

from vllm_omni.diffusion.models.sana_wm.request import (
    SANA_WM_DEFAULT_HEIGHT,
    SANA_WM_DEFAULT_NUM_FRAMES,
    SANA_WM_DEFAULT_WIDTH,
    normalize_sana_wm_payload,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _prompt(**sana_wm):
    return {
        "prompt": "a quiet city street",
        "multi_modal_data": {"image": object()},
        "sana_wm": {"action": "w-8", **sana_wm},
    }


def test_defaults_are_latent_aligned():
    payload = normalize_sana_wm_payload(_prompt())["additional_information"]["sana_wm"]
    assert payload["num_frames"] == SANA_WM_DEFAULT_NUM_FRAMES
    assert payload["height"] == SANA_WM_DEFAULT_HEIGHT
    assert payload["width"] == SANA_WM_DEFAULT_WIDTH


@pytest.mark.parametrize("num_frames", [9, 33, 161])
def test_accepts_aligned_geometry(num_frames):
    payload = normalize_sana_wm_payload(_prompt(num_frames=num_frames))["additional_information"]["sana_wm"]
    assert payload["num_frames"] == num_frames


@pytest.mark.parametrize("num_frames", [8, 10, 160])
def test_rejects_unaligned_num_frames(num_frames):
    # The VAE compresses 8x temporally; an unaligned count would be floored.
    with pytest.raises(ValueError, match="num_frames"):
        normalize_sana_wm_payload(_prompt(num_frames=num_frames))


@pytest.mark.parametrize(("height", "width"), [(700, 1280), (704, 1270), (703, 1279)])
def test_rejects_unaligned_resolution(height, width):
    with pytest.raises(ValueError, match="divisible by 32"):
        normalize_sana_wm_payload(_prompt(height=height, width=width))


def test_normalization_is_idempotent():
    once = normalize_sana_wm_payload(_prompt(num_frames=33))
    twice = normalize_sana_wm_payload(once)
    assert once["additional_information"]["sana_wm"] == twice["additional_information"]["sana_wm"]
