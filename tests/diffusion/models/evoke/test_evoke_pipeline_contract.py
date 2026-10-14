# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Evoke request/output contract that does not need weights."""

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.models.evoke.pipeline_evoke import (
    RECIPE,
    EvokePipeline,
    chunk_media_frames,
    num_chunks_for_frames,
    prompt_clean,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize(
    ("num_frames", "chunks"),
    [(1, 1), (33, 1), (34, 2), (66, 2), (67, 3), (721, 22), (1437, 44)],
)
def test_chunk_count_matches_upstream_launchers(num_frames, chunks):
    assert num_chunks_for_frames(num_frames) == chunks


def test_cold_t2v_chunk_decodes_33_frames_and_every_other_chunk_36():
    assert chunk_media_frames(0, has_prior=False) == 33
    assert chunk_media_frames(1, has_prior=False) == 36
    assert chunk_media_frames(0, has_prior=True) == 36


def test_recipe_is_the_shipped_post_distill_recipe():
    assert RECIPE.pyramid_steps == (1, 1, 1)
    assert RECIPE.history_sizes == (16, 2, 1)
    assert RECIPE.warp_noise_sigma == (0.0, 0.135)
    assert RECIPE.warp_noise_sigma_invisible == 1.0
    assert RECIPE.max_sequence_length == 512


def test_prompt_clean_unescapes_and_collapses_whitespace():
    assert prompt_clean("  a &amp;amp; b\n\tc ") == "a & b c"


def _state(**sampling):
    defaults = {
        "guidance_scale_provided": False,
        "guidance_scale": 1.0,
        "num_inference_steps": None,
        "fps": None,
    }
    return SimpleNamespace(sampling=SimpleNamespace(**{**defaults, **sampling}))


@pytest.mark.parametrize(
    ("sampling", "extra", "message"),
    [
        ({"guidance_scale_provided": True, "guidance_scale": 5.0}, {}, "CFG-free"),
        ({"num_inference_steps": 50}, {}, "exactly 3 steps"),
        ({"fps": 30}, {}, "24 fps"),
        ({}, {"history_sizes": [8, 2, 1]}, "pinned by the checkpoint recipe"),
    ],
)
def test_recipe_knobs_are_not_request_tunable(sampling, extra, message):
    with pytest.raises(ValueError, match=message):
        EvokePipeline._validate_request(None, _state(**sampling), extra)
