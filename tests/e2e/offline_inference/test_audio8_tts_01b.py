# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""E2E offline tests for Audio8 TTS Preview 0.1B (Falcon-H1 hybrid Slow AR).

Same protocol as the 0.6b: prompts are pre-tokenized, since text-only prompts
are plain token ids and voice-clone prompts are a fixed-length placeholder whose
embeddings are built model-side.

What is specific here is the Slow AR backbone -- attention KV in paged blocks
plus a per-request Mamba2 state slot -- so the concurrency case matters more
than it does for a dense model: a shared or misindexed SSM state is invisible
at ``max_num_seqs: 1``.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner, OmniRunnerHandler
from tests.helpers.stage_config import get_deploy_config_path

MODEL = os.environ.get("AUDIO8_TTS_01B_MODEL_PATH", "Audio8/Audio8-TTS-Preview-0.1b")
DEPLOY_CONFIG = get_deploy_config_path("audio8_tts_01b.yaml")
SAMPLE_RATE = 44100
TEXT = "The weather is nice today, perfect for a walk in the park."
TEXT_ZH = "今天天气很好，很适合去公园散步。"

# Distinct inputs so that leaked per-request state -- codec buffers or, unique
# to this checkpoint, Mamba SSM slots -- fails the pairwise-difference check.
CONCURRENT_TEXTS = [
    "The weather is nice today, perfect for a walk in the park.",
    "She sold seashells by the seashore all through the summer.",
    "Our train departs at a quarter past nine tomorrow morning.",
    "He planted rows of tomatoes and basil behind the old house.",
]

# Remote code stays off: the checkpoint's auto_map would otherwise win over the
# registered arktts config, and its modelling code additionally pins
# transformers 4.57 (it imports a cache class later releases removed).
_OMNI_RUNNER_PARAM = (MODEL, DEPLOY_CONFIG, {"trust_remote_code": False})

pytestmark = pytest.mark.parametrize("omni_runner", [_OMNI_RUNNER_PARAM], indirect=True)


def _text_only_prompt(text: str = TEXT) -> dict:
    from transformers import AutoTokenizer

    from vllm_omni.model_executor.models.audio8_tts.prompt_utils import build_text_only_prompt_ids

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompt_ids, normalized_text = build_text_only_prompt_ids(tokenizer, text)
    return {"prompt_token_ids": prompt_ids, "additional_information": {"text": [normalized_text]}}


# At ``core_model`` the harness patches every stage to ``load_format: dummy``,
# so the AR runs on random weights, never emits EOS, and always generates to
# ``max_tokens``. Duration bounds only mean anything once real weights load.
_REAL_WEIGHT_RUN_LEVELS = frozenset({"advanced_model", "full_model"})


def _duration_bounds(run_level: str) -> dict[str, float]:
    if run_level not in _REAL_WEIGHT_RUN_LEVELS:
        return {}
    return {"min_duration_s": 0.5, "max_duration_s": 30.0}


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_text_to_audio_001(omni_runner: OmniRunner, run_level: str) -> None:
    """Default deploy, single request, text-only synthesis.

    Deploy Setting: audio8_tts_01b.yaml
    Input Modal: text
    Output Modal: audio
    """
    prompt = _text_only_prompt()
    request_config = {
        "input": TEXT,
        "prompt_token_ids": prompt["prompt_token_ids"],
        "additional_information": prompt["additional_information"],
        "response_format": "wav",
        "expected_sample_rate": SAMPLE_RATE,
        **_duration_bounds(run_level),
    }
    OmniRunnerHandler(omni_runner).send_tokenized_tts_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_text_to_audio_zh_002(omni_runner: OmniRunner, run_level: str) -> None:
    """Chinese input: the 0.1b ships a different tokenizer from the 0.6b."""
    prompt = _text_only_prompt(TEXT_ZH)
    request_config = {
        "input": TEXT_ZH,
        "prompt_token_ids": prompt["prompt_token_ids"],
        "additional_information": prompt["additional_information"],
        "response_format": "wav",
        "expected_sample_rate": SAMPLE_RATE,
        **_duration_bounds(run_level),
    }
    OmniRunnerHandler(omni_runner).send_tokenized_tts_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_text_to_audio_concurrent_003(omni_runner: OmniRunner, run_level: str) -> None:
    """Four concurrent requests: stage 0 runs ``max_num_seqs: 4``.

    The decoded outputs are asserted pairwise different, which is what catches a
    Mamba state slot shared or misindexed across sequences -- a failure mode a
    single-sequence run cannot see.
    """
    prompts = [_text_only_prompt(text) for text in CONCURRENT_TEXTS]
    request_config = {
        "input": CONCURRENT_TEXTS[0],
        "prompt_token_ids": [p["prompt_token_ids"] for p in prompts],
        "additional_information": [p["additional_information"] for p in prompts],
        "response_format": "wav",
        "expected_sample_rate": SAMPLE_RATE,
        "assert_distinct_outputs": True,
        **_duration_bounds(run_level),
    }
    OmniRunnerHandler(omni_runner).send_tokenized_tts_request(request_config)
