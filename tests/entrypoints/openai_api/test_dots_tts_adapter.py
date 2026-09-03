# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""dots.tts serving adapter: detection, prompt sizing, and request validation.

Pure-CPU: the prompt builder and the adapter are exercised against a stub
tokenizer and a stub server, so no checkpoint, engine, or GPU is needed.
The point of most of these is the one invariant that ties the serving layer
to the talker — ``prompt_token_ids`` must carry exactly as many
``<audio_gen_span>`` slots as the talker will produce prompt patches.
"""

import asyncio
from types import SimpleNamespace

import pytest

from vllm_omni.entrypoints.openai.tts_adapters import detect_tts_model_type, resolve_adapter
from vllm_omni.entrypoints.openai.tts_adapters.dots_tts import DotsTTSAdapter
from vllm_omni.model_executor.models.dots_tts.dots_tts_prompt import (
    build_dots_tts_prompt,
    prompt_audio_plan,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_AUDIO_GEN_START_ID = 151_668
_AUDIO_GEN_SPAN_ID = 151_669
# dots.tts-soar: patch_size=4 latent frames x hop_size=1920 samples.
_SAMPLES_PER_PATCH = 7680
_SAMPLE_RATE = 48_000


class _StubTokenizer:
    unk_token_id = 0

    def encode(self, text, add_special_tokens=False):  # noqa: ARG002
        return [1000 + ord(ch) % 100 for ch in text]

    def convert_tokens_to_ids(self, token):
        return {
            "<|audio_gen_start|>": _AUDIO_GEN_START_ID,
            "<|audio_gen_span|>": _AUDIO_GEN_SPAN_ID,
        }.get(token, self.unk_token_id)


def _plan(num_samples, sample_rate=_SAMPLE_RATE):
    return prompt_audio_plan(
        num_samples,
        sample_rate,
        samples_per_patch=_SAMPLES_PER_PATCH,
        target_sample_rate=_SAMPLE_RATE,
    )


# ── detection ──


def test_dots_tts_detected_by_architecture():
    """The talker architecture wins the shared ``latent_generator`` stage."""
    assert detect_tts_model_type("latent_generator", "DotsTTSForConditionalGeneration") == "dots_tts"
    assert resolve_adapter("dots_tts") is DotsTTSAdapter


def test_dots_tts_does_not_take_the_latent_generator_stage_from_voxcpm2():
    """Claiming no stage key keeps VoxCPM2's stage-key-only deployments intact."""
    assert detect_tts_model_type("latent_generator", None) == "voxcpm2"
    assert DotsTTSAdapter.stage_keys == frozenset()


# ── prompt-audio planning ──


@pytest.mark.parametrize(
    ("num_samples", "expected_patches"),
    [
        (1, 0),  # shorter than one patch: no prompt prefill possible
        (_SAMPLES_PER_PATCH, 0),  # exactly one patch, all of it is the dropped tail
        (_SAMPLES_PER_PATCH + 1, 1),
        (10 * _SAMPLES_PER_PATCH, 9),
    ],
)
def test_prompt_audio_plan_drops_the_regenerated_tail_patch(num_samples, expected_patches):
    patches, target = _plan(num_samples)
    assert patches == expected_patches
    assert target == (expected_patches + 1) * _SAMPLES_PER_PATCH
    assert target >= num_samples


def test_prompt_audio_plan_accounts_for_resampling():
    """A 16 kHz reference is planned at its 48 kHz length, not its own."""
    patches_16k, target_16k = _plan(3 * _SAMPLES_PER_PATCH, sample_rate=16_000)
    patches_48k, target_48k = _plan(9 * _SAMPLES_PER_PATCH)
    assert (patches_16k, target_16k) == (patches_48k, target_48k)


def test_prompt_audio_plan_rejects_empty_audio():
    with pytest.raises(ValueError):
        _plan(0)


# ── prompt building ──


def test_zero_shot_prompt_has_no_audio_spans():
    prompt = build_dots_tts_prompt(_StubTokenizer(), "hello")
    assert prompt["prompt_token_ids"][-1] == _AUDIO_GEN_START_ID
    assert _AUDIO_GEN_SPAN_ID not in prompt["prompt_token_ids"]
    assert prompt["additional_information"] == {}


def test_reference_audio_only_keeps_the_zero_shot_token_sequence():
    """Without a transcript the reference conditions the DiT, not the prefill."""
    zero_shot = build_dots_tts_prompt(_StubTokenizer(), "hello")
    ref_only = build_dots_tts_prompt(
        _StubTokenizer(),
        "hello",
        ref_audio=[0.0] * (4 * _SAMPLES_PER_PATCH),
        ref_sr=_SAMPLE_RATE,
        ref_audio_key="cache-key",
    )
    assert ref_only["prompt_token_ids"] == zero_shot["prompt_token_ids"]
    additional = ref_only["additional_information"]
    assert "reference_audio" in additional
    assert "prompt_audio" not in additional
    assert additional["ref_audio_key"] == "cache-key"


def test_voice_clone_prompt_reserves_one_span_per_prompt_patch():
    patches, target = _plan(4 * _SAMPLES_PER_PATCH)
    prompt = build_dots_tts_prompt(
        _StubTokenizer(),
        "hello",
        ref_audio=[0.0] * (4 * _SAMPLES_PER_PATCH),
        ref_sr=_SAMPLE_RATE,
        ref_text="reference transcript",
        prompt_patch_count=patches,
        prompt_audio_samples=target,
    )
    ids = prompt["prompt_token_ids"]
    assert ids.count(_AUDIO_GEN_SPAN_ID) == patches
    assert ids[-patches - 1] == _AUDIO_GEN_START_ID
    assert ids[-patches:] == [_AUDIO_GEN_SPAN_ID] * patches
    additional = prompt["additional_information"]
    assert additional["prompt_patch_count"] == patches
    assert additional["prompt_audio_samples"] == target
    assert additional["prompt_text"] == "reference transcript"


def test_voice_clone_prepends_the_reference_transcript():
    """Upstream concatenates prompt_text and text into one template slot."""
    tok = _StubTokenizer()
    clone = build_dots_tts_prompt(
        tok,
        "world",
        ref_audio=[0.0] * (2 * _SAMPLES_PER_PATCH),
        ref_sr=_SAMPLE_RATE,
        ref_text="hello ",
        prompt_patch_count=1,
        prompt_audio_samples=2 * _SAMPLES_PER_PATCH,
    )
    joined = build_dots_tts_prompt(tok, "hello world")
    assert clone["prompt_token_ids"][: len(joined["prompt_token_ids"]) - 1] == joined["prompt_token_ids"][:-1]


def test_ref_text_without_audio_is_rejected():
    with pytest.raises(ValueError, match="ref_text requires ref_audio"):
        build_dots_tts_prompt(_StubTokenizer(), "hello", ref_text="transcript")


def test_voice_clone_needs_at_least_one_prompt_patch():
    with pytest.raises(ValueError, match="at least one prompt patch"):
        build_dots_tts_prompt(
            _StubTokenizer(),
            "hello",
            ref_audio=[0.0] * 128,
            ref_sr=_SAMPLE_RATE,
            ref_text="transcript",
            prompt_patch_count=0,
        )


# ── adapter ──


def _make_adapter(ref_audio_samples=None, ref_sr=_SAMPLE_RATE):
    async def _resolve_ref_audio(_locator):
        return list(ref_audio_samples or []), ref_sr, "ref-cache-key"

    server = SimpleNamespace(
        engine_client=SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    patch_size=4,
                    vocoder={"downsample_rates": [2, 2, 2, 4, 6, 10], "sample_rate": _SAMPLE_RATE},
                )
            )
        ),
        _resolve_ref_audio=_resolve_ref_audio,
        _apply_uploaded_speaker=lambda _request: None,
        _validate_ref_audio_format=lambda _ref: None,
        _get_available_speakers=lambda: {"default"},
    )
    adapter = DotsTTSAdapter(SimpleNamespace(server=server, engine_client=server.engine_client))
    adapter._tokenizer = _StubTokenizer()
    return adapter


def _request(**overrides):
    fields = {
        "input": "hello",
        "voice": None,
        "ref_audio": None,
        "ref_text": None,
        "max_new_tokens": None,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_adapter_patch_geometry_matches_the_checkpoint():
    samples_per_patch, sample_rate = _make_adapter()._audio_patch_geometry()
    assert (samples_per_patch, sample_rate) == (_SAMPLES_PER_PATCH, _SAMPLE_RATE)


def test_adapter_build_sizes_spans_from_the_resolved_reference():
    ref = [0.0] * (5 * _SAMPLES_PER_PATCH)
    adapter = _make_adapter(ref_audio_samples=ref)
    request = _request(ref_audio="file:///ref.wav", ref_text="transcript")
    prepared = asyncio.run(adapter.build(request, [], has_inline_ref_audio=True))
    expected_patches, _ = _plan(len(ref))
    assert prepared.model_type == "dots_tts"
    assert prepared.prompt["prompt_token_ids"].count(_AUDIO_GEN_SPAN_ID) == expected_patches
    assert prepared.prompt["additional_information"]["ref_audio_key"] == "ref-cache-key"


def test_adapter_build_without_ref_text_stays_reference_only():
    adapter = _make_adapter(ref_audio_samples=[0.0] * (5 * _SAMPLES_PER_PATCH))
    prepared = asyncio.run(adapter.build(_request(ref_audio="file:///ref.wav"), [], has_inline_ref_audio=True))
    assert _AUDIO_GEN_SPAN_ID not in prepared.prompt["prompt_token_ids"]
    assert "reference_audio" in prepared.prompt["additional_information"]


def test_adapter_build_rejects_an_over_long_reference():
    adapter = _make_adapter(ref_audio_samples=[0.0] * (_SAMPLE_RATE * 45))
    with pytest.raises(ValueError, match="at most"):
        asyncio.run(adapter.build(_request(ref_audio="file:///ref.wav"), [], has_inline_ref_audio=True))


def test_adapter_rejects_ref_text_without_ref_audio():
    adapter = _make_adapter()
    assert "ref_text requires ref_audio" in adapter.validate(_request(ref_text="transcript"))


def test_adapter_rejects_empty_input():
    assert _make_adapter().validate(_request(input="   ")) == "Input text cannot be empty"


def test_adapter_rejects_unknown_voice():
    error = _make_adapter().validate(_request(voice="nonexistent"))
    assert "Invalid voice" in error


def test_adapter_accepts_the_zero_shot_default_voice():
    assert _make_adapter().validate(_request(voice="DEFAULT")) is None


def test_max_tokens_budget_shrinks_by_the_prompt_patches():
    """Prompt patches and generated patches share the talker's FM buffer."""
    adapter = _make_adapter()
    params = [SimpleNamespace(max_tokens=4096)]
    prompt = {"additional_information": {"prompt_patch_count": 100}}
    updated = adapter.apply_sampling_overrides(params, _request(), prompt=prompt)
    assert updated[0].max_tokens == 1024 - 100
    assert params[0].max_tokens == 4096  # caller's list untouched


def test_max_tokens_honours_a_smaller_request_limit():
    adapter = _make_adapter()
    updated = adapter.apply_sampling_overrides(
        [SimpleNamespace(max_tokens=4096)],
        _request(max_new_tokens=64),
        prompt={"additional_information": {"prompt_patch_count": 10}},
    )
    assert updated[0].max_tokens == 64
