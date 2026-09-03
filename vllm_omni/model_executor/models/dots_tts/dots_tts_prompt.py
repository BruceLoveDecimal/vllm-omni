# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""dots.tts prompt builder for vllm-omni's Omni engine + serving layer.

Mirrors upstream ``build_generation_schedule`` (rednote-hilab/dots.tts @
a393d2e data/pipelines/tokenizing.py:169) for the audio-placeholder
template, covering all three conditioning modes:

* **zero-shot** — ``[文本]<text>[文本对应语音]<audio_gen_start>``.
* **reference audio only** (``ref_audio``, no ``ref_text``) — identical
  token sequence; the reference waveform rides in
  ``additional_information`` and the talker turns it into a CAM++
  x-vector that conditions the DiT (``g_cond``).  Upstream calls this
  ``use_prompt_prefill=False`` (model.py:1403).
* **prompt prefill / voice clone** (``ref_audio`` + ``ref_text``) —
  reference text is prepended to the target text and the sequence is
  extended by ``prompt_patch_count`` ``<audio_gen_span>`` slots, whose
  embeddings the talker overwrites with patch-encoder outputs of the
  reference latents (upstream ``_build_prefill_inputs_embeds``,
  model.py:1116).

The token IDs are real Qwen2 IDs (not placeholders ``[1] * N`` like
voxcpm2's builder): the talker's preprocess() runs
``self.model.embed_tokens(input_ids)`` directly, then patches only the
prompt-span rows, so real IDs flow through naturally.
"""

from __future__ import annotations

import math
from typing import Any

from vllm.inputs import tokens_input

# Upstream rednote-hilab/dots.tts @ a393d2e
# data/pipelines/tts_pipeline.py:17-18
_TTS_TEXT_PREFIX = "[文本]"
_TTS_AUDIO_PREFIX = "[文本对应语音]"
_AUDIO_GEN_START_TOKEN = "<|audio_gen_start|>"
_AUDIO_GEN_SPAN_TOKEN = "<|audio_gen_span|>"

#: Per-request FM workspace capacity in the talker, in audio patches.  Prompt
#: patches and generated patches share it, so the serving layer sizes a
#: request's token budget against this too — hence its home here, in the
#: module both the talker and the serving adapter already import.
MAX_AUDIO_PATCHES = 1024


def prompt_audio_plan(
    num_samples: int,
    sample_rate: int,
    *,
    samples_per_patch: int,
    target_sample_rate: int,
) -> tuple[int, int]:
    """Plan the reference-audio prefill: ``(prompt_patch_count, target_samples)``.

    Both the serving layer (which sizes ``prompt_token_ids``) and the
    talker (which resamples and encodes the waveform) call this, so the
    number of ``<audio_gen_span>`` slots always matches the number of
    prompt patches the talker produces.  Resampling length is *not*
    inferred from a resampler's rounding: the talker pads or truncates to
    exactly ``target_samples``, which is the whole point of returning it.

    Mirrors upstream ``_prepare_prompt_audio_for_conditioning``
    (model.py:794): pad up to a whole number of audio patches, encode,
    then drop the final patch (``prompt_latents_sampled[:, :-patch_size]``
    at model.py:924) — the model regenerates that tail patch as its first
    decode step and it is discarded.
    """
    if num_samples <= 0 or sample_rate <= 0:
        raise ValueError(f"Invalid reference audio: num_samples={num_samples} sample_rate={sample_rate}")
    resampled = math.ceil(num_samples * target_sample_rate / sample_rate)
    num_patches = max(1, math.ceil(resampled / samples_per_patch))
    return num_patches - 1, num_patches * samples_per_patch


def build_dots_tts_prompt(
    tokenizer: Any,
    text: str,
    *,
    ref_audio: list[float] | None = None,
    ref_sr: int | None = None,
    ref_text: str | None = None,
    prompt_patch_count: int = 0,
    prompt_audio_samples: int = 0,
    ref_audio_key: str | None = None,
) -> dict[str, Any]:
    """Build a dots.tts prefill prompt dict for ``Omni.generate()``.

    ``prompt_patch_count`` / ``prompt_audio_samples`` come from
    :func:`prompt_audio_plan` and are required together with
    ``ref_text``; they are ignored in the reference-audio-only mode.
    ``ref_audio_key`` is an opaque content identity for the waveform (the
    serving layer's resolved-ref-audio cache key) and keys the talker's
    cross-request conditioning cache — omit it and the talker re-encodes
    the reference on every request.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"text must be a non-empty string, got {text!r}")
    if ref_text is not None and ref_audio is None:
        raise ValueError("ref_text requires ref_audio (upstream: prompt_text requires prompt_audio_path).")
    if ref_audio is not None and not ref_sr:
        raise ValueError("ref_sr is required when ref_audio is provided.")

    use_prompt_prefill = ref_audio is not None and bool(ref_text and ref_text.strip())
    if use_prompt_prefill and prompt_patch_count < 1:
        raise ValueError(
            f"Prompt prefill needs at least one prompt patch, got {prompt_patch_count}. "
            "Reference audio shorter than one audio patch cannot seed the AR loop; "
            "drop ref_text to fall back to reference-audio-only conditioning."
        )

    # Upstream concatenates prompt_text and text into a single template
    # slot (runtime.py:558 `text=f"{prompt_text}{text}"`), so the LM reads
    # the reference transcript and the target text as one utterance.
    body = f"{ref_text}{text}" if use_prompt_prefill else text
    text_ids = tokenizer.encode(f"{_TTS_TEXT_PREFIX}{body}{_TTS_AUDIO_PREFIX}", add_special_tokens=False)
    prompt_token_ids = list(text_ids) + [_require_token_id(tokenizer, _AUDIO_GEN_START_TOKEN)]

    additional: dict[str, Any] = {}
    if ref_audio is not None:
        if use_prompt_prefill:
            prompt_token_ids += [_require_token_id(tokenizer, _AUDIO_GEN_SPAN_TOKEN)] * prompt_patch_count
            additional["prompt_audio"] = [[ref_audio, int(ref_sr)]]
            additional["prompt_text"] = ref_text
            additional["prompt_patch_count"] = int(prompt_patch_count)
            additional["prompt_audio_samples"] = int(prompt_audio_samples)
        else:
            additional["reference_audio"] = [[ref_audio, int(ref_sr)]]
        if ref_audio_key:
            additional["ref_audio_key"] = ref_audio_key

    prompt = tokens_input(prompt_token_ids=prompt_token_ids)
    prompt["additional_information"] = additional
    return prompt


def _require_token_id(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if not isinstance(token_id, int) or token_id == tokenizer.unk_token_id:
        raise ValueError(
            f"Tokenizer does not know {token!r} (got id={token_id!r}).  Did you "
            "load the dots.tts tokenizer (with added_tokens.json)?"
        )
    return token_id
