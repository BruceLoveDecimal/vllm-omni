# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the dots.tts talker's stop-signal pairing and
per-request state lifecycle.

Pure-CPU tests on a bare talker (``__new__`` + the handful of attributes
the methods under test read) — no weights, no GPU, no engine.  Mirrors
tests/model_executor/models/voxcpm2/test_talker_state_eviction.py.
"""

from __future__ import annotations

import functools
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


torch = pytest.importorskip("torch")

_VOCAB_SIZE = 151672


@functools.lru_cache(maxsize=1)
def _dots_tts_talker_mod():
    """Defer talker import (pulls vLLM model_executor) until first use."""
    from vllm_omni.model_executor.models.dots_tts.dots_tts_talker import (
        DotsTTSForConditionalGeneration,
        _RequestState,
    )

    return DotsTTSForConditionalGeneration, _RequestState


def _make_bare_talker():
    DotsTTSForConditionalGeneration, _ = _dots_tts_talker_mod()
    talker = DotsTTSForConditionalGeneration.__new__(DotsTTSForConditionalGeneration)
    talker.config = SimpleNamespace(vocab_size=_VOCAB_SIZE)
    talker._active_states = {}
    talker._pending_requests = []
    talker._results_queue = []
    talker._audio_queue = []
    talker._deferred_cleanup_ids = set()
    talker._beta_trace = False
    return talker


def _add_state(talker, req_id: str, *, is_stopping: bool = False):
    _, _RequestState = _dots_tts_talker_mod()
    state = _RequestState(request_id=req_id)
    state.prefill_completed = True
    state.is_stopping = is_stopping
    talker._active_states[req_id] = state
    return state


def _hidden(bsz: int) -> torch.Tensor:
    return torch.zeros(bsz, 1536, dtype=torch.float32)


class TestComputeLogitsPairing:
    """compute_logits pairs _results_queue entries with batch rows
    positionally, so every request must push exactly one entry per step —
    including prefills, which push a (req_id, None) placeholder."""

    def test_mixed_batch_prefill_placeholder_keeps_rows_aligned(self) -> None:
        talker = _make_bare_talker()
        _add_state(talker, "prefill-req")
        decode_state = _add_state(talker, "decode-req", is_stopping=True)
        decode_state.precomputed_stop_logits = torch.tensor([[0.1, 0.9]])

        # Row 0 is the prefill (None placeholder), row 1 the stopping decode.
        talker._results_queue = [
            ("prefill-req", None),
            ("decode-req", torch.tensor([[0.1, 0.9]])),
        ]
        logits = talker.compute_logits(_hidden(2))

        assert logits.shape == (2, _VOCAB_SIZE)
        # Prefill row: forced continue.
        assert logits[0, 0].item() == 1.0
        assert logits[0, 1].item() == float("-inf")
        # Stopping decode row: stop slot wins.
        assert logits[1, 1].item() == 1.0
        assert logits[1, 0].item() == 0.0
        # Everything else stays masked.
        assert logits[0, 2].item() == float("-inf")
        assert logits[1, 2].item() == float("-inf")
        # Queue drained.
        assert talker._results_queue == []

    def test_below_threshold_stop_logits_force_continue(self) -> None:
        """Raw (continue, stop) softmax must not reach the sampler: a
        prob_stop in (0.5, 0.8] would flip greedy argmax even though
        is_stopping stayed False (the 0.8-threshold contract)."""
        talker = _make_bare_talker()
        _add_state(talker, "req", is_stopping=False)
        talker._results_queue = [("req", torch.tensor([[0.3, 0.7]]))]

        logits = talker.compute_logits(_hidden(1))

        assert logits[0, 0].item() == 1.0
        assert logits[0, 1].item() == float("-inf")

    def test_empty_queue_defaults_to_all_continue(self) -> None:
        talker = _make_bare_talker()
        logits = talker.compute_logits(_hidden(3))

        assert torch.all(logits[:, 0] == 1.0)
        assert torch.all(logits[:, 1] == float("-inf"))


class TestStateEviction:
    """on_requests_finished fires BEFORE forward(), so eviction must be
    deferred to _flush_deferred_cleanup at the end of the step — the
    finishing request's audio still drains through the current forward."""

    def test_finished_request_survives_until_flush(self) -> None:
        talker = _make_bare_talker()
        _add_state(talker, "req-a")
        _add_state(talker, "req-b")

        talker.on_requests_finished(["req-a"])
        assert "req-a" in talker._active_states  # not evicted yet

        talker._flush_deferred_cleanup()
        assert "req-a" not in talker._active_states
        assert "req-b" in talker._active_states
        assert talker._deferred_cleanup_ids == set()

    def test_unknown_request_id_is_ignored(self) -> None:
        talker = _make_bare_talker()
        talker.on_requests_finished(["never-seen"])
        talker._flush_deferred_cleanup()
        assert talker._active_states == {}


class TestPromptPrefillSeeding:
    """Voice-clone prompt prefill seeds the DiT history with the reference's
    latents before the AR loop starts.  The buffer must reach the decode
    loop in the same ``[hidden, latent, hidden, latent, ...]`` layout a
    zero-shot request builds one step at a time (upstream ``_prefill``,
    model.py:1163) — a shifted hidden here silently detunes every patch."""

    @staticmethod
    def _record_appends(talker):
        calls: list[tuple[str, object]] = []
        talker._append_hidden_chunk = lambda _state, chunk: calls.append(("hidden", chunk))
        talker._append_history_chunk = lambda _state, chunk: calls.append(("latent", chunk))
        return calls

    def test_interleaves_prompt_hiddens_and_latents(self) -> None:
        talker = _make_bare_talker()
        state = _add_state(talker, "req")
        state.fm_seq_len = 0
        state.fm_capacity = 1024 * 5
        # Distinguishable per-position hiddens: 8 prefill tokens, of which
        # the last 4 are <audio_gen_start> + 3 <audio_gen_span>.
        req_hidden = torch.arange(8, dtype=torch.float32).reshape(8, 1).repeat(1, 1536)
        state.prompt_patches = torch.arange(3, dtype=torch.float32).reshape(1, 3, 1, 1).repeat(1, 1, 4, 128)
        calls = self._record_appends(talker)

        talker._seed_prompt_fm_history(state, req_hidden)

        assert [kind for kind, _ in calls] == ["hidden", "latent"] * 3
        # Hidden #i is the token *before* prompt span #i: positions 4, 5, 6.
        # Position 7 (the last span) is appended by _finish_decode itself.
        assert [chunk[0, 0, 0].item() for kind, chunk in calls if kind == "hidden"] == [4.0, 5.0, 6.0]
        assert [chunk[0, 0, 0].item() for kind, chunk in calls if kind == "latent"] == [0.0, 1.0, 2.0]
        # Consumed exactly once — a second prefill step must not re-seed.
        assert state.prompt_patches is None

    def test_rejects_a_prefill_shorter_than_the_prompt_spans(self) -> None:
        talker = _make_bare_talker()
        state = _add_state(talker, "req")
        state.fm_capacity = 1024 * 5
        state.prompt_patches = torch.zeros(1, 3, 4, 128)
        self._record_appends(talker)

        with pytest.raises(ValueError, match="expected at least 4 prefill hiddens"):
            talker._seed_prompt_fm_history(state, torch.zeros(2, 1536))

    def test_rejects_a_reference_that_overflows_the_fm_buffer(self) -> None:
        talker = _make_bare_talker()
        state = _add_state(talker, "req")
        state.fm_seq_len = 0
        state.fm_capacity = 10  # two patches' worth
        state.prompt_patches = torch.zeros(1, 8, 4, 128)
        self._record_appends(talker)

        with pytest.raises(ValueError, match="reference audio is too long"):
            talker._seed_prompt_fm_history(state, torch.zeros(9, 1536))


class TestPromptTailPatchIsDropped:
    """Prompt prefill regenerates the reference's final patch as its first
    sampled patch; upstream discards it (model.py:1459).  It must still
    drive the AR loopback and the DiT history, and the vocoder must not see
    it — otherwise the reply opens with a re-synthesis of the reference."""

    @staticmethod
    def _stub_side_path(talker):
        vocoder_calls: list[object] = []
        talker._initialize_request_fm_state = lambda state, *, device, dtype: None
        talker._append_hidden_chunk = lambda *_args: None
        talker._append_history_chunk = lambda *_args: None
        talker._run_dit_n_step_euler = lambda _state: torch.zeros(1, 4, 128)
        talker._io_helper = SimpleNamespace(denormalize=lambda x: x)
        talker._run_patch_encoder_loopback = lambda _state, _patch: torch.zeros(1, 1, 1536)

        def _vocoder(state, patch):
            vocoder_calls.append(patch)
            return torch.ones(1, 1, 128)

        talker._run_vocoder_stream_step = _vocoder
        return vocoder_calls

    def _prefill_state(self, talker, *, drop: bool):
        state = _add_state(talker, "req")
        state.prefill_completed = False
        state.fm_sequence = torch.zeros(1, 5120, 1024)
        state.fm_seq_len = 0
        state.fm_capacity = 5120
        state.drop_next_patch = drop
        return state

    def test_dropped_patch_emits_no_audio_but_advances_the_ar_loop(self) -> None:
        talker = _make_bare_talker()
        vocoder_calls = self._stub_side_path(talker)
        state = self._prefill_state(talker, drop=True)

        talker._finish_decode("req", torch.zeros(1, 1536), is_prefill=True)

        assert talker._audio_queue == []
        assert vocoder_calls == []
        assert state.curr_embed_for_next is not None  # AR loopback still ran
        assert state.drop_next_patch is False  # only the first patch is dropped
        assert talker._results_queue == [("req", None)]

    def test_zero_shot_prefill_still_emits_its_first_patch(self) -> None:
        talker = _make_bare_talker()
        vocoder_calls = self._stub_side_path(talker)
        self._prefill_state(talker, drop=False)

        talker._finish_decode("req", torch.zeros(1, 1536), is_prefill=True)

        assert len(vocoder_calls) == 1
        assert [req_id for req_id, _ in talker._audio_queue] == ["req"]


class TestReferenceAudioUnwrapping:
    def test_accepts_the_nested_serving_envelope(self) -> None:
        from vllm_omni.model_executor.models.dots_tts.dots_tts_talker import (
            _unwrap_reference_audio,
        )

        samples, sample_rate = _unwrap_reference_audio([[[0.1, 0.2], 24000]])
        assert (samples, sample_rate) == ([0.1, 0.2], 24000)

    @pytest.mark.parametrize("ref", [[[[0.1], 0]], "not-audio", [[0.1, 0.2, 0.3]]])
    def test_rejects_malformed_references(self, ref) -> None:
        from vllm_omni.model_executor.models.dots_tts.dots_tts_talker import (
            _unwrap_reference_audio,
        )

        with pytest.raises(ValueError):
            _unwrap_reference_audio(ref)
