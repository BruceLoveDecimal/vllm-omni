# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ring-buffer audio-code history for MOSS-TTS Local/Realtime talkers (RFC #4676).

Stage-0's repetition-penalty window only ever looks at the last
``AUDIO_HISTORY_WINDOW`` frames, but the old code grew a full ``(T, n_vq)``
tensor with ``torch.cat`` every step -- O(T) append + O(T) GPU-resident clone
per step, i.e. O(T^2) per utterance. A fixed-length ring buffer keeps the exact
same window at O(1) per step.

These tests lock the two properties that matter:

* **Parity** -- the materialised window is byte-for-byte the old
  ``acc[-window:]`` per-codebook lists (Realtime's penalty is live at 1.1, so any
  drift would change generated audio).
* **Boundedness** -- the buffer never grows past ``AUDIO_HISTORY_WINDOW`` frames.

CPU-only and weight-free: the ring helpers are module-level pure functions, and
``make_omni_output`` is driven on a bare talker for the payload-shape check.
"""

from __future__ import annotations

import functools

import pytest

torch = pytest.importorskip("torch")

N_VQ = 4


@functools.lru_cache(maxsize=1)
def _talker_module():
    from vllm_omni.model_executor.models.moss_tts import modeling_moss_tts_talker as mod

    return mod


def _reference_window(all_frames: list[torch.Tensor], window: int) -> list[list[int]]:
    """Old semantics: ``acc[-window:]`` reshaped to per-codebook lists."""
    if not all_frames:
        return [[] for _ in range(N_VQ)]
    acc = torch.stack(all_frames, dim=0)
    tail = acc[-window:].long().cpu().tolist()
    return [[row[cb] for row in tail] for cb in range(N_VQ)]


class TestRingBufferParity:
    @pytest.mark.parametrize("num_frames", [0, 1, 5, 49, 50, 51, 123, 500])
    def test_materialised_window_matches_growing_acc(self, num_frames):
        """For any utterance length, the ring window == the old ``acc[-W:]``."""
        mod = _talker_module()
        window = mod.AUDIO_HISTORY_WINDOW
        gen = torch.Generator().manual_seed(1234 + num_frames)

        history, hist_len, hist_pos = None, 0, 0
        grown: list[torch.Tensor] = []
        for _ in range(num_frames):
            frame = torch.randint(0, 1024, (N_VQ,), generator=gen, dtype=torch.long)
            # Both views must agree on the window built from prior frames...
            ring_now = mod._materialize_code_history(history, hist_len, hist_pos, n_vq=N_VQ)
            assert ring_now == _reference_window(grown, window)
            # ...then append this frame to both.
            history, hist_len, hist_pos = mod._push_code_history(
                history, hist_len, hist_pos, frame, n_vq=N_VQ, device=torch.device("cpu")
            )
            grown.append(frame)

        assert (
            mod._materialize_code_history(history, hist_len, hist_pos, n_vq=N_VQ)
            == _reference_window(grown, window)
        )

    def test_buffer_is_fixed_size_regardless_of_length(self):
        mod = _talker_module()
        window = mod.AUDIO_HISTORY_WINDOW
        history, hist_len, hist_pos = None, 0, 0
        for i in range(5 * window):
            frame = torch.full((N_VQ,), i % 1000, dtype=torch.long)
            history, hist_len, hist_pos = mod._push_code_history(
                history, hist_len, hist_pos, frame, n_vq=N_VQ, device=torch.device("cpu")
            )
        assert tuple(history.shape) == (window, N_VQ)  # never grew
        assert hist_len == window  # saturated
        assert 0 <= hist_pos < window

    def test_wrapped_window_is_chronological(self):
        """After wrap, materialised order is oldest -> newest."""
        mod = _talker_module()
        window = mod.AUDIO_HISTORY_WINDOW
        history, hist_len, hist_pos = None, 0, 0
        total = window + 7  # force a wrap
        for i in range(total):
            frame = torch.full((N_VQ,), i, dtype=torch.long)  # frame value == its index
            history, hist_len, hist_pos = mod._push_code_history(
                history, hist_len, hist_pos, frame, n_vq=N_VQ, device=torch.device("cpu")
            )
        window_cb0 = mod._materialize_code_history(history, hist_len, hist_pos, n_vq=N_VQ)[0]
        # Last ``window`` indices, in order.
        assert window_cb0 == list(range(total - window, total))

    def test_empty_history_returns_empty_lists(self):
        mod = _talker_module()
        assert mod._materialize_code_history(None, 0, 0, n_vq=N_VQ) == [[] for _ in range(N_VQ)]
        # A live buffer with zero recorded frames is also empty.
        buf = torch.zeros((mod.AUDIO_HISTORY_WINDOW, N_VQ), dtype=torch.long)
        assert mod._materialize_code_history(buf, 0, 0, n_vq=N_VQ) == [[] for _ in range(N_VQ)]

    def test_push_allocates_long_buffer_on_first_frame(self):
        mod = _talker_module()
        frame = torch.tensor([3, 1, 4, 1], dtype=torch.long)
        history, hist_len, hist_pos = mod._push_code_history(
            None, 0, 0, frame, n_vq=N_VQ, device=torch.device("cpu")
        )
        assert history.dtype == torch.long
        assert tuple(history.shape) == (mod.AUDIO_HISTORY_WINDOW, N_VQ)
        assert hist_len == 1 and hist_pos == 1
        assert torch.equal(history[0], frame)


class TestLocalPayloadShape:
    """W2a: Local make_omni_output emits only ``codes.audio`` (no dead ref echo)."""

    def _bare_local_talker(self):
        from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_talker import (
            MossTTSLocalTalkerForGeneration,
        )

        talker = MossTTSLocalTalkerForGeneration.__new__(MossTTSLocalTalkerForGeneration)
        talker.n_vq = N_VQ
        talker._batch_state = None
        talker._batch_state_spans = None
        return talker

    def test_output_has_audio_only_no_ref(self):
        talker = self._bare_local_talker()
        hidden = torch.zeros((1, 1), dtype=torch.float32)
        # One request emitting a single frame this step.
        info = {
            "audio_state": {"is_stopping": False, "step": 1, "max_new_frames": -1},
            "audio_codes": {"current": torch.arange(N_VQ, dtype=torch.long).reshape(1, N_VQ), "emit": True},
            # A ref grid is present on the request but must NOT be echoed out.
            "codes": {"ref": torch.ones((3, N_VQ), dtype=torch.long)},
        }
        result = talker.make_omni_output(hidden, runtime_additional_information=[info])
        codes = result.multimodal_outputs["codes"]
        assert set(codes.keys()) == {"audio"}  # no "ref" key
        assert len(codes["audio"]) == 1
        assert torch.equal(codes["audio"][0], torch.arange(N_VQ, dtype=torch.long).reshape(1, N_VQ))

    def test_no_emit_returns_empty_payload(self):
        talker = self._bare_local_talker()
        hidden = torch.zeros((1, 1), dtype=torch.float32)
        info = {
            "audio_state": {"is_stopping": True, "step": 5, "max_new_frames": -1},
            "audio_codes": {"current": torch.empty((0, N_VQ), dtype=torch.long), "emit": False},
        }
        result = talker.make_omni_output(hidden, runtime_additional_information=[info])
        assert result.multimodal_outputs == {}
