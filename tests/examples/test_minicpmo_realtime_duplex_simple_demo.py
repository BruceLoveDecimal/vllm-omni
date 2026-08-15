import asyncio
import base64
import importlib.util
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

DEMO_PATH = Path(__file__).resolve().parents[2] / "examples/online_serving/minicpmo/realtime_duplex_demo.py"


def _load_demo_module():
    spec = importlib.util.spec_from_file_location(
        "minicpmo_realtime_duplex_simple_demo_test",
        DEMO_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_post_commit_decision_ignores_streaming_listens_before_commit_ack():
    demo = _load_demo_module()
    events = [
        {"type": "session.created"},
        {"type": "response.listen"},
        {"type": "response.listen"},
        {"type": "input_audio_buffer.committed"},
    ]

    commit_index = demo._input_committed_index(events, after_index=0)

    assert commit_index == 3
    assert demo._post_commit_model_decision(events, commit_index) is None


def test_post_commit_decision_accepts_listen_after_commit_ack():
    demo = _load_demo_module()
    events = [
        {"type": "response.listen"},
        {"type": "input_audio_buffer.committed"},
        {"type": "response.listen"},
    ]

    commit_index = demo._input_committed_index(events, after_index=0)

    assert demo._post_commit_model_decision(events, commit_index) == "listen"


def test_post_commit_decision_accepts_drained_speak_after_commit_ack():
    demo = _load_demo_module()
    events = [
        {"type": "input_audio_buffer.committed"},
        {"type": "response.created", "response": {"id": "resp-a"}},
        {"type": "response.audio.delta", "response_id": "resp-a", "delta": "AAAA"},
        {"type": "response.done", "response_id": "resp-a"},
    ]

    commit_index = demo._input_committed_index(events, after_index=0)

    assert demo._post_commit_model_decision(events, commit_index) == "speak"


def test_post_commit_decision_ignores_cancelled_response():
    demo = _load_demo_module()
    events = [
        {"type": "input_audio_buffer.committed"},
        {"type": "response.done", "response": {"status": "cancelled"}},
    ]

    commit_index = demo._input_committed_index(events, after_index=0)

    assert demo._post_commit_model_decision(events, commit_index) is None


def test_commit_ack_search_starts_after_stream_cursor():
    demo = _load_demo_module()
    events = [
        {"type": "input_audio_buffer.committed"},
        {"type": "response.listen"},
        {"type": "input_audio_buffer.committed"},
    ]

    assert demo._input_committed_index(events, after_index=1) == 2


def test_exact_model_unit_boundary_does_not_require_an_extra_decision():
    demo = _load_demo_module()
    unit_bytes = demo.PCM16_SAMPLE_RATE * demo.PCM16_BYTES_PER_SAMPLE

    assert not demo._has_residual_model_unit(b"\0" * (unit_bytes * 6), chunk_period_ms=1000)


def test_partial_model_unit_requires_a_post_commit_decision():
    demo = _load_demo_module()
    unit_bytes = demo.PCM16_SAMPLE_RATE * demo.PCM16_BYTES_PER_SAMPLE

    assert demo._has_residual_model_unit(b"\0" * (unit_bytes * 6 + 512), chunk_period_ms=1000)


def test_latest_streaming_decision_is_used_at_exact_boundary():
    demo = _load_demo_module()
    events = [
        {"type": "response.listen"},
        {"type": "response.listen"},
        {"type": "input_audio_buffer.committed"},
    ]

    assert demo._latest_model_decision(events, after_index=0) == "listen"


def test_ref_audio_data_url_encodes_explicit_wav(tmp_path):
    demo = _load_demo_module()
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFFfake")

    assert demo._ref_audio_data_url(str(ref)) == "data:audio/wav;base64,UklGRmZha2U="
    assert demo._ref_audio_data_url(None) is None


def test_realtime_duplex_demo_requires_ref_audio(monkeypatch):
    demo = _load_demo_module()
    monkeypatch.setattr(
        demo.sys,
        "argv",
        [
            "realtime_duplex_demo.py",
            "--input-wav",
            "input.wav",
        ],
    )

    with pytest.raises(SystemExit):
        demo.parse_args()


def test_realtime_duplex_demo_accepts_explicit_ref_audio(monkeypatch):
    demo = _load_demo_module()
    monkeypatch.setattr(
        demo.sys,
        "argv",
        [
            "realtime_duplex_demo.py",
            "--input-wav",
            "input.wav",
            "--ref-audio",
            "ref.wav",
        ],
    )

    args = demo.parse_args()

    assert args.ref_audio == "ref.wav"
    assert args.await_responses == 0
    assert args.max_trailing_silence_ms == 30_000


def test_realtime_duplex_demo_rejects_negative_await_responses(monkeypatch):
    demo = _load_demo_module()
    monkeypatch.setattr(
        demo.sys,
        "argv",
        [
            "realtime_duplex_demo.py",
            "--input-wav",
            "input.wav",
            "--ref-audio",
            "ref.wav",
            "--await-responses",
            "-1",
        ],
    )

    with pytest.raises(SystemExit):
        demo.parse_args()


def test_realtime_duplex_demo_handoff_completion_requires_done_and_trailing_listen():
    demo = _load_demo_module()
    listen = {"type": "response.listen"}
    created_a = {"type": "response.created", "response": {"id": "resp-a"}}
    done_a = {"type": "response.done", "response_id": "resp-a"}
    created_b = {"type": "response.created", "response": {"id": "resp-b"}}
    done_b = {"type": "response.done", "response_id": "resp-b"}

    assert not demo._handoff_complete([listen, created_a, done_a, listen], 2)
    assert not demo._handoff_complete([created_a, done_a, listen, created_b], 2)
    assert not demo._handoff_complete([listen, created_a, done_a, listen, created_b, done_b], 2)
    assert demo._handoff_complete([created_a, done_a, listen, created_b, done_b, listen], 2)


def test_realtime_duplex_demo_trailing_silence_runway_paces_until_handoff(monkeypatch):
    demo = _load_demo_module()

    class _FakeClient:
        def __init__(self) -> None:
            self.events = SimpleNamespace(events=[])
            self.sent: list[dict[str, object]] = []

        async def send(self, event: dict[str, object]) -> None:
            self.sent.append(event)
            if len(self.sent) == 2:
                self.events.events = [
                    {"type": "response.created", "response": {"id": "resp-a"}},
                    {"type": "response.done", "response_id": "resp-a"},
                    {"type": "response.created", "response": {"id": "resp-b"}},
                    {"type": "response.done", "response_id": "resp-b"},
                    {"type": "response.listen"},
                ]

    async def _instant_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(demo.asyncio, "sleep", _instant_sleep)
    client = _FakeClient()

    trailing, error = asyncio.run(
        demo._stream_trailing_silence_until_handoff(
            client,
            input_audio_end_ms=25_600,
            min_responses=2,
            chunk_ms=200,
            max_trailing_silence_ms=1_000,
        )
    )

    assert error is None
    assert trailing == {"sent_ms": 400, "max_ms": 1_000, "handoff_complete": True}
    assert all(event["type"] == "input_audio_buffer.append" for event in client.sent)
    assert [event["audio_end_ms"] for event in client.sent] == [25_800, 26_000]


def test_realtime_duplex_demo_trailing_silence_runway_caps_out(monkeypatch):
    demo = _load_demo_module()

    class _FakeClient:
        def __init__(self) -> None:
            self.events = SimpleNamespace(events=[])
            self.sent: list[dict[str, object]] = []

        async def send(self, event: dict[str, object]) -> None:
            self.sent.append(event)

    async def _instant_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(demo.asyncio, "sleep", _instant_sleep)
    client = _FakeClient()

    trailing, error = asyncio.run(
        demo._stream_trailing_silence_until_handoff(
            client,
            input_audio_end_ms=0,
            min_responses=2,
            chunk_ms=200,
            max_trailing_silence_ms=600,
        )
    )

    assert error is not None and "handoff incomplete" in error
    assert trailing == {"sent_ms": 600, "max_ms": 600, "handoff_complete": False}
    assert len(client.sent) == 3


def test_open_streaming_response_requires_post_commit_drain():
    demo = _load_demo_module()

    assert demo._response_in_progress([{"type": "response.created"}])
    assert not demo._response_in_progress(
        [
            {"type": "response.created"},
            {"type": "response.done"},
        ]
    )


def test_streaming_output_writer_persists_audio_deltas_as_they_arrive(tmp_path, capsys):
    demo = _load_demo_module()
    writer = demo._StreamingOutputWriter(tmp_path)
    first_pcm = b"\x01\x00\x02\x00"
    second_pcm = b"\x03\x00"

    writer.handle(
        {
            "type": "response.audio.delta",
            "response_id": "resp-a",
            "delta": base64.b64encode(first_pcm).decode("ascii"),
            "sample_rate_hz": 24_000,
        }
    )
    writer.handle(
        {
            "type": "response.audio_transcript.delta",
            "response_id": "resp-a",
            "delta": "你好",
        }
    )
    writer.handle(
        {
            "type": "response.audio.delta",
            "response_id": "resp-a",
            "delta": base64.b64encode(second_pcm).decode("ascii"),
            "sample_rate_hz": 24_000,
        }
    )

    assert (tmp_path / "output.pcm").read_bytes() == first_pcm + second_pcm
    chunk_paths = sorted((tmp_path / "audio_chunks").glob("*.wav"))
    assert [path.name for path in chunk_paths] == ["chunk_0001.wav", "chunk_0002.wav"]
    with wave.open(str(chunk_paths[0]), "rb") as chunk_wav:
        assert chunk_wav.getframerate() == 24_000
        assert chunk_wav.readframes(chunk_wav.getnframes()) == first_pcm
    assert writer.audio_chunk_paths == chunk_paths
    stderr = capsys.readouterr().err
    assert "audio chunk 1" in stderr
    assert "你好" in stderr
