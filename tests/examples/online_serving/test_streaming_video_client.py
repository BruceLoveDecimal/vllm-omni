# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared client keepalive protocol and task lifecycle."""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.example]


@pytest.fixture
def client(monkeypatch):
    pytest.importorskip("av")
    pytest.importorskip("websockets")
    path = Path(__file__).resolve().parents[3] / "examples/online_serving/streaming_video_generation"
    monkeypatch.syspath_prepend(str(path))
    spec = importlib.util.spec_from_file_location("streaming_video_client_test", path / "streaming_video_client.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "-inf"])
def test_invalid_ping_interval(client, monkeypatch, value):
    monkeypatch.setattr(sys, "argv", ["client", "--ping-interval", value])
    with pytest.raises(SystemExit):
        client.parse_args()


@pytest.mark.parametrize("interval", [0.0, 0.001])
def test_pings_start_after_video_start_and_stop_on_cleanup(client, monkeypatch, tmp_path, interval):
    monkeypatch.setattr(sys, "argv", ["client", "--output", str(tmp_path / "video.mp4")])
    args = client.parse_args()
    assert args.ping_interval == 0
    args.ping_interval = interval
    sent = []
    ping_tasks = []
    original_pinger = client._run_session_pings

    async def pinger(*args):
        task = asyncio.current_task()
        assert task is not None
        ping_tasks.append(task)
        await original_pinger(*args)

    class WebSocket:
        def __init__(self):
            self.incoming = self.messages()

        async def recv(self):
            return await anext(self.incoming)

        async def send(self, payload):
            sent.append(json.loads(payload))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def messages(self):
            await asyncio.sleep(0.01)
            assert [msg["type"] for msg in sent] == ["session.start"]
            yield json.dumps({"type": "video.start"})
            await asyncio.sleep(0.03)
            yield json.dumps({"type": "session.done"})

        def __aiter__(self):
            return self.messages()

    monkeypatch.setattr(client, "connect", lambda *args, **kwargs: WebSocket())
    monkeypatch.setattr(client, "save_video", lambda *args, **kwargs: None)
    monkeypatch.setattr(client, "_run_session_pings", pinger)

    async def run():
        await client.stream_video(args)
        assert all(task.done() for task in ping_tasks)
        count = len(sent)
        await asyncio.sleep(0.01)
        assert len(sent) == count

    asyncio.run(run())
    assert any(msg["type"] == "session.ping" for msg in sent) == (interval > 0)
    assert len(ping_tasks) == int(interval > 0)
