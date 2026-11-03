# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the LingBot Director pipeline.

The world is scripted, so these tests pin the order of overlapping work: which
model runs while which other model is still busy. The HTTP clients run against
mock transports and the WebSocket client against a real server sending real
fragmented MP4.
"""

from __future__ import annotations

# ruff: noqa: E402, I001
import asyncio
import base64
import io
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from PIL import Image

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.lingbot_director.clients import (
    Chunk,
    EditClient,
    SegmentClient,
    VlmClient,
    WorldSession,
    decode_frames,
    fit_image,
    parse_json_reply,
)
from benchmarks.lingbot_director.director import ChunkRecord, Director, longest_run, summarize
from benchmarks.lingbot_director.state_machine import CONFIRM_CHUNKS, TRANSITION_LIMIT, DirectorStateMachine
from benchmarks.lingbot_director.tasks import TASKS, mask_span

OPEN = 1
CLOSED = 0
HEIGHT = 8
WIDTH = 100


def frame_with(state: int) -> np.ndarray:
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[0, 0, 0] = state
    return frame


def state_of(frame: np.ndarray) -> int:
    return int(frame[0, 0, 0])


class FakeSegmenter:
    """A wide mask for a closed door, a narrow one for an open door."""

    def __init__(self) -> None:
        self.calls = 0

    async def best_mask(self, image: np.ndarray, concept: str) -> np.ndarray:
        self.calls += 1
        mask = np.zeros(image.shape[:2], dtype=bool)
        mask[:, : (40 if state_of(image) == OPEN else 100)] = True
        await asyncio.sleep(0)
        return mask


class FakeVlm:
    def __init__(self, *, select_gate: asyncio.Event | None = None, check: dict | None = None) -> None:
        self.select_gate = select_gate
        self.check = check or {"state_reached": True, "scene_preserved": True, "feedback": ""}

    async def ask_json(self, system: str, images: list[np.ndarray], text: str) -> dict[str, Any]:
        if "execute_prompt" in system:
            return {"execute_prompt": "open", "hold_prompt": "hold"}
        if "candidate frames" in system:
            if self.select_gate is not None:
                await self.select_gate.wait()
            return {"best": len(images) - 1, "scene_preserved": True}
        if "edit_prompt" in system:
            return {"edit_prompt": "swing the door open"}
        return self.check


class FakeEditor:
    def __init__(self, gate: asyncio.Event | None = None) -> None:
        self.gate = gate
        self.calls = 0

    async def edit(self, image: np.ndarray, prompt: str, *, seed: int) -> np.ndarray:
        self.calls += 1
        if self.gate is not None:
            await self.gate.wait()
        return frame_with(OPEN)


class FakeWorldSession:
    """Streams one scripted state per chunk and logs everything in shared order."""

    def __init__(self, world: FakeWorld, url: str, image: np.ndarray, prompt: str, num_chunks: int) -> None:
        self.world = world
        self.url = url
        self.image = image
        self.prompt = prompt
        self.num_chunks = num_chunks
        self.number = len(world.sessions)
        self.prompts: list[str] = [prompt]
        self.stopped = False
        world.sessions.append(self)

    async def __aenter__(self) -> FakeWorldSession:
        self.world.log.append(f"open-{self.number}")
        if self.url in self.world.busy:
            raise AssertionError(f"{self.url} already serves a session; stepwise worlds serve one at a time.")
        self.world.busy.add(self.url)
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        self.world.busy.discard(self.url)
        self.world.log.append(f"close-{self.number}")

    async def set_prompt(self, event_id: str, prompt: str, *, transition_chunks: int = 0) -> None:
        self.prompts.append(prompt)

    async def stop(self) -> None:
        if self.stopped:
            raise AssertionError("session stopped twice")
        self.stopped = True
        self.world.log.append(f"stop-{self.number}")

    async def chunks(self):
        states = self.world.script(self.number)
        for index in range(self.num_chunks):
            await asyncio.sleep(0.002)
            if self.stopped:
                return
            self.world.log.append(f"chunk-{self.number}")
            yield Chunk(index=index, frames=[frame_with(states[min(index, len(states) - 1)])])


class FakeWorld:
    def __init__(self, scripts: list[list[int]]) -> None:
        self.scripts = scripts
        self.sessions: list[FakeWorldSession] = []
        self.log: list[str] = []
        self.busy: set[str] = set()

    def script(self, number: int) -> list[int]:
        return self.scripts[min(number, len(self.scripts) - 1)]

    def open(self, url: str, image: np.ndarray, prompt: str, num_chunks: int) -> FakeWorldSession:
        return FakeWorldSession(self, url, image, prompt, num_chunks)


def make_director(world: FakeWorld, arm: str, *, total: int = 8, urls=("w0",), vlm=None, editor=None, **kwargs):
    return Director(
        task=TASKS["door-open"],
        arm=arm,
        seed=7,
        total_chunks=total,
        world_urls=urls,
        open_world=world.open,
        vlm=vlm or FakeVlm(),
        segmenter=FakeSegmenter(),
        editor=editor or FakeEditor(),
        **kwargs,
    )


async def chunks_reach(director: Director, count: int) -> None:
    while len(director.chunk_frames) < count:
        await asyncio.sleep(0.001)


def run_director(director: Director) -> Director:
    asyncio.run(asyncio.wait_for(director.run(frame_with(CLOSED)), timeout=5))
    return director


# ── state machine ────────────────────────────────────────────────────────


def test_machine_needs_confirmed_chunks_to_hold_and_to_recover() -> None:
    machine = DirectorStateMachine("open", "hold")
    machine.observe(True)
    machine.observe(False)
    machine.observe(True)
    assert machine.state == "TRANSITION"
    machine.observe(True)
    assert (machine.state, machine.prompt) == ("HOLD", "hold")
    for _ in range(CONFIRM_CHUNKS):
        machine.observe(False)
    assert (machine.state, machine.prompt) == ("RECOVER", "open")


def test_machine_fails_after_the_transition_budget() -> None:
    machine = DirectorStateMachine("open", "hold")
    for _ in range(TRANSITION_LIMIT):
        machine.observe(False)
    assert machine.state == "FAILED"


# ── measurement ──────────────────────────────────────────────────────────


def test_mask_span_treats_a_missing_mask_as_empty() -> None:
    assert mask_span(None, 1) == 0.0
    assert mask_span(np.zeros((4, 4), dtype=bool), 1) == 0.0


def test_door_counts_as_open_when_its_mask_disappears() -> None:
    assert TASKS["door-open"].achieved(0.0)
    assert not TASKS["beach-stand"].achieved(0.0)


def test_summary_never_counts_a_run_across_a_restart() -> None:
    chunks = [
        ChunkRecord(
            index=i, segment=segment, arrived_s=float(i), frames=1, prompt="p", achieved=True, observed_s=i + 0.5
        )
        for i, segment in enumerate([0, 0, 0, 1, 1])
    ]
    summary = summarize(chunks)
    assert summary["longest_held_in_one_segment"] == 3
    assert summary["segments"] == 2
    assert summary["median_sensor_lag_s"] == 0.5
    assert longest_run([True, None, True, True]) == 2


# ── arms ─────────────────────────────────────────────────────────────────


def test_fixed_arm_measures_but_never_controls() -> None:
    world = FakeWorld([[OPEN]])
    director = run_director(make_director(world, "fixed"))
    assert world.sessions[0].prompts == ["open"]
    assert len(world.sessions) == 1
    assert all(chunk.achieved for chunk in director.trace.chunks)


def test_fsm_arm_switches_to_the_hold_prompt_once() -> None:
    world = FakeWorld([[OPEN]])
    run_director(make_director(world, "fsm"))
    assert world.sessions[0].prompts == ["open", "hold"]
    assert len(world.sessions) == 1


def test_rebase_on_one_world_stops_before_it_reopens() -> None:
    world = FakeWorld([[CLOSED, OPEN, OPEN, OPEN, OPEN, OPEN], [OPEN]])
    director = run_director(make_director(world, "rebase"))
    assert len(world.sessions) == 2
    log = world.log
    assert log.index("stop-0") < log.index("close-0") < log.index("open-1")
    assert world.sessions[1].prompt == "hold"
    assert state_of(world.sessions[1].image) == OPEN
    assert len(director.chunk_frames) == 8
    assert sum(event["kind"] == "restart" for event in director.trace.events) == 1


def test_rebase_with_a_standby_overlaps_until_its_first_chunk() -> None:
    world = FakeWorld([[CLOSED, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN], [OPEN]])
    director = run_director(make_director(world, "rebase", total=12, urls=("w0", "w1")))
    log = world.log
    assert log.index("open-1") < log.index("stop-0")
    # The old segment stops on the very chunk that proves the standby is live, not at the end of its budget.
    assert log[log.index("chunk-1") + 1] == "stop-0"
    assert world.sessions[1].url == "w1"
    assert len(director.chunk_frames) == 12
    assert [event for event in director.trace.events if event["kind"] == "restart"][0]["overlapped"] is True


def test_slow_vlm_does_not_stop_the_world_or_the_sensor() -> None:
    async def scenario() -> tuple[int, int]:
        gate = asyncio.Event()
        world = FakeWorld([[OPEN]])
        segmenter = FakeSegmenter()
        director = make_director(world, "rebase", total=20, vlm=FakeVlm(select_gate=gate))
        director.segmenter = segmenter
        run = asyncio.create_task(director.run(frame_with(CLOSED)))
        await asyncio.wait_for(chunks_reach(director, 10), timeout=2)
        progress = (len(director.chunk_frames), segmenter.calls)
        gate.set()
        await asyncio.wait_for(run, timeout=5)
        return progress

    chunks, sensed = asyncio.run(scenario())
    # The state machine confirmed at chunk 2, so everything after that ran beside the blocked VLM call.
    assert chunks >= 10
    assert sensed >= 8


def test_grounded_world_starts_before_the_edit_finishes() -> None:
    async def scenario() -> tuple[int, Director, FakeWorld]:
        gate = asyncio.Event()
        world = FakeWorld([[CLOSED], [OPEN]])
        director = make_director(world, "grounded", total=12, editor=FakeEditor(gate))
        run = asyncio.create_task(director.run(frame_with(CLOSED)))
        await asyncio.wait_for(chunks_reach(director, 3), timeout=2)
        played_during_edit = len(director.chunk_frames)
        gate.set()
        await asyncio.wait_for(run, timeout=5)
        return played_during_edit, director, world

    played, director, world = asyncio.run(scenario())
    assert played >= 3
    assert len(world.sessions) == 2
    assert state_of(world.sessions[1].image) == OPEN
    assert world.sessions[1].prompt == "hold"
    assert len(director.chunk_frames) == 12


def test_rejected_edits_leave_the_running_world_alone() -> None:
    world = FakeWorld([[CLOSED]])
    check = {"state_reached": False, "scene_preserved": True, "feedback": "door still shut"}
    editor = FakeEditor()
    run_director(make_director(world, "grounded", vlm=FakeVlm(check=check), editor=editor))
    assert editor.calls == 3
    assert len(world.sessions) == 1


def test_chunks_of_the_old_segment_do_not_drive_the_new_hold() -> None:
    # The old segment closes the door again while the standby warms up.
    world = FakeWorld([[OPEN, OPEN, OPEN, CLOSED, CLOSED, CLOSED, CLOSED, CLOSED, CLOSED], [OPEN]])

    class SlowStandby(FakeWorldSession):
        async def __aenter__(self) -> FakeWorldSession:
            if self.number == 1:
                await asyncio.sleep(0.02)
            return await super().__aenter__()

    world.open = lambda url, image, prompt, num_chunks: SlowStandby(world, url, image, prompt, num_chunks)
    director = run_director(make_director(world, "rebase", total=12, urls=("w0", "w1")))
    assert director.machine.state == "HOLD"
    assert not any(event.get("state") == "RECOVER" for event in director.trace.events)


def test_missing_reference_concept_fails_loudly() -> None:
    class Blind(FakeSegmenter):
        async def best_mask(self, image: np.ndarray, concept: str) -> None:
            return None

    director = make_director(FakeWorld([[OPEN]]), "fixed")
    director.segmenter = Blind()
    with pytest.raises(RuntimeError, match="was not found"):
        asyncio.run(director.run(frame_with(CLOSED)))


# ── clients ──────────────────────────────────────────────────────────────


def test_parse_json_reply_accepts_a_fenced_block() -> None:
    assert parse_json_reply('```json\n{"a": 1}\n```') == {"a": 1}


def test_fit_image_keeps_the_center_of_a_portrait_source() -> None:
    source = Image.new("RGB", (300, 300), (0, 0, 0))
    source.paste((255, 0, 0), (100, 0, 200, 300))
    fitted = fit_image(source, 100, 300)
    assert fitted.shape == (300, 100, 3)
    assert fitted[150, 50, 0] > 200


def png_b64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def test_vlm_client_sends_images_as_data_urls() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200, json={"choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}]}
        )

    async def scenario() -> dict[str, Any]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await VlmClient(http, "http://vlm", "qwen").ask_json("sys", [frame_with(OPEN)], "goal")

    assert asyncio.run(scenario()) == {"ok": True}
    assert seen["model"] == "qwen"
    assert seen["chat_template_kwargs"] == {"enable_thinking": False}
    content = seen["messages"][1]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1] == {"type": "text", "text": "goal"}


def test_vlm_client_reports_a_reply_cut_off_by_max_tokens() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["max_tokens"] == 64
        return httpx.Response(
            200, json={"choices": [{"message": {"content": '{"scene": "a la'}, "finish_reason": "length"}]}
        )

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            await VlmClient(http, "http://vlm", "qwen", max_tokens=64).ask_json("sys", [frame_with(OPEN)], "goal")

    with pytest.raises(RuntimeError, match="max_tokens=64"):
        asyncio.run(scenario())


def test_edit_client_posts_multipart_and_restores_the_frame_size() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/images/edits"
        assert b'name="prompt"' in request.content
        assert b"100x8" in request.content
        return httpx.Response(200, json={"data": [{"b64_json": png_b64(Image.new("RGB", (64, 64), (9, 9, 9)))}]})

    async def scenario() -> np.ndarray:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await EditClient(http, "http://edit").edit(frame_with(OPEN), "open it", seed=1)

    assert asyncio.run(scenario()).shape == (HEIGHT, WIDTH, 3)


def test_segment_client_returns_the_top_instance_or_none() -> None:
    mask = Image.new("L", (WIDTH, HEIGHT), 255)
    replies = [{"data": [{"score": 0.9, "box": [0, 0, 1, 1], "mask": png_b64(mask)}]}, {"data": []}]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/images/segmentations"
        return httpx.Response(200, json=replies.pop(0))

    async def scenario() -> tuple[np.ndarray | None, np.ndarray | None]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = SegmentClient(http, "http://seg")
            return await client.best_mask(frame_with(OPEN), "door"), await client.best_mask(frame_with(OPEN), "door")

    found, missing = asyncio.run(scenario())
    assert found is not None and found.all()
    assert missing is None


def fragmented_mp4(num_frames: int) -> tuple[bytes, list[int]]:
    """A real fragmented stream, with the byte offset of every fragment."""
    import av

    buffer = io.BytesIO()
    container = av.open(
        buffer, mode="w", format="mp4", options={"movflags": "+frag_every_frame+empty_moov+default_base_moof"}
    )
    stream = container.add_stream("libx264", rate=16)
    stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
    stream.options = {"tune": "zerolatency", "g": "1"}
    for index in range(num_frames):
        frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), index * 30, dtype=np.uint8), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()
    media = buffer.getvalue()
    offsets = []
    position = media.find(b"moof")
    while position != -1:
        offsets.append(position - 4)
        position = media.find(b"moof", position + 4)
    return media, offsets


def test_world_session_streams_new_frames_and_sends_controls() -> None:
    from websockets.asyncio.server import serve

    media, offsets = fragmented_mp4(6)
    received: list[dict[str, Any]] = []

    async def server(websocket: Any) -> None:
        received.append(json.loads(await websocket.recv()))
        await websocket.send(json.dumps({"type": "video.start", "request_id": "r"}))
        cuts = [0, offsets[3], len(media)]
        for number, (start, end) in enumerate(zip(cuts, cuts[1:])):
            metadata = {
                "type": "video.chunk_metadata",
                "kind": "media",
                "generation_chunk_index": number,
                "started_event_ids": ["e1"] if number == 1 else [],
            }
            await websocket.send(json.dumps(metadata))
            await websocket.send(media[start:end])
            received.append(json.loads(await websocket.recv()))
        await websocket.send(json.dumps({"type": "session.done"}))

    async def scenario() -> list[Chunk]:
        chunks = []
        async with serve(server, "127.0.0.1", 0, max_size=None) as listener:
            port = listener.sockets[0].getsockname()[1]
            async with WorldSession(f"ws://127.0.0.1:{port}", {"type": "session.start"}) as session:
                async for chunk in session.chunks():
                    chunks.append(chunk)
                    if chunk.index == 0:
                        await session.set_prompt("e1", "hold")
                    else:
                        await session.stop()
        return chunks

    chunks = asyncio.run(asyncio.wait_for(scenario(), timeout=15))
    all_frames = decode_frames(media)
    assert sum(len(chunk.frames) for chunk in chunks) == len(all_frames) > 0
    assert len(chunks) == 2
    assert [chunk.started_event_ids for chunk in chunks] == [[], ["e1"]]
    assert received[0] == {"type": "session.start"}
    assert received[1]["interaction"] == {"event_id": "e1", "event": {"prompt": "hold"}, "transition_chunks": 0}
    assert received[2] == {"type": "session.stop"}


# ── calibration and the command line ─────────────────────────────────────


def test_chunk_end_frames_follow_the_short_opening_chunk() -> None:
    from benchmarks.lingbot_director.calibrate_sensor import chunk_end_frames

    assert chunk_end_frames(3) == [8, 20, 32]


def test_balanced_accuracy_weighs_both_classes() -> None:
    from benchmarks.lingbot_director.calibrate_sensor import balanced_accuracy

    assert balanced_accuracy([True, True, False, False], [True, False, False, False]) == 0.75


def test_command_line_runs_a_restart_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Real sockets and real fMP4 for the world; the three HTTP models answer from a mock transport."""
    from websockets.asyncio.server import serve

    from benchmarks.lingbot_director import run_director as cli
    from benchmarks.lingbot_world.workload import chunks_for_num_frames

    media, offsets = fragmented_mp4(6)
    starts: list[dict[str, Any]] = []

    async def world(websocket: Any) -> None:
        start = json.loads(await websocket.recv())
        starts.append(start)
        await websocket.send(json.dumps({"type": "video.start", "request_id": "r"}))
        stopped = asyncio.Event()

        async def listen() -> None:
            async for message in websocket:
                if json.loads(message)["type"] == "session.stop":
                    stopped.set()

        listener = asyncio.create_task(listen())
        cuts = [0, *offsets[1:], len(media)]
        for number in range(chunks_for_num_frames(start["num_frames"])):
            if stopped.is_set():
                break
            await asyncio.sleep(0.01)
            cut = number % (len(cuts) - 1)
            payload = media[: cuts[1]] if cut == 0 else media[cuts[cut] : cuts[cut + 1]]
            metadata = {"type": "video.chunk_metadata", "kind": "media", "generation_chunk_index": number}
            await websocket.send(json.dumps(metadata))
            await websocket.send(payload)
            if cut == len(cuts) - 2:
                break
        await websocket.send(json.dumps({"type": "session.done"}))
        listener.cancel()

    widths = iter([64, 64, 20])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/images/segmentations":
            mask = Image.new("L", (64, 64), 0)
            mask.paste(255, (0, 0, next(widths, 20), 64))
            return httpx.Response(200, json={"data": [{"score": 0.9, "box": [0, 0, 1, 1], "mask": png_b64(mask)}]})
        system = json.loads(request.content)["messages"][0]["content"]
        if "execute_prompt" in system:
            content = {"execute_prompt": "open it", "hold_prompt": "stay open"}
        else:
            content = {"best": 1, "scene_preserved": True}
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(content)}, "finish_reason": "stop"}]}
        )

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        cli.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    image = tmp_path / "door.png"
    Image.new("RGB", (128, 64), (40, 80, 120)).save(image)

    async def scenario() -> dict[str, Any]:
        async with serve(world, "127.0.0.1", 0, max_size=None) as listener:
            port = listener.sockets[0].getsockname()[1]
            args = cli.parse_args(
                [
                    "--task", "door-open", "--arm", "rebase", "--seed", "11", "--chunks", "5",
                    "--image", str(image), "--output", str(tmp_path / "runs"),
                    "--width", "64", "--height", "64",
                    "--world-url", f"ws://127.0.0.1:{port}/v1/realtime/video",
                ]
            )  # fmt: skip
            return await cli.run(args)

    summary = asyncio.run(asyncio.wait_for(scenario(), timeout=30))
    episode = tmp_path / "runs" / "door-open-11-rebase"
    trace = json.loads((episode / "trace.json").read_text())
    assert summary["chunks"] == 5
    assert summary["segments"] == 2
    assert (episode / "rollout.mp4").stat().st_size > 0
    assert (episode / "keyframe-0.png").exists()
    assert starts[0]["prompt"] == "open it"
    assert starts[1]["prompt"] == "stay open"
    assert starts[1]["image_reference"]["image_url"].startswith("data:image/png;base64,")
    assert len(starts[1]["extra_params"]["camera_action_script"]) == chunks_for_num_frames(starts[1]["num_frames"])
    assert any(event["kind"] == "restart" for event in trace["events"])


def test_world_session_pings_and_reports_a_server_side_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stall timeout ends a rollout with a stopped session.done; it must not pass for a normal end."""
    from websockets.asyncio.server import serve

    monkeypatch.setattr(WorldSession, "keepalive_seconds", 0.01)
    received: list[str] = []

    async def server(websocket: Any) -> None:
        received.append(json.loads(await websocket.recv())["type"])
        await websocket.send(json.dumps({"type": "video.start", "request_id": "r"}))
        received.append(json.loads(await websocket.recv())["type"])
        await websocket.send(json.dumps({"type": "session.done", "chunks": 0, "stopped": True}))

    async def scenario() -> None:
        async with serve(server, "127.0.0.1", 0, max_size=None) as listener:
            port = listener.sockets[0].getsockname()[1]
            async with WorldSession(f"ws://127.0.0.1:{port}", {"type": "session.start"}) as session:
                async for _ in session.chunks():
                    pass

    with pytest.raises(RuntimeError, match="stopped the session on its own"):
        asyncio.run(asyncio.wait_for(scenario(), timeout=10))
    assert received == ["session.start", "session.ping"]
