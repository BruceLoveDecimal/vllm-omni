# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the live Director loop: event scheduling, object cards and the browser service.

The world, the VLM and SAM are scripted, so the tests pin what reaches the
world and when, and that slow Director work never holds the world up.
"""

from __future__ import annotations

# ruff: noqa: E402, I001
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.lingbot_director.agent import DirectorAgent, Discovery, Resolution
from benchmarks.lingbot_director.clients import Chunk
from benchmarks.lingbot_director.events import CameraControl, EventScheduler, WorldEvent
from benchmarks.lingbot_director.live import LiveSession, scene_difference
from benchmarks.lingbot_director.tracker import ObjectTracker, iou

HEIGHT = 48
WIDTH = 80


def frame(value: int = 0) -> np.ndarray:
    return np.full((HEIGHT, WIDTH, 3), value, dtype=np.uint8)


def event(source: str, prompt: str, *, duration: int = 2, facts: list[str] | None = None) -> WorldEvent:
    return WorldEvent(
        event_id=prompt,
        source=source,
        request=prompt,
        prompt=prompt,
        duration_chunks=duration,
        facts_after=facts or [],
    )


# ── scheduler and camera ─────────────────────────────────────────────────


def test_prompt_is_scene_then_facts_then_active_event() -> None:
    scheduler = EventScheduler(scene="A hallway.")
    scheduler.facts = ["The door is open."]
    scheduler.submit(event("hotkey", "Birds fly past."), chunk=0)
    assert scheduler.prompt() == "A hallway. The door is open. Birds fly past."


def test_event_runs_its_duration_then_leaves_its_facts() -> None:
    scheduler = EventScheduler(scene="A hallway.")
    scheduler.submit(event("object", "The door swings open.", duration=2, facts=["The door is open."]), chunk=0)
    scheduler.on_chunk(1)
    assert scheduler.active is not None
    scheduler.on_chunk(2)
    assert scheduler.active is None
    assert scheduler.prompt() == "A hallway. The door is open."


def test_higher_priority_preempts_and_equal_priority_waits() -> None:
    scheduler = EventScheduler(scene="A hallway.")
    scheduler.submit(event("director", "Rain starts.", facts=["It rains."]), chunk=0)
    scheduler.submit(event("hotkey", "Birds fly past."), chunk=0)
    assert scheduler.active.prompt == "Birds fly past."
    # A preempted event leaves no facts behind.
    assert scheduler.facts == []
    scheduler.submit(event("slot", "A lamp turns on."), chunk=0)
    assert scheduler.active.prompt == "Birds fly past."
    assert [queued.prompt for queued in scheduler.queue] == ["A lamp turns on."]
    assert [entry["what"] for entry in scheduler.history] == ["started", "preempted", "started"]


def test_camera_sends_only_when_the_held_set_changes() -> None:
    camera = CameraControl()
    forward = camera.key("w", True)
    assert forward["mode"] == "velocity"
    assert forward["data"]["translation"][2] > 0
    assert camera.key("w", True) is None
    assert camera.key("u", True) is None
    stop = camera.key("w", False)
    assert stop["data"]["translation"] == [0.0, 0.0, 0.0]
    assert stop["data"]["rotation"] == [0.0, 0.0, 0.0, 1.0]


# ── tracker ──────────────────────────────────────────────────────────────


class ScriptedSegmenter:
    def __init__(self, replies: list[list[dict[str, Any]]]) -> None:
        self.replies = replies
        self.calls: list[tuple[str, list]] = []

    async def instances(self, image: np.ndarray, concept: str, *, boxes=None) -> list[dict[str, Any]]:
        self.calls.append((concept, boxes))
        return self.replies.pop(0) if self.replies else []


def test_tracker_follows_with_an_exemplar_and_drops_lost_objects() -> None:
    segmenter = ScriptedSegmenter([[{"score": 0.9, "box": [12, 10, 32, 40]}], [], [], []])
    tracker = ObjectTracker(segmenter)
    tracker.merge([{"name": "door", "box": [10, 10, 30, 40], "actions": ["open"]}])
    asyncio.run(tracker.update(frame()))
    assert tracker.objects["obj-0"].box == (12.0, 10.0, 32.0, 40.0)
    assert segmenter.calls[0] == ("door", [[10.0, 10.0, 30.0, 40.0]])
    for _ in range(3):
        asyncio.run(tracker.update(frame()))
    assert tracker.objects == {}


def test_tracker_merges_rediscovered_objects_and_hits_the_smallest_box() -> None:
    tracker = ObjectTracker(ScriptedSegmenter([]))
    tracker.merge([{"name": "door", "box": [10, 0, 50, 40], "actions": ["open"]}])
    tracker.merge(
        [
            {"name": "white door", "box": [11, 0, 50, 40], "actions": ["close"]},
            {"name": "handle", "box": [40, 20, 46, 26], "actions": ["press"]},
        ]
    )
    assert len(tracker.objects) == 2
    assert tracker.objects["obj-0"].actions == ["close"]
    assert tracker.hit(42, 22).name == "handle"
    assert tracker.hit(20, 5).name == "door"
    assert tracker.hit(70, 5) is None
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


# ── Director agent ───────────────────────────────────────────────────────


class ScriptedVlm:
    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self.replies = replies
        self.requests: list[tuple[str, int, str]] = []

    async def ask_json(self, system: str, images: list[np.ndarray], text: str) -> dict[str, Any]:
        self.requests.append((system, len(images), text))
        return self.replies.pop(0)


def test_discover_scales_boxes_from_the_thousand_grid() -> None:
    vlm = ScriptedVlm(
        [
            {
                "scene": "A hallway.",
                "objects": [{"name": "door", "box": [500, 0, 1000, 500], "actions": ["open"]}],
                "character_actions": ["wave"],
                "environment_events": ["rain"],
            }
        ]
    )
    discovery = asyncio.run(DirectorAgent(vlm).discover([frame(), frame()]))
    assert discovery.objects[0]["box"] == [40.0, 0.0, 80.0, 24.0]
    assert vlm.requests[0][1] == 2


def test_resolve_clamps_duration_and_reports_failed_preconditions() -> None:
    reply = {
        "target": "door",
        "action": "open",
        "preconditions": [
            {"condition": "door visible", "holds": True},
            {"condition": "handle reachable", "holds": False},
        ],
        "visual_event": "A hand presses the handle and pulls the door inward.",
        "duration_chunks": 40,
        "facts_after": ["The door is open."],
    }
    resolution = asyncio.run(DirectorAgent(ScriptedVlm([reply])).resolve(frame(), "open the door", scene="", facts=[]))
    assert resolution.duration_chunks == 6
    assert resolution.failed_preconditions == ["handle reachable"]


# ── the live loop ────────────────────────────────────────────────────────


class FakeSession:
    def __init__(self, world: FakeWorld, image: np.ndarray, prompt: str, num_chunks: int) -> None:
        self.world = world
        self.image = image
        self.prompt = prompt
        self.num_chunks = num_chunks
        self.interactions: list[dict[str, Any]] = []
        self.stopped = False
        world.sessions.append(self)

    async def __aenter__(self) -> FakeSession:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        return None

    async def interact(self, event_id: str, *, prompt=None, camera=None, transition_chunks=None) -> None:
        self.interactions.append({"prompt": prompt, "camera": camera})

    async def stop(self) -> None:
        self.stopped = True

    async def chunks(self):
        for index in range(self.num_chunks):
            await self.world.pace()
            if self.stopped:
                return
            self.world.generated += 1
            yield Chunk(index=index, frames=[frame(self.world.generated)])


class FakeWorld:
    def __init__(self, *, delay: float = 0.001) -> None:
        self.sessions: list[FakeSession] = []
        self.generated = 0
        self.delay = delay

    async def pace(self) -> None:
        await asyncio.sleep(self.delay)

    def open(self, image: np.ndarray, prompt: str, num_chunks: int) -> FakeSession:
        return FakeSession(self, image, prompt, num_chunks)


class FakeAgent:
    def __init__(self, *, resolve_gate: asyncio.Event | None = None, failed: list[str] | None = None) -> None:
        self.discoveries = 0
        self.resolve_gate = resolve_gate
        self.failed = failed or []

    async def discover(self, frames: list[np.ndarray]) -> Discovery:
        self.discoveries += 1
        return Discovery(
            scene=f"Scene {self.discoveries}.",
            objects=[{"name": "door", "box": [10, 10, 30, 40], "actions": ["open"]}],
            character_actions=["wave", "jump"],
            environment_events=["rain", "snow"],
        )

    async def resolve(self, frame: np.ndarray, request: str, *, scene: str, facts: list[str]) -> Resolution:
        if self.resolve_gate is not None:
            await self.resolve_gate.wait()
        return Resolution(
            target="door",
            action="open",
            preconditions=[{"condition": item, "holds": False} for item in self.failed],
            visual_event=f"EVENT {request}.",
            duration_chunks=2,
            facts_after=[f"After {request}."],
        )


class StillTracker(ObjectTracker):
    async def update(self, frame: np.ndarray) -> None:
        await asyncio.sleep(0)


def make_live(world: FakeWorld, agent: FakeAgent, **kwargs) -> tuple[LiveSession, list[dict[str, Any]]]:
    published: list[dict[str, Any]] = []

    async def publish(state: dict[str, Any], frames: list[np.ndarray]) -> None:
        published.append({**state, "frame_count": len(frames)})

    live = LiveSession(
        open_world=world.open,
        agent=agent,
        tracker=StillTracker(segmenter=None),
        publish=publish,
        **kwargs,
    )
    return live, published


async def drive(live: LiveSession, script) -> list[dict[str, Any]]:
    """Run the loop while ``script`` sends inputs; ``script`` ends the session with a stop."""
    inputs: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    run = asyncio.create_task(live.run(frame(), inputs))
    await script(live, inputs)
    return await asyncio.wait_for(run, timeout=5)


async def until(condition, timeout: float = 2.0) -> None:
    async def poll() -> None:
        while not condition():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(poll(), timeout)


def sent_prompts(world: FakeWorld) -> list[str]:
    prompts = []
    for session in world.sessions:
        prompts.append(session.prompt)
        prompts.extend(item["prompt"] for item in session.interactions if item["prompt"] is not None)
    return prompts


def test_opening_scene_becomes_the_background_prompt() -> None:
    world = FakeWorld()
    live, _ = make_live(world, FakeAgent(), segment_chunks=4)

    async def script(live: LiveSession, inputs: asyncio.Queue) -> None:
        await until(lambda: live.chunk >= 2)
        inputs.put_nowait({"type": "stop"})

    asyncio.run(drive(live, script))
    assert world.sessions[0].prompt == "Scene 1."
    assert live.proposals == {"u": "wave", "o": "jump", "f": "rain", "g": "snow"}


def test_hotkey_event_plays_for_its_duration_and_leaves_facts() -> None:
    world = FakeWorld()
    live, _ = make_live(world, FakeAgent(), segment_chunks=50, discover_every=100)

    async def script(live: LiveSession, inputs: asyncio.Queue) -> None:
        await until(lambda: live.chunk >= 1)
        inputs.put_nowait({"type": "hotkey", "key": "f"})
        await until(lambda: "After rain." in live.scheduler.facts)
        await until(lambda: live.chunk >= live.trace[-1]["chunk"] + 1)
        inputs.put_nowait({"type": "stop"})

    asyncio.run(drive(live, script))
    assert sent_prompts(world) == ["Scene 1.", "Scene 1. EVENT rain.", "Scene 1. After rain."]


def test_failed_preconditions_reject_the_event_with_a_notice() -> None:
    world = FakeWorld()
    live, _ = make_live(world, FakeAgent(failed=["handle reachable"]), segment_chunks=50, discover_every=100)

    async def script(live: LiveSession, inputs: asyncio.Queue) -> None:
        await until(lambda: live.chunk >= 1)
        inputs.put_nowait({"type": "object_action", "object_id": "obj-0", "action": "open"})
        await until(lambda: len(live.notices) == 1)
        inputs.put_nowait({"type": "stop"})

    asyncio.run(drive(live, script))
    assert sent_prompts(world) == ["Scene 1."]
    assert live.notices[0]["text"] == "open the door: handle reachable"


def test_held_camera_keys_carry_into_the_next_segment() -> None:
    world = FakeWorld()
    live, _ = make_live(world, FakeAgent(), segment_chunks=3, discover_every=100)

    async def script(live: LiveSession, inputs: asyncio.Queue) -> None:
        await until(lambda: live.chunk >= 1)
        inputs.put_nowait({"type": "key", "key": "w", "down": True})
        await until(lambda: len(world.sessions) >= 2 and world.sessions[1].interactions)
        inputs.put_nowait({"type": "stop"})

    asyncio.run(drive(live, script))
    first_camera = world.sessions[0].interactions[0]["camera"]
    assert first_camera["data"]["translation"][2] > 0
    assert world.sessions[1].interactions[0]["camera"] == first_camera
    # The next segment continues from the last generated frame.
    assert int(world.sessions[1].image[0, 0, 0]) == 3


def test_a_slow_director_never_holds_the_world() -> None:
    async def scenario() -> tuple[int, LiveSession]:
        gate = asyncio.Event()
        world = FakeWorld()
        live, _ = make_live(world, FakeAgent(resolve_gate=gate), segment_chunks=100, discover_every=1000)
        inputs: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        run = asyncio.create_task(live.run(frame(), inputs))
        await until(lambda: live.chunk >= 1)
        inputs.put_nowait({"type": "text", "text": "a bird lands"})
        await until(lambda: live.pending)
        started = live.chunk
        await until(lambda: live.chunk >= started + 10)
        progressed = live.chunk - started
        gate.set()
        await until(lambda: live.scheduler.active is not None)
        inputs.put_nowait({"type": "stop"})
        await asyncio.wait_for(run, timeout=5)
        return progressed, live

    progressed, live = asyncio.run(scenario())
    assert progressed >= 10
    assert any(entry["kind"] == "resolved" for entry in live.trace)


def test_director_refreshes_on_schedule_and_only_a_scene_change_replaces_the_background() -> None:
    world = FakeWorld()
    live, _ = make_live(world, FakeAgent(), segment_chunks=50, discover_every=4)

    async def script(live: LiveSession, inputs: asyncio.Queue) -> None:
        await until(lambda: live.agent.discoveries >= 3)
        inputs.put_nowait({"type": "stop"})

    asyncio.run(drive(live, script))
    # Every fake frame differs from the last only slightly, so refreshes keep the opening scene.
    assert live.scheduler.scene == "Scene 1."
    assert live.agent.discoveries < live.chunk
    assert scene_difference(frame(0), frame(255)) == pytest.approx(1.0)


# ── browser service ──────────────────────────────────────────────────────


def test_browser_session_streams_frames_and_forwards_inputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from PIL import Image
    from starlette.testclient import TestClient

    from benchmarks.lingbot_director import live_app

    world = FakeWorld(delay=0.005)
    monkeypatch.setattr(live_app, "DirectorAgent", lambda vlm: FakeAgent())
    monkeypatch.setattr(live_app, "ObjectTracker", lambda segmenter: StillTracker(segmenter=None))
    monkeypatch.setattr(
        live_app.Hub, "open_world", lambda self, image, prompt, num_chunks: world.open(image, prompt, num_chunks)
    )
    image = tmp_path / "scene.png"
    Image.new("RGB", (160, 96), (20, 40, 60)).save(image)
    args = live_app.parse_args(
        ["--image", str(image), "--width", str(WIDTH), "--height", str(HEIGHT), "--segment-chunks", "30"]
    )

    with TestClient(live_app.build_app(args)) as client:
        with client.websocket_connect("/ws") as websocket:
            assert json.loads(websocket.receive_text()) == {"type": "state", "running": False}
            websocket.send_text(json.dumps({"type": "start"}))
            header = None
            while header is None:
                message = websocket.receive()
                if message.get("text") and json.loads(message["text"])["type"] == "frames":
                    header = json.loads(message["text"])
            jpeg = websocket.receive_bytes()
            assert jpeg[:2] == b"\xff\xd8"
            websocket.send_text(json.dumps({"type": "key", "key": "w", "down": True}))
            websocket.send_text(json.dumps({"type": "stop"}))
            while not any(item["camera"] for session in world.sessions for item in session.interactions):
                websocket.receive()

    assert header["count"] == 1
    assert header["chunk"] >= 1
