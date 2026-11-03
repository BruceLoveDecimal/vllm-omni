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

from benchmarks.lingbot_director.agent import DirectorAgent, Discovery, Resolution, Verdict
from benchmarks.lingbot_director.clients import Chunk
from benchmarks.lingbot_director.events import CameraControl, EventScheduler, WorldEvent
from benchmarks.lingbot_director.live import LiveSession, scene_difference
from benchmarks.lingbot_director.tracker import ObjectTracker, iou

HEIGHT = 48
WIDTH = 80


def frame(value: int = 0) -> np.ndarray:
    return np.full((HEIGHT, WIDTH, 3), value, dtype=np.uint8)


def event(
    source: str, prompt: str, *, duration: int = 2, facts: list[str] | None = None, kind: str = "action"
) -> WorldEvent:
    return WorldEvent(
        event_id=prompt,
        source=source,
        request=prompt,
        prompt=prompt,
        duration_chunks=duration,
        facts_after=facts or [],
        kind=kind,
        success_check=f"Did {prompt}",
        scene_after="A changed hallway.",
    )


def play_out(scheduler: EventScheduler, *, first_chunk: int, carrier: str = "p1") -> WorldEvent:
    """Send the active event's prompt, apply it on ``first_chunk`` and run it to its check."""
    scheduler.sent(carrier)
    chunk = first_chunk
    played_out = scheduler.on_chunk(chunk, [carrier])
    while played_out is None:
        chunk += 1
        played_out = scheduler.on_chunk(chunk, [])
    return played_out


# ── scheduler and camera ─────────────────────────────────────────────────


def test_prompt_is_scene_then_facts_then_active_event() -> None:
    scheduler = EventScheduler(scene="A hallway.")
    scheduler.facts = ["The door is open."]
    scheduler.submit(event("hotkey", "Birds fly past."), chunk=0)
    assert scheduler.prompt() == "A hallway. The door is open. Birds fly past."


def test_duration_counts_from_the_chunk_the_server_applied_the_prompt() -> None:
    scheduler = EventScheduler(scene="A hallway.")
    scheduler.submit(event("object", "The door swings open.", duration=2, facts=["The door is open."]), chunk=0)
    scheduler.sent("p1")
    # Chunks generated before the server reports the prompt do not count toward the event.
    assert scheduler.on_chunk(1, []) is None
    assert scheduler.on_chunk(2, ["camera-9"]) is None
    assert scheduler.active.status == "sent"
    assert scheduler.on_chunk(3, ["p1"]) is None
    assert (scheduler.active.status, scheduler.active.applied_chunk) == ("playing", 3)
    played_out = scheduler.on_chunk(4, [])
    assert played_out is scheduler.active and played_out.status == "verifying"
    # Facts wait for the check.
    assert scheduler.facts == []
    assert scheduler.on_chunk(5, []) is None


def test_confirmed_event_keeps_its_facts_and_scene() -> None:
    scheduler = EventScheduler(scene="A hallway.")
    scheduler.submit(event("object", "The door swings open.", facts=["The door is open."]), chunk=0)
    played_out = play_out(scheduler, first_chunk=1)
    assert scheduler.settle(played_out, "yes", chunk=3) == "succeeded"
    assert scheduler.active is None
    assert scheduler.prompt() == "A changed hallway. The door is open."


def test_failed_checks_climb_the_rungs_then_drop_the_event_without_facts() -> None:
    scheduler = EventScheduler(scene="A hallway at sunset.")
    scheduler.facts = ["The lamp is off."]
    scheduler.submit(event("text", "Snow falls.", facts=["Snow covers the floor."]), chunk=0)
    assert scheduler.prompt() == "A hallway at sunset. The lamp is off. Snow falls."

    played_out = play_out(scheduler, first_chunk=1, carrier="p1")
    assert scheduler.settle(played_out, "partial", chunk=2) == "extended"
    # One extension on the same rung, counted from where it is.
    assert scheduler.active.status == "playing"
    played_out = scheduler.on_chunk(3, []) or scheduler.on_chunk(4, [])
    assert played_out is not None
    assert scheduler.settle(played_out, "partial", chunk=4) == "rewrite"
    # The rewrite drops the old scene and facts that contradict the event.
    assert scheduler.prompt() == "A changed hallway. Snow covers the floor. Snow falls."
    assert scheduler.awaiting_apply() and not scheduler.active.restart_pending

    played_out = play_out(scheduler, first_chunk=5, carrier="p2")
    assert scheduler.settle(played_out, "no", chunk=6) == "reanchor"
    assert scheduler.active.restart_pending

    played_out = play_out(scheduler, first_chunk=7, carrier="segment-3")
    assert scheduler.settle(played_out, "no", chunk=8) == "failed"
    assert scheduler.active is None
    assert scheduler.facts == ["The lamp is off."]
    assert scheduler.prompt() == "A hallway at sunset. The lamp is off."
    assert [entry["what"] for entry in scheduler.history] == [
        "started", "applied", "verifying", "extended", "verifying", "escalated", "applied", "verifying",
        "escalated", "applied", "verifying", "failed",
    ]  # fmt: skip


def test_appearance_events_start_with_a_restart() -> None:
    scheduler = EventScheduler(scene="A beach at sunset.")
    scheduler.submit(event("hotkey", "Snow falls.", kind="appearance"), chunk=0)
    assert scheduler.active.rung == "reanchor"
    assert scheduler.active.restart_pending


def test_a_check_for_a_replaced_event_is_ignored() -> None:
    scheduler = EventScheduler(scene="A hallway.")
    scheduler.submit(event("director", "Rain starts.", facts=["It rains."]), chunk=0)
    played_out = play_out(scheduler, first_chunk=1)
    scheduler.submit(event("text", "Birds fly past."), chunk=3)
    assert scheduler.settle(played_out, "yes", chunk=4) == "stale"
    assert scheduler.facts == []


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
        "kind": "something else",
        "success_check": "Is the door open?",
        "scene_after": "A hallway with an open door.",
        "edit_instruction": "Open the door.",
    }
    resolution = asyncio.run(DirectorAgent(ScriptedVlm([reply])).resolve(frame(), "open the door", scene="", facts=[]))
    assert resolution.duration_chunks == 6
    assert resolution.failed_preconditions == ["handle reachable"]
    # Anything but an explicit appearance change starts as a prompt-only action.
    assert resolution.kind == "action"


def test_verify_asks_the_events_question_and_rejects_unknown_verdicts() -> None:
    vlm = ScriptedVlm([{"verdict": "Partial", "observed": "A few flakes."}, {"verdict": "maybe", "observed": ""}])
    agent = DirectorAgent(vlm)
    verdict = asyncio.run(agent.verify([frame(), frame()], "Is snow on the sand?"))
    assert verdict == Verdict(verdict="partial", observed="A few flakes.")
    assert vlm.requests[0][1:] == (2, "Question: Is snow on the sand?")
    with pytest.raises(ValueError, match="maybe"):
        asyncio.run(agent.verify([frame()], "Is snow on the sand?"))


# ── the live loop ────────────────────────────────────────────────────────


class FakeSession:
    def __init__(self, world: FakeWorld, image: np.ndarray, prompt: str, num_chunks: int) -> None:
        self.world = world
        self.image = image
        self.prompt = prompt
        self.num_chunks = num_chunks
        self.interactions: list[dict[str, Any]] = []
        # Interactions not yet applied, each with the number of chunks still to generate before it is.
        self.unapplied: list[list] = []
        self.stopped = False
        self.yielded = 0
        world.sessions.append(self)

    async def __aenter__(self) -> FakeSession:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        return None

    async def interact(self, event_id: str, *, prompt=None, camera=None, transition_chunks=None) -> None:
        self.interactions.append({"event_id": event_id, "prompt": prompt, "camera": camera})
        self.unapplied.append([event_id, self.world.apply_lag])

    async def stop(self) -> None:
        self.stopped = True

    async def chunks(self):
        for index in range(self.num_chunks):
            await self.world.pace()
            if self.stopped:
                return
            self.world.generated += 1
            # Like the server, report the interactions this chunk is the first to use.
            started = [event_id for event_id, wait in self.unapplied if wait == 0]
            self.unapplied = [[event_id, wait - 1] for event_id, wait in self.unapplied if wait > 0]
            self.yielded += 1
            yield Chunk(index=index, frames=[frame(self.world.generated)], started_event_ids=started)


class FakeWorld:
    def __init__(self, *, delay: float = 0.001, apply_lag: int = 0) -> None:
        self.sessions: list[FakeSession] = []
        self.generated = 0
        self.delay = delay
        # Chunks already in flight when an interaction arrives, so it applies that many chunks late.
        self.apply_lag = apply_lag

    async def pace(self) -> None:
        await asyncio.sleep(self.delay)

    def open(self, image: np.ndarray, prompt: str, num_chunks: int) -> FakeSession:
        return FakeSession(self, image, prompt, num_chunks)


class FakeAgent:
    def __init__(
        self,
        *,
        resolve_gate: asyncio.Event | None = None,
        failed: list[str] | None = None,
        kind: str = "action",
        verdicts: list[str] | None = None,
        facts_after: list[str] | None = None,
    ) -> None:
        self.discoveries = 0
        self.resolve_gate = resolve_gate
        self.failed = failed or []
        self.kind = kind
        # Answers to successive checks; once used up, every check passes.
        self.verdicts = list(verdicts or [])
        self.questions: list[str] = []
        self.facts_after = facts_after

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
            facts_after=[f"After {request}."] if self.facts_after is None else self.facts_after,
            kind=self.kind,
            success_check=f"Did {request} happen?",
            scene_after="",
            edit_instruction=f"Make {request} visible.",
        )

    async def verify(self, frames: list[np.ndarray], question: str) -> Verdict:
        self.questions.append(question)
        verdict = self.verdicts.pop(0) if self.verdicts else "yes"
        return Verdict(verdict=verdict, observed=f"saw {verdict}")


class FakeEditor:
    def __init__(self) -> None:
        self.instructions: list[str] = []

    async def edit(self, image: np.ndarray, prompt: str, *, seed: int) -> np.ndarray:
        self.instructions.append(prompt)
        return frame(200)


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
    assert live.agent.questions == ["Did rain happen?"]


def test_event_waits_for_the_server_to_apply_its_prompt() -> None:
    world = FakeWorld(apply_lag=2)
    live, _ = make_live(world, FakeAgent(), segment_chunks=50, discover_every=100)

    async def script(live: LiveSession, inputs: asyncio.Queue) -> None:
        await until(lambda: live.chunk >= 1)
        inputs.put_nowait({"type": "hotkey", "key": "f"})
        await until(lambda: "After rain." in live.scheduler.facts)
        inputs.put_nowait({"type": "stop"})

    asyncio.run(drive(live, script))
    history = {entry["what"]: entry["chunk"] for entry in live.scheduler.history}
    sent = next(entry["chunk"] for entry in live.trace if entry["kind"] == "prompt_sent")
    # Two chunks were in flight, so the third chunk after sending is the first with the event.
    assert history["applied"] == sent + 3
    assert history["verifying"] == history["applied"] + 1


def test_unconfirmed_events_escalate_to_a_restart_from_the_edited_frame() -> None:
    world = FakeWorld()
    editor = FakeEditor()
    live, _ = make_live(
        world,
        FakeAgent(verdicts=["no", "no", "no"]),
        segment_chunks=50,
        discover_every=100,
        editor=editor,
    )

    async def script(live: LiveSession, inputs: asyncio.Queue) -> None:
        await until(lambda: live.chunk >= 1)
        inputs.put_nowait({"type": "text", "text": "snow falls"})
        await until(lambda: any(notice["level"] == "failed" for notice in live.notices))
        inputs.put_nowait({"type": "stop"})

    asyncio.run(drive(live, script))
    assert live.scheduler.facts == []
    assert [notice["level"] for notice in live.notices] == ["retry", "retry", "failed"]
    # Rewrite: the empty scene_after keeps the scene, and facts_after stands in for the facts.
    rewritten = "Scene 1. After snow falls. EVENT snow falls."
    assert sent_prompts(world)[:3] == ["Scene 1.", "Scene 1. EVENT snow falls.", rewritten]
    assert len(world.sessions) == 2
    assert world.sessions[0].stopped
    restarted = world.sessions[1]
    assert restarted.prompt == rewritten
    assert int(restarted.image[0, 0, 0]) == 200
    assert editor.instructions == ["Make snow falls visible."]
    assert sent_prompts(world)[-1] == "Scene 1."


def test_appearance_event_restarts_the_world_from_the_current_frame() -> None:
    world = FakeWorld()
    live, _ = make_live(world, FakeAgent(kind="appearance"), segment_chunks=50, discover_every=100)

    async def script(live: LiveSession, inputs: asyncio.Queue) -> None:
        await until(lambda: live.chunk >= 1)
        inputs.put_nowait({"type": "hotkey", "key": "g"})
        await until(lambda: "After snow." in live.scheduler.facts)
        inputs.put_nowait({"type": "stop"})

    asyncio.run(drive(live, script))
    assert len(world.sessions) == 2
    first, restarted = world.sessions
    # The event's prompt goes out in the restart, not as an interaction on the old segment.
    assert all(item["prompt"] is None for item in first.interactions)
    assert restarted.prompt == "Scene 1. After snow. EVENT snow."
    # Without an editor the restart continues from the frame as it is.
    assert int(restarted.image[0, 0, 0]) == first.yielded


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


def test_a_rewrite_that_leaves_the_prompt_unchanged_is_still_sent_and_confirmed() -> None:
    world = FakeWorld()
    # No facts and no new scene: the rewritten prompt is the one already playing.
    live, _ = make_live(world, FakeAgent(verdicts=["no", "yes"], facts_after=[]), segment_chunks=50, discover_every=100)

    async def script(live: LiveSession, inputs: asyncio.Queue) -> None:
        await until(lambda: live.chunk >= 1)
        inputs.put_nowait({"type": "text", "text": "a bird lands"})
        await until(lambda: any(entry["what"] == "succeeded" for entry in live.scheduler.history))
        inputs.put_nowait({"type": "stop"})

    asyncio.run(drive(live, script))
    assert sent_prompts(world) == [
        "Scene 1.",
        "Scene 1. EVENT a bird lands.",
        "Scene 1. EVENT a bird lands.",
        "Scene 1.",
    ]
