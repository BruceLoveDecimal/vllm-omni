# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Serve the live Director loop to a browser.

Start the models with ``serve_stack.sh``, then::

    python benchmarks/lingbot_director/live_app.py --image /path/scene.jpg --port 7860

and open http://127.0.0.1:7860. Over SSH, forward the port first.
"""

from __future__ import annotations

# ruff: noqa: E402, I001
import argparse
import asyncio
import io
import json
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.lingbot_director.agent import DirectorAgent
from benchmarks.lingbot_director.clients import (
    EditClient,
    SegmentClient,
    VlmClient,
    WorldSession,
    data_url,
    fit_image,
)
from benchmarks.lingbot_director.live import LiveSession
from benchmarks.lingbot_director.tracker import ObjectTracker
from benchmarks.lingbot_world.workload import num_frames_for_chunks

STATIC = Path(__file__).with_name("live_static")
# Messages a slow viewer may fall behind by before its oldest frames are dropped.
VIEWER_BACKLOG = 512


def jpeg_bytes(frame: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


class Viewer:
    """One browser. Its own queue keeps a slow connection from stalling the world."""

    def __init__(self, websocket: Any) -> None:
        self.websocket = websocket
        self.queue: asyncio.Queue[str | bytes] = asyncio.Queue(maxsize=VIEWER_BACKLOG)

    def offer(self, message: str | bytes) -> None:
        if self.queue.full():
            self.queue.get_nowait()
        self.queue.put_nowait(message)

    async def drain(self) -> None:
        while True:
            message = await self.queue.get()
            if isinstance(message, bytes):
                await self.websocket.send_bytes(message)
            else:
                await self.websocket.send_text(message)


class Hub:
    def __init__(self, args: argparse.Namespace, http: httpx.AsyncClient) -> None:
        self.args = args
        self.http = http
        self.viewers: set[Viewer] = set()
        self.session: LiveSession | None = None
        self.inputs: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.task: asyncio.Task[None] | None = None
        self.last_state: str = json.dumps({"type": "state", "running": False})
        self.writer: Any = None

    async def publish(self, state: dict[str, Any], frames: list[np.ndarray]) -> None:
        self.last_state = json.dumps({**state, "running": True}, ensure_ascii=False)
        messages: list[str | bytes] = [self.last_state]
        if frames:
            encoded = await asyncio.to_thread(lambda: [jpeg_bytes(frame) for frame in frames])
            header = {"type": "frames", "chunk": state["frames_chunk"], "count": len(encoded), "fps": self.args.fps}
            messages.append(json.dumps(header))
            messages.extend(encoded)
            if self.writer is not None:
                for frame in frames:
                    self.writer.append_data(frame)
        for viewer in list(self.viewers):
            for message in messages:
                viewer.offer(message)

    def open_world(self, image: np.ndarray, prompt: str, num_chunks: int) -> WorldSession:
        args = self.args
        start = {
            "type": "session.start",
            "model": args.world_model,
            "prompt": prompt,
            "image_reference": {"image_url": data_url(image)},
            "width": args.width,
            "height": args.height,
            "num_frames": num_frames_for_chunks(num_chunks),
            "fps": args.fps,
            "seed": args.seed,
            # No camera script: the camera follows session.interaction events.
            "extra_params": {"flow_shift": 5.0},
        }
        return WorldSession(args.world_url, start)

    def start(self) -> None:
        if self.task is not None and not self.task.done():
            return
        args = self.args
        self.inputs = asyncio.Queue()
        self.session = LiveSession(
            open_world=self.open_world,
            agent=DirectorAgent(VlmClient(self.http, args.vlm_url, args.vlm_model)),
            tracker=ObjectTracker(SegmentClient(self.http, args.segment_url)),
            publish=self.publish,
            segment_chunks=args.segment_chunks,
            discover_every=args.discover_every,
            editor=EditClient(self.http, args.edit_url) if args.edit_url else None,
            edit_seed=args.seed,
        )
        self.task = asyncio.create_task(self._run(self.session))

    async def _run(self, session: LiveSession) -> None:
        reference = fit_image(Image.open(self.args.image), self.args.width, self.args.height)
        record = None
        if self.args.record is not None:
            import imageio.v2 as imageio

            record = self.args.record / time.strftime("live-%Y%m%d-%H%M%S")
            record.mkdir(parents=True, exist_ok=True)
            self.writer = imageio.get_writer(str(record / "rollout.mp4"), fps=self.args.fps, codec="libx264", quality=7)
        try:
            trace = await session.run(reference, self.inputs)
        finally:
            if self.writer is not None:
                self.writer.close()
                self.writer = None
        if record is not None:
            (record / "trace.json").write_text(json.dumps(trace, indent=2, ensure_ascii=False, default=str))
        self.last_state = json.dumps({"type": "state", "running": False})
        for viewer in list(self.viewers):
            viewer.offer(self.last_state)


def build_app(args: argparse.Namespace) -> FastAPI:
    # FastAPI resolves the endpoint annotations from this module's globals.
    hub: Hub | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal hub
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as http:
            hub = Hub(args, http)
            app.state.hub = hub
            yield
            if hub.session is not None:
                await hub.session.stop()

    app = FastAPI(lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        viewer = Viewer(websocket)
        hub.viewers.add(viewer)
        viewer.offer(hub.last_state)
        sender = asyncio.create_task(viewer.drain())
        try:
            while True:
                message = json.loads(await websocket.receive_text())
                if message["type"] == "start":
                    hub.start()
                else:
                    hub.inputs.put_nowait(message)
        except WebSocketDisconnect:
            pass
        finally:
            hub.viewers.discard(viewer)
            sender.cancel()

    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--segment-chunks", type=int, default=40)
    parser.add_argument("--discover-every", type=int, default=8)
    parser.add_argument("--record", type=Path, default=None, help="Directory for rollout.mp4 and trace.json.")
    parser.add_argument("--world-url", default="ws://127.0.0.1:8000/v1/realtime/video")
    parser.add_argument("--world-model", default="robbyant/lingbot-world-v2-14b-causal-fast-diffusers")
    parser.add_argument("--vlm-url", default="http://127.0.0.1:8001")
    parser.add_argument("--vlm-model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--segment-url", default="http://127.0.0.1:8003")
    parser.add_argument(
        "--edit-url",
        default=None,
        help="Image-edit server for re-anchoring failed events; without it a re-anchor restarts from the frame as is.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    args = parse_args(argv)
    uvicorn.run(build_app(args), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
