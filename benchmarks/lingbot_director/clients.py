# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Clients for the four served models. Each one is an ordinary HTTP or WebSocket call."""

from __future__ import annotations

import asyncio
import base64
import io
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np
from PIL import Image


def png_bytes(frame: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="PNG")
    return buffer.getvalue()


def data_url(frame: np.ndarray) -> str:
    return f"data:image/png;base64,{base64.b64encode(png_bytes(frame)).decode()}"


def decode_image(encoded: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB"))


def fit_image(image: Image.Image, width: int, height: int) -> np.ndarray:
    """Center-crop to the target aspect, then resize. The world cache geometry is fixed at load time."""
    scale = max(width / image.width, height / image.height)
    crop_width = round(width / scale)
    crop_height = round(height / scale)
    left = (image.width - crop_width) // 2
    top = (image.height - crop_height) // 2
    cropped = image.convert("RGB").crop((left, top, left + crop_width, top + crop_height))
    return np.asarray(cropped.resize((width, height), Image.BICUBIC))


def parse_json_reply(text: str) -> dict[str, Any]:
    payload = text.strip()
    if payload.startswith("```") and payload.endswith("```"):
        payload = "\n".join(payload.splitlines()[1:-1])
    return json.loads(payload)


class VlmClient:
    """OpenAI-compatible vision chat, from ``vllm serve`` or any compatible endpoint."""

    def __init__(
        self,
        http: httpx.AsyncClient,
        base_url: str,
        model: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> None:
        self.http = http
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        # Qwen3.8 pretty-prints JSON; a four-object discovery took 438 tokens, so 400 cut it off.
        self.max_tokens = max_tokens

    async def ask_json(self, system: str, images: list[np.ndarray], text: str) -> dict[str, Any]:
        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": data_url(image)}} for image in images
        ]
        content.append({"type": "text", "text": text})
        response = await self.http.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                # Qwen3.8 thinks by default, which would put reasoning in front of the JSON answer.
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        response.raise_for_status()
        choice = response.json()["choices"][0]
        if choice["finish_reason"] == "length":
            raise RuntimeError(f"VLM reply hit max_tokens={self.max_tokens} before the JSON closed.")
        return parse_json_reply(choice["message"]["content"])


class EditClient:
    """``POST /v1/images/edits`` on a served image-edit pipeline."""

    def __init__(self, http: httpx.AsyncClient, base_url: str, *, num_inference_steps: int = 40) -> None:
        self.http = http
        self.base_url = base_url
        self.num_inference_steps = num_inference_steps

    async def edit(self, image: np.ndarray, prompt: str, *, seed: int) -> np.ndarray:
        height, width = image.shape[:2]
        response = await self.http.post(
            f"{self.base_url}/v1/images/edits",
            files={"image": ("frame.png", png_bytes(image), "image/png")},
            data={
                "prompt": prompt,
                "size": f"{width}x{height}",
                "output_format": "png",
                "num_inference_steps": str(self.num_inference_steps),
                "seed": str(seed),
            },
        )
        response.raise_for_status()
        edited = Image.fromarray(decode_image(response.json()["data"][0]["b64_json"]))
        return np.asarray(edited.resize((width, height), Image.BICUBIC))


class SegmentClient:
    """``POST /v1/images/segmentations`` on a served SAM 3 detector."""

    def __init__(self, http: httpx.AsyncClient, base_url: str, *, score_threshold: float = 0.3) -> None:
        self.http = http
        self.base_url = base_url
        self.score_threshold = score_threshold

    async def _segment(
        self, image: np.ndarray, concept: str, boxes: list[list[float]] | None = None
    ) -> list[dict[str, Any]]:
        data = {"prompt": concept, "score_threshold": str(self.score_threshold), "mask_format": "png"}
        if boxes:
            # Exemplar boxes steer the detector toward the instance being tracked.
            data["boxes"] = json.dumps(boxes)
        response = await self.http.post(
            f"{self.base_url}/v1/images/segmentations",
            files={"image": ("frame.png", png_bytes(image), "image/png")},
            data=data,
        )
        response.raise_for_status()
        return response.json()["data"]

    async def instances(
        self, image: np.ndarray, concept: str, *, boxes: list[list[float]] | None = None
    ) -> list[dict[str, Any]]:
        """Every instance above the score threshold, as ``{"score", "box"}`` in pixels."""
        return [{"score": item["score"], "box": item["box"]} for item in await self._segment(image, concept, boxes)]

    async def best_mask(self, image: np.ndarray, concept: str) -> np.ndarray | None:
        """The highest-scoring instance of ``concept``, or None when nothing matches."""
        instances = await self._segment(image, concept)
        if not instances:
            return None
        mask = Image.open(io.BytesIO(base64.b64decode(instances[0]["mask"])))
        return np.asarray(mask) > 127


@dataclass
class Chunk:
    index: int
    frames: list[np.ndarray]


def decode_frames(media: bytes) -> list[np.ndarray]:
    import av

    with av.open(io.BytesIO(media), format="mp4", mode="r") as container:
        stream = container.streams.video[0]
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]


class WorldSession:
    """One ``WS /v1/realtime/video`` rollout. Chunks arrive while controls are sent."""

    # The server aborts a session after about 60 s without progress or a ping; a cold
    # start can take that long before the first chunk.
    keepalive_seconds = 10.0

    def __init__(self, url: str, start_payload: dict[str, Any]) -> None:
        self.url = url
        self.start_payload = start_payload
        self._connection: Any = None
        self._websocket: Any = None
        self._send_lock = asyncio.Lock()
        self._keepalive: asyncio.Task[None] | None = None
        self._stop_requested = False

    async def __aenter__(self) -> WorldSession:
        from websockets.asyncio.client import connect

        self._connection = connect(self.url, max_size=None, ping_interval=None)
        self._websocket = await self._connection.__aenter__()
        await self._send(self.start_payload)
        self._keepalive = asyncio.create_task(self._ping())
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        self._keepalive.cancel()
        await self._connection.__aexit__(*exc_info)

    async def _ping(self) -> None:
        while True:
            await asyncio.sleep(self.keepalive_seconds)
            await self._send({"type": "session.ping"})

    async def _send(self, payload: dict[str, Any]) -> None:
        from websockets.exceptions import ConnectionClosed

        async with self._send_lock:
            try:
                await self._websocket.send(json.dumps(payload, ensure_ascii=False))
            except ConnectionClosed:
                # The rollout already ended; a late control has nothing left to act on.
                pass

    async def set_prompt(self, event_id: str, prompt: str, *, transition_chunks: int = 0) -> None:
        await self.interact(event_id, prompt=prompt, transition_chunks=transition_chunks)

    async def interact(
        self,
        event_id: str,
        *,
        prompt: str | None = None,
        camera: dict[str, Any] | None = None,
        transition_chunks: int | None = None,
    ) -> None:
        """Queue a prompt and/or a structural camera command for the next chunk boundary."""
        event: dict[str, Any] = {}
        if prompt is not None:
            event["prompt"] = prompt
        if camera is not None:
            event["multi_modal_data"] = {"camera": camera}
        interaction: dict[str, Any] = {"event_id": event_id, "event": event}
        if transition_chunks is not None:
            interaction["transition_chunks"] = transition_chunks
        await self._send({"type": "session.interaction", "interaction": interaction})

    async def stop(self) -> None:
        self._stop_requested = True
        await self._send({"type": "session.stop"})

    async def chunks(self) -> AsyncIterator[Chunk]:
        media = bytearray()
        pending: dict[str, Any] | None = None
        decoded = 0
        while True:
            message = await self._websocket.recv()
            if isinstance(message, (bytes, bytearray)):
                assert pending is not None, "binary frame without video.chunk_metadata"
                media.extend(message)
                if pending.get("kind") == "media":
                    # Fragments share one init segment, so decode the whole stream and keep the new frames.
                    frames = await asyncio.to_thread(decode_frames, bytes(media))
                    yield Chunk(index=int(pending["generation_chunk_index"]), frames=frames[decoded:])
                    decoded = len(frames)
                pending = None
                continue
            event = json.loads(message)
            if event["type"] == "video.chunk_metadata":
                pending = event
            elif event["type"] == "error":
                raise RuntimeError(f"World session failed: {event}")
            elif event["type"] == "session.done":
                if event.get("stopped") and not self._stop_requested:
                    # A stall timeout or server-side abort ends the rollout early without an error frame.
                    raise RuntimeError(f"World stopped the session on its own: {event}")
                return
