# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Online realtime-streaming smoke for the distilled SANA-WM Stage-1
(``SanaWmStreamingPipeline`` over ``WS /v1/realtime/video``).

Boots the server from the streaming repo with ``--diffusion-streaming-output``
(the runner then drives the pipeline through step execution), opens one
``session.start`` with a first-frame reference image and a forward-move camera
action, and checks that the fragmented-MP4 chunks arrive one generated latent
block at a time with contiguous ``generation_chunk_index`` values and that the
frame counts add up to ``num_frames``.

From ``tests/``::

    pytest -s -v e2e/online_serving/test_sana_wm_streaming.py -m "advanced_model and diffusion" --run-level=advanced_model
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from io import BytesIO
from typing import Any

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler
from vllm_omni.diffusion.models.sana_wm import (
    SANA_WM_OUTPUT_HEIGHT,
    SANA_WM_OUTPUT_WIDTH,
    SANA_WM_STREAMING_MODEL_ID,
)

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = os.environ.get("SANA_WM_STREAMING_E2E_MODEL", SANA_WM_STREAMING_MODEL_ID)
PROMPT = "A slow forward camera move through a quiet city street."

# Two generated latent blocks: 8 * 3 * 2 + 1 pixel frames -> chunks of 25 + 24.
SMOKE_NUM_FRAMES = int(os.environ.get("SANA_WM_STREAMING_E2E_NUM_FRAMES", "49"))
SMOKE_CHUNK_SIZE = 3
SMOKE_TOTAL_CHUNKS = (SMOKE_NUM_FRAMES - 1) // (8 * SMOKE_CHUNK_SIZE)
SANA_WM_PARAMS: dict[str, Any] = {
    "action": f"w-{SMOKE_NUM_FRAMES - 1}",
    "translation_speed": 0.055,
    "rotation_speed_deg": 1.2,
    "intrinsics": {
        "fx": SANA_WM_OUTPUT_WIDTH / 2,
        "fy": SANA_WM_OUTPUT_WIDTH / 2,
        "cx": SANA_WM_OUTPUT_WIDTH / 2,
        "cy": SANA_WM_OUTPUT_HEIGHT / 2,
    },
}

SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"})


def _first_frame_data_url() -> str:
    from PIL import Image

    image = Image.new("RGB", (SANA_WM_OUTPUT_WIDTH, SANA_WM_OUTPUT_HEIGHT), (96, 128, 160))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _session_start_payload(model: str) -> dict[str, Any]:
    return {
        "type": "session.start",
        "model": model,
        "prompt": PROMPT,
        "image_reference": {"image_url": _first_frame_data_url()},
        "width": SANA_WM_OUTPUT_WIDTH,
        "height": SANA_WM_OUTPUT_HEIGHT,
        "num_frames": SMOKE_NUM_FRAMES,
        "fps": 16,
        "num_inference_steps": 4,
        "guidance_scale": 1.0,
        "seed": 42,
        "format": "m4s",
        "extra_params": {"sana_wm": SANA_WM_PARAMS},
    }


async def _stream_session(url: str, payload: dict[str, Any], *, timeout_seconds: float) -> dict[str, Any]:
    """Collect every control message and binary chunk of one realtime session."""
    import websockets

    chunks: list[dict[str, Any]] = []
    binary_bytes = 0
    started_at = time.perf_counter()
    first_media_at: float | None = None
    done: dict[str, Any] | None = None
    deadline = started_at + timeout_seconds
    async with websockets.connect(url, max_size=None) as websocket:
        await websocket.send(json.dumps(payload))
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError(f"Streaming session did not finish within {timeout_seconds}s")
            message = await asyncio.wait_for(websocket.recv(), timeout=remaining)
            if isinstance(message, bytes):
                binary_bytes += len(message)
                continue
            msg = json.loads(message)
            msg_type = msg.get("type")
            if msg_type == "video.chunk_metadata":
                if msg.get("kind") == "media" and first_media_at is None:
                    first_media_at = time.perf_counter() - started_at
                chunks.append(msg)
            elif msg_type == "session.done":
                done = msg
                break
            elif msg_type == "error":
                raise RuntimeError(str(msg.get("message", msg)))
    return {
        "chunks": chunks,
        "binary_bytes": binary_bytes,
        "done": done,
        "first_media_seconds": first_media_at,
        "total_seconds": time.perf_counter() - started_at,
    }


def _get_diffusion_feature_cases(model: str):
    return [
        pytest.param(
            OmniServerParams(model=model, server_args=["--diffusion-streaming-output"]),
            id="streaming",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
    ]


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", _get_diffusion_feature_cases(MODEL), indirect=True)
def test_realtime_streaming_001(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    """One session streams ``total_chunks`` media chunks whose frames add up to ``num_frames``."""
    request_config = {
        "model": omni_server.model,
        "form_data": {
            key: value
            for key, value in _session_start_payload(omni_server.model).items()
            if key not in ("type", "model")
        },
    }
    # The helper assembles the fragments into one MP4 and validates it.
    responses = openai_client.send_streaming_video_diffusion_request(request_config, timeout_seconds=900.0)
    assert responses and responses[0].success

    # Second session: inspect the chunk cadence and metadata directly.
    url = openai_client._build_ws_url("/v1/realtime/video")
    result = asyncio.run(_stream_session(url, _session_start_payload(omni_server.model), timeout_seconds=900.0))
    media = [chunk for chunk in result["chunks"] if chunk.get("kind") == "media"]
    assert result["done"] is not None
    assert result["binary_bytes"] > 0
    assert [chunk.get("generation_chunk_index") for chunk in media] == list(range(SMOKE_TOTAL_CHUNKS))
    frame_counts = [int(chunk.get("num_frames") or 0) for chunk in media]
    assert frame_counts[0] == 1 + 8 * SMOKE_CHUNK_SIZE
    assert all(count == 8 * SMOKE_CHUNK_SIZE for count in frame_counts[1:])
    assert sum(frame_counts) == SMOKE_NUM_FRAMES
    print(
        f"[sana_wm.stream] chunks={len(media)} frames={frame_counts} "
        f"first_media={result['first_media_seconds']:.2f}s total={result['total_seconds']:.2f}s"
    )
