# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wiring the real model into Mage-VL's duplex adapter.

The adapter takes two callables -- score a segment, generate an answer -- so its streaming
behaviour can be tested without a GPU. This is the other side of that seam: the scorer
that actually runs the vision tower and the cognition gate, and the text stream that
actually runs the decoder.

Both are per session. The gate carries recurrent state, so a scorer instance belongs to
one stream and no other; the generator is stateless but its cancellation is what makes
barge-in real -- dropping the async iterator aborts the request in the engine rather than
letting a superseded answer run to completion.
"""

from __future__ import annotations

import base64
import io
from collections.abc import AsyncIterator
from typing import Any

import torch

VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"


class MageVLSegmentScorer:
    """One video segment in, ``p_speak`` out, advancing this session's gate state.

    A segment is whatever the stream delivers for one time slice -- a list of sampled
    frames, or codec canvases. They go through the same image path the model uses
    everywhere else; the gate then pools the segment's visual tokens into the single
    perception token its recurrence consumes.
    """

    def __init__(
        self,
        processor: Any,
        vision_tower: torch.nn.Module,
        gate: Any,
        *,
        device: Any = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self._processor = processor
        self._tower = vision_tower
        self._gate = gate
        self._device = device
        self._dtype = dtype
        self._state = gate.new_state(device=device, dtype=torch.float32)

    def reset(self) -> None:
        """Start a new stream. The gate's state is the session; it does not carry over."""
        self._state = self._gate.new_state(device=self._device, dtype=torch.float32)

    @torch.inference_mode()
    def __call__(self, segment: Any) -> float:
        images = _as_images(segment)
        if not images:
            raise ValueError("a segment must carry at least one frame")

        processed = self._processor(
            text=_prompt_for(len(images), "."),
            images=images,
            return_tensors="pt",
        )
        embeds = self._tower(
            processed["pixel_values"].to(device=self._device, dtype=self._dtype),
            processed["image_grid_thw"].to(self._device),
            processed["patch_positions"].to(self._device),
        )

        # [1, 1, visual tokens, hidden]: one segment, its tokens pooled by the gate.
        vision_tokens = embeds.unsqueeze(0).unsqueeze(0).float()
        score = self._gate.score(vision_tokens, self._state)
        return float(score.reshape(-1)[-1])


def build_text_stream(
    async_llm: Any,
    *,
    sampling_params: Any = None,
    max_tokens: int = 96,
    request_prefix: str = "mage-vl-duplex",
):
    """Return the adapter's ``generate``: messages in, text deltas out.

    Cancelling the returned iterator (which is what a barge-in does) propagates into
    ``AsyncLLM``, so a superseded answer stops occupying the engine instead of finishing
    invisibly.
    """
    from vllm import SamplingParams

    params = sampling_params or SamplingParams(temperature=0.0, max_tokens=max_tokens)
    counter = {"n": 0}

    async def generate(messages: list[dict[str, Any]]) -> AsyncIterator[str]:
        images, query = _split_messages(messages)
        counter["n"] += 1
        request_id = f"{request_prefix}-{counter['n']}"

        prompt: dict[str, Any] = {"prompt": _prompt_for(len(images), query)}
        if images:
            prompt["multi_modal_data"] = {"image": images}

        emitted = 0
        async for output in async_llm.generate(prompt, params, request_id):
            text = output.outputs[0].text
            if len(text) > emitted:
                yield text[emitted:]
                emitted = len(text)

    return generate


def _prompt_for(num_images: int, query: str) -> str:
    return (
        "<|im_start|>user\n"
        + VISION_BLOCK * num_images
        + query
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


def _split_messages(messages: list[dict[str, Any]]) -> tuple[list[Any], str]:
    """Pull the images and the question back out of the adapter's message list."""
    images: list[Any] = []
    query = ""
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") == "image_url":
                images.extend(_as_images(part["image_url"]["url"]))
            elif part.get("type") == "text":
                query = part["text"]
    return images, query


def _as_images(segment: Any) -> list[Any]:
    """Accept a frame, a list of frames, or a ``data:`` URI carrying one."""
    from PIL import Image

    if isinstance(segment, (list, tuple)):
        return [image for item in segment for image in _as_images(item)]
    if isinstance(segment, Image.Image):
        return [segment]
    if isinstance(segment, (bytes, bytearray)):
        return [Image.open(io.BytesIO(bytes(segment))).convert("RGB")]
    if isinstance(segment, str) and segment.startswith("data:"):
        payload = segment.split(",", 1)[1]
        return [Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")]
    if isinstance(segment, str):
        return [Image.open(segment).convert("RGB")]
    raise TypeError(f"cannot read a video segment from {type(segment).__name__}")


def sample_segments(video_path: str, *, segment_seconds: float, frames_per_segment: int) -> list[list[Any]]:
    """Cut a video into fixed-length segments of uniformly sampled frames.

    This is the stream a client would push; keeping it here means the e2e drives the same
    framing the policy's ``segment_seconds`` default describes.
    """
    import cv2
    import numpy as np
    from PIL import Image

    capture = cv2.VideoCapture(video_path)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    if total <= 0:
        capture.release()
        raise ValueError(f"could not read video: {video_path}")

    per_segment = max(int(round(segment_seconds * fps)), 1)
    segments: list[list[Any]] = []
    for start in range(0, total, per_segment):
        stop = min(start + per_segment, total)
        indices = np.linspace(start, stop - 1, min(frames_per_segment, stop - start), dtype=int)
        frames = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if ok:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        if frames:
            segments.append(frames)
    capture.release()
    return segments


__all__ = [
    "MageVLSegmentScorer",
    "build_text_stream",
    "sample_segments",
]
