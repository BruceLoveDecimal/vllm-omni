# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ordered image references and the encoder inputs of a TaoMate-R1 session.

The release encodes references exactly as given: each image is resized to a
2048-pixel short edge (both sides on the 32-pixel grid, Lanczos) and becomes
``<Picture i>`` in request order, in front of the prompt. The same resized
image feeds both Qwen3-VL and the video VAE anchor encode. This differs from
MiniMax-H3 Ref2VA serving, which keeps the original reference resolution.

Video and audio references are not served in this version; their release
preprocessing (24 fps, 768 short edge, session-length clips, discarded
reference audio) has not been ported.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image

from vllm_omni.errors import OmniClientError
from vllm_omni.model_executor.models.minimax_h3.conditioning import MiniMaxH3EncoderMediaInput
from vllm_omni.model_executor.models.minimax_h3.encoder_processing import PreparedEncoderInputs
from vllm_omni.model_executor.models.minimax_h3.preprocessing import load_minimax_h3_images

from .chunks import R1SessionPlan

REFERENCE_SHORT_EDGE = 2048
REFERENCE_MULTIPLE = 32
MAX_IMAGE_REFERENCES = 9
_SUPPORTED_MODALITIES = frozenset({"image"})


def reference_size(width: int, height: int, short_edge: int = REFERENCE_SHORT_EDGE) -> tuple[int, int]:
    """Release resize rule: scale to ``short_edge`` and round each side to 32 (at least 32)."""
    if width <= 0 or height <= 0:
        raise OmniClientError(f"reference image must have a positive size, got {width}x{height}")
    scale = short_edge / min(width, height)
    return tuple(
        max(REFERENCE_MULTIPLE, round(value * scale / REFERENCE_MULTIPLE) * REFERENCE_MULTIPLE)
        for value in (width, height)
    )  # type: ignore[return-value]


def extract_reference_images(multi_modal_data: Mapping[str, Any] | None) -> list[Image.Image]:
    """Return the session's ordered RGB reference images; reject other modalities."""
    data = dict(multi_modal_data or {})
    unsupported = sorted(key for key, value in data.items() if value is not None and key not in _SUPPORTED_MODALITIES)
    if unsupported:
        raise OmniClientError(f"TaoMate-R1 accepts ordered image references only; unsupported inputs: {unsupported}")
    raw = data.get("image")
    if raw is None:
        raise OmniClientError("TaoMate-R1 requires at least one image reference (image_reference)")
    images = load_minimax_h3_images(raw)
    if not 1 <= len(images) <= MAX_IMAGE_REFERENCES:
        raise OmniClientError(f"TaoMate-R1 accepts 1 to {MAX_IMAGE_REFERENCES} image references, got {len(images)}")
    return images


def resize_references(images: Sequence[Image.Image]) -> list[Image.Image]:
    return [image.convert("RGB").resize(reference_size(*image.size), Image.Resampling.LANCZOS) for image in images]


def condition_labels(count: int) -> list[tuple[str, int]]:
    """``<Picture 1> ... <Picture n>`` in request order."""
    return [("image", index) for index in range(1, int(count) + 1)]


def _image_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(np.array(array, copy=True)).contiguous()


def build_encoder_inputs(
    prompt: str,
    images: Sequence[Image.Image],
    *,
    plan: R1SessionPlan,
    height: int,
    width: int,
) -> PreparedEncoderInputs:
    """Qwen presentation and VAE media inputs of one session (images already resized)."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise OmniClientError("TaoMate-R1 requires a non-empty prompt")
    media = MiniMaxH3EncoderMediaInput(
        task="ref2va",
        height=int(height),
        width=int(width),
        num_frames=plan.frame_count,
        latent_t=plan.video_latent_t,
        audio_t=plan.audio_latent_t,
        images=tuple(_image_tensor(image) for image in images),
    )
    return PreparedEncoderInputs(
        prompt=prompt,
        media=media,
        images=list(images),
        qwen_videos=[],
        video_timestamps=[],
        condition_labels=condition_labels(len(images)),
    )


__all__ = [
    "MAX_IMAGE_REFERENCES",
    "REFERENCE_SHORT_EDGE",
    "build_encoder_inputs",
    "condition_labels",
    "extract_reference_images",
    "reference_size",
    "resize_references",
]
