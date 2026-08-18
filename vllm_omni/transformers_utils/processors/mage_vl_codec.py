# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Codec-native video preprocessing for Mage-VL.

Under codec sampling a video does not become one image per frame. An external codec
pass groups frames by "readiness" (how much new information a frame carries, read off
the codec's own bit-cost map) and tiles the selected patches onto a small number of
square **canvases**. Each canvas therefore carries patches from several source frames,
which is why the vision tower needs an explicit ``(t, h, w)`` table rather than deriving
positions from ``grid_thw``: ``t`` is the *source frame index*, and it is not dense.

That pass emits an asset directory:

    canvas_00000.jpg ...     the canvases, already aligned to the patch grid
    src_patch_position.npy   ``[sum(t*h*w), 3]`` (t, h, w) per patch, row-major
    meta.json                at least ``fps``; ``decimals`` when timestamps need it

This module turns such a directory into the tensors the model consumes, and rewrites the
prompt's video placeholder into per-timestamp ``<|image_pad|>`` runs. Producing the
directory is the codec engine's job -- see :func:`process_codec_video`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm_omni.transformers_utils.processors.mage_vl import (
    convert_positions_to_block_layout,
)

VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"
VIDEO_PAD = "<|video_pad|>"

# Exactly one chat-template video placeholder block, so image blocks elsewhere in the
# prompt are left alone.
_VIDEO_BLOCK_RE = re.compile(
    re.escape(VISION_START) + r"\s*" + re.escape(VIDEO_PAD) + r"\s*" + re.escape(VISION_END)
)


@dataclass
class DcvcConfig:
    """Knobs for the DCVC-RT neural-codec engine, nested under :class:`CodecConfig`."""

    qp: int = 42
    reset_interval: int = 64
    intra_period: int = -1
    max_side: int = 0
    seq_len_frames: int = 0
    patch: int = 16
    canvas_token_side: int | None = None
    readiness_sum_threshold_mode: str = "auto"
    readiness_coverage_bins: int = 3
    readiness_delta_ratio: float = 0.05
    bitcost_grid: str = "sub"
    bitcost_pct: int = 99
    decode_backsearch_max: int = 16
    max_group_frames: int = 128


@dataclass
class CodecConfig:
    """Codec sampling budget, mirroring the checkpoint's ``preprocessor_config.json``.

    ``max_pixels`` is shared with the image processor's pixel budget so canvas size and
    the smart-resize budget cannot drift apart.
    """

    target_canvas: int = 32
    group_size: int = 32
    images_per_group: int = 4
    patch: int = 14
    max_pixels: int = 150000
    min_group_frames: int = 8
    max_group_frames: int = 64
    spatial_mask_mode: str = "off"
    engine: str = "hevc"
    dcvc: DcvcConfig = field(default_factory=DcvcConfig)

    def __post_init__(self) -> None:
        if isinstance(self.dcvc, dict):
            # The checkpoint's ``codec.dcvc`` also carries selection knobs consumed by the
            # engine itself (per_frame_cap_ratio, bottom_atten, ...); keep only ours.
            known = {f.name for f in dataclass_fields(DcvcConfig)}
            self.dcvc = DcvcConfig(**{k: v for k, v in self.dcvc.items() if k in known})

    def validate(self) -> None:
        if self.target_canvas <= 0:
            raise ValueError("CodecConfig.target_canvas must be > 0")
        if self.target_canvas % self.images_per_group != 0:
            raise ValueError("CodecConfig.target_canvas must be divisible by images_per_group")
        if self.group_size % self.images_per_group != 0:
            raise ValueError("CodecConfig.group_size must be divisible by images_per_group")

    def num_sampled_frames(self) -> int:
        return (self.target_canvas // self.images_per_group) * self.group_size


def codec_positions_for_processor(
    src_positions: np.ndarray,
    image_grid_thw: torch.Tensor,
) -> torch.Tensor:
    """Reorder the codec's row-major position table into the processor's block layout.

    The codec emits positions in raster order per canvas; ``pixel_values`` comes out of
    the image processor in 2x2 block order. Row *i* of the result must line up with patch
    *i* of the pixels, so each canvas is reordered independently.
    """
    positions = torch.as_tensor(src_positions, dtype=torch.int64)
    expected_total = int(image_grid_thw.prod(dim=1).sum().item())
    if expected_total != positions.shape[0]:
        raise ValueError(
            "codec patch position length mismatch: "
            f"thw_total={expected_total}, positions={positions.shape[0]}"
        )

    chunks: list[torch.Tensor] = []
    offset = 0
    for row in image_grid_thw:
        t, h, w = int(row[0]), int(row[1]), int(row[2])
        count = t * h * w
        chunks.append(
            convert_positions_to_block_layout(positions[offset : offset + count], t, h, w)
        )
        offset += count
    return torch.cat(chunks, dim=0)


def drop_padding_canvases(
    images: list[Any],
    src_positions: np.ndarray,
) -> tuple[list[Any], np.ndarray, int]:
    """Drop canvases that are entirely padding, along with their patch rows.

    The codec pads its output up to ``target_canvas``; a padding canvas is marked by a
    negative source-frame index on every one of its patches. Padding is canvas-granular
    by construction, so a canvas that is only *partly* negative means the asset and the
    position table disagree -- worth failing on rather than silently trimming.
    """
    n_canvas = len(images)
    if n_canvas == 0:
        return images, src_positions, 0

    total_patches = src_positions.shape[0]
    if total_patches % n_canvas != 0:
        raise ValueError(
            f"src_positions length {total_patches} not divisible by canvas count {n_canvas}"
        )

    per_canvas = total_patches // n_canvas
    positions = src_positions.reshape(n_canvas, per_canvas, 3)
    canvas_t = positions[..., 0]
    keep_mask = (canvas_t >= 0).any(axis=1)
    if bool((keep_mask & ~(canvas_t >= 0).all(axis=1)).any()):
        raise ValueError("half-padding canvas: padding is expected to be canvas-granular")

    dropped = int(n_canvas - int(keep_mask.sum()))
    if dropped == 0:
        return images, src_positions, 0

    kept_images = [image for image, keep in zip(images, keep_mask.tolist()) if keep]
    kept_positions = positions[keep_mask].reshape(-1, 3)
    return kept_images, kept_positions, dropped


def _timestamp_runs(
    patch_positions: torch.Tensor,
    fps: float,
    decimals: int,
    spatial_merge_size: int = 2,
) -> list[tuple[str, int]]:
    """Group consecutive patches by source frame into (timestamp, token count) runs."""
    unique_t, counts = torch.unique_consecutive(patch_positions[:, 0], return_counts=True)
    merge_factor = int(spatial_merge_size) ** 2

    runs: list[tuple[str, int]] = []
    for t_value, count in zip(unique_t.tolist(), counts.tolist()):
        if int(t_value) < 0:
            continue
        token_count = int(count) // merge_factor
        if token_count <= 0:
            continue
        runs.append((f"<{float(t_value) / float(fps):.{decimals}f} seconds>", token_count))
    return runs


def rewrite_text_with_codec_positions(
    text: str,
    patch_positions: torch.Tensor,
    fps: float,
    decimals: int = 1,
) -> str:
    """Replace the prompt's video placeholder with timestamped image-pad runs.

    Frame sampling can leave the placeholder alone because every frame contributes the
    same token count. Codec sampling cannot: each source frame contributes a different
    number of patches, and the model is told *when* each run happened. Returns ``text``
    unchanged when it carries no video placeholder.
    """
    match = _VIDEO_BLOCK_RE.search(text)
    if match is None:
        return text

    parts: list[str] = []
    for timestamp, token_count in _timestamp_runs(patch_positions, fps, decimals):
        parts.extend([timestamp, VISION_START, IMAGE_PAD * token_count, VISION_END, "\n"])

    tail_start = match.end()
    if tail_start < len(text) and text[tail_start] == "\n":
        tail_start += 1
    return text[: match.start()] + "".join(parts) + text[tail_start:]


def codec_image_processor_outputs(image_processor, images: list[Any], max_pixels: int) -> Any:
    """Run the image processor over codec canvases without letting it resize them.

    The canvases already sit on the patch grid, so any smart-resize would desynchronise
    ``image_grid_thw`` from the codec's position table. Clamping the budget to the actual
    canvas sizes -- up for ``max_pixels``, down for ``min_pixels`` -- pins the resize to a
    no-op. The ``min_pixels`` clamp is the load-bearing half: the processor's default
    (200704) would otherwise *upscale* small canvases into extra patches.
    """
    canvas_pixels = [image.width * image.height for image in images]
    return image_processor(
        images=images,
        min_pixels=min(canvas_pixels) if canvas_pixels else 1,
        max_pixels=max(int(max_pixels), max(canvas_pixels, default=int(max_pixels))),
        return_tensors="pt",
    )


def load_codec_result(asset_dir: str | Path) -> dict[str, Any]:
    """Read a codec asset directory into canvases, positions and metadata."""
    from PIL import Image

    asset_dir = Path(asset_dir)
    positions_path = asset_dir / "src_patch_position.npy"
    meta_path = asset_dir / "meta.json"
    if not positions_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"{asset_dir} is not a codec asset directory: expected src_patch_position.npy "
            "and meta.json alongside the canvas images"
        )

    meta = json.loads(meta_path.read_text())
    canvases = sorted(asset_dir.glob("canvas_*.jpg")) or sorted(asset_dir.glob("canvas_*.png"))
    if not canvases:
        raise FileNotFoundError(f"no canvas_*.jpg / canvas_*.png found in {asset_dir}")

    return {
        "images": [Image.open(path).convert("RGB") for path in canvases],
        "src_positions": np.load(positions_path),
        "fps": float(meta["fps"]),
        "decimals": int(meta.get("decimals", 1)),
    }


def process_codec_video(video_url: str, cfg: CodecConfig) -> dict[str, Any]:
    """Produce a codec asset directory for ``video_url``.

    Not implemented: both engines need tooling that this repository does not ship. The
    error names what is missing rather than letting the request fail somewhere downstream
    with a shape mismatch. Point ``codec_config={"asset_dir": ...}`` at a directory the
    reference pipeline already produced to use the codec path today.
    """
    cfg.validate()
    if cfg.engine in ("hevc", "cv-preinfer"):
        raise NotImplementedError(
            "engine='hevc' needs the external 'cv-preinfer' binary, which is not bundled "
            "with the checkpoint or with vllm-omni. Pass an already-produced asset "
            "directory via codec_config={'asset_dir': ...}, or use frame sampling "
            "(video_backend='frames')."
        )
    if cfg.engine == "dcvc-rt":
        raise NotImplementedError(
            "engine='dcvc-rt' needs the checkpoint's bundled neural_codec package and its "
            "compiled CUDA extensions, which vllm-omni does not vendor. Pass an "
            "already-produced asset directory via codec_config={'asset_dir': ...}, or use "
            "frame sampling (video_backend='frames')."
        )
    raise ValueError(f"unknown codec engine: {cfg.engine!r} (use 'hevc' or 'dcvc-rt')")


__all__ = [
    "CodecConfig",
    "DcvcConfig",
    "codec_image_processor_outputs",
    "codec_positions_for_processor",
    "drop_padding_canvases",
    "load_codec_result",
    "process_codec_video",
    "rewrite_text_with_codec_positions",
]
