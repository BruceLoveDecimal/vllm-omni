# SPDX-License-Identifier: Apache-2.0
"""Window planning for MiniMax H3 sliding-window long-video generation.

Spec: ``docs/design/feature/minimax_h3_sliding_window.md`` (issue #6737).

Planning works in frames on the model's native grids so downstream shapes
never need re-alignment:

- window frame counts sit on the 17n+5 boundary (``align_frame_count``);
- the overlap sits on whole video latent groups (17 frames -> 5 latent
  frames) *and* whole audio latent rows (40 Hz over 24 FPS -> 3-frame
  grid), i.e. on multiples of lcm(17, 3) = 51 frames, so tail slices and
  stitch points are exact for video latents, audio latents, and the
  32 kHz waveform alike.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from vllm_omni.errors import OmniClientError

from .time_request import MINIMAX_H3_SHAPE_PLANNER

MINIMAX_H3_WINDOW_FPS = 24
MINIMAX_H3_MAX_WINDOWED_OUTPUT_SECONDS = 60.0
MINIMAX_H3_DEFAULT_WINDOW_OVERLAP_SECONDS = 1.0
MINIMAX_H3_MIN_WINDOW_OVERLAP_SECONDS = 0.5
MINIMAX_H3_MAX_WINDOW_OVERLAP_SECONDS = 3.0
MINIMAX_H3_MIN_MAX_WINDOW_SECONDS = 8.0
MINIMAX_H3_MAX_MAX_WINDOW_SECONDS = 15.0

# lcm of the 17-frame video latent group and the 3-frame audio latent grid.
MINIMAX_H3_WINDOW_OVERLAP_FRAME_UNIT = 51
_VIDEO_LATENT_PER_GROUP = 5
_VIDEO_FRAMES_PER_GROUP = 17
_AUDIO_LATENT_HZ = 40

# Flipped on by milestone M3 once the windowed forward path exists; until
# then long durations are rejected at validation time.
MINIMAX_H3_WINDOWED_GENERATION_ENABLED = False

_NATIVE_MIN_SECONDS = 4.0
_NATIVE_MAX_SECONDS = 15.0


@dataclass(frozen=True)
class WindowSpec:
    """One generation window; all counts sit on the model's native grids."""

    index: int
    num_frames: int
    net_frames: int
    overlap_frames: int
    latent_t: int
    audio_t: int
    overlap_latent_t: int
    overlap_audio_t: int


@dataclass(frozen=True)
class WindowPlan:
    windows: tuple[WindowSpec, ...]
    total_frames: int
    overlap_frames: int

    @property
    def num_windows(self) -> int:
        return len(self.windows)


def validate_output_duration(duration_seconds: float) -> None:
    """Reject durations the current build cannot serve.

    Raises :class:`OmniClientError` outside [4, 15] seconds natively, or
    outside [4, 60] once ``MINIMAX_H3_WINDOWED_GENERATION_ENABLED`` is on.
    """
    duration = float(duration_seconds)
    if not math.isfinite(duration) or not (_NATIVE_MIN_SECONDS <= duration <= MINIMAX_H3_MAX_WINDOWED_OUTPUT_SECONDS):
        raise OmniClientError(
            f"MiniMax H3 output duration must be in [4, 15] seconds natively "
            f"(up to {MINIMAX_H3_MAX_WINDOWED_OUTPUT_SECONDS:g} with sliding-window generation), "
            f"got {duration}"
        )
    if duration > _NATIVE_MAX_SECONDS and not MINIMAX_H3_WINDOWED_GENERATION_ENABLED:
        raise OmniClientError(
            f"MiniMax H3 sliding-window generation for durations in "
            f"(15, {MINIMAX_H3_MAX_WINDOWED_OUTPUT_SECONDS:g}] seconds is not yet enabled; "
            f"output duration must be in [4, 15] seconds, got {duration}"
        )


def _overlap_frames_from_seconds(window_overlap_seconds: float | None) -> int:
    if window_overlap_seconds is None:
        seconds = MINIMAX_H3_DEFAULT_WINDOW_OVERLAP_SECONDS
    else:
        if isinstance(window_overlap_seconds, bool):
            raise OmniClientError(f"MiniMax H3 window_overlap_seconds must be a number, got {window_overlap_seconds!r}")
        try:
            seconds = float(window_overlap_seconds)
        except (TypeError, ValueError) as exc:
            raise OmniClientError(
                f"MiniMax H3 window_overlap_seconds must be a number, got {window_overlap_seconds!r}"
            ) from exc
        if not math.isfinite(seconds) or seconds <= 0:
            raise OmniClientError(f"MiniMax H3 window_overlap_seconds must be a positive number, got {seconds}")
    seconds = min(
        max(seconds, MINIMAX_H3_MIN_WINDOW_OVERLAP_SECONDS),
        MINIMAX_H3_MAX_WINDOW_OVERLAP_SECONDS,
    )
    unit = MINIMAX_H3_WINDOW_OVERLAP_FRAME_UNIT
    return unit * max(1, round(seconds * MINIMAX_H3_WINDOW_FPS / unit))


def _max_window_frames_from_seconds(max_window_seconds: float | None) -> int:
    if max_window_seconds is None:
        seconds = MINIMAX_H3_MAX_MAX_WINDOW_SECONDS
    else:
        if isinstance(max_window_seconds, bool):
            raise OmniClientError(f"MiniMax H3 max_window_seconds must be a number, got {max_window_seconds!r}")
        try:
            seconds = float(max_window_seconds)
        except (TypeError, ValueError) as exc:
            raise OmniClientError(
                f"MiniMax H3 max_window_seconds must be a number, got {max_window_seconds!r}"
            ) from exc
        if not math.isfinite(seconds) or not (
            MINIMAX_H3_MIN_MAX_WINDOW_SECONDS <= seconds <= MINIMAX_H3_MAX_MAX_WINDOW_SECONDS
        ):
            raise OmniClientError(
                f"MiniMax H3 max_window_seconds must be in "
                f"[{MINIMAX_H3_MIN_MAX_WINDOW_SECONDS:g}, {MINIMAX_H3_MAX_MAX_WINDOW_SECONDS:g}] seconds, "
                f"got {max_window_seconds}"
            )
    return MINIMAX_H3_SHAPE_PLANNER.align_frame_count(round(seconds * MINIMAX_H3_WINDOW_FPS))


def _overlap_latent_t(overlap_frames: int) -> int:
    return overlap_frames // _VIDEO_FRAMES_PER_GROUP * _VIDEO_LATENT_PER_GROUP


def _overlap_audio_t(overlap_frames: int) -> int:
    # Exact for multiples of 3 frames; overlap sits on the 51-frame grid.
    return overlap_frames * _AUDIO_LATENT_HZ // MINIMAX_H3_WINDOW_FPS


def _window_spec(index: int, num_frames: int, overlap_frames: int) -> WindowSpec:
    return WindowSpec(
        index=index,
        num_frames=num_frames,
        net_frames=num_frames - overlap_frames,
        overlap_frames=overlap_frames,
        latent_t=MINIMAX_H3_SHAPE_PLANNER.video_latent_t(num_frames),
        audio_t=MINIMAX_H3_SHAPE_PLANNER.audio_latent_t(num_frames / MINIMAX_H3_WINDOW_FPS),
        overlap_latent_t=_overlap_latent_t(overlap_frames),
        overlap_audio_t=_overlap_audio_t(overlap_frames),
    )


def plan_windows(
    *,
    requested_frames: int,
    overlap_frames: int = MINIMAX_H3_WINDOW_OVERLAP_FRAME_UNIT,
    min_window_frames: int | None = None,
    max_window_frames: int | None = None,
) -> WindowPlan:
    """Decompose ``requested_frames`` into overlapped windows.

    Every window's ``num_frames`` is 17n+5-aligned and lies in
    ``[min_window_frames, max_window_frames]``; windows after the first
    re-cover the previous window's last ``overlap_frames`` frames, so the
    net frames sum to at least ``requested_frames`` (alignment may add up
    to one grid step per window, mirroring the single-pass align-up).
    """
    align = MINIMAX_H3_SHAPE_PLANNER.align_frame_count
    if requested_frames <= 0:
        raise OmniClientError(f"MiniMax H3 requested_frames must be positive, got {requested_frames}")
    if overlap_frames <= 0 or overlap_frames % MINIMAX_H3_WINDOW_OVERLAP_FRAME_UNIT != 0:
        raise OmniClientError(
            f"MiniMax H3 window overlap must be a positive multiple of "
            f"{MINIMAX_H3_WINDOW_OVERLAP_FRAME_UNIT} frames, got {overlap_frames}"
        )
    min_frames = (
        align(round(_NATIVE_MIN_SECONDS * MINIMAX_H3_WINDOW_FPS)) if min_window_frames is None else min_window_frames
    )
    max_frames = (
        align(round(_NATIVE_MAX_SECONDS * MINIMAX_H3_WINDOW_FPS)) if max_window_frames is None else max_window_frames
    )
    if align(min_frames) != min_frames or align(max_frames) != max_frames:
        raise OmniClientError("MiniMax H3 window frame bounds must sit on the 17n+5 grid")
    if not 0 < min_frames <= max_frames:
        raise OmniClientError(f"MiniMax H3 window frame bounds are invalid: min={min_frames}, max={max_frames}")
    if overlap_frames >= min_frames:
        raise OmniClientError(
            f"MiniMax H3 window overlap ({overlap_frames} frames) must be smaller than "
            f"the minimum window ({min_frames} frames)"
        )

    if requested_frames <= max_frames:
        num_frames = align(requested_frames)
        if num_frames < min_frames:
            raise OmniClientError(
                f"MiniMax H3 requested_frames={requested_frames} is below the minimum window of {min_frames} frames"
            )
        window = _window_spec(0, num_frames, overlap_frames=0)
        return WindowPlan(windows=(window,), total_frames=window.num_frames, overlap_frames=0)

    stride = max_frames - overlap_frames
    num_windows = 1 + math.ceil((requested_frames - max_frames) / stride)
    while True:
        gross_total = requested_frames + (num_windows - 1) * overlap_frames
        uniform = align(math.ceil(gross_total / num_windows))
        if uniform <= max_frames:
            break
        num_windows += 1

    last_raw = gross_total - (num_windows - 1) * uniform
    last = align(max(last_raw, min_frames))

    windows = [
        _window_spec(index, uniform, overlap_frames=0 if index == 0 else overlap_frames)
        for index in range(num_windows - 1)
    ]
    windows.append(_window_spec(num_windows - 1, last, overlap_frames=overlap_frames))
    total_frames = sum(spec.net_frames for spec in windows)

    for spec in windows:
        if not min_frames <= spec.num_frames <= max_frames:
            raise AssertionError(
                f"window {spec.index} of {num_windows} has {spec.num_frames} frames, "
                f"outside [{min_frames}, {max_frames}] for requested_frames={requested_frames}"
            )
    if total_frames < requested_frames:
        raise AssertionError(
            f"planned {total_frames} net frames < requested {requested_frames} "
            f"({num_windows} windows, overlap {overlap_frames})"
        )
    return WindowPlan(windows=tuple(windows), total_frames=total_frames, overlap_frames=overlap_frames)


def plan_windows_for_duration(
    duration_seconds: float,
    *,
    window_overlap_seconds: float | None = None,
    max_window_seconds: float | None = None,
) -> WindowPlan:
    """Plan windows for a duration request, resolving the user-facing knobs."""
    duration = float(duration_seconds)
    if not math.isfinite(duration) or not (_NATIVE_MIN_SECONDS <= duration <= MINIMAX_H3_MAX_WINDOWED_OUTPUT_SECONDS):
        raise OmniClientError(
            f"MiniMax H3 output duration must be in "
            f"[{_NATIVE_MIN_SECONDS:g}, {MINIMAX_H3_MAX_WINDOWED_OUTPUT_SECONDS:g}] seconds "
            f"for windowed planning, got {duration}"
        )
    return plan_windows(
        requested_frames=round(duration * MINIMAX_H3_WINDOW_FPS),
        overlap_frames=_overlap_frames_from_seconds(window_overlap_seconds),
        max_window_frames=_max_window_frames_from_seconds(max_window_seconds),
    )
