# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L1 CPU tests for MiniMax H3 sliding-window planning (issue #6737, M1)."""

import math

import pytest

from vllm_omni.diffusion.models.minimax_h3.sliding_window import (
    MINIMAX_H3_MAX_WINDOWED_OUTPUT_SECONDS,
    MINIMAX_H3_WINDOW_FPS,
    MINIMAX_H3_WINDOW_OVERLAP_FRAME_UNIT,
    WindowPlan,
    plan_windows,
    plan_windows_for_duration,
    validate_output_duration,
)
from vllm_omni.diffusion.models.minimax_h3.time_request import MINIMAX_H3_SHAPE_PLANNER
from vllm_omni.errors import OmniClientError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

FPS = MINIMAX_H3_WINDOW_FPS
MIN_WINDOW_FRAMES = MINIMAX_H3_SHAPE_PLANNER.align_frame_count(round(4.0 * FPS))
MAX_WINDOW_FRAMES = MINIMAX_H3_SHAPE_PLANNER.align_frame_count(round(15.0 * FPS))


def assert_plan_invariants(plan: WindowPlan, requested_frames: int, overlap_frames: int) -> None:
    align = MINIMAX_H3_SHAPE_PLANNER.align_frame_count
    assert plan.num_windows >= 1
    assert plan.total_frames >= requested_frames
    # Align-up may add at most one 17-frame grid step per window, mirroring
    # the single-pass path, plus a min-window clamp on the last window.
    assert plan.total_frames <= requested_frames + 17 * plan.num_windows + MIN_WINDOW_FRAMES
    assert plan.total_frames == sum(spec.net_frames for spec in plan.windows)

    for position, spec in enumerate(plan.windows):
        assert spec.index == position
        assert align(spec.num_frames) == spec.num_frames
        assert MIN_WINDOW_FRAMES <= spec.num_frames <= MAX_WINDOW_FRAMES
        expected_overlap = 0 if position == 0 else overlap_frames
        assert spec.overlap_frames == expected_overlap
        assert spec.net_frames == spec.num_frames - expected_overlap
        assert spec.net_frames > 0
        assert spec.latent_t == MINIMAX_H3_SHAPE_PLANNER.video_latent_t(spec.num_frames)
        assert spec.audio_t == MINIMAX_H3_SHAPE_PLANNER.audio_latent_t(spec.num_frames / FPS)
        assert spec.overlap_latent_t == expected_overlap // 17 * 5
        assert spec.overlap_audio_t == expected_overlap * 40 // FPS
        # The overlap grid keeps latent and waveform slices exact.
        assert expected_overlap * 40 % FPS == 0
        assert expected_overlap % 17 == 0


class TestPlanWindows:
    def test_native_duration_stays_single_window(self):
        plan = plan_windows(requested_frames=round(12.0 * FPS))
        assert plan.num_windows == 1
        assert plan.windows[0].overlap_frames == 0
        assert plan.total_frames == MINIMAX_H3_SHAPE_PLANNER.align_frame_count(288)

    def test_16s_is_never_planned_as_15_plus_1(self):
        requested = round(16.0 * FPS)
        plan = plan_windows(requested_frames=requested)
        assert plan.num_windows == 2
        assert_plan_invariants(plan, requested, MINIMAX_H3_WINDOW_OVERLAP_FRAME_UNIT)
        # Even distribution: no degenerate tail window near the minimum, and
        # no head window pinned at the maximum.
        first, second = plan.windows
        assert first.num_frames < MAX_WINDOW_FRAMES
        assert second.num_frames > MIN_WINDOW_FRAMES
        assert abs(first.num_frames - second.num_frames) <= 17

    def test_30s_plan(self):
        requested = round(30.0 * FPS)
        plan = plan_windows(requested_frames=requested)
        assert_plan_invariants(plan, requested, MINIMAX_H3_WINDOW_OVERLAP_FRAME_UNIT)
        assert plan.num_windows == 3

    def test_60s_plan(self):
        requested = round(60.0 * FPS)
        plan = plan_windows(requested_frames=requested)
        assert_plan_invariants(plan, requested, MINIMAX_H3_WINDOW_OVERLAP_FRAME_UNIT)

    @pytest.mark.parametrize("duration_ds", range(41, 601))
    def test_every_decisecond_duration_plans_validly(self, duration_ds):
        """Property sweep: all durations in [4.1, 60.0] s at 0.1 s steps."""
        requested = round(duration_ds / 10.0 * FPS)
        plan = plan_windows(requested_frames=requested)
        assert_plan_invariants(plan, requested, MINIMAX_H3_WINDOW_OVERLAP_FRAME_UNIT)
        if requested <= MAX_WINDOW_FRAMES:
            assert plan.num_windows == 1
        else:
            assert plan.num_windows >= 2

    @pytest.mark.parametrize("overlap_units", [1, 2])
    def test_wider_overlap_still_plans_validly(self, overlap_units):
        overlap = overlap_units * MINIMAX_H3_WINDOW_OVERLAP_FRAME_UNIT
        for seconds in (16.0, 30.0, 45.0, 60.0):
            requested = round(seconds * FPS)
            plan = plan_windows(requested_frames=requested, overlap_frames=overlap)
            assert_plan_invariants(plan, requested, overlap)

    def test_boundary_just_above_native_max(self):
        # 15.04 s: barely over one window, must still split evenly.
        requested = round(15.04 * FPS)
        assert requested > MAX_WINDOW_FRAMES - 17  # sits at the boundary band
        plan = plan_windows(requested_frames=requested + 17)
        if plan.num_windows > 1:
            assert_plan_invariants(plan, requested + 17, MINIMAX_H3_WINDOW_OVERLAP_FRAME_UNIT)

    def test_rejects_bad_overlap(self):
        with pytest.raises(OmniClientError, match="multiple of 51"):
            plan_windows(requested_frames=1000, overlap_frames=24)
        with pytest.raises(OmniClientError, match="multiple of 51"):
            plan_windows(requested_frames=1000, overlap_frames=0)

    def test_rejects_overlap_not_smaller_than_min_window(self):
        with pytest.raises(OmniClientError, match="smaller than"):
            plan_windows(requested_frames=1000, overlap_frames=51 * 3)

    def test_rejects_misaligned_bounds(self):
        with pytest.raises(OmniClientError, match="17n\\+5 grid"):
            plan_windows(requested_frames=1000, max_window_frames=360)

    def test_rejects_nonpositive_frames(self):
        with pytest.raises(OmniClientError, match="positive"):
            plan_windows(requested_frames=0)

    def test_rejects_tiny_single_window(self):
        with pytest.raises(OmniClientError, match="below the"):
            plan_windows(requested_frames=50)


class TestPlanWindowsForDuration:
    def test_default_knobs(self):
        plan = plan_windows_for_duration(30.0)
        assert_plan_invariants(plan, round(30.0 * FPS), MINIMAX_H3_WINDOW_OVERLAP_FRAME_UNIT)

    def test_overlap_seconds_snaps_to_frame_unit(self):
        for seconds in (0.5, 1.0, 2.0, 3.0):
            plan = plan_windows_for_duration(30.0, window_overlap_seconds=seconds)
            assert plan.overlap_frames % MINIMAX_H3_WINDOW_OVERLAP_FRAME_UNIT == 0
            assert plan.overlap_frames >= MINIMAX_H3_WINDOW_OVERLAP_FRAME_UNIT

    def test_overlap_seconds_clamped_not_rejected(self):
        # The spec's table clamps this knob into [0.5, 3.0].
        low = plan_windows_for_duration(30.0, window_overlap_seconds=0.5)
        clamped = plan_windows_for_duration(30.0, window_overlap_seconds=0.05)
        assert clamped.overlap_frames == low.overlap_frames

    def test_overlap_seconds_rejects_nonsense(self):
        for bad in (True, "fast", float("nan"), -1.0, 0.0):
            with pytest.raises(OmniClientError):
                plan_windows_for_duration(30.0, window_overlap_seconds=bad)

    def test_max_window_seconds_bounds(self):
        plan = plan_windows_for_duration(30.0, max_window_seconds=8.0)
        max_frames = MINIMAX_H3_SHAPE_PLANNER.align_frame_count(round(8.0 * FPS))
        assert all(spec.num_frames <= max_frames for spec in plan.windows)
        for bad in (7.9, 15.1, True, "big", float("inf")):
            with pytest.raises(OmniClientError):
                plan_windows_for_duration(30.0, max_window_seconds=bad)

    def test_duration_out_of_planning_range(self):
        for bad in (3.9, 60.1, float("nan"), float("inf")):
            with pytest.raises(OmniClientError, match="windowed planning"):
                plan_windows_for_duration(bad)

    def test_full_knob_sweep_stays_valid(self):
        for duration in (16.0, 20.0, 30.0, 45.0, 60.0):
            for overlap in (0.5, 1.5, 3.0):
                for max_window in (8.0, 12.0, 15.0):
                    plan = plan_windows_for_duration(
                        duration,
                        window_overlap_seconds=overlap,
                        max_window_seconds=max_window,
                    )
                    requested = round(duration * FPS)
                    assert plan.total_frames >= requested
                    max_frames = MINIMAX_H3_SHAPE_PLANNER.align_frame_count(round(max_window * FPS))
                    for spec in plan.windows:
                        assert MIN_WINDOW_FRAMES <= spec.num_frames <= max_frames


class TestValidateOutputDuration:
    def test_native_range_passes(self):
        for duration in (4.0, 5.0, 8.7, 15.0):
            validate_output_duration(duration)

    def test_windowed_band_reports_not_enabled(self):
        for duration in (15.1, 16.0, 30.0, MINIMAX_H3_MAX_WINDOWED_OUTPUT_SECONDS):
            with pytest.raises(OmniClientError, match="not yet enabled"):
                validate_output_duration(duration)

    def test_out_of_range_names_both_limits(self):
        for duration in (0.0, 3.99, 60.01, math.inf, math.nan):
            with pytest.raises(OmniClientError, match="sliding-window"):
                validate_output_duration(duration)


class TestPipelineIntegration:
    """The pipeline's duration checks route through the validator."""

    def _resolve(self, sampling_kwargs):
        from types import SimpleNamespace

        from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

        pipeline = object.__new__(MiniMaxH3Pipeline)
        defaults = {"fps": None, "height": None, "width": None, "num_frames": None, "extra_args": {}}
        sampling = SimpleNamespace(**{**defaults, **sampling_kwargs})
        return MiniMaxH3Pipeline._resolve_shape(pipeline, "t2va", sampling, None)

    def test_native_duration_still_resolves(self):
        height, width, num_frames, latent_t, audio_t = self._resolve(
            {"extra_args": {"duration": 5.0, "aspect_ratio": "16:9"}}
        )
        assert num_frames == MINIMAX_H3_SHAPE_PLANNER.align_frame_count(120)
        assert latent_t == MINIMAX_H3_SHAPE_PLANNER.video_latent_t(num_frames)

    def test_long_duration_reports_windowed_status(self):
        with pytest.raises(OmniClientError, match="not yet enabled"):
            self._resolve({"extra_args": {"duration": 30.0, "aspect_ratio": "16:9"}})

    def test_long_num_frames_reports_windowed_status(self):
        with pytest.raises(OmniClientError, match="not yet enabled"):
            self._resolve({"num_frames": 500, "extra_args": {"aspect_ratio": "16:9"}})

    def test_far_out_of_range_duration_rejected(self):
        with pytest.raises(OmniClientError, match="sliding-window"):
            self._resolve({"extra_args": {"duration": 100.0, "aspect_ratio": "16:9"}})
