# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Evoke geometry: bank, renderers and latent visibility on synthetic scenes."""

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.evoke.geometry.cloud import (
    DegenerateGeometryError,
    FrameBank,
    build_single_source_warp,
    render_backward_zbuf,
    render_intrinsics,
    umeyama_scale,
    visibility_to_latent,
)
from vllm_omni.diffusion.models.evoke.geometry.session import GeometryRecipe, GeometrySession

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

H, W = 48, 80
K = render_intrinsics(60.0, 60.0, 40.0, 24.0, H, W)


def _pose(x: float = 0.0, z: float = 0.0) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = x
    pose[2, 3] = z
    return pose


class _PlaneEstimator:
    """A fronto-parallel textured plane at a fixed depth, at the input resolution."""

    def __init__(self, depth: float = 4.0, skip: bool = False):
        self.depth = depth
        self.skip = skip

    def depth_window(self, frames, c2w, ks):
        if self.skip:
            raise DegenerateGeometryError("zero baseline")
        n, h, w, _ = frames.shape
        return (
            np.full((n, h, w), self.depth, np.float32),
            np.asarray(ks, np.float32),
            None,
            np.asarray(frames, np.float32),
        )

    def depth_single(self, frame, intrinsics):
        h, w, _ = frame.shape
        return (
            np.full((h, w), self.depth, np.float32),
            np.asarray(intrinsics, np.float32),
            np.asarray(frame, np.float32),
        )


def _frames(n: int) -> np.ndarray:
    return np.random.default_rng(0).random((n, H, W, 3)).astype(np.float32)


def test_cold_start_renders_black_and_invisible():
    warp, vis = render_backward_zbuf(FrameBank("cpu"), [], torch.eye(4)[None].repeat(3, 1, 1), K, H, W)
    assert warp.shape == (1, 3, 3, H, W) and vis.shape == (1, 1, 3, H, W)
    assert torch.all(warp == -1) and torch.all(vis == 0)


def test_same_pose_reprojection_recovers_the_source_frame():
    bank = FrameBank("cpu")
    # Flat colour: upstream samples source colours with a sub-pixel offset
    # (pixel index / (w - 1) under align_corners=False), so texture would blur.
    frames = np.full((3, H, W, 3), 0.3, np.float32)
    poses = np.stack([_pose(), _pose(x=0.1), _pose(x=0.2)])
    assert bank.ingest(_PlaneEstimator(), frames, poses, K, [0, 1, 2])
    target = torch.as_tensor(poses[2])[None]
    warp, vis = render_backward_zbuf(bank, [2], target, K, H, W)
    assert vis.mean() > 0.95
    expected = torch.as_tensor(frames[2]).permute(2, 0, 1) * 2 - 1
    covered = vis[0, 0, 0].bool()
    assert torch.allclose(warp[0, :, 0][:, covered], expected[:, covered], atol=1e-4)


def test_lateral_motion_disoccludes_one_border():
    bank = FrameBank("cpu")
    bank.ingest(_PlaneEstimator(), _frames(3), np.stack([_pose()] * 3), K, [0, 1, 2])
    # Moving right by a quarter of the plane width uncovers the right border.
    warp, vis = render_backward_zbuf(bank, [0, 1, 2], torch.as_tensor(_pose(x=1.0))[None], K, H, W)
    column_cover = vis[0, 0, 0].mean(0)
    assert column_cover[: W // 4].min() == 1 and column_cover[-W // 8 :].max() == 0


def test_skipped_window_is_not_ingested_and_other_errors_propagate():
    bank = FrameBank("cpu")
    assert not bank.ingest(_PlaneEstimator(skip=True), _frames(3), np.stack([_pose()] * 3), K, [0, 1, 2])
    assert len(bank) == 0

    class _Broken(_PlaneEstimator):
        def depth_window(self, *args):
            raise KeyError("missing head")

    with pytest.raises(KeyError):
        bank.ingest(_Broken(), _frames(3), np.stack([_pose()] * 3), K, [0, 1, 2])


def test_bounded_bank_evicts_oldest_frames():
    bank = FrameBank("cpu", max_frames=36)
    for chunk in range(3):
        gids = [chunk * 36 + i for i in range(0, 36, 12)]
        bank.ingest(_PlaneEstimator(), _frames(3), np.stack([_pose()] * 3), K, gids)
    assert min(bank.frames) > 108 - 36 - 36


def test_umeyama_scale_recovers_trajectory_scale_and_rejects_static():
    gt = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 1]], np.float64)
    assert umeyama_scale(gt, 0.25 * gt + 3.0) == pytest.approx(0.25)
    assert umeyama_scale(np.zeros((3, 3)), np.zeros((3, 3))) is None


def test_single_source_warp_matches_reference_at_its_own_pose():
    # Flat colour: equal-depth neighbours tie in the splat z-buffer.
    ref = torch.full((3, H, W), 0.25)
    poses = torch.as_tensor(np.stack([_pose(), _pose(z=0.2)]))
    warp, vis = build_single_source_warp(
        _PlaneEstimator(), ref, poses[0], K, poses, height=H, width=W, chunk0_target_disparity_px=0
    )
    covered = vis[0, 0, 0].bool()
    assert covered.float().mean() > 0.95
    assert torch.allclose(warp[0, :, 0][:, covered], ref[:, covered], atol=1e-2)


def test_visibility_to_latent_is_constant_per_token():
    vis = (torch.rand(1, 1, 33, H, W) > 0.5).float()
    latent = visibility_to_latent(vis, 9, H // 8, W // 8)
    assert latent.shape == (1, 1, 9, H // 8, W // 8)
    blocks = latent.unfold(3, 2, 2).unfold(4, 2, 2)
    assert torch.all(blocks.amax((-1, -2)) == blocks.amin((-1, -2)))


def test_session_pool_respects_lag():
    session = GeometrySession(
        _PlaneEstimator(), K, H, W, torch.device("cpu"), GeometryRecipe(lag_chunks=1), reference_frame=None
    )
    session.estimator.reset_stream = lambda: None
    frames = torch.rand(3, 36, H, W) * 2 - 1
    c2ws = torch.as_tensor(np.stack([_pose()] * 72), dtype=torch.float64)
    session.ingest(frames, 0, c2ws)
    session.ingest(frames, 36, c2ws)
    # lag=1: only frames at least one chunk older than the newest gid may be recalled.
    _, vis = session.render_chunk(2, c2ws[:2].float())
    assert vis.mean() > 0.9
    session.recipe = GeometryRecipe(lag_chunks=3)
    _, vis = session.render_chunk(2, c2ws[:2].float())
    assert vis.sum() == 0
    session.close()
    assert len(session.bank) == 0
