# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from Alaya-EVOKE (https://github.com/AlayaLab/Evoke @ 1d271be),
# evoke/modules/geometric_state/vigeo_cloud.py, Apache-2.0.
"""Depth from ViGeo, rescaled into the commanded camera trajectory's units.

ViGeo (https://github.com/aigc3d/ViGeo) is an optional, separately installed
package; it is imported lazily on the geometry owner rank only. Its network is
not vendored here. The ViGeo *weights* are CC-BY-NC-4.0.

ViGeo predicts depth up to scale and its own camera poses. The adapter fixes
the one free scalar and hands the bank depth at the commanded trajectory's
scale, unprojected with the commanded intrinsics:

* ``depth_median``: ``c = median(depth) / depth_median_target`` from the first
  window, locked for the session. Baseline-free, so it works for a still first
  chunk and for pure rotation. Default for i2v.
* ``anchor``: the Umeyama scale between commanded and predicted camera centres,
  median of the first ``anchor_windows`` windows, then locked. Default for v2v.
* ``per_window`` / ``fixed``: ablation / reproduction modes.

The adapter is stateful while streaming (ViGeo kv-cache, locked scale);
:meth:`reset_stream` must run at every session boundary.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.models.evoke.geometry.cloud import DegenerateGeometryError, umeyama_scale

logger = logging.getLogger(__name__)

PATCH = 14
# 1 cls + 4 register + 1 camera token precede every frame's patch grid.
EXTRA_TOKENS = 6
# Smallest kept-context the weakest global block may receive before ViGeo's
# eviction silently becomes a no-op (budget <= one frame of tokens).
MIN_KEEP_FRAMES = 4

SCALE_MODES = ("per_window", "anchor", "depth_median", "fixed")
VIGEO_INSTALL_HINT = (
    "Evoke geometry requires the ViGeo package: "
    'pip install --no-deps "vigeo @ git+https://github.com/aigc3d/ViGeo.git@87ee8fbe1f8f4ddda473b4f2562b80979a76305f"'
)


def fit_patch_resolution(h: int, w: int, long_side: int, patch: int = PATCH) -> tuple[int, int]:
    s = float(long_side) / float(max(h, w))
    return max(patch, int(round(h * s / patch)) * patch), max(patch, int(round(w * s / patch)) * patch)


def resolve_vigeo_weights(path: str | os.PathLike[str]) -> Path:
    """``vigeo.pt`` file, or a directory containing it."""
    path = Path(path)
    if path.is_dir():
        path = path / "vigeo.pt"
    if not path.is_file():
        raise FileNotFoundError(f"ViGeo weights not found at {path}; set model_config.evoke_vigeo_path.")
    return path


class ViGeoDepthEstimator:
    def __init__(
        self,
        weights: str | os.PathLike[str],
        device: torch.device | str,
        *,
        process_res: int = 644,
        mode: str = "chunk",
        chunk_size: int = 16,
        scale_mode: str = "depth_median",
        depth_median_target: float = 5.0,
        anchor_windows: int = 4,
        cache_keep_frames: int = 6,
        scale_value: float = 0.0,
    ) -> None:
        if mode not in ("offline", "chunk", "online"):
            raise ValueError(f"ViGeo mode must be offline/chunk/online, got {mode!r}")
        if scale_mode not in SCALE_MODES:
            raise ValueError(f"ViGeo scale_mode must be one of {SCALE_MODES}, got {scale_mode!r}")
        if scale_mode == "fixed" and not scale_value > 0:
            raise ValueError("ViGeo scale_mode=fixed needs scale_value > 0")
        if not depth_median_target > 0:
            raise ValueError("ViGeo depth_median_target must be > 0")
        self.weights = resolve_vigeo_weights(weights)
        self.device = torch.device(device)
        self.process_res = int(process_res)
        self.mode = mode
        self.chunk_size = int(chunk_size)
        self.scale_mode = scale_mode
        self.depth_median_target = float(depth_median_target)
        self.anchor_windows = int(anchor_windows)
        self.cache_keep_frames = int(cache_keep_frames)
        self.scale_value = float(scale_value)
        self._model = None
        self._kv = None
        self._anchor_scales: list[float] = []
        self._scale_locked: float | None = None

    @property
    def streaming(self) -> bool:
        return self.mode in ("chunk", "online")

    def load(self) -> None:
        """Build ViGeo in fp32 regardless of the engine's default dtype."""
        if self._model is not None:
            return
        try:
            from vigeo import ViGeo
        except ImportError as exc:
            raise ImportError(VIGEO_INSTALL_HINT) from exc
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            model = ViGeo.from_pretrained(str(self.weights))
        finally:
            torch.set_default_dtype(previous)
        self._model = model.to(self.device).eval()
        logger.info("Loaded ViGeo from %s (scale_mode=%s mode=%s)", self.weights, self.scale_mode, self.mode)

    def configure_session(self, *, scale_mode: str | None = None) -> None:
        """Start a new stream, optionally switching the scale mode for this session."""
        if scale_mode is not None:
            if scale_mode not in SCALE_MODES:
                raise ValueError(f"ViGeo scale_mode must be one of {SCALE_MODES}, got {scale_mode!r}")
            self.scale_mode = scale_mode
        self.reset_stream()

    def reset_stream(self) -> None:
        """Drop the kv-cache and the locked scale together: one is meaningless without the other."""
        self._kv = None
        self._anchor_scales = []
        self._scale_locked = None
        if self._model is not None:
            self._model.reset_cache_state()

    def _budget(self, h: int, w: int) -> int:
        tokens_per_frame = (h // PATCH) * (w // PATCH) + EXTRA_TOKENS
        num_blocks = int(getattr(self._model.pretrained, "num_global_blocks", 0))
        if num_blocks <= 0:
            raise RuntimeError("ViGeo reports no global blocks; cannot size the streaming kv-cache budget.")
        return num_blocks * max(MIN_KEEP_FRAMES, self.cache_keep_frames) * tokens_per_frame

    def _infer(self, frames_rgb: np.ndarray):
        x = torch.as_tensor(np.ascontiguousarray(frames_rgb), dtype=torch.float32).permute(0, 3, 1, 2).clamp(0, 1)
        _, _, H, W = x.shape
        h, w = fit_patch_resolution(H, W, self.process_res)
        if (h, w) != (H, W):
            x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        x = x.to(self.device)
        if self.streaming:
            out = self._model.infer(
                x,
                mode=self.mode,
                chunk_size=self.chunk_size,
                num_tokens=None,
                resize_output=True,
                kv_caches=self._kv,
                reset_cache=self._kv is None,
                total_budget=self._budget(int(x.shape[-2]), int(x.shape[-1])),
            )
            self._kv = out["kv_caches"]
        else:
            out = self._model.infer(x, mode=self.mode, num_tokens=None, resize_output=True)
        depth = out["depth_pred"][:, 0].float()
        conf = out["conf_pred"]
        conf = None if conf is None else conf[:, 0].float()
        pose = out["pose_pred"].float().cpu().numpy()
        n = depth.shape[0]
        c2w_pred = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
        c2w_pred[:, :3, :4] = pose
        rgb = x.permute(0, 2, 3, 1).float().cpu().numpy()
        return depth, conf, c2w_pred, rgb, (int(depth.shape[-2]), int(depth.shape[-1]))

    def _baseline_free_scale(self, depth: torch.Tensor) -> float:
        if self._scale_locked is None:
            if self.scale_mode == "fixed":
                self._scale_locked = self.scale_value
            else:
                median = float(torch.median(depth.detach().float()).item())
                if not np.isfinite(median) or median <= 0:
                    raise DegenerateGeometryError(f"ViGeo depth median is {median!r}")
                self._scale_locked = median / self.depth_median_target
        return self._scale_locked

    def _trajectory_scale(self, raw: float | None) -> float | None:
        if raw is None:
            return None
        if self.scale_mode == "per_window":
            return raw
        if self._scale_locked is not None:
            return self._scale_locked
        self._anchor_scales.append(float(raw))
        if len(self._anchor_scales) >= max(1, self.anchor_windows):
            self._scale_locked = float(np.median(self._anchor_scales))
            return self._scale_locked
        return raw

    @torch.no_grad()
    def depth_single(self, frame_rgb: np.ndarray, k_gt: np.ndarray):
        """One reference image -> (depth ``[h, w]``, intrinsics at depth res, rgb ``[h, w, 3]``).

        Isolated from the stream: the reference-image forward neither leaks its
        kv-cache nor its scale into the rollout, which re-anchors on its own.
        """
        if self.scale_mode not in ("depth_median", "fixed"):
            raise NotImplementedError("the single-image warp needs scale_mode depth_median or fixed")
        self.load()
        frame = np.asarray(frame_rgb, np.float32)
        H, W = int(frame.shape[0]), int(frame.shape[1])
        self.reset_stream()
        try:
            depth, _conf, _pose, rgb, (h, w) = self._infer(frame[None])
            scale = self._baseline_free_scale(depth)
            out_depth = (depth / scale)[0].cpu().numpy().astype(np.float32)
            intr = _scale_intrinsics(k_gt, (H, W), (h, w))[0]
            return out_depth, intr, np.asarray(rgb[0], np.float32)
        finally:
            self.reset_stream()

    @torch.no_grad()
    def depth_window(self, frames_rgb: np.ndarray, c2w_gt: np.ndarray, k_gt: np.ndarray):
        """Window ``[N, H, W, 3]`` -> (depth, intrinsics, conf, rgb) at depth resolution, GT scale."""
        frames = np.asarray(frames_rgb, np.float32)
        n, H, W = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
        if n < 3:
            raise ValueError(f"ViGeo windows need >= 3 posed frames, got {n}")
        self.load()
        depth, conf_t, c2w_pred, rgb, (h, w) = self._infer(frames)
        raw = umeyama_scale(np.asarray(c2w_gt, np.float64)[:, :3, 3], c2w_pred[:, :3, 3].astype(np.float64))
        if self.scale_mode in ("depth_median", "fixed"):
            scale = self._baseline_free_scale(depth)
        else:
            scale = self._trajectory_scale(raw)
        if scale is None:
            raise DegenerateGeometryError("zero camera baseline; no solvable depth scale")
        out_depth = (depth / float(scale)).cpu().numpy().astype(np.float32)
        intr = _scale_intrinsics(k_gt, (H, W), (h, w))
        if intr.shape[0] == 1 and n > 1:
            intr = np.repeat(intr, n, axis=0)
        conf = None if conf_t is None else conf_t.clamp(-30, 30).exp().cpu().numpy().astype(np.float32)
        return out_depth, intr, conf, rgb


def _scale_intrinsics(intrinsics: np.ndarray, src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> np.ndarray:
    (H, W), (h, w) = src_hw, dst_hw
    out = np.asarray(intrinsics, np.float32).copy()
    if out.ndim == 2:
        out = out[None]
    out[:, 0, 0] *= w / float(W)
    out[:, 0, 2] *= w / float(W)
    out[:, 1, 1] *= h / float(H)
    out[:, 1, 2] *= h / float(H)
    return out
