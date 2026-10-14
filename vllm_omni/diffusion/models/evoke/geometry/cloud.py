# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from Alaya-EVOKE (https://github.com/AlayaLab/Evoke @ 1d271be),
# evoke/modules/geometric_state/da3_cloud.py and dataset/online_materialize.py, Apache-2.0.
"""Pose-indexed point-cloud bank and the warp renderers of the shipped recipe.

Only the code paths the real-time recipe (``infer_post_distill.sh``) executes
are ported: the per-frame bank with a confidence-percentile gate, the
``backward_zbuf`` per-pixel multi-source renderer, and the i2v chunk-0
single-image warp. The upstream hygiene / consistency-gate / carving /
re-anchoring experiments default to off and are not carried over.

Conventions: poses are OpenCV camera-to-world ``[4, 4]`` (x right, y down,
z forward); intrinsics are pixel ``[3, 3]``; warps are ``[1, 3, F, H, W]`` in
``[-1, 1]`` with visibility ``[1, 1, F, H, W]`` in ``{0, 1}``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Calibrated upstream (12 i2v references x 5 budgets): the interior optimum of
# chunk-0 / later-chunk parallax agreement. 0 disables the rescale.
CHUNK0_TARGET_DISPARITY_PX_DEFAULT = 135.0


def unproject_depth(depth: torch.Tensor, intr: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    """depth ``[h, w]`` + intrinsics at depth resolution + c2w -> world points ``[h, w, 3]``."""
    h, w = depth.shape
    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]
    ys, xs = torch.meshgrid(
        torch.arange(h, device=depth.device, dtype=torch.float32),
        torch.arange(w, device=depth.device, dtype=torch.float32),
        indexing="ij",
    )
    z = depth.float()
    cam = torch.stack([(xs - cx) / fx * z, (ys - cy) / fy * z, z], dim=-1)
    rotation = c2w[:3, :3].float()
    translation = c2w[:3, 3].float()
    return cam @ rotation.T + translation


def umeyama_scale(src_centers: np.ndarray, dst_centers: np.ndarray) -> float | None:
    """Scale ``c`` of the Sim(3) Umeyama fit ``dst ~= c R src + t`` without a rank check.

    Collinear (straight-line) camera motion has a well-defined scale even though
    the rotation is degenerate. Returns ``None`` for a zero baseline.
    """
    src = np.asarray(src_centers, np.float64)
    dst = np.asarray(dst_centers, np.float64)
    n = int(src.shape[0])
    if n < 2:
        return None
    src_c = src - src.mean(0)
    dst_c = dst - dst.mean(0)
    var_src = float((src_c**2).sum() / n)
    if not np.isfinite(var_src) or var_src < 1e-12:
        return None
    sigma = (dst_c.T @ src_c) / n
    u, d, vt = np.linalg.svd(sigma)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1.0
    scale = float((d * np.diag(sign)).sum() / var_src)
    if not np.isfinite(scale) or scale <= 1e-9:
        return None
    return scale


def scale_intrinsics(intrinsics: np.ndarray, src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> np.ndarray:
    """Rescale pixel intrinsics ``[..., 3, 3]`` from ``src_hw`` to ``dst_hw``."""
    (src_h, src_w), (dst_h, dst_w) = src_hw, dst_hw
    sx, sy = dst_w / float(src_w), dst_h / float(src_h)
    out = np.array(intrinsics, np.float32).copy()
    out[..., 0, 0] *= sx
    out[..., 0, 2] *= sx
    out[..., 1, 1] *= sy
    out[..., 1, 2] *= sy
    return out


def render_intrinsics(fx: float, fy: float, cx: float, cy: float, height: int, width: int) -> np.ndarray:
    """Render-resolution K the upstream pipeline derives from ``[fx, fy, cx, cy]``.

    The source resolution is backed out from the principal point (assumed
    centred) and K is rescaled to the render size.
    """
    src_w = max(2.0 * cx, 1.0)
    src_h = max(2.0 * cy, 1.0)
    sx = float(width) / src_w
    sy = float(height) / src_h
    return np.array([[fx * sx, 0.0, cx * sx], [0.0, fy * sy, cy * sy], [0.0, 0.0, 1.0]], np.float32)


@dataclass
class BankFrame:
    depth: torch.Tensor  # [h, w] at depth resolution, GT scale
    intr: torch.Tensor  # [3, 3] at depth resolution
    c2w: torch.Tensor  # [4, 4]
    rgb: torch.Tensor  # [3, h, w] in [0, 1]


class FrameBank:
    """Per-frame geometry bank keyed by absolute pixel-frame index (``gid``).

    Each ingest is one depth-estimator window with its own scale; rendering
    picks sources per target pixel, so windows never need to be fused into one
    global cloud. ``max_frames > 0`` bounds residency by evicting the oldest
    gids (a recipe change: it alters what can be recalled).
    """

    def __init__(self, device: torch.device | str, max_frames: int = 0):
        self.device = torch.device(device)
        self.max_frames = int(max_frames)
        self.frames: dict[int, BankFrame] = {}

    def __len__(self) -> int:
        return len(self.frames)

    def clear(self) -> None:
        self.frames.clear()

    def nbytes(self) -> int:
        total = 0
        for frame in self.frames.values():
            for tensor in (frame.depth, frame.intr, frame.c2w, frame.rgb):
                total += tensor.numel() * tensor.element_size()
        return total

    @torch.no_grad()
    def ingest(
        self,
        estimator,
        frames_rgb: np.ndarray,
        c2w: np.ndarray,
        k_pix: np.ndarray,
        frame_ids: Sequence[int],
    ) -> bool:
        """frames ``[N, H, W, 3]`` in ``[0, 1]`` + poses ``[N, 4, 4]`` -> stored depth.

        Returns ``False`` when the estimator reports a window without solvable
        geometry (zero baseline); the window is skipped, matching upstream.
        Any other estimator failure propagates instead of silently producing an
        empty warp.
        """
        frames_rgb = np.asarray(frames_rgb, np.float32)
        c2w = np.asarray(c2w, np.float32)
        n = frames_rgb.shape[0]
        if n < 3:
            return False
        ks = np.stack([np.asarray(k_pix, np.float32)] * n)
        try:
            depth, intr, conf, rgb = estimator.depth_window(frames_rgb, c2w, ks)
        except DegenerateGeometryError as exc:
            logger.warning("Evoke geometry ingest skipped a window: %s", exc)
            return False
        del conf  # the confidence gate only shaped upstream's sparse cloud, which no shipped renderer reads
        for i, gid in enumerate(frame_ids):
            color = (
                torch.as_tensor(np.asarray(rgb[i], np.float32), device=self.device)
                if rgb is not None
                else torch.zeros(*np.asarray(depth[i]).shape, 3, device=self.device)
            )
            self.frames[int(gid)] = BankFrame(
                depth=torch.as_tensor(depth[i], device=self.device),
                intr=torch.as_tensor(intr[i], device=self.device),
                c2w=torch.as_tensor(c2w[i], device=self.device),
                rgb=color.permute(2, 0, 1).contiguous(),
            )
        if self.max_frames > 0 and self.frames:
            newest = max(self.frames)
            for gid in [g for g in self.frames if g < newest - self.max_frames + 1]:
                del self.frames[gid]
        return True


class DegenerateGeometryError(RuntimeError):
    """A depth window carried no solvable geometry (e.g. zero camera baseline)."""


@torch.no_grad()
def render_cloud_batched(
    xyz: torch.Tensor,
    rgb: torch.Tensor,
    c2w_targets: torch.Tensor,
    k_render: torch.Tensor,
    height: int,
    width: int,
    *,
    splat_radius: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Z-buffered point splat of one cloud into F views (black where empty)."""
    device = xyz.device
    points = xyz.reshape(-1, 3).float()
    colors = rgb.reshape(-1, 3).float()
    npt = points.shape[0]
    if npt >= (1 << 24):
        raise ValueError(f"cloud has {npt} points, over the 2^24 render index limit")
    c2w = c2w_targets.to(device=device, dtype=torch.float32)
    Kt = k_render.to(device=device, dtype=torch.float32)
    views = c2w.shape[0]
    hw = height * width
    fill = torch.zeros(3, device=device)
    if npt == 0:
        warp = fill[None, :, None, None, None].expand(1, 3, views, height, width) * 2 - 1
        return warp.contiguous(), torch.zeros((1, 1, views, height, width), device=device)
    rotation = c2w[:, :3, :3]
    translation = c2w[:, :3, 3]
    cam = torch.einsum("vij,vpj->vpi", rotation.transpose(1, 2), points[None] - translation[:, None])
    z = cam[..., 2]
    fx = Kt[:, 0, 0][:, None]
    fy = Kt[:, 1, 1][:, None]
    cx = Kt[:, 0, 2][:, None]
    cy = Kt[:, 1, 2][:, None]
    px = torch.round((cam[..., 0] / z) * fx + cx).long()
    py = torch.round((cam[..., 1] / z) * fy + cy).long()
    front = z > 1e-4
    zq = (z.clamp(0, 1e6) * 1000.0).long().clamp(0, (1 << 38) - 1)
    idx = torch.arange(npt, device=device)[None].expand(views, npt)
    vidx = torch.arange(views, device=device)[:, None].expand(views, npt)
    best = torch.full((views * hw,), 1 << 62, dtype=torch.long, device=device)
    for dy in range(-splat_radius, splat_radius + 1):
        for dx in range(-splat_radius, splat_radius + 1):
            qx = px + dx
            qy = py + dy
            ok = front & (qx >= 0) & (qx < width) & (qy >= 0) & (qy < height)
            key = (vidx * hw + qy * width + qx)[ok]
            packed = (zq[ok] << 24) | idx[ok]
            best.scatter_reduce_(0, key, packed, reduce="amin", include_self=True)
    valid = best < (1 << 62)
    winner = (best & ((1 << 24) - 1)).clamp(max=npt - 1)
    img = fill[None].expand(views * hw, 3).clone()
    img[valid] = colors[winner[valid]]
    img = img.reshape(views, height, width, 3).clamp(0, 1)
    warp = img.permute(3, 0, 1, 2).unsqueeze(0) * 2.0 - 1.0
    vis = valid.reshape(views, height, width).float()[None, None]
    return warp.contiguous(), vis.contiguous()


@torch.no_grad()
def render_backward_zbuf(
    bank: FrameBank,
    gids: Sequence[int],
    target_c2ws: torch.Tensor,
    k_pix: np.ndarray,
    height: int,
    width: int,
    *,
    nearby: int = 16,
    depth_thresh: float = 0.02,
    topk: int = 8,
    covis_points: int = 2000,
    fg_covis: float = 0.3,
    fg_factor: float = 1.5,
    fg_scale_exempt: bool = True,
    despeckle: bool = False,
    despeckle_ksize: int = 3,
    despeckle_fill_iters: int = 4,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-pixel top-K multi-source z-buffer render (upstream ``render_mode=backward_zbuf``).

    For every target frame the K sources whose subsampled clouds best cover the
    target frustum are forward-splatted; each pixel keeps the nearest surface.
    Uncovered pixels stay black and invisible. Older (recalled) sources are
    depth-aligned to the most recent ``nearby`` sources first, and far,
    low-coverage sources that are not geometrically consistent are dropped.
    """
    device = bank.device
    H, W = int(height), int(width)
    num_targets = int(target_c2ws.shape[0])
    Kt = torch.as_tensor(np.asarray(k_pix, np.float32), device=device)
    fx, fy, cx, cy = Kt[0, 0], Kt[1, 1], Kt[0, 2], Kt[1, 2]
    gids = sorted(int(g) for g in gids)
    if not gids:
        return (
            torch.full((1, 3, num_targets, H, W), -1.0, device=device),
            torch.zeros((1, 1, num_targets, H, W), device=device),
        )
    far = 1e6
    row_of = {g: i for i, g in enumerate(gids)}

    subsampled = torch.full((len(gids), covis_points, 3), far, device=device)
    for g in gids:
        frame = bank.frames[g]
        world = unproject_depth(frame.depth, frame.intr, frame.c2w).reshape(-1, 3)
        world = world[(frame.depth.reshape(-1) > 1e-4) & torch.isfinite(world).all(-1)]
        if world.shape[0] > 0:
            pick = torch.randint(0, world.shape[0], (covis_points,), device=device, generator=generator)
            subsampled[row_of[g]] = world[pick]

    def covis(rows: list[int], target: torch.Tensor) -> torch.Tensor:
        pts = subsampled[torch.as_tensor(rows, device=device)]
        w2c = torch.linalg.inv(target)
        cam = torch.einsum("cmj,kj->cmk", pts - target[:3, 3], w2c[:3, :3])
        z = cam[..., 2]
        u = cam[..., 0] / z.clamp(min=1e-6) * fx + cx
        v = cam[..., 1] / z.clamp(min=1e-6) * fy + cy
        ok = (z > 1e-4) & (u >= 0) & (u < W) & (v >= 0) & (v < H) & (pts[..., 0] < far * 0.5)
        return ok.float().mean(1)

    def splat(g: int, target: torch.Tensor, scale: float):
        frame = bank.frames[g]
        h, w = frame.depth.shape
        ys, xs = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )
        z = frame.depth.float()
        if scale != 1.0:
            z = z * scale
        it = frame.intr
        xc = (xs - it[0, 2]) / it[0, 0] * z
        yc = (ys - it[1, 2]) / it[1, 1] * z
        cam = torch.stack([xc, yc, z, torch.ones_like(z)], -1).reshape(-1, 4)
        world = (frame.c2w @ cam.T).T[:, :3]
        w2c = torch.linalg.inv(target)
        ct = (w2c[:3, :3] @ world.T).T + w2c[:3, 3]
        zt = ct[:, 2]
        xt = torch.round(ct[:, 0] / zt.clamp(min=1e-6) * fx + cx).long()
        yt = torch.round(ct[:, 1] / zt.clamp(min=1e-6) * fy + cy).long()
        src_index = torch.arange(h * w, device=device)
        ok = (z.reshape(-1) > 1e-4) & (zt > 1e-4) & (xt >= 0) & (xt < W) & (yt >= 0) & (yt < H)
        zbuf = torch.full((H * W,), float("inf"), device=device)
        if not bool(ok.any()):
            return torch.zeros(3, H, W, device=device), zbuf
        key = (yt * W + xt)[ok]
        zq = (zt[ok].clamp(0, 1e6) * 1000).long().clamp(0, (1 << 38) - 1)
        packed = (zq << 24) | src_index[ok]
        best = torch.full((H * W,), 1 << 62, dtype=torch.long, device=device)
        best.scatter_reduce_(0, key, packed, reduce="amin", include_self=True)
        valid = best < (1 << 62)
        owner = (best & ((1 << 24) - 1)).clamp(max=h * w - 1)
        zbuf[valid] = (best[valid] >> 24).float() / 1000.0
        us = (owner % w).float()
        vs = (owner // w).float()
        gx = us / max(w - 1, 1) * 2 - 1
        gy = vs / max(h - 1, 1) * 2 - 1
        grid = torch.stack([gx.reshape(H, W), gy.reshape(H, W)], -1)[None]
        sampled = F.grid_sample(
            frame.rgb[None].float(), grid, mode="bilinear", padding_mode="border", align_corners=False
        )[0]
        valid_hw = valid.reshape(H, W)
        color = torch.where(valid_hw[None].expand(3, H, W), sampled, torch.zeros_like(sampled))
        return color, zbuf

    def estimate_scale(g: int, ref_points: torch.Tensor | None, min_points: int = 200) -> tuple[float, int]:
        frame = bank.frames[g]
        h, w = frame.depth.shape
        if ref_points is None or ref_points.shape[0] == 0:
            return 1.0, 0
        w2c = torch.linalg.inv(frame.c2w)
        cam = (w2c[:3, :3] @ ref_points.T).T + w2c[:3, 3]
        z = cam[:, 2]
        it = frame.intr
        u = (cam[:, 0] / z.clamp(min=1e-6) * it[0, 0] + it[0, 2]).round().long()
        v = (cam[:, 1] / z.clamp(min=1e-6) * it[1, 1] + it[1, 2]).round().long()
        ok = (z > 1e-4) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if int(ok.sum()) < min_points:
            return 1.0, int(ok.sum())
        nearest = torch.full((h * w,), float("inf"), device=device)
        nearest.scatter_reduce_(0, v[ok] * w + u[ok], z[ok], reduce="amin", include_self=True)
        flat = frame.depth.reshape(-1)
        overlap = (nearest < float("inf")) & (flat > 1e-4)
        count = int(overlap.sum())
        if count < min_points:
            return 1.0, count
        scale = float((nearest[overlap] / flat[overlap]).median())
        return (scale if 0.2 < scale < 5.0 else 1.0), count

    nearby_ids = gids[-int(nearby) :]
    nearby_set = set(nearby_ids)
    nearby_centres = torch.stack([bank.frames[g].c2w[:3, 3] for g in nearby_ids]).float() if fg_covis > 0 else None
    centres = {g: bank.frames[g].c2w[:3, 3].float() for g in gids}
    ref_parts = [subsampled[row_of[g]][subsampled[row_of[g]][:, 0] < far * 0.5] for g in nearby_ids]
    ref_parts = [p for p in ref_parts if p.shape[0] > 0]
    ref_points = torch.cat(ref_parts) if ref_parts else None
    alignment = {g: ((1.0, 0) if g in nearby_set else estimate_scale(g, ref_points)) for g in gids}
    source_scale = {g: alignment[g][0] for g in gids}
    self_consistent = (
        {g: g not in nearby_set and alignment[g][1] >= 200 and 0.8 <= alignment[g][0] <= 1.25 for g in gids}
        if fg_scale_exempt
        else {}
    )

    targets = target_c2ws.to(device=device, dtype=torch.float32)
    rows = [row_of[g] for g in gids]
    warps: list[torch.Tensor] = []
    visibility: list[torch.Tensor] = []
    for f in range(num_targets):
        target = targets[f]
        scores = covis(rows, target)
        order = torch.topk(scores, k=min(int(topk), len(gids))).indices.tolist()
        if fg_covis > 0 and nearby_centres is not None:
            centre = target[:3, 3]
            nearest = float((nearby_centres - centre).norm(dim=1).min())
            order = [
                i
                for i in order
                if float(scores[i]) >= fg_covis
                or float((centres[gids[i]] - centre).norm()) <= fg_factor * nearest
                or self_consistent.get(gids[i], False)
            ]
        fused = torch.zeros(3, H, W, device=device)
        fused_depth = torch.full((H * W,), float("inf"), device=device)
        covered = torch.zeros(H, W, dtype=torch.bool, device=device)
        for i in order:
            g = gids[i]
            color, zbuf = splat(g, target, source_scale[g])
            update = torch.isfinite(zbuf) & (zbuf < fused_depth - depth_thresh)
            if not bool(update.any()):
                update = torch.isfinite(zbuf) & (zbuf < fused_depth)
                if not bool(update.any()):
                    continue
            update_hw = update.reshape(H, W)
            fused = torch.where(update_hw[None].expand(3, H, W), color, fused)
            fused_depth = torch.where(update, zbuf, fused_depth)
            covered = covered | update_hw
        if despeckle:
            fused, covered = _despeckle(fused, covered, despeckle_ksize, despeckle_fill_iters)
        warps.append(fused.clamp(0, 1) * 2 - 1)
        visibility.append(covered.float())
    return torch.stack(warps, 1)[None].contiguous(), torch.stack(visibility)[None, None].contiguous()


def _despeckle(color: torch.Tensor, covered: torch.Tensor, ksize: int, fill_iters: int):
    """Morphological open -> close on the coverage mask; fill newly closed pixels from neighbours."""

    def erode(x):
        return -F.max_pool2d(-x, ksize, 1, ksize // 2)

    def dilate(x):
        return F.max_pool2d(x, ksize, 1, ksize // 2)

    mask = covered.float()[None, None]
    closed = erode(dilate(dilate(erode(mask))))[0, 0] > 0.5
    fill = closed & ~covered
    removed = covered & ~closed
    out = color.clone()
    known = covered.float()[None, None]
    for _ in range(int(fill_iters)):
        num = F.avg_pool2d(out[None] * known, 3, 1, 1)
        den = F.avg_pool2d(known, 3, 1, 1)
        update = fill & (den[0, 0] > 1e-6) & (known[0, 0] < 0.5)
        out = torch.where(update[None].expand(3, -1, -1), (num / den.clamp(min=1e-6))[0], out)
        known[0, 0] = torch.where(update, torch.ones_like(known[0, 0]), known[0, 0])
    out[:, removed] = 0.0
    return out, closed


@torch.no_grad()
def solve_chunk0_disparity_scale(
    depth: torch.Tensor,
    intr_depth: torch.Tensor,
    ref_c2w: torch.Tensor,
    target_c2ws: torch.Tensor,
    k_out: torch.Tensor,
    target_px: float,
    *,
    quantile: float = 0.9,
    k_limits: tuple[float, float] = (1.0 / 64.0, 64.0),
) -> float | None:
    """Depth scale putting the q-quantile of chunk-0 translational parallax at ``target_px``.

    Closed form per pixel: ``|dx| = C / (k d + dz)``; see upstream
    ``solve_chunk0_disparity_scale``. Returns ``None`` when no translation is
    commanded (the solve is degenerate) so the caller keeps the median scale.
    """
    d = depth.float()
    h, w = int(d.shape[0]), int(d.shape[1])
    if not target_px > 0:
        return None
    ys, xs = torch.meshgrid(
        torch.arange(h, device=d.device, dtype=torch.float32),
        torch.arange(w, device=d.device, dtype=torch.float32),
        indexing="ij",
    )
    m_x = (xs - float(intr_depth[0, 2])) / float(intr_depth[0, 0])
    m_y = (ys - float(intr_depth[1, 2])) / float(intr_depth[1, 1])
    fx_o, fy_o = float(k_out[0, 0]), float(k_out[1, 1])
    r0 = ref_c2w[:3, :3].float()
    t0 = ref_c2w[:3, 3].float()
    valid = torch.isfinite(d) & (d > 0)
    if not bool(valid.any()):
        return None
    d_safe = torch.where(valid, d, torch.ones_like(d))
    best = 0.0
    for i in range(int(target_c2ws.shape[0])):
        delta = r0.transpose(0, 1) @ (t0 - target_c2ws[i, :3, 3].float())
        dx, dy, dz = float(delta[0]), float(delta[1]), float(delta[2])
        c = torch.sqrt((fx_o * (dx - m_x * dz)) ** 2 + (fy_o * (dy - m_y * dz)) ** 2)
        k_p = ((c / target_px) - dz) / d_safe
        k_p = torch.where(valid, k_p, torch.zeros_like(k_p)).clamp_min(0.0)
        best = max(best, float(torch.quantile(k_p.reshape(-1), float(quantile))))
    if not np.isfinite(best) or best <= 0:
        return None
    return float(min(max(best, k_limits[0]), k_limits[1]))


@torch.no_grad()
def build_single_source_warp(
    estimator,
    ref_frame: torch.Tensor,
    ref_c2w: torch.Tensor,
    k_pix: np.ndarray,
    target_c2ws: torch.Tensor,
    *,
    height: int,
    width: int,
    splat_radius: int = 1,
    chunk0_target_disparity_px: float = CHUNK0_TARGET_DISPARITY_PX_DEFAULT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """i2v chunk-0 warp from the reference image alone (monocular, baseline-free scale).

    ``ref_frame`` ``[3, H, W]`` in ``[-1, 1]``. Errors surface: the caller
    decides whether a blank warp is acceptable.
    """
    device = target_c2ws.device
    targets = target_c2ws.to(dtype=torch.float32)
    frame = (ref_frame.detach().permute(1, 2, 0).float() * 0.5 + 0.5).clamp(0, 1).cpu().numpy()
    k_np = np.asarray(k_pix, np.float32)
    depth_np, intr_np, rgb_np = estimator.depth_single(frame, k_np)
    depth = torch.as_tensor(depth_np, device=device)
    intr = torch.as_tensor(intr_np, device=device)
    ref_pose = ref_c2w.to(device=device, dtype=torch.float32)
    if chunk0_target_disparity_px > 0:
        k = solve_chunk0_disparity_scale(
            depth, intr, ref_pose, targets, torch.as_tensor(k_np, device=device), chunk0_target_disparity_px
        )
        if k is not None:
            depth = depth * k
    xyz = unproject_depth(depth, intr, ref_pose).reshape(-1, 3)
    rgb = torch.as_tensor(np.asarray(rgb_np, np.float32), device=device).reshape(-1, 3)
    k_render = torch.as_tensor(k_np, device=device)[None].expand(targets.shape[0], 3, 3)
    return render_cloud_batched(xyz, rgb, targets, k_render, int(height), int(width), splat_radius=splat_radius)


def visibility_to_latent(
    visibility_pix: torch.Tensor,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    *,
    vae_t_stride: int = 4,
    patch_hw: tuple[int, int] = (2, 2),
    cov_thresh: float = 0.5,
) -> torch.Tensor:
    """Pixel visibility ``[1, 1, F, H, W]`` -> blocky latent mask ``[1, 1, T, h, w]``.

    One pixel frame per latent frame (stride ``vae_t_stride``) is pooled to the
    DiT patch grid and binarized, so the mask is constant within each token.
    """
    ids = (torch.arange(num_latent_frames, device=visibility_pix.device) * int(vae_t_stride)).clamp(
        max=visibility_pix.shape[2] - 1
    )
    sampled = visibility_pix.index_select(2, ids)
    ph, pw = max(1, int(patch_hw[0])), max(1, int(patch_hw[1]))
    gh, gw = max(1, latent_height // ph), max(1, latent_width // pw)
    coverage = F.adaptive_avg_pool3d(sampled, (num_latent_frames, gh, gw))
    binary = (coverage > float(cov_thresh)).to(sampled.dtype)
    return F.interpolate(binary, size=(num_latent_frames, latent_height, latent_width), mode="nearest")
