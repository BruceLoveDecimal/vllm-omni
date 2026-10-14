# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from Alaya-EVOKE (https://github.com/AlayaLab/Evoke @ 1d271be),
# evoke/pipelines/pipeline_evoke.py (_geo_da3_* helpers), Apache-2.0.
"""Per-session geometric state: what has been seen, and how to render the next chunk.

One session owns one :class:`FrameBank` keyed by absolute 24 fps frame index
(``gid``). Chunk ``k`` covers gids ``[36k, 36k + 36)``; its warp is rendered at
the first 33 of those poses (the frames a 9-latent window encodes) from every
bank frame at least ``lag`` chunks old, and its decoded frames are ingested
after the chunk is generated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from vllm_omni.diffusion.models.evoke.geometry.cloud import (
    CHUNK0_TARGET_DISPARITY_PX_DEFAULT,
    FrameBank,
    build_single_source_warp,
    render_backward_zbuf,
)


@dataclass(frozen=True)
class GeometryRecipe:
    """Warp-render knobs of the shipped real-time recipe (``infer_post_distill.sh``)."""

    pix_stride: int = 36
    window_pix: int = 33
    ingest_frames: int = 12
    lag_chunks: int = 0
    nearby: int = 16
    despeckle: bool = False
    chunk0_reference_warp: bool = True
    chunk0_target_disparity_px: float = CHUNK0_TARGET_DISPARITY_PX_DEFAULT
    chunk0_splat_radius: int = 1
    max_bank_frames: int = 0


class GeometrySession:
    def __init__(
        self,
        estimator,
        k_pix: np.ndarray,
        height: int,
        width: int,
        device: torch.device,
        recipe: GeometryRecipe,
        *,
        reference_frame: torch.Tensor | None,
    ) -> None:
        self.estimator = estimator
        self.k_pix = np.asarray(k_pix, np.float32)
        self.height = int(height)
        self.width = int(width)
        self.device = device
        self.recipe = recipe
        self.bank = FrameBank(device, max_frames=recipe.max_bank_frames)
        # i2v: the conditioning image is the only source for chunk 0's warp.
        self.reference_frame = reference_frame
        self.last_coverage: float | None = None

    def close(self) -> None:
        self.bank.clear()
        self.reference_frame = None
        self.estimator.reset_stream()

    @torch.no_grad()
    def render_chunk(self, chunk_index: int, target_c2ws: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Warp ``[1, 3, F, H, W]`` in ``[-1, 1]`` and visibility for ``target_c2ws`` ``[F, 4, 4]``."""
        targets = target_c2ws.to(device=self.device, dtype=torch.float32)
        gids = list(self.bank.frames)
        if gids:
            cutoff = max(gids) - self.recipe.lag_chunks * self.recipe.pix_stride
            pool = [g for g in gids if g <= cutoff]
        else:
            pool = []
        if chunk_index == 0 and not pool and self.reference_frame is not None and self.recipe.chunk0_reference_warp:
            warp, vis = build_single_source_warp(
                self.estimator,
                self.reference_frame.to(self.device),
                targets[0],
                self.k_pix,
                targets,
                height=self.height,
                width=self.width,
                splat_radius=self.recipe.chunk0_splat_radius,
                chunk0_target_disparity_px=self.recipe.chunk0_target_disparity_px,
            )
        else:
            warp, vis = render_backward_zbuf(
                self.bank,
                pool,
                targets,
                self.k_pix,
                self.height,
                self.width,
                nearby=self.recipe.nearby,
                despeckle=self.recipe.despeckle,
            )
        warp = warp.to(device=self.device, dtype=torch.float32)
        vis = vis.to(device=self.device, dtype=torch.float32)
        self.last_coverage = float(vis[0, 0].mean())
        return warp, vis

    @torch.no_grad()
    def ingest(self, frames: torch.Tensor, pix_start: int, c2ws: torch.Tensor) -> bool:
        """Ingest decoded frames ``[3, N, H, W]`` in ``[-1, 1]`` whose first frame is gid ``pix_start``.

        ``c2ws`` holds the pose of every gid of the session so far.
        """
        num = int(frames.shape[1])
        if num < 3:
            return False
        local = torch.unique(torch.linspace(0, num - 1, self.recipe.ingest_frames).round().long().clamp(0, num - 1))
        if local.numel() < 3:
            return False
        gids = [int(pix_start) + int(i) for i in local.tolist()]
        pose_index = torch.tensor([min(g, c2ws.shape[0] - 1) for g in gids])
        poses = c2ws[pose_index].float().cpu().numpy()
        rgb = (frames[:, local.to(frames.device)].permute(1, 2, 3, 0) * 0.5 + 0.5).clamp(0, 1).float().cpu().numpy()
        return self.bank.ingest(self.estimator, rgb, poses, self.k_pix, gids)
