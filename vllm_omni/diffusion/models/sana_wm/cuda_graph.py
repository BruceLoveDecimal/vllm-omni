# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA graph helpers for the SANA-WM native Stage-1 smoke path."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import torch

SANA_WM_CUDAGRAPH_ENV = "VLLM_OMNI_SANA_WM_CUDAGRAPH"
SANA_WM_CUDAGRAPH_BUCKETS = (65, 129, 193, 257, 321)


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def sana_wm_cudagraph_requested(extra_args: dict[str, Any] | None = None) -> bool:
    """Return whether Sana-WM Stage-1 CUDA graph replay was requested."""

    if _truthy(os.environ.get(SANA_WM_CUDAGRAPH_ENV, "")):
        return True
    return _truthy((extra_args or {}).get("sana_wm_cudagraph", ""))


def parse_sana_wm_cudagraph_buckets(extra_args: dict[str, Any] | None = None) -> tuple[int, ...]:
    """Parse frame buckets for per-shape graph capture.

    The default buckets match the production SANA-WM frame counts discussed in
    the integration plan. Runtime overrides are useful for small unit or smoke
    tests and intentionally do not change the exported default.
    """

    raw = os.environ.get("VLLM_OMNI_SANA_WM_CUDAGRAPH_BUCKETS")
    if not raw and extra_args is not None:
        raw = extra_args.get("sana_wm_cudagraph_buckets")
    if raw is None or raw == "":
        return SANA_WM_CUDAGRAPH_BUCKETS
    if isinstance(raw, (list, tuple)):
        values = raw
    else:
        values = str(raw).replace(";", ",").split(",")
    buckets = tuple(sorted({int(value) for value in values if str(value).strip()}))
    if not buckets:
        raise ValueError("Sana-WM CUDA graph buckets must contain at least one frame count.")
    return buckets


def sana_wm_cudagraph_bucket(num_frames: int, buckets: tuple[int, ...] = SANA_WM_CUDAGRAPH_BUCKETS) -> int | None:
    """Return the exact frame bucket for graph capture, or None for eager."""

    return int(num_frames) if int(num_frames) in buckets else None


@dataclass
class _SanaWmCudaGraphEntry:
    graph: torch.cuda.CUDAGraph
    latents: torch.Tensor
    timestep: torch.Tensor
    encoder_hidden_states: torch.Tensor
    plucker: torch.Tensor
    raymap: torch.Tensor
    output: torch.Tensor


class SanaWmCudaGraphDenoiser:
    """Per-bucket CUDA graph replay for SANA-WM denoising steps.

    The helper is intentionally conservative: it captures only exact configured
    frame buckets, only on CUDA, and falls back to eager execution if capture
    fails. Graph replay is disabled by default until the native path meets the
    PSNR/SSIM parity gate without graphing.
    """

    def __init__(self) -> None:
        self.graphs: dict[tuple[Any, ...], _SanaWmCudaGraphEntry] = {}
        self.disabled_keys: set[tuple[Any, ...]] = set()
        self.capture_failures: dict[tuple[Any, ...], str] = {}
        self.replay_count = 0
        self.capture_count = 0

    @staticmethod
    def _eager(
        transformer: torch.nn.Module,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        *,
        encoder_hidden_states: torch.Tensor,
        camera_encoder: torch.nn.Module | None,
        plucker: torch.Tensor,
        raymap: torch.Tensor,
    ) -> torch.Tensor:
        return transformer(
            latents,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            camera_encoder=camera_encoder,
            plucker=plucker,
            raymap=raymap,
        )

    @staticmethod
    def _key(
        *,
        bucket: int,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        plucker: torch.Tensor,
        raymap: torch.Tensor,
        transformer: torch.nn.Module,
        camera_encoder: torch.nn.Module | None,
    ) -> tuple[Any, ...]:
        return (
            bucket,
            id(transformer),
            id(camera_encoder),
            latents.device.type,
            latents.device.index,
            latents.dtype,
            tuple(latents.shape),
            tuple(timestep.shape),
            tuple(encoder_hidden_states.shape),
            tuple(plucker.shape),
            tuple(raymap.shape),
        )

    def run(
        self,
        transformer: torch.nn.Module,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        *,
        encoder_hidden_states: torch.Tensor,
        camera_encoder: torch.nn.Module | None,
        plucker: torch.Tensor,
        raymap: torch.Tensor,
        num_frames: int,
        buckets: tuple[int, ...] = SANA_WM_CUDAGRAPH_BUCKETS,
    ) -> tuple[torch.Tensor, bool]:
        bucket = sana_wm_cudagraph_bucket(num_frames, buckets)
        if bucket is None or not latents.is_cuda:
            return (
                self._eager(
                    transformer,
                    latents,
                    timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    camera_encoder=camera_encoder,
                    plucker=plucker,
                    raymap=raymap,
                ),
                False,
            )

        key = self._key(
            bucket=bucket,
            latents=latents,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            plucker=plucker,
            raymap=raymap,
            transformer=transformer,
            camera_encoder=camera_encoder,
        )
        if key in self.disabled_keys:
            return (
                self._eager(
                    transformer,
                    latents,
                    timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    camera_encoder=camera_encoder,
                    plucker=plucker,
                    raymap=raymap,
                ),
                False,
            )

        entry = self.graphs.get(key)
        if entry is None:
            try:
                entry = self._capture(
                    transformer,
                    latents,
                    timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    camera_encoder=camera_encoder,
                    plucker=plucker,
                    raymap=raymap,
                )
                self.graphs[key] = entry
                self.capture_count += 1
            except Exception as exc:
                self.disabled_keys.add(key)
                self.capture_failures[key] = f"{type(exc).__name__}: {exc}"
                return (
                    self._eager(
                        transformer,
                        latents,
                        timestep,
                        encoder_hidden_states=encoder_hidden_states,
                        camera_encoder=camera_encoder,
                        plucker=plucker,
                        raymap=raymap,
                    ),
                    False,
                )
        else:
            entry.latents.copy_(latents)
            entry.timestep.copy_(timestep)
            entry.encoder_hidden_states.copy_(encoder_hidden_states)
            entry.plucker.copy_(plucker)
            entry.raymap.copy_(raymap)
            entry.graph.replay()
            self.replay_count += 1

        return entry.output.clone(), True

    def _capture(
        self,
        transformer: torch.nn.Module,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        *,
        encoder_hidden_states: torch.Tensor,
        camera_encoder: torch.nn.Module | None,
        plucker: torch.Tensor,
        raymap: torch.Tensor,
    ) -> _SanaWmCudaGraphEntry:
        static_latents = latents.detach().clone()
        static_timestep = timestep.detach().clone()
        static_encoder_hidden_states = encoder_hidden_states.detach().clone()
        static_plucker = plucker.detach().clone()
        static_raymap = raymap.detach().clone()

        with torch.inference_mode():
            static_output = self._eager(
                transformer,
                static_latents,
                static_timestep,
                encoder_hidden_states=static_encoder_hidden_states,
                camera_encoder=camera_encoder,
                plucker=static_plucker,
                raymap=static_raymap,
            )
            torch.cuda.synchronize(static_latents.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_output = self._eager(
                    transformer,
                    static_latents,
                    static_timestep,
                    encoder_hidden_states=static_encoder_hidden_states,
                    camera_encoder=camera_encoder,
                    plucker=static_plucker,
                    raymap=static_raymap,
                )

        return _SanaWmCudaGraphEntry(
            graph=graph,
            latents=static_latents,
            timestep=static_timestep,
            encoder_hidden_states=static_encoder_hidden_states,
            plucker=static_plucker,
            raymap=static_raymap,
            output=static_output,
        )
