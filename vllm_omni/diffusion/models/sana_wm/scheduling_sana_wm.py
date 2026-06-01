# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SANA-WM Stage-1 flow scheduler helpers."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch

SANA_WM_DEFAULT_INFERENCE_FLOW_SHIFT = 9.8


def shift_flow_timestep(
    timestep: torch.Tensor,
    shift: float = SANA_WM_DEFAULT_INFERENCE_FLOW_SHIFT,
) -> torch.Tensor:
    """Apply the inference flow-shift transform used by SANA-WM."""

    return shift * timestep / (1.0 + (shift - 1.0) * timestep)


class SanaWmFlowMatchScheduler:
    """Native flow-matching Euler scheduler matching NVlabs ``LTXFlowEuler``.

    Reproduces ``diffusers.FlowMatchEulerDiscreteScheduler(shift=shift)``
    timestep + sigma tables (which NVlabs uses) without depending on
    diffusers at runtime. The schedule has a subtle but load-bearing
    quirk: ``sigma_min`` is set at construction time to the
    shift-transformed ``1/num_train`` (not the raw ``1/num_train``), so
    ``set_timesteps`` ends up applying the shift formula **twice** — once
    implicitly via the shifted ``sigma_min`` linspace endpoint, once
    explicitly. For ``shift=9.8, N=3`` this yields timesteps
    ``[1000, 909.0, 87.7]`` (matches NVlabs) rather than the near-linear
    ``[999, 951, 830]`` that the prior ``DPMSolverMultistep`` wrapper
    produced (see audit §6.13k).
    """

    NUM_TRAIN_TIMESTEPS = 1000

    def __init__(
        self,
        num_inference_steps: int,
        shift: float = SANA_WM_DEFAULT_INFERENCE_FLOW_SHIFT,
    ) -> None:
        if num_inference_steps <= 0:
            raise ValueError("Sana-WM scheduler num_inference_steps must be positive.")
        self.num_inference_steps = num_inference_steps
        self.shift = float(shift)
        self.num_train_timesteps = self.NUM_TRAIN_TIMESTEPS
        # Shifted sigma_min: raw min is 1/num_train; the FlowMatchEuler
        # __init__ shifts the full sigma array, so the persisted sigma_min
        # is the shifted value. set_timesteps then linspaces from
        # sigma_max=1.0 down to this shifted sigma_min.
        raw_min = 1.0 / self.num_train_timesteps
        self._sigma_min_shifted = (
            self.shift * raw_min / (1.0 + (self.shift - 1.0) * raw_min)
        )
        self._timesteps_tensor: torch.Tensor | None = None  # (N,)
        self._sigmas_tensor: torch.Tensor | None = None     # (N+1,)
        self._timesteps_device: torch.device | None = None

    def _build_schedule(self, device: torch.device) -> None:
        # Replay diffusers FlowMatchEulerDiscreteScheduler.set_timesteps:
        # ts_pre = linspace(num_train * sigma_max, num_train * sigma_min_shifted, N)
        # sig_pre = ts_pre / num_train
        # sigmas = shift * sig_pre / (1 + (shift-1) * sig_pre)   <-- second shift
        # timesteps = sigmas * num_train
        # sigmas = cat([sigmas, 0])
        N = self.num_inference_steps
        ts_pre = torch.linspace(
            float(self.num_train_timesteps),
            float(self.num_train_timesteps) * self._sigma_min_shifted,
            N,
            dtype=torch.float32,
            device=device,
        )
        sig_pre = ts_pre / float(self.num_train_timesteps)
        sigmas = self.shift * sig_pre / (1.0 + (self.shift - 1.0) * sig_pre)
        timesteps = sigmas * float(self.num_train_timesteps)

        # Audit §6.13s: env-gated subdivision of the biggest sigma jump.
        # The flow-shift schedule front-loads tiny early steps then takes a
        # huge final jump (e.g. for shift=9.8, N=20: ..., t=392 → t=87.7).
        # Oracle-snap probe in §6.13p showed this single step is the
        # amplifier that turns the ~0.04 same-latent noise_pred residual
        # into the 321f trajectory collapse. Subdividing it into
        # K halfway steps (geometric mean) reduces the per-step dt and
        # the trajectory perturbation.
        # Env var:
        #   SANA_WM_SCHED_SUBDIVIDE_BIG_STEP=K  (K extra sigmas inserted at
        #   the largest dt position; K=1 by default if set to "1"/"true").
        # NOTE: this makes the native schedule diverge from NVlabs by K
        # extra steps. Use only when latent parity vs NVlabs is no longer
        # the gating concern (the 321f case in §6.13o/p).
        subdiv_env = os.environ.get("SANA_WM_SCHED_SUBDIVIDE_BIG_STEP", "")
        subdiv_k = 0
        if subdiv_env:
            if subdiv_env.lower() in {"1", "true", "yes", "on"}:
                subdiv_k = 1
            else:
                try:
                    subdiv_k = max(0, int(subdiv_env))
                except ValueError:
                    subdiv_k = 0
        if subdiv_k > 0 and sigmas.numel() >= 2:
            # find adjacent (cur, next) pair with largest |sigma_next - sigma_cur|.
            # next-step sigmas come from cat([sigmas[1:], 0]); we want the
            # largest jump *into* the terminal zero or *between* schedule sigmas.
            sigmas_with_zero = torch.cat(
                [sigmas, torch.zeros(1, dtype=torch.float32, device=device)]
            )
            jumps = (sigmas_with_zero[:-1] - sigmas_with_zero[1:]).abs()
            big_idx = int(jumps.argmax().item())
            sigma_cur = float(sigmas_with_zero[big_idx].item())
            sigma_next = float(sigmas_with_zero[big_idx + 1].item())
            # geometric mean(s) between sigma_cur and sigma_next.
            # K=1: insert one sigma = sqrt(cur * next).
            # K>1: insert K equally-log-spaced sigmas.
            # Clamp sigma_next at 1e-6 if it's 0 so geomspace is well-defined,
            # then drop the inserted final-zero copy.
            log_cur = math.log(max(sigma_cur, 1e-9))
            log_next = math.log(max(sigma_next, 1e-9))
            new_logs = torch.linspace(
                log_cur, log_next, subdiv_k + 2, dtype=torch.float32, device=device
            )[1:-1]
            inserted = new_logs.exp()
            print(
                f"[sana-wm scheduler] subdivide_big_step k={subdiv_k} "
                f"big_jump=sigma[{big_idx}]={sigma_cur:.4f}→{sigma_next:.4f} "
                f"inserted={inserted.tolist()}",
                flush=True,
            )
            # If the big jump was at the LAST schedule sigma (index N-1)
            # into the terminal 0, then big_idx == N-1; the inserted sigmas
            # go between sigmas[N-1] and 0 — i.e., append them, then append 0.
            # Otherwise the inserted sigmas go between sigmas[big_idx] and
            # sigmas[big_idx+1] in the non-terminal portion of the schedule.
            if big_idx == sigmas.numel() - 1:
                # Big jump is the final-step jump into 0.
                sigmas = torch.cat([sigmas, inserted])
            else:
                sigmas = torch.cat(
                    [sigmas[: big_idx + 1], inserted, sigmas[big_idx + 1 :]]
                )
            timesteps = sigmas * float(self.num_train_timesteps)

        sigmas_with_terminal = torch.cat(
            [sigmas, torch.zeros(1, dtype=torch.float32, device=device)]
        )
        self._timesteps_tensor = timesteps
        self._sigmas_tensor = sigmas_with_terminal

    def _ensure_timesteps(self, device: torch.device) -> None:
        if self._timesteps_device != device or self._timesteps_tensor is None:
            self._build_schedule(device)
            self._timesteps_device = device

    def timesteps(self, *, device: torch.device) -> torch.Tensor:
        self._ensure_timesteps(device)
        return self._timesteps_tensor

    @property
    def sigmas(self) -> torch.Tensor:
        if self._sigmas_tensor is None:
            self._build_schedule(torch.device("cpu"))
        return self._sigmas_tensor

    def step(
        self,
        noise_pred: torch.Tensor,
        timestep: torch.Tensor,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """Single-token flow-matching Euler step.

        ``prev = latents + (sigma_next - sigma_cur) * (-noise_pred)`` —
        the NVlabs ``LTXFlowEuler`` sign convention. Used by the legacy
        non-per-token path; the per-token path is
        :meth:`step_flow_euler_per_token`.
        """
        self._ensure_timesteps(latents.device)
        sigmas = self._sigmas_tensor
        idx = self._sigma_index_for(timestep)
        sigma_cur = sigmas[idx]
        sigma_next = sigmas[idx + 1] if idx + 1 < sigmas.numel() else sigmas[-1]
        dt = (sigma_next - sigma_cur).to(latents.dtype)
        return (latents + dt * (-noise_pred.to(latents.dtype))).to(latents.dtype)

    def _sigma_index_for(self, timestep: torch.Tensor) -> int:
        """Return the index ``i`` such that ``self.timesteps_tensor[i] == timestep``.

        ``argmin(abs(diff))`` so the lookup survives fp16/bf16/fp32
        round-trips of ``timestep``.
        """
        ts = self._timesteps_tensor
        t_val = timestep.to(device=ts.device, dtype=ts.dtype)
        return int((ts - t_val).abs().argmin().item())

    def step_flow_euler_per_token(
        self,
        noise_pred: torch.Tensor,           # (B, C, F, H, W)
        timestep: torch.Tensor,             # scalar current step
        latents: torch.Tensor,              # (B, C, F, H, W)
        per_token_timesteps: torch.Tensor,  # (B, F*H*W) per-token integer timestep
    ) -> torch.Tensor:
        """Per-token flow-matching Euler step matching NVlabs ``LTXFlowEuler.sample``.

        NVlabs invokes diffusers
        ``FlowMatchEulerDiscreteScheduler.step`` with the noise prediction
        sign-flipped and ``per_token_timesteps``. In that branch, diffusers
        does not use ``self.sigmas[step_index]`` as the current sigma; it
        derives the current sigma directly from each token's timestep
        (``per_token_t / num_train_timesteps``), then selects the largest
        scheduler sigma strictly below it as the next sigma. This matters at
        ``t=1000``: the first full-noise step advances the scheduler index,
        but NVlabs' post-step token mask can still discard the update.

        Args:
            noise_pred: ``(B, C, F, H, W)`` model output.
            timestep: scalar integer timestep at this sampling step.
            latents: ``(B, C, F, H, W)`` current latent.
            per_token_timesteps: ``(B, F*H*W)`` per-token current
                timestep, with conditioning tokens set to 0.

        Returns:
            ``(B, C, F, H, W)`` updated latent (``prev_sample``).
        """
        if noise_pred.shape != latents.shape:
            raise ValueError(
                "Sana-WM per-token step: noise_pred / latents shape mismatch "
                f"({tuple(noise_pred.shape)} vs {tuple(latents.shape)})."
            )
        if latents.ndim != 5:
            raise ValueError(
                f"Sana-WM per-token step expects (B, C, F, H, W); got {tuple(latents.shape)}."
            )
        self._ensure_timesteps(latents.device)
        sigmas = self._sigmas_tensor.to(device=latents.device, dtype=torch.float32)
        per_token_sigmas = (
            per_token_timesteps.to(device=latents.device, dtype=torch.float32)
            / float(self.num_train_timesteps)
        )
        scheduler_sigmas = sigmas[:, None, None]
        lower_mask = scheduler_sigmas < per_token_sigmas[None] - 1e-6
        lower_sigmas = (lower_mask * scheduler_sigmas).max(dim=0).values
        # Matches diffusers' per-token branch: dt is positive and
        # ``model_output`` already carries the sign selected by the caller.
        dt = per_token_sigmas - lower_sigmas  # (B, FHW)

        # Sign convention follows NVlabs `LTXFlowEuler.sample`, which calls
        # diffusers with `model_output=-noise_pred`. Diffusers then applies
        # `prev = sample + dt * model_output`.
        #
        # 2026-05-28 GPU A/B (audit §6.13c verification): the
        # `-noise_pred` direction gives MAE=69.06 / PSNR=9.68 on the
        # 9-frame harness. `+noise_pred` gives MAE=112.5 / PSNR=6.0 —
        # clearly the wrong direction. `VLLM_OMNI_SANA_WM_FLIP_FLOW_SIGN=1`
        # swaps to the opposite convention for future verification.
        flip = os.environ.get("VLLM_OMNI_SANA_WM_FLIP_FLOW_SIGN", "").lower() in {
            "1", "true", "yes", "on"
        }
        sign = 1.0 if flip else -1.0  # default: NVlabs -noise_pred
        b, c, f, h, w = latents.shape
        latents_flat = latents.permute(0, 2, 3, 4, 1).reshape(b, -1, c).float()
        noise_flat = noise_pred.permute(0, 2, 3, 4, 1).reshape(b, -1, c).float()
        prev_flat = latents_flat + dt.unsqueeze(-1) * (sign * noise_flat)
        prev = prev_flat.reshape(b, f, h, w, c).permute(0, 4, 1, 2, 3).contiguous()
        return prev.to(latents.dtype)

    def add_noise(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Linear flow interpolation: sigma*noise + (1-sigma)*sample.

        ``sigma = timestep / num_train_timesteps`` maps the scheduler's integer
        timestep (0–1000 range) back to the [0, 1] noise level.
        """
        num_train = float(self.num_train_timesteps)
        sigma = (timestep.float() / num_train).clamp(0.0, 1.0)
        while sigma.ndim < sample.ndim:
            sigma = sigma.unsqueeze(-1)
        return (sigma * noise + (1.0 - sigma) * sample).to(sample.dtype)


@dataclass(frozen=True)
class SanaWmFlowDpmScheduler:
    """Shifted-Euler smoke scheduler — kept for backward-compat with existing tests.

    New code should use :class:`SanaWmFlowMatchScheduler`.
    """

    num_inference_steps: int
    shift: float = SANA_WM_DEFAULT_INFERENCE_FLOW_SHIFT

    def __post_init__(self) -> None:
        if self.num_inference_steps <= 0:
            raise ValueError("Sana-WM scheduler num_inference_steps must be positive.")

    def timesteps(self, *, device: torch.device) -> torch.Tensor:
        base = torch.linspace(1.0, 0.0, self.num_inference_steps + 1, device=device, dtype=torch.float32)[:-1]
        return shift_flow_timestep(base, self.shift)

    def deltas(self, *, device: torch.device) -> torch.Tensor:
        base = torch.linspace(1.0, 0.0, self.num_inference_steps + 1, device=device, dtype=torch.float32)
        shifted = shift_flow_timestep(base, self.shift)
        return shifted[:-1] - shifted[1:]

    def step(self, latents: torch.Tensor, noise_pred: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        while delta.ndim < latents.ndim:
            delta = delta.unsqueeze(-1)
        return latents - delta.to(device=latents.device, dtype=latents.dtype) * noise_pred
