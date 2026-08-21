# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""StreamMind cognition gate for Mage-VL: the "should I speak yet" recurrence.

The gate watches a video stream one segment at a time and decides, per segment, whether
the model should answer now or keep listening. It is a small recurrent network -- mean-pool
each segment's patches into one token, project it, run it through a single Mamba block, and
classify -- and it ships in the checkpoint as ``streammind_gate.safetensors``.

What makes it worth its own module is the recurrence: the Mamba state carries the whole
session's history, so a segment must be folded into the state exactly once. Running the
gate over the full stream again on every new segment would both cost O(n^2) and, being a
different arithmetic path, drift from the incremental answer. This module therefore keeps
the convolution and SSM state per session and advances it segment by segment.

The kernels come from vLLM (``causal_conv1d_fn`` / ``selective_scan_fn``); the reference
implementation uses ``mamba_ssm``, which this deliberately does not depend on -- its CUDA
extension has no wheel for the versions of torch this project targets.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MageVLGateConfig:
    """Shapes of the gate as it ships in the checkpoint."""

    hidden_size: int = 2560
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: int = 160

    @property
    def d_inner(self) -> int:
        return self.hidden_size * self.expand


class MageVLGateState:
    """One session's recurrent state: the conv window and the SSM state.

    Both are updated **in place** by the kernels, which is what makes advancing a session
    cheap -- and what makes sharing an instance between sessions a bug.
    """

    def __init__(self, config: MageVLGateConfig, device, dtype: torch.dtype):
        self.conv_state = torch.zeros(
            1, config.d_inner, config.d_conv - 1, device=device, dtype=dtype
        )
        self.ssm_state = torch.zeros(
            1, config.d_inner, config.d_state, device=device, dtype=torch.float32
        )
        self.seen_segments = 0

    def reset(self) -> None:
        self.conv_state.zero_()
        self.ssm_state.zero_()
        self.seen_segments = 0


class MageVLGateMamba(nn.Module):
    """The checkpoint's Mamba-1 mixer, as an explicit recurrence over segments.

    Parameter names match the checkpoint (``in_proj`` / ``conv1d`` / ``x_proj`` /
    ``dt_proj`` / ``A_log`` / ``D`` / ``out_proj``), so weights load without a mapping.

    On not using vLLM's Mamba kernels: they are written for the engine's paged state
    cache and expect block bookkeeping (``block_idx_*``, ``initial_state_idx``) that only
    the engine's state manager produces -- called without it they silently return their
    input untouched, which was measured, not assumed. The gate runs outside the engine and
    advances one segment at a time, so the recurrence below (the same arithmetic
    ``mamba_ssm`` documents as its reference implementation) is both correct and cheap:
    one step is a handful of small matmuls per second of video. If the gate ever moves
    inside the engine, its kernels come with it.
    """

    def __init__(self, config: MageVLGateConfig):
        super().__init__()
        self.config = config
        d_inner, d_state = config.d_inner, config.d_state

        self.in_proj = nn.Linear(config.hidden_size, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=config.d_conv, groups=d_inner, bias=True
        )
        self.x_proj = nn.Linear(d_inner, config.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(config.dt_rank, d_inner, bias=True)
        self.A_log = nn.Parameter(torch.zeros(d_inner, d_state))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, config.hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor, state: MageVLGateState) -> torch.Tensor:
        """``hidden``: ``[1, T, hidden_size]`` -- one session's next ``T`` segments."""
        if hidden.shape[0] != 1:
            raise ValueError(
                f"the gate advances one session at a time, got batch {hidden.shape[0]}"
            )
        config = self.config
        seq_len = hidden.shape[1]

        projected = self.in_proj(hidden)
        x = projected[..., : config.d_inner].transpose(1, 2)
        z = projected[..., config.d_inner :]

        # Causal convolution, continued from the session's window rather than zero-padded.
        padded = torch.cat([state.conv_state.to(x.dtype), x], dim=-1)
        convolved = F.conv1d(
            padded,
            self.conv1d.weight,
            self.conv1d.bias,
            groups=config.d_inner,
        )
        state.conv_state.copy_(padded[..., -(config.d_conv - 1) :])
        convolved = F.silu(convolved)

        ssm_params = self.x_proj(convolved.transpose(1, 2))
        delta, b_proj, c_proj = ssm_params.split(
            [config.dt_rank, config.d_state, config.d_state], dim=-1
        )
        delta = F.softplus(F.linear(delta, self.dt_proj.weight, self.dt_proj.bias).float())
        a_matrix = -torch.exp(self.A_log.float())

        ssm_state = state.ssm_state
        outputs = []
        for step in range(seq_len):
            delta_t = delta[:, step].unsqueeze(-1)  # [1, d_inner, 1]
            x_t = convolved[..., step].float()
            ssm_state.mul_(torch.exp(delta_t * a_matrix)).add_(
                delta_t * b_proj[:, step].float().unsqueeze(1) * x_t.unsqueeze(-1)
            )
            y = (ssm_state * c_proj[:, step].float().unsqueeze(1)).sum(-1)
            y = y + self.D.float() * x_t
            outputs.append(y * F.silu(z[:, step].float()))

        state.seen_segments += seq_len
        scanned = torch.stack(outputs, dim=1).to(hidden.dtype)
        return self.out_proj(scanned)


class MageVLPerceptionEncoder(nn.Module):
    """Segments in, one perception token per segment out.

    Mirrors the checkpoint's ``pre_net -> VideoMamba -> post_net``: the Mamba block is
    pre-normed with a residual around it, then the whole stream is normed once more
    (``norm_fn``) before the projection.
    """

    def __init__(self, config: MageVLGateConfig | None = None):
        super().__init__()
        self.config = config or MageVLGateConfig()
        hidden = self.config.hidden_size

        self.pre_net = nn.Linear(hidden, hidden)
        self.block_norm = nn.LayerNorm(hidden)
        self.mixer = MageVLGateMamba(self.config)
        self.norm_fn = nn.LayerNorm(hidden)
        self.post_net = nn.Linear(hidden, hidden)

    def new_state(self, device=None, dtype: torch.dtype = torch.float32) -> MageVLGateState:
        device = device or self.pre_net.weight.device
        return MageVLGateState(self.config, device, dtype)

    def forward(
        self,
        vision_tokens: torch.Tensor,
        state: MageVLGateState,
    ) -> torch.Tensor:
        """``vision_tokens``: ``[1, T, patches, hidden]`` (or already pooled ``[1, T, hidden]``).

        ``state`` is advanced in place, so feeding segments one at a time and feeding them
        all at once give the same tokens -- which is the property the streaming path relies
        on and ``tests`` pins.
        """
        if vision_tokens.dim() == 4:
            vision_tokens = vision_tokens.mean(dim=2)
        if vision_tokens.dim() != 3:
            raise ValueError(f"expected [B, T, P, D] or [B, T, D], got {vision_tokens.shape}")

        batch, segments, hidden = vision_tokens.shape
        flat = vision_tokens.reshape(batch * segments, hidden)
        pre = F.leaky_relu(self.pre_net(flat)).reshape(batch, segments, hidden)

        normed = self.block_norm(pre)
        residual = self.mixer(normed, state) + pre
        tokens = self.norm_fn(residual)

        flat_tokens = tokens.reshape(batch * segments, hidden)
        return self.post_net(F.leaky_relu(flat_tokens)).reshape(batch, segments, hidden)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load from the checkpoint's ``streammind_gate.safetensors`` key names.

        Only the perception half is consumed here; ``cls_net.*`` belongs to the classifier
        and is returned unloaded rather than silently dropped.
        """
        mapping = {
            "pre_net.fc3.": "pre_net.",
            "post_net.fc3.": "post_net.",
            "mamba_model.ssms.0.norm.": "block_norm.",
            "mamba_model.ssms.0.mixer.": "mixer.",
            "mamba_model.norm_fn.": "norm_fn.",
        }
        params = dict(self.named_parameters())
        loaded: set[str] = set()

        for name, tensor in weights:
            for prefix, replacement in mapping.items():
                if name.startswith(prefix):
                    target = replacement + name[len(prefix) :]
                    param = params.get(target)
                    if param is None:
                        raise ValueError(f"checkpoint key {name!r} has no parameter {target!r}")
                    if param.shape != tensor.shape:
                        raise ValueError(
                            f"{target}: checkpoint {tuple(tensor.shape)} vs "
                            f"module {tuple(param.shape)}"
                        )
                    param.data.copy_(tensor.to(param.dtype))
                    loaded.add(target)
                    break

        expected = set(params)
        if loaded != expected:
            raise ValueError(f"gate weights missing: {sorted(expected - loaded)}")
        return loaded


__all__ = [
    "MageVLGateConfig",
    "MageVLGateMamba",
    "MageVLGateState",
    "MageVLPerceptionEncoder",
]
