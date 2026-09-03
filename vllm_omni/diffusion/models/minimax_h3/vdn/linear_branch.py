# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""VDN's bidirectional linear-attention branch: everything the window cannot see.

For a query frame ``t`` whose softmax window is ``[lo, hi]``, this summarises
frames ``[0, lo)`` and ``(hi, F]`` - the exact complement - into one ``d x d``
state per head, reads it with the query, and adds the result to the video rows.

    1. features        per token: [short conv ->] SiLU, L2-norm on q/k, no RoPE
    2. frame statistics per frame: S tokens -> A = k^T diag(b) k, B = v^T diag(b) k
    3. delta rule      transition = diag(alpha) (I + A)^-1, injection = B (I + A)^-1
    4. two scans       forward and reverse state banks over frames
    5. boundary gather the state just outside the window, decayed in to frame t
    6. readout         q . state, RMSNorm, output gate

Ported from the reference implementation rather than reinvented, and the
precision choices are load-bearing rather than defensive:

* ``alpha`` is an fp32 island end to end. The scan multiplies it across every
  frame, so a bf16 rounding compounds: after ~100 frames the worst channels'
  retention moves by tens of percent.
* ``A`` is accumulated in fp32 and symmetrised. It is the matrix the delta rule
  inverts, and computed as ``(k b)^T k`` in bf16 its ``(i, j)`` and ``(j, i)``
  entries round differently; on real activations - where patches within a frame
  are strongly correlated - that asymmetry pushes the smallest eigenvalue of
  ``I + A`` below the 1 the maths guarantees, and the Cholesky then factorises
  an indefinite matrix and fails outright. Random test inputs do not reproduce
  it.
* ``B`` stays in bf16 for its GEMM. It is a plain readout, never inverted, and
  its error enters the state linearly.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch
import torch.nn.functional as F
from torch import nn
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader

from .config import MiniMaxH3HybridAttentionConfig

#: The separable short conv's kernel, in both space and time. Not a knob: the
#: released weights are ``[C, 1, 5, 5]`` and ``[C, 1, 5]``.
SHORT_CONV_KERNEL = 5
#: The state both directional scans start from is half the prompt state, so the
#: two directions together carry roughly one copy of it while each frame's own
#: injection stays at full weight.
TEXT_STATE_SCALE = 0.5
RMS_NORM_EPS = 1e-6
L2_NORM_EPS = 1e-6
#: The low-rank width of the two gates. ``rank = head_dim`` in the release.
GATE_BOTTLENECK = 128


def _tensor_parallel_world_size() -> int:
    """The TP world size, or 1 where there is no process group.

    Reading it asserts when vLLM's parallel state was never initialised, which
    is every CPU unit test and any standalone build. Nothing is sharded there,
    so 1 is the honest answer rather than a failure - the same accommodation
    ``_sequence_parallel_local_span`` makes in the transformer.
    """
    try:
        return get_tensor_model_parallel_world_size()
    except AssertionError:
        return 1


def _shard_param_across_tp(param: torch.Tensor | None, dim: int = 0) -> None:
    """Attach vLLM's TP shard loader to a plain (non-parallel-layer) parameter.

    vLLM parallel layers narrow full checkpoint tensors to the local shard in
    their own ``weight_loader``. The branch is head-parallel but built from
    plain modules - the recurrence needs the tensors, not a distributed matmul -
    so its per-head vectors and depthwise convs need the same behaviour at load
    time. Attached only under TP > 1: at TP = 1 the checkpoint tensor already
    matches the parameter. (Same arrangement as SANA-WM's gated DeltaNet, which
    is TP-sharded the same way.)
    """
    if param is None or _tensor_parallel_world_size() <= 1:
        return
    param.weight_loader = sharded_weight_loader(dim)


@contextlib.contextmanager
def _tf32_matmul(enabled: bool) -> Iterator[None]:
    """TF32 for the one fp32 GEMM inside, restored on the way out.

    ``A`` is computed in fp32 rather than bf16 because bf16's 8 mantissa bits
    break the conditioning ``I + A`` depends on. TF32 has 10, and on real
    activations that is enough: the smallest eigenvalue of ``I + A`` is
    unchanged to many decimals where bf16 moves it by enough to fail. Scoped to
    the one matmul on purpose - the flag is global, and turning it on for the
    whole branch would also put the delta rule's solve and the scan's fp32 bmms
    on tensor cores, which is a much larger numerical change for a much smaller
    saving.
    """
    backend = getattr(torch.backends.cuda, "matmul", None)
    if not enabled or backend is None or not hasattr(backend, "allow_tf32"):
        yield
        return
    previous = backend.allow_tf32
    backend.allow_tf32 = True
    try:
        yield
    finally:
        backend.allow_tf32 = previous


class MiniMaxH3LinearSepConv(nn.Module):
    """Depthwise 5x5 spatial then 5-tap temporal conv on k and v.

    Separable rather than a dense ``5^3`` depthwise Conv3d because the halves
    ride tuned paths - cuDNN's NHWC depthwise 2-D and a fused shift-multiply-add
    over frames - while the full 3-D kernel has no fast implementation anywhere.
    The effective kernel is the outer product of the two, i.e. rank-1 in
    space-time. Temporal padding is zero and bidirectional: this branch is not
    causal and the stencil deliberately crosses VAE chunk boundaries.
    """

    def __init__(self, channels: int, targets: tuple[str, ...]) -> None:
        super().__init__()
        self.targets = tuple(targets)
        for name in self.targets:
            spatial = nn.Conv2d(
                channels,
                channels,
                SHORT_CONV_KERNEL,
                padding=SHORT_CONV_KERNEL // 2,
                groups=channels,
                bias=False,
            )
            temporal = nn.Conv1d(
                channels,
                channels,
                SHORT_CONV_KERNEL,
                padding=SHORT_CONV_KERNEL // 2,
                groups=channels,
                bias=False,
            )
            setattr(self, f"{name}_sp", spatial)
            setattr(self, f"{name}_tm", temporal)
            _shard_param_across_tp(spatial.weight)
            _shard_param_across_tp(temporal.weight)

    def forward(
        self,
        tokens: torch.Tensor,
        proj: str,
        *,
        num_frames: int,
        frame_size: tuple[int, int],
        channels: slice | None = None,
    ) -> torch.Tensor:
        """``[F*S, heads, dim]`` -> the same, convolved over (t, h, w).

        ``channels`` is the head range this rank is computing, when sequence
        parallelism has given it fewer heads than the parameters hold.
        """
        if proj not in self.targets:
            return tokens
        heads, head_dim = tokens.shape[-2], tokens.shape[-1]
        grid_h, grid_w = frame_size
        width = heads * head_dim
        rows_in = slice(None) if channels is None else channels
        # The token layout [F*S, C] read as [F, H, W, C] IS the channels-last
        # memory format of [F, C, H, W], so the depthwise conv consumes a
        # permuted view and its output permutes back for free.
        volume = tokens.reshape(num_frames, grid_h, grid_w, width).permute(0, 3, 1, 2)
        spatial = getattr(self, f"{proj}_sp").weight[rows_in]
        volume = F.conv2d(volume, spatial, padding=SHORT_CONV_KERNEL // 2, groups=width)
        rows = volume.permute(0, 2, 3, 1).reshape(num_frames, grid_h * grid_w, width)
        # The fp32 master weight is cast explicitly: an elementwise multiply is
        # not an autocast op, so fp32 x bf16 would promote the whole pass.
        taps = getattr(self, f"{proj}_tm").weight[rows_in].squeeze(1).to(rows.dtype)
        return _temporal_shift(rows, taps).reshape(-1, heads, head_dim)


def _temporal_shift(rows: torch.Tensor, taps: torch.Tensor) -> torch.Tensor:
    """Depthwise 5-tap conv over frames as shift-multiply-add.

    ``rows`` is ``[F, S, C]`` and ``taps`` ``[C, 5]``, zero-padded and
    symmetric. Written as shifts rather than ``conv1d``/``conv3d`` because the
    taps overlap almost completely, so this fuses into one pass over the tensor
    where a depthwise temporal convolution is a black box that runs well below
    bandwidth at this shape.
    """
    pad = SHORT_CONV_KERNEL // 2
    padded = F.pad(rows, (0, 0, 0, 0, pad, pad))
    frames = rows.shape[0]
    out: torch.Tensor | None = None
    for tap in range(SHORT_CONV_KERNEL):
        part = padded[tap : tap + frames] * taps[:, tap].view(1, 1, -1)
        out = part if out is None else out + part
    assert out is not None
    return out


class MiniMaxH3FrameGate(nn.Module):
    """``alpha``: KDA's double-exponential retention gate, per frame and channel.

    ``alpha = exp(-exp(A_log) * softplus(up(down(mean_frame(x))) + dt_bias))``.
    ``A_log`` is per head and the per-channel freedom is ``dt_bias``, which is
    the layout the released weights use.
    """

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.down = nn.Linear(hidden_size, GATE_BOTTLENECK, bias=False)
        self.up = nn.Linear(GATE_BOTTLENECK, num_heads * head_dim, bias=False)
        self.A_log = nn.Parameter(torch.zeros(num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.zeros(num_heads * head_dim, dtype=torch.float32))
        # ``down`` is shared by every head; everything else is per head.
        _shard_param_across_tp(self.up.weight)
        _shard_param_across_tp(self.A_log)
        _shard_param_across_tp(self.dt_bias)

    def forward(self, frame_mean: torch.Tensor, *, heads: slice | None = None) -> torch.Tensor:
        """``[F, hidden]`` fp32 -> ``[F, heads, dim]`` fp32 retention per frame.

        The weights are promoted too, not just the input: an fp32 activation
        against an 8-mantissa-bit weight was never fp32 arithmetic, and with
        autocast off nothing would reconcile the dtypes anyway.

        ``heads`` evaluates one head range: ``down`` is shared by every head, so
        only ``up``'s rows, ``dt_bias`` and ``A_log`` take the slice.
        """
        if heads is None:
            up_weight, dt_bias, a_log, num_heads = self.up.weight, self.dt_bias, self.A_log, self.num_heads
        else:
            channels = slice(heads.start * self.head_dim, heads.stop * self.head_dim)
            up_weight, dt_bias, a_log = self.up.weight[channels], self.dt_bias[channels], self.A_log[heads]
            num_heads = heads.stop - heads.start
        delta = F.linear(frame_mean.float(), self.down.weight.float())
        delta = F.linear(delta, up_weight.float())
        delta = delta + dt_bias.float()
        scale = torch.exp(a_log.float())[:, None]
        delta = delta.view(-1, num_heads, self.head_dim)
        return torch.exp(-scale * F.softplus(delta))


class MiniMaxH3BranchNorm(nn.Module):
    """RMSNorm over the head dimension with an fp32 second moment.

    ``vector_norm(dtype=fp32)`` squares and accumulates in fp32 without
    materialising an fp32 copy of a ~1.4 GiB readout, where ``pow(2)`` would
    round every square to bf16 before the accumulation starts and promoting the
    input first would cost the copy.
    """

    def __init__(self, dim: int, eps: float = RMS_NORM_EPS) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_square = torch.linalg.vector_norm(x, dim=-1, keepdim=True, dtype=torch.float32).pow(2) / x.shape[-1]
        return x * torch.rsqrt(mean_square + self.eps).to(x.dtype) * self.weight.to(x.dtype)


class MiniMaxH3LinearBranch(nn.Module):
    """The whole branch for one attention layer, on this rank's heads."""

    def __init__(
        self,
        config: MiniMaxH3HybridAttentionConfig,
        *,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        use_tf32_statistics: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.use_tf32_statistics = use_tf32_statistics
        channels = num_heads * head_dim

        self.short_conv = MiniMaxH3LinearSepConv(channels, config.short_conv_targets)
        self.alpha = MiniMaxH3FrameGate(hidden_size, num_heads, head_dim)
        # Per-head retention weight on the keys, centred on 0.5 at init.
        self.beta_proj = nn.Linear(hidden_size, num_heads, bias=False)
        # The branch's own output projection input gate, low rank per channel.
        self.output_gate = nn.ModuleDict(
            {
                "down": nn.Linear(hidden_size, GATE_BOTTLENECK, bias=False),
                "up": nn.Linear(GATE_BOTTLENECK, channels, bias=True),
            }
        )
        self.norm = MiniMaxH3BranchNorm(head_dim)
        _shard_param_across_tp(self.beta_proj.weight)
        _shard_param_across_tp(self.output_gate["up"].weight)
        _shard_param_across_tp(self.output_gate["up"].bias)

    # -- pieces the sequence owner may compute before a dispatch --------------

    def beta(self, x: torch.Tensor) -> torch.Tensor:
        """``[rows, hidden]`` -> per-head key weight ``[rows, heads]``."""
        return torch.sigmoid(self.beta_proj(x))

    def gate_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """The low-rank half of the output gate, ``[rows, 128]``.

        Split out because under sequence parallelism only this half has to
        travel: the receiving rank applies ``up`` for its own heads, so the
        payload is 128 wide instead of ``heads * dim``.
        """
        return self.output_gate["down"](x)

    def gate_from_hidden(self, hidden: torch.Tensor, *, heads: slice | None = None) -> torch.Tensor:
        """``up`` for one head range, so a rank builds only the gate it uses."""
        rows = hidden.shape[0]
        up = self.output_gate["up"]
        if heads is None:
            return torch.sigmoid(up(hidden)).view(rows, self.num_heads, self.head_dim)
        channels = slice(heads.start * self.head_dim, heads.stop * self.head_dim)
        bias = None if up.bias is None else up.bias[channels]
        gate = torch.sigmoid(F.linear(hidden, up.weight[channels], bias))
        return gate.view(rows, heads.stop - heads.start, self.head_dim)

    def gate(self, x: torch.Tensor, *, heads: slice | None = None) -> torch.Tensor:
        return self.gate_from_hidden(self.gate_hidden(x), heads=heads)

    @staticmethod
    def frame_mean(video_x: torch.Tensor, *, num_frames: int) -> torch.Tensor:
        """Per-frame mean of the residual stream, ``[F, hidden]`` fp32.

        fp32 on the reduction itself, not just inside the gate: the input is
        bf16, so a bf16 mean would round before the fp32 island starts and
        ``.float()`` in there cannot recover what the mean already discarded.
        """
        return video_x.view(num_frames, -1, video_x.shape[-1]).mean(dim=1, dtype=torch.float32)

    # -- the branch itself ----------------------------------------------------

    @torch.compiler.disable
    def forward(
        self,
        *,
        video_x: torch.Tensor | None,
        video_qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        text_qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        num_frames: int,
        tokens_per_frame: int,
        frame_size: tuple[int, int],
        bounds: list[tuple[int, int]] | tuple[tuple[int, int], ...],
        text_x: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
        frame_mean: torch.Tensor | None = None,
        text_beta: torch.Tensor | None = None,
        heads: slice | None = None,
    ) -> torch.Tensor:
        """``[F*S, heads*dim]``: the readout for every video row.

        The two anchor frames are dropped from the INPUT rather than masked out
        of the output: under ``anchor_frames="both"`` the softmax already covers
        them exactly in both directions, so including them here would double
        count. Their readout rows are exactly zero, which keeps the softmax and
        the branch an exact partition.

        ``heads`` is the head range this rank computes. Sequence parallelism
        redistributes heads at run time rather than sharding the parameters, so
        a rank holds every tensor-parallel head and reads only its own slice.
        """
        reference = video_x if video_x is not None else gate
        if reference is None:
            raise ValueError("the linear branch needs the residual stream, or the gate a dispatch computed from it")
        total_rows = num_frames * tokens_per_frame
        num_heads = self.num_heads if heads is None else heads.stop - heads.start
        width = num_heads * self.head_dim

        if num_frames <= 2:
            # The anchors are the whole clip; the window already covers it.
            return reference.new_zeros(total_rows, width)

        inner = slice(tokens_per_frame, (num_frames - 1) * tokens_per_frame)
        inner_frames = slice(1, num_frames - 1)
        readout = self._readout(
            video_x=None if video_x is None else video_x[inner],
            video_qkv=tuple(tensor[inner] for tensor in video_qkv),
            text_qkv=text_qkv,
            text_x=text_x,
            num_frames=num_frames - 2,
            tokens_per_frame=tokens_per_frame,
            frame_size=frame_size,
            # Dropping the two anchors rebases the window by one frame: the
            # complement of [lo, hi] inside frames 1..F-2 is the same arithmetic
            # on (lo - 1, hi - 1).
            bounds=[(low - 1, high - 1) for low, high in bounds[inner_frames]],
            beta=None if beta is None else beta[inner],
            gate=None if gate is None else gate[inner],
            frame_mean=None if frame_mean is None else frame_mean[inner_frames],
            text_beta=text_beta,
            heads=heads,
        )
        # Zero the two anchor frames rather than the whole tensor: at 768p this
        # is ~1.4 GiB and only 2 of its 102 frames need clearing.
        out = readout.new_empty(total_rows, readout.shape[-1])
        out[:tokens_per_frame].zero_()
        out[(num_frames - 1) * tokens_per_frame :].zero_()
        out[inner] = readout
        return out

    def _readout(
        self,
        *,
        video_x: torch.Tensor | None,
        video_qkv: tuple[torch.Tensor, ...],
        text_qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        text_x: torch.Tensor | None,
        num_frames: int,
        tokens_per_frame: int,
        frame_size: tuple[int, int],
        bounds: list[tuple[int, int]],
        beta: torch.Tensor | None,
        gate: torch.Tensor | None,
        frame_mean: torch.Tensor | None,
        text_beta: torch.Tensor | None,
        heads: slice | None,
    ) -> torch.Tensor:
        """The algorithm over exactly the frames it owns (anchors already gone)."""
        head_dim = self.head_dim
        num_local_heads = self.num_heads if heads is None else heads.stop - heads.start
        shape_per_frame = (num_frames, tokens_per_frame, num_local_heads, head_dim)

        query, key, value = self._features(video_qkv, num_frames=num_frames, frame_size=frame_size, heads=heads)
        query_by_frame = query.view(shape_per_frame).permute(0, 2, 1, 3)  # [F, H, S, d]
        key_by_frame = key.view(shape_per_frame).permute(0, 2, 1, 3)
        value_by_frame = value.view(shape_per_frame).permute(0, 2, 1, 3)

        if beta is None:
            if video_x is None:
                raise ValueError("beta must be supplied when the residual stream is not local")
            beta = self.beta(video_x)
        beta = beta.view(num_frames, tokens_per_frame, num_local_heads).permute(0, 2, 1)  # [F, H, S]

        if frame_mean is None:
            if video_x is None:
                raise ValueError("frame_mean must be supplied when the residual stream is not local")
            frame_mean = self.frame_mean(video_x, num_frames=num_frames)
        if gate is None:
            if video_x is None:
                raise ValueError("the output gate must be supplied when the residual stream is not local")
            gate = self.gate(video_x, heads=heads)

        with torch.autocast(device_type=key.device.type, enabled=False):
            statistics_a, statistics_b = self._frame_statistics(key_by_frame, value_by_frame, beta)
            alpha = self.alpha(frame_mean, heads=heads)
            text_state = self._text_state(text_x, text_qkv, text_beta=text_beta, heads=heads)
            transitions, injections = self._factor_apply(alpha, statistics_a, statistics_b)
            prefix, suffix = _run_scans(transitions, injections, text_state)
            state = _gather_linear_state(prefix, suffix, alpha, bounds, text_state=text_state)

        readout = torch.matmul(query_by_frame, state.to(query.dtype).transpose(-1, -2))  # [F, H, S, d]
        readout = readout.permute(0, 2, 1, 3).reshape(num_frames * tokens_per_frame, num_local_heads, head_dim)
        return (self.norm(readout) * gate).reshape(num_frames * tokens_per_frame, num_local_heads * head_dim)

    def _feature_one(
        self,
        tokens: torch.Tensor,
        proj: str,
        *,
        num_frames: int = 1,
        frame_size: tuple[int, int] = (1, 1),
        use_conv: bool = True,
        heads: slice | None = None,
    ) -> torch.Tensor:
        """One projection's features: ``[short conv ->] SiLU [-> L2-norm]``.

        The branch shares the softmax branch's QKV projections: it takes the raw
        pre-QK-norm, pre-RoPE tensors and applies its own post-processing, so it
        adds no projection cost and sees the Stage-B adapted projections.

        ``use_conv=False`` is the text chunk: the short conv is a (t, h, w)
        stencil over the video volume and a prompt has no such grid. Text keeps
        the rest of the pipeline, so the delta rule it feeds is the one the
        video frames feed.
        """
        if use_conv:
            channels = None if heads is None else slice(heads.start * self.head_dim, heads.stop * self.head_dim)
            tokens = self.short_conv(tokens, proj, num_frames=num_frames, frame_size=frame_size, channels=channels)
        activated = F.silu(tokens)
        if proj == "v":
            return activated
        # The cast back matters: ``normalize`` computes its norm in fp32 under
        # autocast, so the division would otherwise promote the whole feature
        # tensor and meet a bf16 value in the statistics.
        return F.normalize(activated, dim=-1, eps=L2_NORM_EPS).to(activated.dtype)

    def _features(
        self,
        qkv: tuple[torch.Tensor, ...],
        *,
        num_frames: int,
        frame_size: tuple[int, int],
        heads: slice | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query, key, value = (
            self._feature_one(tokens, proj, num_frames=num_frames, frame_size=frame_size, heads=heads)
            for proj, tokens in zip(("q", "k", "v"), qkv, strict=True)
        )
        return query, key, value

    def _frame_statistics(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        beta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Collapse each frame's S tokens into two ``d x d`` matrices.

            A[f,h,k,l] = sum_s k[f,h,s,k] beta[f,h,s] k[f,h,s,l]
            B[f,h,v,k] = sum_s v[f,h,s,v] beta[f,h,s] k[f,h,s,k]

        The ``.contiguous()`` calls are load-bearing rather than defensive: the
        inputs arrive from ``permute(0, 2, 1, 3)``, so the contraction axis
        carries stride ``H*d`` and a batched GEMM handed those strides drops to
        a fraction of its throughput.
        """
        key_compact = key.contiguous()
        key_fp32 = key_compact.float()
        scaled = (key_fp32 * beta.unsqueeze(-1).float()).contiguous()
        value_scaled = (value * beta.unsqueeze(-1).to(value.dtype)).contiguous()
        with _tf32_matmul(self.use_tf32_statistics and key.is_cuda):
            statistics_a = torch.matmul(scaled.transpose(-1, -2), key_fp32)
        # Free, and independent of dtype: guarantees the Cholesky below
        # factorises the matrix this function means rather than its lower half.
        statistics_a = 0.5 * (statistics_a + statistics_a.transpose(-1, -2))
        statistics_b = torch.matmul(value_scaled.transpose(-1, -2), key_compact).float()
        return statistics_a, statistics_b

    @staticmethod
    def _factor_apply(
        alpha: torch.Tensor,
        statistics_a: torch.Tensor,
        statistics_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The delta rule: ``S_out = (S_in diag(alpha) + B) (I + A)^-1``.

        ``I + A`` is symmetric positive definite, so the inverse is an exact
        Cholesky. It is formed explicitly rather than solved against because the
        serial scan applies the same operator once per frame anyway, and it is
        built as one triangular solve plus a product rather than
        ``cholesky_solve``'s two solves: a batched triangular solve at 128x128
        runs an order of magnitude below a batched GEMM at the same shape.
        """
        matrix = statistics_a.float()
        eye = torch.eye(matrix.shape[-1], device=matrix.device, dtype=torch.float32).expand_as(matrix)
        chol = torch.linalg.cholesky(matrix + eye)
        lower_inverse = torch.linalg.solve_triangular(chol, eye, upper=False, left=True)
        inverse = lower_inverse.transpose(-1, -2) @ lower_inverse
        transitions = alpha.unsqueeze(-1) * inverse
        injections = statistics_b.float() @ inverse
        return transitions, injections

    def _text_state(
        self,
        text_x: torch.Tensor | None,
        text_qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        *,
        text_beta: torch.Tensor | None,
        heads: slice | None = None,
    ) -> torch.Tensor | None:
        """The state both scans start from: half the prompt, written as one chunk.

        The whole prompt is injected into a zero state exactly the way a video
        frame is, with two differences: there is no causal scan inside the text
        (the encoder and the token refiner have already written word order into
        every hidden state) and alpha plays no part, because the old state is
        zero so the transition multiplies nothing.

        This is deliberately no longer the exact complement of the window - the
        softmax already sees every text row densely, so the prompt is read
        twice, once exactly and once as a recurrent state. That is the point of
        it: the branch is being given a condition, not a missing input.
        """
        if text_qkv is None or (text_x is None and text_beta is None):
            return None
        _, key_raw, value_raw = text_qkv
        length = key_raw.shape[0]
        head_dim = self.head_dim
        num_local_heads = self.num_heads if heads is None else heads.stop - heads.start
        key = self._feature_one(key_raw, "k", use_conv=False)
        value = self._feature_one(value_raw, "v", use_conv=False)
        key = key.view(1, length, num_local_heads, head_dim).permute(0, 2, 1, 3)
        value = value.view(1, length, num_local_heads, head_dim).permute(0, 2, 1, 3)
        beta = self.beta(text_x) if text_beta is None else text_beta
        beta = beta.view(1, length, num_local_heads).permute(0, 2, 1)
        statistics_a, statistics_b = self._frame_statistics(key, value, beta)
        ones = torch.ones(1, num_local_heads, head_dim, device=statistics_a.device, dtype=statistics_a.dtype)
        _, injection = self._factor_apply(ones, statistics_a, statistics_b)
        return TEXT_STATE_SCALE * injection[0]


def _run_scans(
    transitions: torch.Tensor,
    injections: torch.Tensor,
    text_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward and reverse state banks over frames.

    ``prefix[t]`` holds frames ``0..t`` and ``suffix[t]`` holds ``t..F-1``. Both
    directions start from the same (text) state, so every frame's two
    directional states carry the prompt while each frame is still injected
    exactly once per direction.

    Written into preallocated banks with ``baddbmm``: the recurrence is
    launch-bound rather than compute-bound, so folding the add into the GEMM's
    epilogue turns three launches per frame into one, and holding one bank per
    direction instead of a list plus its stack halves the live memory.
    """
    num_frames = transitions.shape[0]
    start = torch.zeros_like(injections[0]) if text_state is None else text_state.to(injections.dtype)
    prefix = torch.empty((num_frames, *start.shape), dtype=injections.dtype, device=injections.device)
    suffix = torch.empty_like(prefix)

    state = start
    for frame in range(num_frames):
        torch.baddbmm(injections[frame], state, transitions[frame], out=prefix[frame])
        state = prefix[frame]

    state = start
    for frame in range(num_frames - 1, -1, -1):
        torch.baddbmm(injections[frame], state, transitions[frame], out=suffix[frame])
        state = suffix[frame]
    return prefix, suffix


def _gather_linear_state(
    prefix: torch.Tensor,
    suffix: torch.Tensor,
    alpha: torch.Tensor,
    bounds: list[tuple[int, int]],
    *,
    text_state: torch.Tensor | None,
) -> torch.Tensor:
    """Everything OUTSIDE the window, in the query frame's frame of reference.

    ``prefix[lo-1]`` holds every frame before the window and ``suffix[hi+1]``
    every frame after it, so their sum is exactly the complement with nothing
    counted twice. Both are then decayed through the frames the window covers -
    advancing the recurrence across them while pretending they wrote nothing,
    which for this branch they did not: the softmax covers them, so applying the
    full transition would double count.

    Ends have no neighbour on one side. The index is clamped so the gather stays
    in range and the contribution is masked, or - when the scans started from a
    prompt state - reads that state instead, decayed over exactly the frames
    between the boundary and ``t``.
    """
    num_frames = prefix.shape[0]
    device = prefix.device
    last_before = torch.tensor([low for low, _ in bounds], device=device) - 1
    first_after = torch.tensor([high for _, high in bounds], device=device) + 1
    has_before = last_before >= 0
    has_after = first_after < num_frames
    frames = torch.arange(num_frames, device=device)

    state_before = prefix[last_before.clamp(min=0)]
    state_after = suffix[first_after.clamp(max=num_frames - 1)]
    if text_state is not None:
        seed = text_state.to(state_before.dtype)
        state_before = torch.where(has_before.view(-1, 1, 1, 1), state_before, seed)
        state_after = torch.where(has_after.view(-1, 1, 1, 1), state_after, seed)

    # prod_{u=a..b} alpha_u as a difference of log-prefix sums, so any (a, b)
    # pair is one subtraction rather than a product loop. The leading zero row
    # makes the prefix exclusive, i.e. the empty product is 1.
    log_alpha = torch.log(alpha.clamp_min(1e-12))
    log_prefix = torch.cat([torch.zeros_like(log_alpha[:1]), log_alpha.cumsum(0)])
    # A boundary row gathers a clamped, then discarded, state but must decay the
    # prompt over the frames it really skipped: from virtual -1 that is [0..t],
    # from virtual F it is [t..F-1]. Clamping both the same way would decay it
    # over one frame too few.
    bridge_before = (last_before + 1).clamp(min=0)
    bridge_after = first_after.clamp(max=num_frames)
    decay_before = torch.exp(log_prefix[frames + 1] - log_prefix[bridge_before])
    decay_after = torch.exp(log_prefix[bridge_after] - log_prefix[frames])
    # alpha is per KEY channel: broadcast it over the value dimension.
    state_before = state_before * decay_before.unsqueeze(2)
    state_after = state_after * decay_after.unsqueeze(2)

    if text_state is not None:
        return state_before + state_after
    return state_before * has_before.view(-1, 1, 1, 1) + state_after * has_after.view(-1, 1, 1, 1)


__all__ = [
    "GATE_BOTTLENECK",
    "SHORT_CONV_KERNEL",
    "TEXT_STATE_SCALE",
    "MiniMaxH3BranchNorm",
    "MiniMaxH3FrameGate",
    "MiniMaxH3LinearBranch",
    "MiniMaxH3LinearSepConv",
]
