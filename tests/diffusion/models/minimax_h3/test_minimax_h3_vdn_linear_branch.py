# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""VDN's linear branch against a naive transcription of the same algorithm."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from tests.diffusion.models.minimax_h3.vdn_linear_reference import reference_forward
from tests.diffusion.models.minimax_h3.vdn_release import TRANSFORM_CONFIG
from vllm_omni.diffusion.models.minimax_h3.vdn.config import MiniMaxH3HybridAttentionConfig
from vllm_omni.diffusion.models.minimax_h3.vdn.linear_branch import MiniMaxH3LinearBranch
from vllm_omni.diffusion.models.minimax_h3.vdn.window import window_bounds

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

HIDDEN = 12
HEADS = 2
HEAD_DIM = 4
GRID = (2, 3)
TOKENS_PER_FRAME = GRID[0] * GRID[1]
TEXT_LEN = 5


def _config(**overrides) -> MiniMaxH3HybridAttentionConfig:
    return MiniMaxH3HybridAttentionConfig.from_transform_config(
        TRANSFORM_CONFIG, attention_head_dim=HEAD_DIM, **overrides
    )


def _branch(seed: int = 5) -> MiniMaxH3LinearBranch:
    torch.manual_seed(seed)
    branch = MiniMaxH3LinearBranch(_config(), hidden_size=HIDDEN, num_heads=HEADS, head_dim=HEAD_DIM)
    # The released weights are trained, not at their init values: give every
    # parameter a distinct scale so a mis-wired one cannot cancel out.
    with torch.no_grad():
        for index, parameter in enumerate(branch.parameters()):
            parameter.copy_(torch.randn_like(parameter) * 0.3 + 0.05 * (index % 5))
        branch.alpha.A_log.copy_(torch.log(torch.empty(HEADS).uniform_(1.0, 4.0)))
        branch.alpha.dt_bias.copy_(torch.empty(HEADS * HEAD_DIM).uniform_(-1.0, 1.0))
    return branch.to(torch.float32).eval()


def _inputs(num_frames: int, *, seed: int = 6, text: bool = True):
    generator = torch.Generator().manual_seed(seed)
    rows = num_frames * TOKENS_PER_FRAME

    def draw(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float32)

    video_x = draw(rows, HIDDEN)
    video_qkv = (draw(rows, HEADS, HEAD_DIM), draw(rows, HEADS, HEAD_DIM), draw(rows, HEADS, HEAD_DIM))
    if not text:
        return video_x, video_qkv, None, None
    text_x = draw(TEXT_LEN, HIDDEN)
    text_qkv = (
        draw(TEXT_LEN, HEADS, HEAD_DIM),
        draw(TEXT_LEN, HEADS, HEAD_DIM),
        draw(TEXT_LEN, HEADS, HEAD_DIM),
    )
    return video_x, video_qkv, text_x, text_qkv


def _run(branch, num_frames, *, text=True, **overrides):
    video_x, video_qkv, text_x, text_qkv = _inputs(num_frames, text=text)
    config = branch.config
    bounds = window_bounds(num_frames, chunk=config.chunk, radius=config.radius)
    call = {
        "video_x": video_x,
        "video_qkv": video_qkv,
        "text_x": text_x,
        "text_qkv": text_qkv,
        "num_frames": num_frames,
        "tokens_per_frame": TOKENS_PER_FRAME,
        "frame_size": GRID,
        "bounds": bounds,
    }
    call.update(overrides)
    with torch.no_grad():
        out = branch(**call)
    return out, call


@pytest.mark.parametrize("num_frames", [6, 13, 17])
@pytest.mark.parametrize("text", [True, False])
def test_the_branch_matches_a_naive_transcription_of_the_algorithm(num_frames, text):
    branch = _branch()
    out, call = _run(branch, num_frames, text=text)

    expected = reference_forward(
        branch,
        video_x=call["video_x"],
        video_qkv=call["video_qkv"],
        text_x=call["text_x"],
        text_qkv=call["text_qkv"],
        num_frames=num_frames,
        tokens_per_frame=TOKENS_PER_FRAME,
        frame_size=GRID,
        bounds=call["bounds"],
    )

    torch.testing.assert_close(out.double(), expected, rtol=2e-4, atol=2e-4)


def test_the_two_anchor_frames_read_exactly_zero():
    """The softmax covers them in both directions, so the branch must not."""
    branch = _branch()
    num_frames = 13
    out, _ = _run(branch, num_frames)

    assert torch.count_nonzero(out[:TOKENS_PER_FRAME]) == 0
    assert torch.count_nonzero(out[(num_frames - 1) * TOKENS_PER_FRAME :]) == 0
    assert torch.count_nonzero(out[TOKENS_PER_FRAME : 2 * TOKENS_PER_FRAME]) > 0


def test_a_clip_that_is_only_anchors_reads_nothing():
    branch = _branch()
    out, _ = _run(branch, 2)

    assert out.shape == (2 * TOKENS_PER_FRAME, HEADS * HEAD_DIM)
    assert torch.count_nonzero(out) == 0


def test_the_prompt_reaches_frames_whose_window_touches_a_clip_end():
    """Without the text state those rows would read nothing from that side."""
    branch = _branch()
    with_text, _ = _run(branch, 13, text=True)
    without_text, _ = _run(branch, 13, text=False)

    first_inner = slice(TOKENS_PER_FRAME, 2 * TOKENS_PER_FRAME)
    assert not torch.allclose(with_text[first_inner], without_text[first_inner])


def _slice_branch(full: MiniMaxH3LinearBranch, start: int, stop: int) -> MiniMaxH3LinearBranch:
    """The branch a tensor-parallel rank owning heads ``[start, stop)`` builds."""
    heads = stop - start
    sliced = MiniMaxH3LinearBranch(full.config, hidden_size=HIDDEN, num_heads=heads, head_dim=HEAD_DIM)
    channels = slice(start * HEAD_DIM, stop * HEAD_DIM)
    with torch.no_grad():
        # Shared by every head: the low-rank halves of both gates and the norm.
        sliced.alpha.down.weight.copy_(full.alpha.down.weight)
        sliced.output_gate["down"].weight.copy_(full.output_gate["down"].weight)
        sliced.norm.weight.copy_(full.norm.weight)
        # Per head, or per head channel.
        sliced.alpha.A_log.copy_(full.alpha.A_log[start:stop])
        sliced.alpha.dt_bias.copy_(full.alpha.dt_bias[channels])
        sliced.alpha.up.weight.copy_(full.alpha.up.weight[channels])
        sliced.beta_proj.weight.copy_(full.beta_proj.weight[start:stop])
        sliced.output_gate["up"].weight.copy_(full.output_gate["up"].weight[channels])
        sliced.output_gate["up"].bias.copy_(full.output_gate["up"].bias[channels])
        for target in full.short_conv.targets:
            for kind in ("sp", "tm"):
                source = getattr(full.short_conv, f"{target}_{kind}").weight
                getattr(sliced.short_conv, f"{target}_{kind}").weight.copy_(source[channels])
    return sliced.to(torch.float32).eval()


def test_a_tensor_parallel_rank_reproduces_its_own_head_slice():
    """Every per-head parameter has to be narrowed on the axis TP shards it on.

    A wrongly sharded ``A_log`` or conv channel still runs and still produces a
    plausible video; this is the check that says which heads it belongs to.
    """
    full = _branch()
    num_frames = 13
    out, call = _run(full, num_frames)

    for start, stop in ((0, 1), (1, 2)):
        sliced = _slice_branch(full, start, stop)
        with torch.no_grad():
            partial = sliced(
                video_x=call["video_x"],
                video_qkv=tuple(tensor[:, start:stop] for tensor in call["video_qkv"]),
                text_x=call["text_x"],
                text_qkv=tuple(tensor[:, start:stop] for tensor in call["text_qkv"]),
                num_frames=num_frames,
                tokens_per_frame=TOKENS_PER_FRAME,
                frame_size=GRID,
                bounds=call["bounds"],
            )
        expected = out.view(-1, HEADS, HEAD_DIM)[:, start:stop].reshape(partial.shape)
        torch.testing.assert_close(partial, expected, rtol=1e-5, atol=1e-5)


def test_a_sequence_parallel_rank_reads_only_its_own_head_slice():
    """Ulysses shards no parameter: it hands each rank a head block at run time.

    So a rank holds every tensor-parallel head and must read its own slice of
    the per-head parameters - the same numbers a tensor-parallel rank gets from
    narrowing them at load, arrived at the other way round.
    """
    full = _branch()
    num_frames = 13
    out, call = _run(full, num_frames)
    video_x = call["video_x"]

    for start, stop in ((0, 1), (1, 2)):
        heads = slice(start, stop)
        with torch.no_grad():
            partial = full(
                video_x=None,
                video_qkv=tuple(tensor[:, start:stop] for tensor in call["video_qkv"]),
                text_x=None,
                text_qkv=tuple(tensor[:, start:stop] for tensor in call["text_qkv"]),
                num_frames=num_frames,
                tokens_per_frame=TOKENS_PER_FRAME,
                frame_size=GRID,
                bounds=call["bounds"],
                beta=full.beta(video_x)[:, start:stop],
                gate=full.gate(video_x, heads=heads),
                frame_mean=full.frame_mean(video_x, num_frames=num_frames),
                text_beta=full.beta(call["text_x"])[:, start:stop],
                heads=heads,
            )
        expected = out.view(-1, HEADS, HEAD_DIM)[:, start:stop].reshape(partial.shape)
        torch.testing.assert_close(partial, expected, rtol=1e-5, atol=1e-5)


def test_a_dispatch_may_supply_the_parts_that_read_the_residual_stream():
    """Under Ulysses the branch runs on a rank that never held ``x``.

    beta, the output gate and the frame mean are computed by the row owners and
    travel beside Q/K/V, so the same numbers must come out either way.
    """
    branch = _branch()
    num_frames = 13
    out, call = _run(branch, num_frames)

    video_x = call["video_x"]
    with torch.no_grad():
        precomputed = branch(
            video_x=None,
            video_qkv=call["video_qkv"],
            text_x=None,
            text_qkv=call["text_qkv"],
            num_frames=num_frames,
            tokens_per_frame=TOKENS_PER_FRAME,
            frame_size=GRID,
            bounds=call["bounds"],
            beta=branch.beta(video_x),
            gate=branch.gate(video_x),
            frame_mean=branch.frame_mean(video_x, num_frames=num_frames),
            text_beta=branch.beta(call["text_x"]),
        )

    torch.testing.assert_close(precomputed, out, rtol=1e-6, atol=1e-6)


def test_the_branch_refuses_to_guess_what_a_dispatch_did_not_send():
    branch = _branch()
    with pytest.raises(ValueError, match="residual stream"):
        branch(
            video_x=None,
            video_qkv=(torch.zeros(6, HEADS, HEAD_DIM),) * 3,
            text_qkv=None,
            num_frames=1,
            tokens_per_frame=TOKENS_PER_FRAME,
            frame_size=GRID,
            bounds=[(0, 0)],
        )


def test_the_gate_splits_into_the_halves_sequence_parallelism_sends():
    """Only the low-rank half travels; the receiving rank applies ``up``.

    Its width is the gate's rank rather than anything per head, which is what
    makes the payload independent of how many heads a rank ends up with - at
    the released shape, 128 against 56 x 128.
    """
    from vllm_omni.diffusion.models.minimax_h3.vdn.linear_branch import GATE_BOTTLENECK

    branch = _branch()
    x = torch.randn(7, HIDDEN)

    torch.testing.assert_close(branch.gate_from_hidden(branch.gate_hidden(x)), branch.gate(x))
    assert branch.gate_hidden(x).shape == (7, GATE_BOTTLENECK)


def test_the_frame_mean_is_accumulated_in_fp32():
    """bf16 would round before the gate's fp32 island ever starts."""
    branch = _branch()
    video_x = torch.randn(2 * TOKENS_PER_FRAME, HIDDEN, dtype=torch.bfloat16)

    assert branch.frame_mean(video_x, num_frames=2).dtype is torch.float32


def test_the_frame_gate_stays_fp32_even_on_bfloat16_weights():
    branch = _branch().to(torch.bfloat16)
    frame_mean = torch.randn(3, HIDDEN, dtype=torch.float32)

    alpha = branch.alpha(frame_mean)

    assert alpha.dtype is torch.float32
    # A retention gate outside (0, 1] would either kill or amplify the scan.
    assert bool((alpha > 0).all() and (alpha <= 1).all())


def test_the_frame_statistics_symmetrise_what_the_delta_rule_inverts():
    """Real activations are correlated enough that the asymmetry is fatal.

    ``(k b)^T k`` computed in a reduced precision is only symmetric to that
    precision, which can push the smallest eigenvalue of ``I + A`` below one and
    make the Cholesky factorise an indefinite matrix.
    """
    branch = _branch()
    generator = torch.Generator().manual_seed(21)
    shared = torch.randn(1, HEADS, 1, HEAD_DIM, generator=generator)
    key = (shared + 0.01 * torch.randn(1, HEADS, 32, HEAD_DIM, generator=generator)).to(torch.bfloat16)
    value = torch.randn(1, HEADS, 32, HEAD_DIM, generator=generator).to(torch.bfloat16)
    beta = torch.rand(1, HEADS, 32, generator=generator).to(torch.bfloat16)

    stats_a, _ = branch._frame_statistics(key, value, beta)

    assert stats_a.dtype is torch.float32
    torch.testing.assert_close(stats_a, stats_a.transpose(-1, -2), rtol=0, atol=0)
    eye = torch.eye(HEAD_DIM).expand_as(stats_a)
    torch.linalg.cholesky(stats_a + eye)


def test_the_short_conv_is_bidirectional_across_frames():
    """The stencil deliberately crosses VAE chunk boundaries in both directions."""
    conv = _branch().short_conv
    frames, channels = 5, HEADS * HEAD_DIM
    tokens = torch.zeros(frames * TOKENS_PER_FRAME, HEADS, HEAD_DIM)
    tokens[2 * TOKENS_PER_FRAME] = 1.0

    out = conv(tokens, "k", num_frames=frames, frame_size=GRID).view(frames, TOKENS_PER_FRAME, channels)

    assert torch.count_nonzero(out[0]) > 0, "a later frame must reach an earlier one"
    assert torch.count_nonzero(out[4]) > 0, "and an earlier frame a later one"


def test_a_projection_the_release_does_not_convolve_passes_through():
    conv = _branch().short_conv
    tokens = torch.randn(2 * TOKENS_PER_FRAME, HEADS, HEAD_DIM)

    assert conv(tokens, "q", num_frames=2, frame_size=GRID) is tokens


def test_the_branch_norm_matches_torch_rms_norm():
    branch = _branch()
    x = torch.randn(4, HEADS, HEAD_DIM)

    expected = nn.functional.rms_norm(x, (HEAD_DIM,), branch.norm.weight, 1e-6)

    torch.testing.assert_close(branch.norm(x), expected, rtol=1e-5, atol=1e-5)
