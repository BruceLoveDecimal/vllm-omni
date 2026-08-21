# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only checks for the Mage-VL cognition gate's perception encoder.

The numbers that matter -- agreement with the checkpoint's own gate -- come from
``parity/run_gate_parity.py`` on a GPU box (`rel_l2` 4.55e-07 against the reference, and
5.99e-07 when fed one segment at a time). What is pinned here is the behaviour that makes
those numbers safe to rely on: the recurrence is per session, advancing it in pieces
matches advancing it at once, and the checkpoint's key names still land on parameters.
"""

import pytest
import torch

from vllm_omni.model_executor.models.mage_vl.gate import (
    MageVLGateConfig,
    MageVLGateState,
    MageVLPerceptionEncoder,
)

# Small enough to be instant, same shape relationships as the checkpoint.
TINY = MageVLGateConfig(hidden_size=32, d_state=4, d_conv=4, expand=2, dt_rank=8)


def _encoder(seed: int = 0) -> MageVLPerceptionEncoder:
    torch.manual_seed(seed)
    encoder = MageVLPerceptionEncoder(TINY).float().eval()
    # Random weights beat the module defaults here: an all-zero A_log would make the
    # recurrence collapse to something these tests could not distinguish from a bug.
    for parameter in encoder.parameters():
        torch.nn.init.normal_(parameter, std=0.05)
    return encoder


@pytest.mark.core_model
@pytest.mark.cpu
def test_streaming_matches_one_shot():
    """Segment-at-a-time must equal the whole stream at once.

    This is the property streaming depends on: a session's gate decisions are made from a
    state built incrementally, so if that diverged from the batched path, every offline
    number measured against the reference would describe a path production never takes.
    """
    encoder = _encoder()
    torch.manual_seed(1)
    vision = torch.randn(1, 6, 5, TINY.hidden_size)

    with torch.inference_mode():
        one_shot = encoder(vision, encoder.new_state())

        state = encoder.new_state()
        streamed = torch.cat(
            [encoder(vision[:, i : i + 1], state) for i in range(vision.shape[1])], dim=1
        )

    assert torch.allclose(one_shot, streamed, atol=1e-5, rtol=1e-4)


@pytest.mark.core_model
@pytest.mark.cpu
def test_sessions_do_not_share_state():
    """Two sessions advanced alternately must not see each other's history."""
    encoder = _encoder()
    torch.manual_seed(2)
    first = torch.randn(1, 3, TINY.hidden_size)
    second = torch.randn(1, 3, TINY.hidden_size)

    with torch.inference_mode():
        state_a, state_b = encoder.new_state(), encoder.new_state()
        interleaved_a = torch.cat(
            [encoder(first[:, i : i + 1], state_a) for i in range(3)], dim=1
        )
        # state_b is advanced in between, on unrelated content.
        for i in range(3):
            encoder(second[:, i : i + 1], state_b)

        alone = encoder(first, encoder.new_state())

    assert torch.allclose(interleaved_a, alone, atol=1e-5, rtol=1e-4)


@pytest.mark.core_model
@pytest.mark.cpu
def test_state_reset_returns_a_fresh_session():
    encoder = _encoder()
    torch.manual_seed(3)
    vision = torch.randn(1, 4, TINY.hidden_size)

    with torch.inference_mode():
        state = encoder.new_state()
        first_pass = encoder(vision, state)
        assert state.seen_segments == 4
        assert state.ssm_state.abs().sum() > 0

        state.reset()
        assert state.seen_segments == 0
        second_pass = encoder(vision, state)

    assert torch.allclose(first_pass, second_pass, atol=1e-6)


@pytest.mark.core_model
@pytest.mark.cpu
def test_patch_dimension_is_mean_pooled():
    """``[B, T, P, D]`` in is the same as pooling the patches first."""
    encoder = _encoder()
    torch.manual_seed(4)
    vision = torch.randn(1, 3, 7, TINY.hidden_size)

    with torch.inference_mode():
        from_patches = encoder(vision, encoder.new_state())
        from_pooled = encoder(vision.mean(dim=2), encoder.new_state())

    assert torch.allclose(from_patches, from_pooled, atol=1e-6)


@pytest.mark.core_model
@pytest.mark.cpu
def test_batched_sessions_are_rejected():
    """The state is one session's; a batch would silently blend histories."""
    encoder = _encoder()

    with pytest.raises(ValueError, match="one session at a time"):
        encoder(torch.randn(2, 3, TINY.hidden_size), encoder.new_state())

    with pytest.raises(ValueError, match=r"expected \[B, T, P, D\]"):
        encoder(torch.randn(3, TINY.hidden_size), encoder.new_state())


@pytest.mark.core_model
@pytest.mark.cpu
def test_checkpoint_key_names_load():
    """``streammind_gate.safetensors`` names map onto this module without a rename step."""
    encoder = MageVLPerceptionEncoder(TINY)
    d_inner = TINY.d_inner
    checkpoint = {
        "pre_net.fc3.weight": torch.randn(TINY.hidden_size, TINY.hidden_size),
        "pre_net.fc3.bias": torch.randn(TINY.hidden_size),
        "post_net.fc3.weight": torch.randn(TINY.hidden_size, TINY.hidden_size),
        "post_net.fc3.bias": torch.randn(TINY.hidden_size),
        "mamba_model.norm_fn.weight": torch.randn(TINY.hidden_size),
        "mamba_model.norm_fn.bias": torch.randn(TINY.hidden_size),
        "mamba_model.ssms.0.norm.weight": torch.randn(TINY.hidden_size),
        "mamba_model.ssms.0.norm.bias": torch.randn(TINY.hidden_size),
        "mamba_model.ssms.0.mixer.in_proj.weight": torch.randn(d_inner * 2, TINY.hidden_size),
        "mamba_model.ssms.0.mixer.conv1d.weight": torch.randn(d_inner, 1, TINY.d_conv),
        "mamba_model.ssms.0.mixer.conv1d.bias": torch.randn(d_inner),
        "mamba_model.ssms.0.mixer.x_proj.weight": torch.randn(
            TINY.dt_rank + TINY.d_state * 2, d_inner
        ),
        "mamba_model.ssms.0.mixer.dt_proj.weight": torch.randn(d_inner, TINY.dt_rank),
        "mamba_model.ssms.0.mixer.dt_proj.bias": torch.randn(d_inner),
        "mamba_model.ssms.0.mixer.A_log": torch.randn(d_inner, TINY.d_state),
        "mamba_model.ssms.0.mixer.D": torch.randn(d_inner),
        "mamba_model.ssms.0.mixer.out_proj.weight": torch.randn(TINY.hidden_size, d_inner),
    }
    # The classifier half of the checkpoint belongs to another module and is ignored here.
    checkpoint["cls_net.cls_model.lm_head.weight"] = torch.randn(2, TINY.hidden_size)

    loaded = encoder.load_weights(checkpoint.items())

    assert loaded == set(dict(encoder.named_parameters()))
    assert torch.equal(encoder.pre_net.weight, checkpoint["pre_net.fc3.weight"])
    assert torch.equal(
        encoder.mixer.A_log, checkpoint["mamba_model.ssms.0.mixer.A_log"]
    )


@pytest.mark.core_model
@pytest.mark.cpu
def test_incomplete_checkpoint_is_an_error():
    """A gate half-loaded from a truncated checkpoint would run and produce noise."""
    encoder = MageVLPerceptionEncoder(TINY)

    with pytest.raises(ValueError, match="gate weights missing"):
        encoder.load_weights(
            [("pre_net.fc3.weight", torch.randn(TINY.hidden_size, TINY.hidden_size))]
        )


@pytest.mark.core_model
@pytest.mark.cpu
def test_state_shapes_follow_the_config():
    state = MageVLGateState(TINY, device="cpu", dtype=torch.float32)

    assert state.conv_state.shape == (1, TINY.d_inner, TINY.d_conv - 1)
    assert state.ssm_state.shape == (1, TINY.d_inner, TINY.d_state)
    # The SSM state stays fp32 even when the model runs in bf16: it accumulates over the
    # whole session, which is exactly where low precision compounds.
    assert state.ssm_state.dtype == torch.float32
