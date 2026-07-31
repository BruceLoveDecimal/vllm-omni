# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _init_distributed():
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29531")
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="env://",
    )
    initialize_model_parallel()
    yield
    cleanup_dist_env_and_memory()


@pytest.fixture(autouse=True)
def _force_default_gemm(monkeypatch):
    from vllm.model_executor.layers.utils import default_unquantized_gemm

    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.dispatch_unquantized_gemm",
        lambda: default_unquantized_gemm,
    )


def _tiny_transformer(**overrides):
    from vllm_omni.diffusion.models.mage_flow.mage_flow_transformer import (
        MageFlowTransformer2DModel,
    )

    kwargs = {
        "in_channels": 4,
        "out_channels": 4,
        "context_in_dim": 6,
        "hidden_size": 8,
        "num_heads": 1,
        "depth": 1,
        "axes_dim": [2, 2, 4],
    }
    kwargs.update(overrides)
    return MageFlowTransformer2DModel(**kwargs)


def _initialize_parameters(module: nn.Module) -> None:
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.uniform_(-0.02, 0.02)


class _GridRope(nn.Module):
    def forward(self, grids, device=None):
        length = sum(frame * height * width for frame, height, width in grids)
        value = complex(length + 1, length)
        return torch.full(
            (length, 4),
            value,
            dtype=torch.complex64,
            device=device,
        )


def test_image_rope_prepare_uses_complex_one_for_padding():
    from vllm_omni.diffusion.models.mage_flow.mage_flow_layers import (
        MageFlowImageRopePrepare,
    )

    prepare = MageFlowImageRopePrepare(nn.Identity(), _GridRope())
    hidden_states = torch.randn(2, 3, 4)
    attention_mask = torch.tensor([[True, True, False], [True, False, False]])

    projected, rotary = prepare(
        hidden_states,
        attention_mask,
        [[(1, 1, 2)], [(1, 1, 1)]],
        max_image_tokens=3,
    )

    assert projected.shape == hidden_states.shape
    assert rotary.shape == (2, 3, 4)
    torch.testing.assert_close(
        rotary[0, 2:],
        torch.ones_like(rotary[0, 2:]),
    )
    torch.testing.assert_close(
        rotary[1, 1:],
        torch.ones_like(rotary[1, 1:]),
    )
    assert not torch.equal(rotary[0, 0], torch.ones_like(rotary[0, 0]))


class _AttentionRecorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.metadata = None

    def forward(self, query, key, value, attn_metadata=None):
        self.metadata = attn_metadata
        return torch.cat([attn_metadata.joint_value, value], dim=1)


def test_sp_attention_metadata_carries_merged_text_and_image_mask(
    monkeypatch,
):
    from vllm_omni.diffusion.models.mage_flow import mage_flow_layers

    attention = mage_flow_layers.MageJointAttention(
        query_dim=8,
        heads=1,
        dim_head=8,
        added_kv_proj_dim=8,
    )
    _initialize_parameters(attention)
    recorder = _AttentionRecorder()
    attention.attention = recorder
    monkeypatch.setattr(
        mage_flow_layers,
        "_sequence_parallel_active",
        lambda: True,
    )

    image_mask = torch.ones(1, 2, dtype=torch.bool)
    text_mask = torch.ones(1, 3, dtype=torch.bool)
    sp_mask = torch.tensor(
        [[True, True, False, True, True, True, False]]
    )
    rotary = torch.ones(1, 2, 4, dtype=torch.complex64)
    image_output, text_output = attention(
        torch.randn(1, 2, 8),
        torch.randn(1, 3, 8),
        rotary,
        image_attention_mask=image_mask,
        encoder_attention_mask=text_mask,
        sp_attention_mask=sp_mask,
    )

    assert image_output.shape == (1, 2, 8)
    assert text_output.shape == (1, 3, 8)
    assert recorder.metadata.attn_mask is sp_mask
    assert recorder.metadata.joint_attn_mask is None
    assert recorder.metadata.joint_strategy == "front"


def test_single_request_explicit_padding_uses_only_valid_attention_tokens():
    from vllm_omni.diffusion.models.mage_flow import mage_flow_layers

    class _LocalAttentionRecorder(nn.Module):
        def __init__(self):
            super().__init__()
            self.query_lengths = []

        def forward(self, query, _key, value, attn_metadata=None):
            assert attn_metadata is None
            self.query_lengths.append(query.shape[1])
            return value

    attention = mage_flow_layers.MageJointAttention(
        query_dim=8,
        heads=1,
        dim_head=8,
        added_kv_proj_dim=8,
    )
    _initialize_parameters(attention)
    recorder = _LocalAttentionRecorder()
    attention.attention = recorder
    image_mask = torch.tensor([[True, True, False]])
    text_mask = torch.tensor([[True, False]])

    image_output, text_output = attention(
        torch.randn(1, 3, 8),
        torch.randn(1, 2, 8),
        torch.ones(1, 3, 4, dtype=torch.complex64),
        image_attention_mask=image_mask,
        encoder_attention_mask=text_mask,
    )

    assert recorder.query_lengths == [3]
    assert not image_output[:, 2:].count_nonzero()
    assert not text_output[:, 1:].count_nonzero()


def test_mage_flow_masked_sp_uses_sdpa_fallback():
    from vllm_omni.diffusion.attention.backends.abstract import (
        AttentionMetadata,
        QueryRange,
    )
    from vllm_omni.diffusion.models.mage_flow.mage_flow_layers import (
        _MageFlowAttention,
    )

    class _Fallback:
        def __init__(self):
            self.called = False

        def forward(self, query, _key, _value, _metadata):
            self.called = True
            return torch.zeros_like(query)

    class _Unexpected:
        def forward(self, *_args):
            raise AssertionError("configured backend must not receive this mask")

    attention = _MageFlowAttention.__new__(_MageFlowAttention)
    nn.Module.__init__(attention)
    attention.attention = _Unexpected()
    attention.sdpa_fallback = _Fallback()
    query = torch.zeros(1, 4, 1, 2)
    key = value = torch.zeros(1, 6, 1, 2)
    metadata = AttentionMetadata(
        attn_mask=torch.ones(1, 6, dtype=torch.bool),
        query_ranges=(QueryRange(0, 4, 0),),
    )

    output = attention._run_local_attention(query, key, value, metadata)

    assert attention.sdpa_fallback.called
    assert output.shape == query.shape
@pytest.mark.parametrize(
    ("mask_sp_padding", "encoder_mask_values", "expected_image_mask"),
    [
        (True, None, [True, True, True, False]),
        (True, [True, False], [True, True, True, False]),
        (False, None, None),
        (False, [True, False], [True, True, True, True]),
    ],
)
def test_sp_auto_padding_mask_comes_from_forward_context(
    mask_sp_padding,
    encoder_mask_values,
    expected_image_mask,
):
    from vllm_omni.diffusion.forward_context import (
        ForwardContext,
        override_forward_context,
    )

    model = _tiny_transformer()
    model.parallel_config = SimpleNamespace(
        mask_sp_padding=mask_sp_padding,
        sequence_parallel_size=2,
    )

    class _RopePrepare(nn.Module):
        def __init__(self, projection):
            super().__init__()
            self.projection = projection

        def forward(self, hidden_states, _mask, _grids, _max_tokens):
            projected = self.projection(hidden_states)
            rotary = torch.ones(
                projected.shape[0],
                projected.shape[1],
                4,
                dtype=torch.complex64,
                device=projected.device,
            )
            return projected, rotary

    class _Timestep(nn.Module):
        def forward(self, timestep, hidden_states):
            return hidden_states.new_zeros(timestep.shape[0], 8)

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.kwargs = None

        def forward(self, **kwargs):
            self.kwargs = kwargs
            return kwargs["encoder_hidden_states"], kwargs["hidden_states"]

    class _OutputNorm(nn.Module):
        def forward(self, hidden_states, _temb):
            return hidden_states

    block = _Block()
    model.image_rope_prepare = _RopePrepare(model.img_in)
    model.time_text_embed = _Timestep()
    model.transformer_blocks = nn.ModuleList([block])
    model.norm_out = _OutputNorm()
    _initialize_parameters(model)

    context = ForwardContext(
        sp_padding_size=1,
        sp_original_seq_len=3,
        sp_plan_hooks_applied=True,
        _sp_shard_depth=1,
    )
    with override_forward_context(context):
        encoder_mask = (
            None
            if encoder_mask_values is None
            else torch.tensor([encoder_mask_values])
        )
        model(
            hidden_states=torch.randn(1, 3, 4),
            encoder_hidden_states=torch.randn(1, 2, 6),
            timestep=torch.zeros(1),
            image_grid_hw=(1, 3),
            encoder_attention_mask=encoder_mask,
        )

    if expected_image_mask is None:
        assert block.kwargs["sp_attention_mask"] is None
    else:
        expected_text_mask = (
            [True, True]
            if encoder_mask_values is None
            else encoder_mask_values
        )
        expected = torch.tensor(
            [expected_text_mask + expected_image_mask]
        )
        torch.testing.assert_close(
            block.kwargs["sp_attention_mask"],
            expected,
        )
    if encoder_mask is None:
        assert block.kwargs["encoder_attention_mask"] is None
    else:
        torch.testing.assert_close(
            block.kwargs["encoder_attention_mask"],
            encoder_mask,
        )
    assert block.kwargs["image_attention_mask"] is None


def test_sp_rejects_request_level_image_padding():
    from vllm_omni.diffusion.forward_context import (
        ForwardContext,
        override_forward_context,
    )

    model = _tiny_transformer()
    context = ForwardContext(
        sp_plan_hooks_applied=True,
        _sp_shard_depth=1,
    )

    with (
        override_forward_context(context),
        pytest.raises(ValueError, match="padded image token"),
    ):
        model(
            hidden_states=torch.randn(1, 3, 4),
            encoder_hidden_states=torch.randn(1, 2, 6),
            timestep=torch.zeros(1),
            image_grid_hw=(1, 2),
            image_attention_mask=torch.tensor([[True, True, False]]),
        )


def test_masked_sp_rejects_ring_attention():
    from vllm_omni.diffusion.forward_context import (
        ForwardContext,
        override_forward_context,
    )

    model = _tiny_transformer()
    model.parallel_config = SimpleNamespace(
        mask_sp_padding=True,
        sequence_parallel_size=2,
        ring_degree=2,
    )
    context = ForwardContext(
        sp_plan_hooks_applied=True,
        _sp_shard_depth=1,
    )

    with (
        override_forward_context(context),
        pytest.raises(ValueError, match="ring_degree"),
    ):
        model(
            hidden_states=torch.randn(1, 3, 4),
            encoder_hidden_states=torch.randn(1, 2, 6),
            timestep=torch.zeros(1),
            image_grid_hw=(1, 3),
            encoder_attention_mask=torch.tensor([[True, False]]),
        )
