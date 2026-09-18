# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AuK encoder layer fusion: the stacked formulation equals the upstream per-layer loop."""

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.model_executor.models.auk.auk import _FUSION_LN_EPS, aux_hidden_state_layers, fuse_layer_outputs

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _reference_fusion(layer_outputs, layer_weights, layer_scale):
    """The upstream loop: LayerNorm each layer in fp32, softmax-weight, sum, scale."""
    weights = torch.softmax(layer_weights, dim=0)
    fused = None
    for idx, layer_output in enumerate(layer_outputs):
        normed = F.layer_norm(layer_output.float(), (layer_output.shape[-1],), eps=_FUSION_LN_EPS)
        contribution = normed * weights[idx]
        fused = contribution if fused is None else fused + contribution
    return fused * layer_scale


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_stacked_fusion_matches_the_per_layer_loop(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    num_layers, tokens, hidden = 6, 11, 32
    layer_outputs = [torch.randn(tokens, hidden, dtype=dtype) * (idx + 1) for idx in range(num_layers)]
    layer_weights = torch.randn(num_layers)
    layer_scale = torch.tensor([0.7])

    fused = fuse_layer_outputs(layer_outputs, layer_weights=layer_weights, layer_scale=layer_scale)

    assert fused.dtype is torch.float32 and fused.shape == (tokens, hidden)
    torch.testing.assert_close(
        fused, _reference_fusion(layer_outputs, layer_weights, layer_scale), atol=1e-5, rtol=1e-5
    )


def test_aux_layers_cover_every_layer_but_the_last() -> None:
    """vLLM records the stream entering layer i, so indices 1..L-1 are the outputs of layers 0..L-2."""
    assert aux_hidden_state_layers(36) == tuple(range(1, 36))
    assert len(aux_hidden_state_layers(36)) == 35
    assert aux_hidden_state_layers(1) == ()


def test_aux_layers_recover_hf_output_hidden_states() -> None:
    """Simulate vLLM's residual-stream bookkeeping and check the fused input equals HF's layout."""
    torch.manual_seed(1)
    num_layers, tokens, hidden = 4, 5, 8
    deltas = [torch.randn(tokens, hidden) for _ in range(num_layers)]

    # HF: output_hidden_states[i + 1] is the residual stream after layer i; the
    # last entry is the final norm applied to it.
    stream = torch.zeros(tokens, hidden)
    hf_layers = []
    for delta in deltas:
        stream = stream + delta
        hf_layers.append(stream.clone())
    hf_layers[-1] = F.layer_norm(hf_layers[-1], (hidden,))

    # vLLM: layer i returns (hidden, residual) with hidden + residual == stream;
    # the aux hook stores hidden + residual when entering an aux index.
    aux = aux_hidden_state_layers(num_layers)
    hidden_states, residual = torch.zeros(tokens, hidden), None
    collected = []
    if 0 in aux:
        collected.append(hidden_states)
    for idx, delta in enumerate(deltas):
        residual = hidden_states if residual is None else hidden_states + residual
        hidden_states = delta  # the layer's contribution; the stream is hidden + residual
        if idx + 1 in aux:
            collected.append(hidden_states + residual)
    final = F.layer_norm(hidden_states + residual, (hidden,))
    recovered = [*collected, final]

    assert len(recovered) == num_layers
    for got, want in zip(recovered, hf_layers):
        torch.testing.assert_close(got, want)
