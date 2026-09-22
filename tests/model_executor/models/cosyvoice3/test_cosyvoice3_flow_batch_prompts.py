# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Batched streaming flow across requests whose prompts differ in length."""

import types

import pytest
import torch
from omegaconf import DictConfig
from torch import nn

from vllm_omni.diffusion.models.cosyvoice3_audio.cosyvoice3_dit import DiT
from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.cfm import (
    CausalConditionalCFM,
    CausalMaskedDiffWithDiT,
    _join_rows,
)
from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.layers import PreLookaheadLayer
from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav import CosyVoice3Code2Wav, _right_pad_cat

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

CFM_PARAMS = DictConfig(
    {
        "sigma_min": 1e-06,
        "solver": "euler",
        "t_scheduler": "cosine",
        "training_cfg_rate": 0.2,
        "inference_cfg_rate": 0.7,
    }
)


def _code2wav() -> CosyVoice3Code2Wav:
    torch.manual_seed(0)
    estimator = DiT(dim=64, depth=2, heads=2, dim_head=32, dropout=0.0, mel_dim=80, mu_dim=80, spk_dim=80)
    # Small blocks, so the short test sequences span several of them.
    estimator.static_chunk_size = 4
    decoder = CausalConditionalCFM(
        in_channels=240, cfm_params=CFM_PARAMS, n_spks=1, spk_emb_dim=80, estimator=estimator
    )
    flow_model = CausalMaskedDiffWithDiT(
        input_size=80,
        output_size=80,
        spk_embed_dim=192,
        vocab_size=64,
        input_frame_rate=25,
        only_mask_loss=True,
        token_mel_ratio=2,
        pre_lookahead_len=2,
        pre_lookahead_layer=PreLookaheadLayer(in_channels=80, channels=80, pre_lookahead_len=2),
        decoder=decoder,
    ).eval()
    model = object.__new__(CosyVoice3Code2Wav)
    nn.Module.__init__(model)
    model.flow_model = flow_model
    # Return the mel itself, so the comparison is the flow's output.
    model._stream_hift_from_feat = types.MethodType(lambda self, feat, **kw: (feat, {}), model)
    return model


def _item(seed: int, prompt_tokens: int, tokens: int, **extra):
    g = torch.Generator().manual_seed(seed)
    return {
        "token": torch.randint(0, 64, (1, tokens), generator=g, dtype=torch.int32),
        "prompt_token": torch.randint(0, 64, (1, prompt_tokens), generator=g, dtype=torch.int32),
        "prompt_feat": torch.randn(1, 2 * prompt_tokens, 80, generator=g),
        "embedding": torch.randn(1, 192, generator=g),
        **extra,
    }


def _individual(model, item):
    return model.forward_streaming(
        token=item["token"],
        prompt_token=item["prompt_token"],
        prompt_feat=item["prompt_feat"],
        embedding=item["embedding"],
        cache_state=item.get("cache_state"),
        n_timesteps=2,
        token_offset_tokens=int(item.get("token_offset_tokens", 0)),
        finalize=bool(item.get("finalize", False)),
    )


@pytest.mark.parametrize("finalize", [False, True])
def test_ragged_prompts_batch_into_one_flow_call_and_match_each_request(finalize):
    model = _code2wav()
    items = [
        _item(0, prompt_tokens=4, tokens=9, finalize=finalize),
        _item(1, prompt_tokens=7, tokens=6, finalize=finalize),
        # A windowed stream: the resent left context and its noise offset.
        _item(
            2,
            prompt_tokens=5,
            tokens=11,
            finalize=finalize,
            token_offset_tokens=3,
            cache_state={"flow_emitted_tokens": 8},
        ),
    ]
    forward_mel_calls = []
    forward_mel = model._forward_mel

    def counting_forward_mel(**kwargs):
        forward_mel_calls.append(kwargs)
        return forward_mel(**kwargs)

    model._forward_mel = counting_forward_mel
    with torch.inference_mode():
        batched = model.forward_streaming_batch(items, n_timesteps=2)
        assert len(forward_mel_calls) == 1  # one group despite three prompt lengths
        individual = [_individual(model, item) for item in items]

    for (batched_mel, batched_state), (individual_mel, individual_state) in zip(batched, individual):
        assert batched_mel.shape == individual_mel.shape
        torch.testing.assert_close(batched_mel, individual_mel, rtol=0, atol=1e-4)
        assert batched_state == individual_state


def test_join_rows_places_each_suffix_right_after_its_own_prefix():
    prefix = torch.tensor([[1, 2, 0], [3, 4, 5]])
    suffix = torch.tensor([[6, 7, 8], [9, 0, 0]])
    joined = _join_rows(prefix, suffix, [2, 3], [3, 1])
    assert torch.equal(joined, torch.tensor([[1, 2, 6, 7, 8], [3, 4, 5, 9, 0]]))


def test_right_pad_cat_pads_the_time_dim():
    feats = _right_pad_cat([torch.ones(1, 2, 3), torch.ones(1, 4, 3)])
    assert feats.shape == (2, 4, 3)
    assert feats[0, 2:].eq(0).all() and feats[1].eq(1).all()
