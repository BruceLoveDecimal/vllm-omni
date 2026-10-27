# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Compact full-attention GQA without replacing PixelUMM's streaming decoder."""

from hymm.models.vae.pixelumm_decoder.model import GenAttention
from torch import nn

from vllm_omni.diffusion.attention.layer import Attention


class HyNomasDecoderAttention(GenAttention):
    def __init__(self, source: GenAttention):
        # Reuse the reference projections and norms without allocating another
        # set of decoder weights. Parameter names and offload ownership stay flat.
        nn.Module.__init__(self)
        self.arch = source.arch
        for name, child in source.named_children():
            self.add_module(name, child)
        self.attention = Attention(
            num_heads=self.arch.num_heads,
            num_kv_heads=self.arch.num_kv_heads,
            head_size=self.arch.head_dim,
            softmax_scale=self.arch.head_dim**-0.5,
            causal=False,
            role="hy_nomas.decoder",
            role_category="self",
            skip_sequence_parallel=True,
        )
        self.train(source.training)

    def forward(self, x, cos, sin, block_mask=None):
        if block_mask is not None:
            # The reference local path already uses FlexAttention with GQA.
            return super().forward(x, cos, sin, block_mask)
        arch = self.arch
        batch, count, _ = x.shape
        query = self.q_proj(x).view(batch, count, arch.num_heads, arch.head_dim)
        key = self.k_proj(x).view(batch, count, arch.num_kv_heads, arch.head_dim)
        value = self.v_proj(x).view(batch, count, arch.num_kv_heads, arch.head_dim)
        # Retain reference FP32 norm/RoPE ordering and bounded intermediates.
        query = self._prepare_qk(query, cos, sin, self.q_norm_t, self.q_norm_hw, value.dtype)
        key = self._prepare_qk(key, cos, sin, self.k_norm_t, self.k_norm_hw, value.dtype)
        output = self.attention(query, key, value)
        return self.o_proj(output.reshape(batch, count, arch.num_heads * arch.head_dim))


def enable_compact_gqa(decoder: nn.Module) -> None:
    for layer in decoder.layers:
        if not isinstance(layer.attn, HyNomasDecoderAttention):
            layer.attn = HyNomasDecoderAttention(layer.attn)
