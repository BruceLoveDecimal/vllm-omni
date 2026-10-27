# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Attention kernels preserving Hy-Nomas' BF16 inference numerics."""

from functools import lru_cache

import torch
from torch import nn


@lru_cache(maxsize=1)
def _flex_attention():
    from torch.nn.attention.flex_attention import flex_attention

    return torch.compile(flex_attention, dynamic=True, fullgraph=True)


class HyNomasTextAttentionKernel(nn.Module):
    """The text refiner uses FA2 with both padded queries and keys removed."""

    def forward(self, query, key, value, attn_metadata=None):
        from flash_attn import flash_attn_varlen_func
        from flash_attn.bert_padding import pad_input, unpad_input

        batch, length, heads, dim = query.shape
        mask = attn_metadata.attn_mask
        if mask is None:
            mask = torch.ones((batch, length), dtype=torch.bool, device=query.device)
        packed = []
        for tensor in (query, key, value):
            tensor, indices, cumulative, maximum, _ = unpad_input(tensor.flatten(2), mask)
            packed.append(tensor.unflatten(-1, (heads, dim)))
        output = flash_attn_varlen_func(*packed, cumulative, cumulative, maximum, maximum, causal=False)
        return pad_input(output.flatten(1), indices, batch, length).unflatten(-1, (heads, dim))


class HyNomasJointAttentionKernel(nn.Module):
    """FlexAttention with independent query/key visibility and shared layer masks."""

    def forward(self, query, key, value, attn_metadata=None):
        from torch.nn.attention.flex_attention import create_block_mask

        extra = attn_metadata.extra
        masks = extra["block_masks"]
        slot = extra["mask_slot"]
        if slot not in masks:
            query_mask = extra["query_mask"]
            key_mask = attn_metadata.attn_mask.reshape(key.shape[0], key.shape[1])

            def visible(batch, head, q, k):
                return query_mask[batch, q] & key_mask[batch, k]

            masks[slot] = create_block_mask(
                visible, query.shape[0], None, query.shape[1], key.shape[1], device=query.device, _compile=True
            )
        output = _flex_attention()(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            block_mask=masks[slot],
            kernel_options={"ROWS_GUARANTEED_SAFE": False, "BLOCKS_ARE_CONTIGUOUS": False},
        )
        return output.transpose(1, 2).contiguous()
