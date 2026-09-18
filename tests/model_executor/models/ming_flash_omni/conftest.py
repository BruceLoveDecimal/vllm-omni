# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def ming_vision_encoder(init_fake_tp_group, monkeypatch):
    """Small real ViT: exercises CUDA kernels without downloading Ming weights."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    from vllm.config import MultiModalConfig
    from vllm.model_executor.models import vision
    from vllm.utils.torch_utils import set_default_torch_dtype

    from vllm_omni.model_executor.models.ming_flash_omni.vision_encoder import MingVisionEncoder

    mm_config = MultiModalConfig(mm_encoder_attn_backend="FLASH_ATTN")
    monkeypatch.setattr(vision, "get_multimodal_config", lambda: mm_config)
    config = SimpleNamespace(
        hidden_size=128,
        num_heads=2,
        image_size=32,
        patch_size=2,
        spatial_merge_size=2,
        temporal_patch_size=2,
        in_channels=3,
        out_hidden_size=64,
        intermediate_size=256,
        hidden_act="gelu",
        depth=2,
        deepstack_visual_indexes=[0],
        apply_vit_abs_pos_embed=True,
    )
    with set_default_torch_dtype(torch.bfloat16):
        encoder = MingVisionEncoder(config).cuda().eval()
    # vLLM linear layers allocate uninitialized checkpoint storage.
    generator = torch.Generator(device="cuda").manual_seed(17)
    with torch.no_grad():
        for name, parameter in encoder.named_parameters():
            if parameter.ndim >= 2:
                parameter.normal_(std=0.02, generator=generator)
            elif name.endswith("weight"):
                parameter.fill_(1)
            else:
                parameter.zero_()
    return encoder
