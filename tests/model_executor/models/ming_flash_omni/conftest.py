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


@pytest.fixture
def ming_vision_thinker(ming_vision_encoder):
    """Use the production thinker methods without allocating its language model."""
    from vllm.config import MultiModalConfig

    from vllm_omni.model_executor.models.ming_flash_omni.ming_flash_omni_thinker import (
        MingFlashOmniThinkerForConditionalGeneration,
    )
    from vllm_omni.model_executor.models.ming_flash_omni.projectors import VisionProjector

    thinker = MingFlashOmniThinkerForConditionalGeneration.__new__(MingFlashOmniThinkerForConditionalGeneration)
    torch.nn.Module.__init__(thinker)
    thinker.vision = ming_vision_encoder
    thinker.config = SimpleNamespace(hidden_size=96)
    thinker.thinker_config = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))
    thinker.model_config = SimpleNamespace(
        max_model_len=512,
        multimodal_config=MultiModalConfig(limit_per_prompt={"video": 0}, mm_encoder_attn_backend="FLASH_ATTN"),
    )
    thinker.linear_proj = VisionProjector(64, 96, mlp_depth=2).to(device="cuda", dtype=torch.bfloat16)
    return thinker.eval()


@pytest.fixture
def ming_encoder_graph_manager(ming_vision_thinker):
    from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

    from vllm_omni.model_executor.models.ming_flash_omni.ming_flash_omni import MingFlashOmniForConditionalGeneration

    wrapper = MingFlashOmniForConditionalGeneration.__new__(MingFlashOmniForConditionalGeneration)
    torch.nn.Module.__init__(wrapper)
    wrapper.thinker = ming_vision_thinker
    wrapper.model = ming_vision_thinker
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            encoder_cudagraph_token_budgets=[4, 8, 16],
            encoder_cudagraph_max_vision_items_per_batch=2,
            encoder_cudagraph_max_frames_per_batch=None,
        ),
        model_config=ming_vision_thinker.model_config,
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )
    manager = EncoderCudaGraphManager(config, torch.device("cuda"), torch.bfloat16, wrapper)
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        manager.capture(graph_pool=torch.cuda.graph_pool_handle())
    torch.cuda.current_stream().wait_stream(capture_stream)
    yield manager
    manager.clear()
