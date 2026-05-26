# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.cuda, pytest.mark.diffusion, pytest.mark.parallel]


def test_sana_wm_stage1_uses_vllm_parallel_layers_inside_tp_context() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("SANA-WM vLLM parallel-layer smoke requires CUDA.")

    import torch.distributed as dist
    from vllm.config import CompilationConfig, DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )

    from vllm_omni.diffusion.models.sana_wm import SanaWmConfig, SanaWmTransformer3DModel

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29681")
    try:
        with set_current_vllm_config(
            VllmConfig(
                compilation_config=CompilationConfig(),
                device_config=DeviceConfig(device="cuda"),
            )
        ):
            init_distributed_environment(world_size=1, rank=0, local_rank=0)
            initialize_model_parallel(tensor_model_parallel_size=1)
            model = SanaWmTransformer3DModel(
                config=SanaWmConfig(
                    num_blocks=1,
                    hidden_size=8,
                    linear_head_dim=4,
                    mlp_ratio=1,
                    model_max_length=3,
                ),
                materialize=True,
            )
            block = model.blocks[0]

            assert model.use_vllm_parallel_layers is True
            assert block.attn.qkv.__class__.__name__ == "QKVParallelLinear"
            assert block.attn.proj.__class__.__name__ == "RowParallelLinear"
            assert block.cross_attn.q_linear.__class__.__name__ == "ColumnParallelLinear"
            assert block.cross_attn.kv_linear.__class__.__name__ == "ColumnParallelLinear"
            assert block.cross_attn.proj.__class__.__name__ == "RowParallelLinear"
            assert model.y_embedder.y_proj.fc1.__class__.__name__ == "ColumnParallelLinear"
            assert model.y_embedder.y_proj.fc2.__class__.__name__ == "RowParallelLinear"
            assert model.t_embedder.mlp[0].__class__.__name__ == "ColumnParallelLinear"
            assert model.t_embedder.mlp[2].__class__.__name__ == "RowParallelLinear"
    finally:
        destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()
