# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.cuda, pytest.mark.diffusion, pytest.mark.parallel]

# Shared tiny config used by all tests in this file.
_TINY_CFG_KWARGS = dict(
    num_blocks=1,
    hidden_size=8,
    linear_head_dim=4,
    mlp_ratio=1,
    model_max_length=3,
    chunk_plucker_channels=6,
)


def _init_vllm_tp(*, world_size: int = 1, rank: int = 0, port: int = 29681):
    """Init a single-process vLLM TP context and return the cleanup callable."""
    import torch.distributed as dist
    from vllm.config import CompilationConfig, DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port))

    ctx = set_current_vllm_config(
        VllmConfig(
            compilation_config=CompilationConfig(),
            device_config=DeviceConfig(device="cuda"),
        )
    )
    ctx.__enter__()
    init_distributed_environment(world_size=world_size, rank=rank, local_rank=rank)
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    def cleanup():
        destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()
        ctx.__exit__(None, None, None)

    return cleanup


def test_sana_wm_stage1_uses_vllm_parallel_layers_inside_tp_context() -> None:
    """All TP-aware linear sites use the correct vLLM parallel layer class."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("SANA-WM vLLM parallel-layer smoke requires CUDA.")

    from vllm_omni.diffusion.models.sana_wm import SanaWmConfig, SanaWmTransformer3DModel

    cleanup = _init_vllm_tp(port=29681)
    try:
        model = SanaWmTransformer3DModel(
            config=SanaWmConfig(**_TINY_CFG_KWARGS),
            materialize=True,
        )
        block = model.blocks[0]

        assert model.use_vllm_parallel_layers is True

        # --- pre-existing main-attention sites ---
        assert block.attn.qkv.__class__.__name__ == "QKVParallelLinear"
        assert block.attn.proj.__class__.__name__ == "RowParallelLinear"
        assert block.cross_attn.q_linear.__class__.__name__ == "ColumnParallelLinear"
        assert block.cross_attn.kv_linear.__class__.__name__ == "ColumnParallelLinear"
        assert block.cross_attn.proj.__class__.__name__ == "RowParallelLinear"
        assert model.y_embedder.y_proj.fc1.__class__.__name__ == "ColumnParallelLinear"
        assert model.y_embedder.y_proj.fc2.__class__.__name__ == "RowParallelLinear"
        assert model.t_embedder.mlp[0].__class__.__name__ == "ColumnParallelLinear"
        assert model.t_embedder.mlp[2].__class__.__name__ == "RowParallelLinear"

        # --- newly parallelised sites (5a completion) ---
        assert block.attn.q_proj_cam.__class__.__name__ == "ColumnParallelLinear"
        assert block.attn.k_proj_cam.__class__.__name__ == "ColumnParallelLinear"
        assert block.attn.v_proj_cam.__class__.__name__ == "ColumnParallelLinear"
        assert block.attn.out_proj_cam.__class__.__name__ == "RowParallelLinear"
        assert block.plucker_proj.__class__.__name__ == "ColumnParallelLinear"
        assert model.final_layer.linear.__class__.__name__ == "ColumnParallelLinear"
        assert model.t_block[1].__class__.__name__ == "ColumnParallelLinear"
        assert "RMSNorm" in model.attention_y_norm.__class__.__name__
    finally:
        cleanup()


def test_sana_wm_stage1_tp1_forward_output_shape() -> None:
    """Stage-1 model produces the correct output shape inside a tp_size=1 vLLM context.

    This exercises the full forward path — including the new t_block, final_layer,
    attention_y_norm, and plucker_proj TP sites — end-to-end under the vLLM
    distributed context that activates parallel layers.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("SANA-WM tp_size=1 forward smoke requires CUDA.")

    from vllm_omni.diffusion.models.sana_wm import SanaWmConfig, SanaWmTransformer3DModel

    cleanup = _init_vllm_tp(port=29682)
    try:
        config = SanaWmConfig(**_TINY_CFG_KWARGS)
        model = SanaWmTransformer3DModel(config=config, materialize=True).cuda()

        latents = torch.randn(1, 4, 1, 4, 4, device="cuda")
        enc = torch.randn(1, 3, 6, device="cuda")
        plucker = torch.randn(1, 6, 1, 4, 4, device="cuda")

        with torch.no_grad():
            out = model(latents, torch.tensor([0.5], device="cuda"), encoder_hidden_states=enc, plucker=plucker)

        assert out.shape == latents.shape, f"tp_size=1 forward: {out.shape} != {latents.shape}"
        assert torch.isfinite(out).all(), "tp_size=1 forward produced non-finite values"
    finally:
        cleanup()


def test_sana_wm_stage1_tp1_output_matches_fallback() -> None:
    """tp_size=1 with TP layers and use_vllm_parallel_layers=False produce identical output shapes.

    With tp_size=1 and disable_tp=True (which _disable_tp_for_vllm_layer returns
    when the TP group is size 1), ColumnParallelLinear behaves like nn.Linear
    modulo internal state.  This test confirms the shapes — not numerics, since
    initialisation differs — are identical.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("SANA-WM tp_size=1 parity smoke requires CUDA.")

    from vllm_omni.diffusion.models.sana_wm import SanaWmConfig, SanaWmTransformer3DModel

    cleanup = _init_vllm_tp(port=29683)
    try:
        config = SanaWmConfig(**_TINY_CFG_KWARGS)
        latents = torch.randn(1, 4, 1, 4, 4, device="cuda")
        enc = torch.randn(1, 3, 6, device="cuda")

        shapes = {}
        for use_parallel in (True, False):
            model = SanaWmTransformer3DModel(
                config=config, materialize=True, use_vllm_parallel_layers=use_parallel
            ).cuda()
            with torch.no_grad():
                out = model(latents, torch.tensor([0.5], device="cuda"), encoder_hidden_states=enc)
            shapes[use_parallel] = out.shape

        assert shapes[True] == shapes[False], (
            f"Shape mismatch: TP={shapes[True]}, fallback={shapes[False]}"
        )
    finally:
        cleanup()
