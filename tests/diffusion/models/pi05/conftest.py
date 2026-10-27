# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""π0.5 runs on vLLM's parallel linear layers, which read the tensor-parallel
group as they are built. Outside a diffusion worker nothing creates it, so these
tests run in a single-rank one, on gloo so that CPU-only hosts work too."""

import os

import pytest
import torch
from vllm.distributed import parallel_state


@pytest.fixture(scope="session", autouse=True)
def _single_rank_tensor_parallel_group():
    if parallel_state.model_parallel_is_initialized():
        yield
        return
    if not torch.distributed.is_initialized():
        port = os.environ.get("PI05_TEST_MASTER_PORT", "29571")
        parallel_state.init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            backend="gloo",
        )
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
    yield
    parallel_state.destroy_model_parallel()
    parallel_state.destroy_distributed_environment()
