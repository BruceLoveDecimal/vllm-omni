# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Local conftest for sana_wm tests.

Overrides the root `default_vllm_config` session fixture to guard against
ROCm/torch.compile module-load errors on CUDA-only machines (the aiter_ops
import path is unconditionally triggered by VllmConfig.__post_init__ in some
vLLM builds regardless of the active backend).
"""

from __future__ import annotations

import os

import pytest
import torch

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")


@pytest.fixture(scope="session", autouse=True)
def default_vllm_config():
    try:
        from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

        has_gpu = torch.cuda.is_available() and torch.accelerator.device_count() > 0
        device_config = DeviceConfig("cuda" if has_gpu else "cpu")
        with set_current_vllm_config(VllmConfig(device_config=device_config)):
            yield
    except Exception:
        # Fallback: yield without a VllmConfig context so that unit tests that
        # don't touch vLLM infra (e.g. scheduler, camera, VAE encode tests) can
        # still run on machines where VllmConfig construction crashes.
        yield
