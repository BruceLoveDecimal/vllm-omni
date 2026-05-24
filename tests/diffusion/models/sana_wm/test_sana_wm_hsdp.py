# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_sana_wm_hsdp_contract_is_pending_gpu_validation() -> None:
    pytest.skip("SANA-WM HSDP validation requires multi-GPU hardware.")
