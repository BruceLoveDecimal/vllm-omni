# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_sana_wm_cfg_parallel_parity_pending_gpu_validation() -> None:
    pytest.skip("SANA-WM offline cfg-parallel parity requires GPU validation.")
