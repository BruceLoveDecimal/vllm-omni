# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_sana_wm_offline_example_documents_cfg_parallel_entrypoint() -> None:
    from pathlib import Path

    example = Path("examples/offline_inference/sana_wm/sana_wm.py")
    text = example.read_text(encoding="utf-8")

    assert "SanaWmTwoStagesPipeline" in text
    assert "sana_wm_inprocess_refiner" in text


def test_sana_wm_perf_config_contains_cfg_parallel_case() -> None:
    import json
    from pathlib import Path

    config = json.loads(Path("tests/dfx/perf/tests/test_sana_wm_vllm_omni.json").read_text(encoding="utf-8"))

    names = {case["test_name"] for case in config}
    assert "test_sana_wm_cfg2_eager" in names
