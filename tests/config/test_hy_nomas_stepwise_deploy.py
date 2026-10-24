# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Explicit deploy resolution must reach streamed AR execution, not generic diffusion."""

from pathlib import Path

import pytest

from vllm_omni.config.pipeline_registry import resolve_pipeline_config
from vllm_omni.config.stage_config import StageExecutionType, load_deploy_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_deploy_resolves_a_registered_realtime_video_topology():
    path = Path(__file__).resolve().parents[2] / "vllm_omni/deploy/hy_nomas_stepwise.yaml"
    deploy = load_deploy_config(path)
    topology = resolve_pipeline_config(deploy.pipeline, None)
    assert topology is not None
    assert len(topology.stages) == len(deploy.stages) == 1
    assert topology.stages[0].execution_type == StageExecutionType.DIFFUSION
    assert topology.stages[0].final_output_type == "video"
    assert topology.stages[0].model_arch == deploy.stages[0].model_class_name
    stage = deploy.stages[0]
    assert stage.max_num_seqs == 1 and stage.step_execution
    assert stage.engine_extras["streaming_output"]
    assert stage.engine_extras["engine_backend"].endswith("ar_diffusion.engine.ARDiffusionEngine")
    # No implicit retargeting of ordinary offline generation onto AR serving.
    assert topology.default_deploy_config_name is None
