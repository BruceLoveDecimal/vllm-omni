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
    assert stage.diffusion_attention_config == {"default": {"backend": "TORCH_SDPA"}}
    assert stage.engine_extras["streaming_output"]
    assert stage.engine_extras["engine_backend"].endswith("ar_diffusion.engine.ARDiffusionEngine")
    # No implicit retargeting of ordinary offline generation onto AR serving.
    assert topology.default_deploy_config_name is None


def test_two_stage_deploy_is_explicit_and_not_realtime():
    path = Path(__file__).resolve().parents[2] / "vllm_omni/deploy/hy_nomas_two_stage.yaml"
    deploy = load_deploy_config(path)
    topology = resolve_pipeline_config(deploy.pipeline, None)
    assert len(topology.stages) == len(deploy.stages) == 2
    assert not deploy.async_chunk
    assert all(not stage.step_execution for stage in deploy.stages)
    assert [stage.final_output for stage in topology.stages] == [False, True]
    assert topology.stages[0].engine_output_type == "custom"
    assert topology.stages[1].input_sources == (0,)
    assert topology.stages[1].custom_process_input_func.endswith("diffusion_stage_handoff")
    assert [stage.model_arch for stage in topology.stages] == [stage.model_class_name for stage in deploy.stages]


def test_two_stage_realtime_uses_streamed_producer_and_stateful_decoder():
    path = Path(__file__).resolve().parents[2] / "vllm_omni/deploy/hy_nomas_two_stage_realtime.yaml"
    deploy = load_deploy_config(path)
    assert deploy.pipeline == "hy_nomas_two_stage"
    assert not deploy.async_chunk  # Segment acknowledgments, not the LLM chunk connector.
    producer, decoder = deploy.stages
    assert producer.step_execution
    assert producer.engine_extras["streaming_output"]
    assert producer.engine_extras["engine_backend"].endswith("ARDiffusionEngine")
    assert not decoder.step_execution
    assert decoder.engine_extras["model_config"]["streaming_stage_input"]
    # Realtime clones the first diffusion stage's request parameters to all
    # diffusion stages, so selecting np only on the decoder is insufficient.
    assert producer.default_sampling_params["output_type"] == decoder.default_sampling_params["output_type"] == "np"
    assert decoder.video_output_transport == {
        "enable_device_postprocess": True,
        "enable_registered_shm": True,
    }


@pytest.mark.parametrize(
    "deploy_name,allows_interaction",
    [("hy_nomas_two_stage_realtime.yaml", True), ("hy_nomas_two_stage.yaml", False), ("wan2_2_egd.yaml", False)],
)
def test_resolved_stage_configs_enforce_segment_interaction_capability(deploy_name, allows_interaction):
    from types import SimpleNamespace

    from vllm_omni.config.omni_config import VllmOmniConfig
    from vllm_omni.entrypoints.async_omni import AsyncOmni

    path = Path(__file__).resolve().parents[2] / "vllm_omni/deploy" / deploy_name
    deploy = load_deploy_config(path)
    topology = resolve_pipeline_config(deploy.pipeline, None)
    config = VllmOmniConfig.from_pipeline_config(topology, user_deploy_config=deploy)
    omni = object.__new__(AsyncOmni)
    omni.engine = SimpleNamespace(
        num_stages=len(config.stage_configs),
        stage_configs=config.stage_configs,
        get_stage_metadata=lambda i: SimpleNamespace(stage_type=config.stage_configs[i].stage_type),
    )
    omni.request_states = {"r": SimpleNamespace(request_id="r", external_request_id="external")}
    if allows_interaction:
        assert omni._resolve_interaction_request("external") == "r"
    else:
        with pytest.raises(ValueError, match="ordered diffusion segment handoff"):
            omni._resolve_interaction_request("external")
