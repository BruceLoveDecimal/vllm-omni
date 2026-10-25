# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Single-stage Hy-Nomas topology; realtime stepwise deployment is explicit."""

from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

HY_NOMAS_PIPELINE = PipelineConfig(
    model_type="hy_nomas",
    model_arch="HyNomasPipeline",
    diffusers_class_name="HyNomasModularPipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="video",
            model_arch="HyNomasPipeline",
        ),
    ),
)

# Full-request handoff first. Chunk transport/backpressure is a separate
# orchestrator extension; do not enable realtime streaming for this topology.
HY_NOMAS_TWO_STAGE_PIPELINE = PipelineConfig(
    model_type="hy_nomas_two_stage",
    model_arch="HyNomasDenoisePipeline",
    default_deploy_config_name="hy_nomas_two_stage.yaml",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=False,
            engine_output_type="custom",
            model_arch="HyNomasDenoisePipeline",
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(0,),
            final_output=True,
            final_output_type="video",
            model_arch="HyNomasDecodePipeline",
            custom_process_input_func=(
                "vllm_omni.model_executor.stage_input_processors.diffusion_disagg.diffusion_stage_handoff"
            ),
        ),
    ),
)
