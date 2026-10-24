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
