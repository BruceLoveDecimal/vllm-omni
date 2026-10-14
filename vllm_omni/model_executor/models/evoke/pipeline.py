# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Alaya-EVOKE single-stage diffusion topology.

Registered so ``--deploy-config vllm_omni/deploy/evoke_realtime.yaml`` can
resolve the stage. No ``default_deploy_config_name``: the realtime topology is
opt-in.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

EVOKE_PIPELINE = PipelineConfig(
    model_type="evoke",
    model_arch="EvokePipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="video",
            model_arch="EvokePipeline",
        ),
    ),
)
