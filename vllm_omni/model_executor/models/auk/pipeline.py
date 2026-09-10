# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AuK topology: semantic conditioning, latent flow matching, and VAE decoding."""

from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

AUK_PIPELINE = PipelineConfig(
    model_type="auk",
    model_arch="AuKPipeline",
    default_deploy_config_name="auk.yaml",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="audio",
            owns_tokenizer=False,
            requires_multimodal_data=False,
            engine_output_type="audio",
        ),
    ),
)
