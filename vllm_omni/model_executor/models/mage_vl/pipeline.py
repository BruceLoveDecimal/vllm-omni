# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline topology for Mage-VL.

Mage-VL is an understanding model: multimodal in, text out, one engine stage. The
proactive-streaming cognition gate is a separate System-1 component that lives outside
this pipeline, so nothing here is stage-routed on its behalf.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

MAGE_VL_PIPELINE = PipelineConfig(
    model_type="mage_vl",
    default_deploy_config_name="mage_vl.yaml",
    model_arch="MageVLForConditionalGeneration",
    hf_architectures=("MageVLForConditionalGeneration",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="ar",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="text",
        ),
    ),
)
