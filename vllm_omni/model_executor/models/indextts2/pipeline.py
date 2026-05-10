# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IndexTTS2 MVP pipeline topology.

Single-stage AR wrapper: text + reference audio -> speech waveform. The
upstream IndexTTS2 runtime owns the GPT, semantic codec, s2mel/CFM, and BigVGAN
submodules inside one stage. This MVP intentionally keeps ``async_chunk`` off
because upstream streams by completed text segment rather than codec frame.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

INDEXTTS2_PIPELINE = PipelineConfig(
    model_type="indextts2",
    model_arch="IndexTTS2ForGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="indextts2",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="audio",
            owns_tokenizer=True,
            engine_output_type="audio",
            sampling_constraints={
                "detokenize": False,
                "stop_token_ids": [2],
            },
        ),
    ),
)
