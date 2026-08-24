# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Audio8 TTS Preview pipeline topology.

Stage 0: ``audio8_tts_slow_ar``       -- text -> semantic tokens + codec codes.
Stage 1: ``audio8_tts_codec_decoder`` -- codec codes -> 44.1 kHz waveform.

The 0.6b and 0.1b checkpoints share ``model_type = "arktts"`` but not a Slow AR
backbone, so ``resolve_audio8_tts_pipeline`` picks the topology from
``slow_backbone``. Stage 1 is byte-identical between them.
"""

from __future__ import annotations

from transformers import PretrainedConfig

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.audio8_tts"

#: ``<|im_end|>``: the Slow AR ends the utterance with an end-of-turn token.
AUDIO8_TTS_EOS_TOKEN_ID = 151645

AUDIO8_TTS_PIPELINE = PipelineConfig(
    model_type="arktts",
    default_deploy_config_name="audio8_tts.yaml",
    model_arch="Audio8TTSSlowARForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="audio8_tts_slow_ar",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=(f"{_PROC}.slow_ar_to_codec_decoder_async_chunk"),
            sampling_constraints={
                "detokenize": False,
                "stop_token_ids": [AUDIO8_TTS_EOS_TOKEN_ID],
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="audio8_tts_codec_decoder",
            model_arch="Audio8TTSCodecDecoder",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            sampling_constraints={"detokenize": True},
        ),
    ),
)

#: The 0.1b tokenizer is not the 0.6b's: end-of-turn is id 228, not 151645.
AUDIO8_TTS_HYBRID_EOS_TOKEN_ID = 228

AUDIO8_TTS_HYBRID_PIPELINE = PipelineConfig(
    model_type="arktts",
    default_deploy_config_name="audio8_tts_01b.yaml",
    model_arch="Audio8TTSHybridSlowARForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="audio8_tts_hybrid_slow_ar",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=(f"{_PROC}.slow_ar_to_codec_decoder_async_chunk"),
            sampling_constraints={
                "detokenize": False,
                "stop_token_ids": [AUDIO8_TTS_HYBRID_EOS_TOKEN_ID],
            },
        ),
        AUDIO8_TTS_PIPELINE.stages[1],
    ),
)


def resolve_audio8_tts_pipeline(hf_config: PretrainedConfig | None) -> PipelineConfig:
    """Pick the Slow AR topology from ``slow_backbone`` (0.1b: ``falcon_h1``)."""
    if hf_config is not None and str(getattr(hf_config, "slow_backbone", "dense")) == "falcon_h1":
        return AUDIO8_TTS_HYBRID_PIPELINE
    return AUDIO8_TTS_PIPELINE
