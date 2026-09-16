# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AuK pipeline topology (frozen).

Stage 0: Encoder — frozen Qwen2.5-Omni thinker, prefill only, emits the
         learned layer fusion as ``multimodal_outputs["hidden_states"]["output"]``
Stage 1: DiT     — rectified-flow transformer (plus the VAE encoder for the
         source clip), emits normalized target latents
Stage 2: Vocoder — BigVGAN-flow VAE decoder, emits a 24 kHz waveform

The codec sits in its own stage so that, under load, one batch's DiT steps
overlap the previous batch's clip rendering instead of queueing behind it.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.auk"

AUK_PIPELINE = PipelineConfig(
    model_type="auk",
    default_deploy_config_name="auk.yaml",
    model_arch="AuKForConditionalGeneration",
    hf_architectures=("AuKForConditionalGeneration",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="encoder",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=False,
            owns_tokenizer=True,
            requires_multimodal_data=True,
            hf_config_name="thinker_config",
            engine_output_type="latent",
            model_arch="AuKForConditionalGeneration",
            # No model_subdir: tools/prepare_auk_checkpoint.py assembles one
            # flat directory, so the encoder finds the thinker shards, the
            # tokenizer, the audio processor and the layer-fusion tensors in
            # auk.safetensors all under the pipeline root.
            # One prefill step produces the text condition; the sampled token
            # is discarded, and detokenizing it would only cost latency.
            sampling_constraints={"max_tokens": 1, "temperature": 0.0, "detokenize": False},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(0,),
            requires_multimodal_data=True,
            final_output=False,
            engine_output_type="latent",
            model_arch="AuKLatentPipeline",
            custom_process_input_func=f"{_PROC}.encoder2dit",
            omni_kv_config={"need_recv_cache": False},
            # Single replica, and the whole ODE runs inside one forward, so
            # the stage stays in the orchestrator process.
            inline_diffusion=True,
        ),
        StagePipelineConfig(
            stage_id=2,
            model_stage="vocoder",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(1,),
            final_output=True,
            final_output_type="audio",
            model_arch="AuKVocoderPipeline",
            custom_process_input_func=f"{_PROC}.dit2vocoder",
            omni_kv_config={"need_recv_cache": False},
            # The diffusion model-parallel state is process-global, so a
            # second inline diffusion stage cannot share the orchestrator
            # process with the DiT stage; the vocoder runs in its own process.
            # The handoff is a [gen_frames, 64] fp32 latent, a few tens of KB.
            inline_diffusion=False,
        ),
    ),
)
