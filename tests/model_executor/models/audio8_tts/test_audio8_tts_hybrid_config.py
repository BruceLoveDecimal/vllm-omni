# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Audio8 TTS 0.1b: arktts -> Falcon-H1 config translation and pipeline dispatch.

The 0.1b shares ``model_type = "arktts"`` with the 0.6b but not a Slow AR
backbone, so these guard the two things that fail silently rather than loudly:
a mistranslated GQA head count (cleanly loaded weights, NaN logits -- #6424) and
a Mamba state shape that disagrees with the checkpoint's conv1d / in_proj.
"""

import pytest

from vllm_omni.model_executor.models.audio8_tts.configuration_audio8_tts import (
    Audio8TTSConfig,
    Audio8TTSHybridSlowARConfig,
    Audio8TTSSlowARConfig,
)

# Verbatim from Audio8/Audio8-TTS-Preview-0.1b config.json, minus fields that
# only matter to the checkpoint's own remote code.
HYBRID_CONFIG_FIELDS = {
    "slow_backbone": "falcon_h1",
    "vocab_size": 69633,
    "dim": 512,
    "n_head": 8,
    "n_local_heads": 2,
    "head_dim": 64,
    "n_layer": 24,
    "intermediate_size": 768,
    "max_seq_len": 2048,
    "rope_base": 1e11,
    "norm_eps": 1e-5,
    "codebook_size": 4096,
    "num_codebooks": 10,
    "semantic_begin_id": 65537,
    "semantic_end_id": 69632,
    "eos_token_id": 228,
    "pad_token_id": 0,
    "embedding_multiplier": 0.10888671875,
    "lm_head_multiplier": 0.078125,
    "expansion_factor": 1.5,
    "mamba_chunk_size": 128,
    "mamba_conv_bias": True,
    "mamba_d_conv": 4,
    "mamba_d_head": 32,
    "mamba_d_ssm": 768,
    "mamba_d_state": 64,
    "mamba_expand": 2,
    "mamba_n_groups": 1,
    "mamba_n_heads": 24,
    "mamba_rms_norm": False,
    "mamba_use_mlp": True,
    "n_fast_layer": 4,
    "fast_dim": 512,
    "fast_n_head": 8,
    "fast_n_local_heads": 2,
    "fast_head_dim": 64,
    "fast_intermediate_size": 4864,
}

# The 0.6b, for the dispatch test: same model_type, no slow_backbone field.
DENSE_CONFIG_FIELDS = {
    "vocab_size": 155776,
    "dim": 896,
    "n_head": 14,
    "n_local_heads": 2,
    "n_layer": 24,
    "semantic_begin_id": 151678,
    "semantic_end_id": 155773,
    "eos_token_id": 151645,
}


def _rope_theta(cfg) -> float:
    """Transformers >= 5 moved ``rope_theta`` inside ``rope_parameters``."""
    params = getattr(cfg, "rope_parameters", None)
    if isinstance(params, dict) and "rope_theta" in params:
        return float(params["rope_theta"])
    return float(cfg.rope_theta)


@pytest.mark.core_model
@pytest.mark.cpu
def test_hybrid_slow_backbone_selects_falcon_config() -> None:
    hybrid = Audio8TTSConfig(**HYBRID_CONFIG_FIELDS).get_text_config()
    dense = Audio8TTSConfig(**DENSE_CONFIG_FIELDS).get_text_config()

    assert isinstance(hybrid, Audio8TTSHybridSlowARConfig)
    assert type(dense) is Audio8TTSSlowARConfig


@pytest.mark.core_model
@pytest.mark.cpu
def test_hybrid_config_translates_arktts_names() -> None:
    cfg = Audio8TTSConfig(**HYBRID_CONFIG_FIELDS).get_text_config()

    assert cfg.hidden_size == 512
    assert cfg.num_hidden_layers == 24
    assert cfg.num_attention_heads == 8
    assert cfg.head_dim == 64
    assert cfg.intermediate_size == 768
    assert cfg.max_position_embeddings == 2048
    assert cfg.rms_norm_eps == pytest.approx(1e-5)
    # 1e11, not vLLM's Falcon-H1 default: a wrong theta still runs and only
    # degrades long-context prosody.
    assert _rope_theta(cfg) == pytest.approx(1e11)
    # Kept on the sub-config so the Slow AR wrapper reads them off text_config.
    assert (cfg.codebook_size, cfg.num_codebooks) == (4096, 10)
    assert (cfg.semantic_begin_id, cfg.semantic_end_id) == (65537, 69632)
    assert cfg.eos_token_id == 228


@pytest.mark.core_model
@pytest.mark.cpu
def test_hybrid_config_maps_gqa_from_n_local_heads() -> None:
    """arktts spells GQA ``n_local_heads``; losing it yields NaN logits (#6424)."""
    fields = dict(HYBRID_CONFIG_FIELDS)
    fields.pop("num_key_value_heads", None)
    cfg = Audio8TTSConfig(**fields).get_text_config()

    assert cfg.num_key_value_heads == 2
    assert cfg.num_key_value_heads != cfg.num_attention_heads

    # An explicit num_key_value_heads in the checkpoint must not override it.
    conflicting = Audio8TTSConfig(**{**fields, "num_key_value_heads": 8}).get_text_config()
    assert conflicting.num_key_value_heads == 2


@pytest.mark.core_model
@pytest.mark.cpu
def test_hybrid_config_carries_mamba_fields() -> None:
    cfg = Audio8TTSConfig(**HYBRID_CONFIG_FIELDS).get_text_config()

    assert cfg.mamba_d_ssm == 768
    assert cfg.mamba_n_heads == 24
    assert cfg.mamba_d_head == 32
    assert cfg.mamba_d_state == 64
    assert cfg.mamba_d_conv == 4
    assert cfg.mamba_n_groups == 1
    assert cfg.mamba_rms_norm is False
    # mamba_d_ssm must factor through the head count, or MambaMixer2 builds a
    # projection the checkpoint's in_proj [1688, 512] cannot fill.
    assert cfg.mamba_n_heads * cfg.mamba_d_head == cfg.mamba_d_ssm
    assert cfg.embedding_multiplier == pytest.approx(0.10888671875)


@pytest.mark.core_model
@pytest.mark.cpu
def test_hybrid_config_round_trips_through_dict() -> None:
    """``AutoConfig`` reloads sub-configs from dicts; the class must survive."""
    original = Audio8TTSConfig(**HYBRID_CONFIG_FIELDS)
    reloaded = Audio8TTSConfig(**original.to_dict())

    text_config = reloaded.get_text_config()
    assert isinstance(text_config, Audio8TTSHybridSlowARConfig)
    assert text_config.num_key_value_heads == 2
    assert text_config.mamba_d_ssm == 768
    assert reloaded.fast_ar_config.hidden_size == 512


@pytest.mark.core_model
@pytest.mark.cpu
def test_mamba_state_shape_matches_checkpoint_tensors() -> None:
    """The hybrid cache slots must match the checkpoint's own mamba tensors.

    ``mamba.conv1d.weight`` is ``[896, 1, 4]``, so the conv state carries
    ``mamba_d_ssm + 2 * n_groups * d_state = 896`` channels over ``d_conv - 1``
    positions; the SSM state is ``n_heads x d_head x d_state``. A mismatch here
    is what silently corrupts decoding after the first frame.
    """
    from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator

    cfg = Audio8TTSConfig(**HYBRID_CONFIG_FIELDS).get_text_config()
    conv_shape, ssm_shape = MambaStateShapeCalculator.mamba2_state_shape(
        intermediate_size=cfg.mamba_d_ssm,
        tp_world_size=1,
        n_groups=cfg.mamba_n_groups,
        num_heads=cfg.mamba_n_heads,
        head_dim=cfg.mamba_d_head,
        state_size=cfg.mamba_d_state,
        conv_kernel=cfg.mamba_d_conv,
    )

    conv_channels = cfg.mamba_d_ssm + 2 * cfg.mamba_n_groups * cfg.mamba_d_state
    assert conv_channels == 896
    # vLLM reports the conv state kernel-major.
    assert conv_shape == (cfg.mamba_d_conv - 1, conv_channels) == (3, 896)
    assert ssm_shape == (cfg.mamba_n_heads, cfg.mamba_d_head, cfg.mamba_d_state) == (24, 32, 64)


@pytest.mark.core_model
@pytest.mark.cpu
def test_pipeline_resolver_dispatches_on_slow_backbone() -> None:
    from vllm_omni.model_executor.models.audio8_tts.pipeline import (
        AUDIO8_TTS_HYBRID_PIPELINE,
        AUDIO8_TTS_PIPELINE,
        resolve_audio8_tts_pipeline,
    )

    hybrid = resolve_audio8_tts_pipeline(Audio8TTSConfig(**HYBRID_CONFIG_FIELDS))
    dense = resolve_audio8_tts_pipeline(Audio8TTSConfig(**DENSE_CONFIG_FIELDS))

    assert hybrid is AUDIO8_TTS_HYBRID_PIPELINE
    assert dense is AUDIO8_TTS_PIPELINE
    assert resolve_audio8_tts_pipeline(None) is AUDIO8_TTS_PIPELINE

    # The 0.1b tokenizer's end-of-turn id, not the 0.6b's: a stale stop id makes
    # generation run to max_tokens instead of ending the utterance.
    assert hybrid.stages[0].sampling_constraints["stop_token_ids"] == [228]
    assert hybrid.stages[0].model_stage == "audio8_tts_hybrid_slow_ar"
    # Stage 1 is shared verbatim.
    assert hybrid.stages[1] is AUDIO8_TTS_PIPELINE.stages[1]
