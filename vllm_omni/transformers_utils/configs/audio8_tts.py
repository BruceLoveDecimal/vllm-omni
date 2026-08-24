# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Audio8 TTS config registration with transformers AutoConfig.

Registers ``model_type = "arktts"`` (and the sub-config types) so
``AutoConfig.from_pretrained("Audio8/Audio8-TTS-Preview-0.6b")`` resolves to
vllm-omni's Qwen2-shaped config instead of the checkpoint's remote code. The
0.1b shares ``model_type`` and dispatches on ``slow_backbone`` to the
Falcon-H1-shaped Slow AR sub-config.

This only wins when ``trust_remote_code`` is **off** -- transformers prefers a
checkpoint's ``auto_map`` over registered classes otherwise -- which is why
``deploy/audio8_tts.yaml`` sets ``trust_remote_code: false``.
"""

from transformers import AutoConfig

from vllm_omni.model_executor.models.audio8_tts.configuration_audio8_tts import (
    Audio8TTSConfig,
    Audio8TTSFastARConfig,
    Audio8TTSHybridSlowARConfig,
    Audio8TTSSlowARConfig,
)

AutoConfig.register("arktts", Audio8TTSConfig)
AutoConfig.register("arktts_slow_ar", Audio8TTSSlowARConfig)
AutoConfig.register("arktts_hybrid_slow_ar", Audio8TTSHybridSlowARConfig)
AutoConfig.register("arktts_fast_ar", Audio8TTSFastARConfig)

__all__ = [
    "Audio8TTSConfig",
    "Audio8TTSFastARConfig",
    "Audio8TTSHybridSlowARConfig",
    "Audio8TTSSlowARConfig",
]
