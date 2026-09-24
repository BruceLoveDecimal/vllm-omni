# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Realtime-Venus pipeline topology (frozen).

Realtime-Venus is fine-tuned from MiniCPM-o 4.5 without architecture changes:
the checkpoint keeps every MiniCPM-o 4.5 weight name and only adds tokenizer
tokens. It therefore runs on the MiniCPM-o 4.5 Thinker -> Talker -> Code2Wav
stages and full-duplex plugin unchanged.

``Realtime-Venus-Omni`` ships its own ``model_type`` and is auto-detected.
``Realtime-Venus-Audio`` reports MiniCPM-o 4.5's ``model_type``/``version``
verbatim, so it resolves to the MiniCPM-o 4.5 pipeline, or to this one
through the deploy YAML's ``pipeline`` key.
"""

from dataclasses import replace

from vllm_omni.model_executor.models.minicpmo_4_5.pipeline import MINICPMO_4_5_PIPELINE

REALTIME_VENUS_OMNI_PIPELINE = replace(
    MINICPMO_4_5_PIPELINE,
    model_type="realtime_venus_omni",
    default_deploy_config_name="realtime_venus_omni.yaml",
    hf_architectures=("RealtimeVenusOmni",),
    hf_config_predicate=None,
)
