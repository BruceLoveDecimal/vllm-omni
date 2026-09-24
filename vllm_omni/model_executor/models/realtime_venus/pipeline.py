# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Realtime-Venus pipeline topology (frozen).

Realtime-Venus is fine-tuned from MiniCPM-o 4.5 without architecture changes:
the checkpoint keeps every MiniCPM-o 4.5 weight name and adds four tokenizer
tokens (``<delegate>``/``</delegate>`` and ``<backend>``/``</backend>``). It
therefore runs on the MiniCPM-o 4.5 Thinker -> Talker -> Code2Wav stages
unchanged; only the full-duplex session policy differs (in-stream text input,
delegation as function calls, packaged default voice).

``Realtime-Venus-Omni`` ships its own ``model_type`` and is auto-detected.
``Realtime-Venus-Audio`` reports MiniCPM-o 4.5's ``model_type``/``version``
verbatim, so it selects this pipeline through the deploy YAML's
``pipeline`` key.
"""

from dataclasses import replace

from vllm_omni.model_executor.models.minicpmo_4_5.pipeline import MINICPMO_4_5_PIPELINE

REALTIME_VENUS_OMNI_PIPELINE = replace(
    MINICPMO_4_5_PIPELINE,
    model_type="realtime_venus_omni",
    default_deploy_config_name="realtime_venus_omni.yaml",
    duplex_plugin="vllm_omni.model_executor.models.realtime_venus.duplex.plugin.RealtimeVenusDuplexPlugin",
    hf_architectures=("RealtimeVenusOmni",),
    hf_config_predicate=None,
)
