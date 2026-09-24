# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Realtime-Venus pipeline registration and deploy configuration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_DEPLOY_DIR = Path(__file__).resolve().parents[4] / "vllm_omni" / "deploy"


def test_pipeline_reuses_the_minicpmo45_stages() -> None:
    venus = OMNI_PIPELINES["realtime_venus_omni"]
    minicpmo = OMNI_PIPELINES["minicpmo_4_5"]

    assert venus.model_type == "realtime_venus_omni"
    assert venus.model_arch == minicpmo.model_arch == "MiniCPMO45OmniForConditionalGeneration"
    assert venus.stages == minicpmo.stages
    assert venus.hf_architectures == ("RealtimeVenusOmni",)
    assert venus.hf_config_predicate is None
    assert venus.default_deploy_config_name == "realtime_venus_omni.yaml"
    assert venus.duplex_plugin == (
        "vllm_omni.model_executor.models.realtime_venus.duplex.plugin.RealtimeVenusDuplexPlugin"
    )


def test_deploy_config_inherits_the_minicpmo45_deployment() -> None:
    deploy = load_deploy_config(_DEPLOY_DIR / "realtime_venus_omni.yaml")
    stages = merge_pipeline_deploy(OMNI_PIPELINES[deploy.pipeline], deploy)
    base = load_deploy_config(_DEPLOY_DIR / "minicpmo_4_5.yaml")
    base_stages = merge_pipeline_deploy(OMNI_PIPELINES[base.pipeline], base)

    assert deploy.pipeline == "realtime_venus_omni"
    assert deploy.session_mode == "duplex"
    assert deploy.connectors == base.connectors
    assert [stage.yaml_engine_args for stage in stages] == [stage.yaml_engine_args for stage in base_stages]


@pytest.mark.parametrize(
    ("model_type", "architectures", "deploy", "expected"),
    [
        # Realtime-Venus-Omni ships its own model_type.
        ("realtime_venus_omni", ["RealtimeVenusOmni"], None, "realtime_venus_omni"),
        # Realtime-Venus-Audio reports MiniCPM-o 4.5's config verbatim.
        ("minicpmo", ["MiniCPMO"], None, "minicpmo_4_5"),
        ("minicpmo", ["MiniCPMO"], "realtime_venus_omni.yaml", "realtime_venus_omni"),
    ],
)
def test_checkpoints_select_their_pipeline(monkeypatch, model_type, architectures, deploy, expected) -> None:
    hf_config = SimpleNamespace(model_type=model_type, architectures=architectures, version="4.5")
    monkeypatch.setattr(StageConfigFactory, "get_hf_config", classmethod(lambda cls, **kwargs: hf_config))
    monkeypatch.setattr(StageConfigFactory, "try_infer_model_type", classmethod(lambda cls, **kwargs: model_type))

    pipeline = StageConfigFactory.get_pipeline_config(model="venus", trust_remote_code=True, deploy_config_path=deploy)

    assert pipeline.model_type == expected
