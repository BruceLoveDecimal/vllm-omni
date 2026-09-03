# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""What a VDN-H3 release has to say before this server will run it."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from tests.diffusion.models.minimax_h3.vdn_release import HEAD_DIM, TRANSFORM_CONFIG, write_release
from vllm_omni.diffusion.models.minimax_h3.vdn.checkpoint import (
    VDN_AUDIO_SHIFT,
    VDN_TURBO_DENOISE_STEPS,
    VDN_VIDEO_SHIFT,
    VdnCheckpointError,
    VdnSpec,
    resolve_vdn_checkpoint,
)
from vllm_omni.diffusion.models.minimax_h3.vdn.config import (
    MiniMaxH3HybridAttentionConfig,
    MiniMaxH3HybridGeometry,
    VdnConfigError,
)
from vllm_omni.diffusion.models.minimax_h3.vdn.serving import VdnServingContract
from vllm_omni.errors import OmniClientError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _resolve(root, **spec):
    spec.setdefault("checkpoint", str(root))
    return resolve_vdn_checkpoint(VdnSpec.from_mapping(spec))


def _arch(config=None, **runtime):
    return MiniMaxH3HybridAttentionConfig.from_transform_config(
        config if config is not None else TRANSFORM_CONFIG,
        attention_head_dim=HEAD_DIM,
        **runtime,
    )


def test_a_released_layout_resolves_to_its_branch_and_both_adapters(tmp_path):
    checkpoint = _resolve(write_release(tmp_path / "release"))

    assert checkpoint.branch_path.is_file()
    assert [adapter.name for adapter in checkpoint.adapters] == ["default", "turbo"]
    assert checkpoint.has_turbo
    # The DMD release is a distilled student, so it pins its ladder: nine
    # positions bounding the eight forwards it was trained for.
    assert len(checkpoint.base_schedule) == VDN_TURBO_DENOISE_STEPS + 1
    assert checkpoint.base_schedule[0] == 1.0
    assert checkpoint.base_schedule[-1] == 0.0


def test_a_stage_b_release_pins_no_schedule(tmp_path):
    """The 50-step checkpoint is not distilled, so it keeps the uniform ladder."""
    checkpoint = _resolve(write_release(tmp_path / "release", adapters=("default",)))

    assert not checkpoint.has_turbo
    assert checkpoint.base_schedule is None


def test_adapters_can_be_selected_and_a_missing_one_is_named(tmp_path):
    root = write_release(tmp_path / "release")

    assert [a.name for a in _resolve(root, adapters=["default"]).adapters] == ["default"]
    with pytest.raises(VdnCheckpointError, match=r"declares no adapter named \['fake'\]"):
        _resolve(root, adapters=["fake"])


def test_the_turbo_adapter_merges_its_adaln_targets_at_their_own_rank(tmp_path):
    """alpha tracks rank per module, so every pair still merges at exactly 1."""
    checkpoint = _resolve(write_release(tmp_path / "release"))
    turbo = next(adapter for adapter in checkpoint.adapters if adapter.name == "turbo")

    assert turbo.scale_for("transformer_blocks.0.attn.orig.to_q") == pytest.approx(1.0)
    assert turbo.scale_for("transformer_blocks.0.adaln_proj.linear") == pytest.approx(1.0)
    # ...and the declaration is what says so, rather than an assumption.
    assert turbo.rank_pattern["transformer_blocks.0.adaln_proj.linear"] != turbo.rank


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"kind": "train_state", "checkpoint_format_version": 2}, "not released weights"),
        ({"kind": "weights", "checkpoint_format_version": 1}, "checkpoint_format_version"),
        (
            {"kind": "weights", "checkpoint_format_version": 2, "metadata": {"truncated_blocks": 2}},
            "truncated smoke-test",
        ),
    ],
)
def test_an_unfinished_artifact_is_refused(metadata, message, tmp_path):
    root = write_release(tmp_path / "release", metadata=metadata)

    with pytest.raises(VdnCheckpointError, match=message):
        _resolve(root)


def test_a_release_without_the_hybrid_transform_is_refused(tmp_path):
    root = write_release(tmp_path / "release")
    spec = json.loads((root / "model_spec.json").read_text())
    spec["transforms"][0]["type"] = "something_else"
    (root / "model_spec.json").write_text(json.dumps(spec))

    with pytest.raises(VdnCheckpointError, match="exactly one 'hybrid_attention' transform"):
        _resolve(root)


def test_a_hub_id_needs_the_release_named(tmp_path):
    with pytest.raises(VdnCheckpointError, match="vdn.subdir must name the release"):
        resolve_vdn_checkpoint(VdnSpec.from_mapping({"checkpoint": "OpenVDN/vdn-minimax-h3"}))


def test_an_unknown_vdn_config_key_is_refused():
    with pytest.raises(VdnCheckpointError, match=r"unknown vdn config keys \['radius'\]"):
        VdnSpec.from_mapping({"checkpoint": "x", "radius": 2})


def test_the_released_architecture_is_read_exactly():
    config = _arch()

    assert (config.chunk, config.radius) == (5, 1)
    assert config.short_conv_targets == ("k", "v")
    assert config.linear_head_dim == HEAD_DIM
    # chunk 5 x (2*1 + 1) chunks: at or below 15 frames every frame sees every
    # chunk, so the window IS full attention and the branch must stay off.
    assert config.full_cover_frames == 15
    assert config.covers_all_frames(15)
    assert not config.covers_all_frames(16)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        ({"anchor_frames": "columns"}, "anchor_frames='columns'"),
        ({"enable_softmax_gate": False}, "enable_softmax_gate=false"),
        ({"linear_attention": {"delta_rule": "sana_scaled"}}, "delta_rule='sana_scaled'"),
        ({"linear_attention": {"bridge": "none"}}, "bridge='none'"),
        ({"linear_attention": {"a_fp32": False}}, "a_fp32=false"),
        ({"linear_attention": {"enable_text_state": False}}, "enable_text_state=false"),
        ({"linear_attention": {"short_conv": {"targets": ["q", "k", "v"]}}}, "short_conv.targets"),
        ({"linear_attention": {"linear_head_dim": HEAD_DIM * 2}}, "linear_head_dim"),
        ({"softmax_attention": {"chunk": 0}}, "chunk must be positive"),
        ({"linear_attention": {"unknown": 1}}, "unknown linear_attention keys"),
        ({"unknown": 1}, "unknown transform config keys"),
    ],
)
def test_an_architecture_this_server_never_validated_is_refused(mutate, message):
    """VDN also trains, so its spec admits values the release never used.

    Implementing them here would add branches that nothing exercises and
    nothing would notice breaking, so each is a refusal instead.
    """
    config = copy.deepcopy(TRANSFORM_CONFIG)
    for key, value in mutate.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key].update(value)
        else:
            config[key] = value

    with pytest.raises(VdnConfigError, match=message):
        _arch(config)


def test_a_missing_architecture_field_is_refused_rather_than_defaulted():
    config = copy.deepcopy(TRANSFORM_CONFIG)
    del config["linear_attention"]["bridge"]

    with pytest.raises(VdnConfigError, match="omits 'bridge'"):
        _arch(config)


def test_runtime_knobs_are_checked_but_never_reach_the_architecture():
    with pytest.raises(VdnConfigError, match="window_group_batch"):
        _arch(window_group_batch=0)
    with pytest.raises(VdnConfigError, match="window_impl='flex'"):
        _arch(window_impl="flex")


def _geometry(**overrides):
    fields = {
        "seq_len": 128,
        "used_len": 120,
        "text_start": 0,
        "text_len": 8,
        "video_start": 16,
        "num_frames": 26,
        "frame_height": 2,
        "frame_width": 2,
    }
    fields.update(overrides)
    return MiniMaxH3HybridGeometry(**fields)


def test_the_geometry_reports_the_rows_the_branch_needs():
    geometry = _geometry()

    assert geometry.tokens_per_frame == 4
    assert geometry.video_end == 120
    assert geometry.num_video_rows == 104
    assert geometry.frame_rows(0) == (16, 20)
    assert geometry.frame_rows(25) == (116, 120)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"text_len": 0}, "carries no text rows"),
        ({"num_frames": 25}, "last content block"),
        ({"used_len": 200}, "last content block"),
        ({"text_start": 4}, "packs text first"),
    ],
)
def test_a_layout_the_branch_cannot_serve_is_refused(overrides, message):
    with pytest.raises(VdnConfigError, match=message):
        _geometry(**overrides)


def _od_config(**overrides):
    parallel = SimpleNamespace(
        ring_degree=overrides.pop("ring_degree", 1),
        allgather_degree=overrides.pop("allgather_degree", 1),
        ulysses_mode=overrides.pop("ulysses_mode", "strict"),
        ulysses_degree=overrides.pop("ulysses_degree", 1),
    )
    fields = {
        "enable_cpu_offload": False,
        "enable_layerwise_offload": False,
        "enable_distributed_layerwise_offload": False,
    }
    fields.update(overrides)
    return SimpleNamespace(parallel_config=parallel, **fields)


def _contract(tmp_path):
    checkpoint = _resolve(write_release(tmp_path / "release"))
    return VdnServingContract(checkpoint, _arch())


def test_a_supported_server_layout_starts(tmp_path):
    _contract(tmp_path).check_serving_contract(partition="fl2va", od_config=_od_config(), lora_path=None)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"partition": "ref2va"}, "distills"),
        ({"partition": "combined"}, "--task-type t2va"),
        ({"lora_path": "some/adapter"}, "--lora-path"),
        ({"od_config": _od_config(enable_distributed_layerwise_offload=True)}, "cannot be combined with"),
        ({"od_config": _od_config(enable_cpu_offload=True)}, "cannot be combined with"),
        ({"od_config": _od_config(ring_degree=2)}, "ring_degree=2"),
        ({"od_config": _od_config(allgather_degree=2)}, "allgather_degree=2"),
        ({"od_config": _od_config(ulysses_mode="advanced_uaa")}, "ulysses_mode='strict'"),
    ],
)
def test_a_layout_that_would_route_around_the_branch_is_refused(kwargs, message, tmp_path):
    call = {"partition": "fl2va", "od_config": _od_config(), "lora_path": None}
    call.update(kwargs)

    with pytest.raises(ValueError, match=message):
        _contract(tmp_path).check_serving_contract(**call)


def _sampling(**overrides):
    fields = {"lora_request": None, "extra_args": {}, "num_inference_steps": VDN_TURBO_DENOISE_STEPS}
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_a_request_on_the_trained_shifts_is_accepted(tmp_path):
    contract = _contract(tmp_path)
    contract.check_task("t2va")
    contract.check_request(
        _sampling(extra_args={"flow_shift": VDN_VIDEO_SHIFT, "audio_flow_shift": VDN_AUDIO_SHIFT}),
        video_shift=VDN_VIDEO_SHIFT,
        audio_shift=VDN_AUDIO_SHIFT,
    )


@pytest.mark.parametrize("task", ["fl2va", "ref2va"])
def test_a_task_vdn_never_trained_is_refused(task, tmp_path):
    with pytest.raises(OmniClientError, match=r"serves \['t2va'\] only"):
        _contract(tmp_path).check_task(task)


@pytest.mark.parametrize(
    ("sampling", "message"),
    [
        (_sampling(lora_request=SimpleNamespace(lora_int_id=1)), "per-request lora is unavailable"),
        (_sampling(extra_args={"flow_shift": 6.0}), "requires flow_shift=12"),
        (_sampling(extra_args={"audio_flow_shift": 1.0}), "requires audio_flow_shift=3"),
    ],
)
def test_a_request_that_would_sample_off_the_trained_rungs_is_refused(sampling, message, tmp_path):
    with pytest.raises(OmniClientError, match=message):
        _contract(tmp_path).check_request(
            sampling,
            video_shift=VDN_VIDEO_SHIFT,
            audio_shift=VDN_AUDIO_SHIFT,
        )
