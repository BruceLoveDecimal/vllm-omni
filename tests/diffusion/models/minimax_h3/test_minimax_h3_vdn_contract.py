# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Where the hybrid meets the rest of the MiniMax-H3 serving path."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from tests.diffusion.models.minimax_h3.vdn_release import HEAD_DIM, TRANSFORM_CONFIG, write_release
from vllm_omni.diffusion.models.minimax_h3.denoise_loop import MiniMaxH3DenoiseBranch
from vllm_omni.diffusion.models.minimax_h3.packed_sequence import minimax_h3_packed_sequence
from vllm_omni.diffusion.models.minimax_h3.vdn.checkpoint import (
    VDN_TURBO_DENOISE_STEPS,
    VdnSpec,
    resolve_vdn_checkpoint,
)
from vllm_omni.diffusion.models.minimax_h3.vdn.config import MiniMaxH3HybridAttentionConfig
from vllm_omni.diffusion.models.minimax_h3.vdn.serving import VdnServingContract
from vllm_omni.errors import OmniClientError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

TEXT_LEN = 4
LATENT_T = 6
LATENT_H = 4
LATENT_W = 4
AUDIO_T = 2


def _config() -> MiniMaxH3HybridAttentionConfig:
    return MiniMaxH3HybridAttentionConfig.from_transform_config(TRANSFORM_CONFIG, attention_head_dim=HEAD_DIM)


def _branch(hybrid: bool):
    packed = minimax_h3_packed_sequence(
        text_len=TEXT_LEN,
        latent_t=LATENT_T,
        latent_h=LATENT_H,
        latent_w=LATENT_W,
        audio_t=AUDIO_T,
        include_keyframe_cond=False,
    )
    tags = packed["token_tags"].clone()
    return MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=torch.zeros(TEXT_LEN, 5120),
        token_tags=tags,
        device=torch.device("cpu"),
        hybrid_config=_config() if hybrid else None,
    )


def test_the_denoise_branch_publishes_the_geometry_beside_the_packed_metadata():
    """It travels in ``packed_seq_params``, which every forward already reads."""
    branch = _branch(hybrid=True)

    geometry = branch.static_kwargs["packed_seq_params"]["hybrid_geometry"]

    assert geometry.text_len == TEXT_LEN
    assert geometry.text_start == 0
    # [text | audio | video | pad]: the audio rows are channel-major, two per
    # latent step, and the target video is the last content block.
    assert geometry.video_start == TEXT_LEN + AUDIO_T * 2
    assert geometry.num_frames == LATENT_T
    assert geometry.frame_size == (LATENT_H // 2, LATENT_W // 2)
    assert geometry.video_end == geometry.used_len
    assert geometry.used_len < geometry.seq_len, "this layout should carry alignment padding"


def test_a_dense_checkpoint_publishes_no_geometry_at_all():
    """The dense path must keep the request contract it has always had."""
    branch = _branch(hybrid=False)

    assert "hybrid_geometry" not in branch.static_kwargs["packed_seq_params"]


def test_the_geometry_is_plain_ints_so_no_layer_syncs_on_it():
    geometry = _branch(hybrid=True).static_kwargs["packed_seq_params"]["hybrid_geometry"]

    for field in ("seq_len", "used_len", "text_len", "video_start", "num_frames"):
        assert isinstance(getattr(geometry, field), int)


def test_the_forward_keyword_contract_is_unchanged():
    """The geometry rides inside packed_seq_params rather than widening it.

    ``MiniMaxH3DiTModel.forward`` refuses any kwarg it does not consume, and
    TeaCache mirrors that set, so a new top-level keyword would have to be
    added in two places to stay consistent. It is not a new keyword.
    """
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import _FORWARD_SUPPORTED_KWARGS

    assert "hybrid_geometry" not in _FORWARD_SUPPORTED_KWARGS


def test_the_teacache_extractor_forwards_the_geometry():
    """A mirror that drops it would cache a dense residual for a hybrid model.

    The extractor reimplements the forward rather than calling it, so this is
    checked where the two are meant to agree: the block call itself.
    """
    from vllm_omni.diffusion.cache.teacache.extractors import extract_minimax_h3_context

    source = inspect.getsource(extract_minimax_h3_context)

    assert 'hybrid_geometry = module._psp_optional(psp, "hybrid_geometry", None)' in source
    assert "hybrid_geometry=hybrid_geometry" in source


def _pipeline(tmp_path, **overrides):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    checkpoint = resolve_vdn_checkpoint(VdnSpec.from_mapping({"checkpoint": str(write_release(tmp_path / "release"))}))
    pipeline._fasth3 = None
    pipeline._vdn = VdnServingContract(checkpoint, _config())
    pipeline._vdn_fusion = None
    pipeline.partition = "fl2va"
    pipeline.supported_tasks = frozenset({"t2va", "fl2va"})
    pipeline.default_video_shift = 12.0
    pipeline.default_audio_shift = 3.0
    for name, value in overrides.items():
        setattr(pipeline, name, value)
    return pipeline


def _sampling(**overrides):
    fields = {"num_inference_steps": None, "lora_request": None, "lora_scale": 0.0, "extra_args": {}}
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_the_distilled_ladder_is_the_checkpoint_s_own(tmp_path):
    """Eight forwards, so nine sigma positions - not the uniform grid a count
    would derive, and not the seven a nine-point uniform ladder would give."""
    pipeline = _pipeline(tmp_path)

    positions, num_steps = pipeline._resolve_sigma_positions("t2va", _sampling())

    assert num_steps == VDN_TURBO_DENOISE_STEPS
    assert len(positions) == VDN_TURBO_DENOISE_STEPS + 1
    assert positions[0] == 1.0 and positions[-1] == 0.0
    assert pipeline._resolve_sigma_positions("t2va", _sampling(num_inference_steps=8))[1] == 8


@pytest.mark.parametrize("requested", [7, 9, 50])
def test_a_request_off_the_distilled_ladder_is_refused(requested, tmp_path):
    pipeline = _pipeline(tmp_path)

    with pytest.raises(OmniClientError, match="8-step student"):
        pipeline._resolve_sigma_positions("t2va", _sampling(num_inference_steps=requested))


def test_a_task_the_checkpoint_never_trained_is_refused(tmp_path):
    pipeline = _pipeline(tmp_path)

    assert pipeline._resolve_task("t2va", {}) == "t2va"
    with pytest.raises(OmniClientError, match=r"serves \['t2va'\] only"):
        pipeline._resolve_task("fl2va", {})


def test_requests_do_not_share_a_forward_under_the_hybrid():
    """The window plan and the frame scans are geometry over ONE sequence."""
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    hybrid = SimpleNamespace(hybrid_config=_config(), modules=lambda: iter(()))

    assert MiniMaxH3Pipeline._packed_batch_supported(hybrid) is False


def test_the_hybrid_attention_is_absent_from_a_dense_block():
    """A server without a VDN checkpoint must build the parameter set it always did."""
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3Attention

    signature = inspect.signature(MiniMaxH3Attention.__init__)

    assert signature.parameters["hybrid_config"].default is None
