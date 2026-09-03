# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Where the hybrid meets the rest of the MiniMax-H3 serving path.

The seam is deliberately one field wide: the branch reads its geometry off the
``VideoTokenLayout`` every attention layer already receives, so no forward
signature, no packed-metadata key and no cache mirror had to learn about it.
These tests hold that seam in place.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from tests.diffusion.models.minimax_h3.vdn_release import HEAD_DIM, TRANSFORM_CONFIG, write_release
from vllm_omni.diffusion.models.minimax_h3.denoise_loop import MiniMaxH3DenoiseBranch
from vllm_omni.diffusion.models.minimax_h3.packed_sequence import minimax_h3_packed_sequence
from vllm_omni.diffusion.models.minimax_h3.vdn.checkpoint import VDN_TURBO_DENOISE_STEPS
from vllm_omni.diffusion.models.minimax_h3.vdn.config import (
    MiniMaxH3HybridAttentionConfig,
    MiniMaxH3HybridGeometry,
    VdnConfigError,
)
from vllm_omni.diffusion.models.minimax_h3.vdn.serving import resolve_vdn_serving
from vllm_omni.errors import OmniClientError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

TEXT_LEN = 4
LATENT_T = 6
LATENT_H = 4
LATENT_W = 4
AUDIO_T = 2


def _config() -> MiniMaxH3HybridAttentionConfig:
    return MiniMaxH3HybridAttentionConfig.from_transform_config(TRANSFORM_CONFIG, attention_head_dim=HEAD_DIM)


def _branch():
    packed = minimax_h3_packed_sequence(
        text_len=TEXT_LEN,
        latent_t=LATENT_T,
        latent_h=LATENT_H,
        latent_w=LATENT_W,
        audio_t=AUDIO_T,
        include_keyframe_cond=False,
    )
    branch = MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=torch.zeros(TEXT_LEN, 5120),
        token_tags=packed["token_tags"].clone(),
        device=torch.device("cpu"),
    )
    return branch, packed


def test_the_packed_layout_states_where_the_prompt_is():
    """The one thing the shared layout did not already say.

    "The prompt" and "everything the softmax keeps dense" are different row
    sets - the latter also holds the soundtrack - and the branch is seeded from
    the former.
    """
    branch, _ = _branch()

    layout = branch.static_kwargs["video_token_layout"]

    assert layout.text_len == TEXT_LEN


def test_the_branch_reads_its_geometry_off_that_layout():
    """No new metadata travels: the geometry is derived from what is there."""
    branch, packed = _branch()
    layout = branch.static_kwargs["video_token_layout"]

    geometry = MiniMaxH3HybridGeometry.from_video_layout(layout, packed_total=branch.seq_len)

    assert geometry.text_start == 0
    assert geometry.text_len == TEXT_LEN
    # [text | audio | video | pad]: audio rows are channel-major, two per step.
    assert geometry.video_start == int(packed["video_row_start"])
    assert geometry.video_start == TEXT_LEN + AUDIO_T * 2
    assert geometry.num_frames == LATENT_T
    assert geometry.frame_size == (LATENT_H // 2, LATENT_W // 2)
    assert geometry.video_end == geometry.used_len == branch.used_len
    assert geometry.seq_len == branch.seq_len
    assert geometry.used_len < geometry.seq_len, "this layout should carry alignment padding"


def test_the_geometry_is_plain_ints_so_no_layer_syncs_on_it():
    branch, _ = _branch()
    layout = branch.static_kwargs["video_token_layout"]

    geometry = MiniMaxH3HybridGeometry.from_video_layout(layout, packed_total=branch.seq_len)

    for field in ("seq_len", "used_len", "text_len", "video_start", "num_frames"):
        assert isinstance(getattr(geometry, field), int)


def test_a_layout_that_does_not_say_where_the_prompt_is_refused():
    """Every other model leaves the field unset, and none of them come here."""
    from vllm_omni.diffusion.attention.backends.abstract import VideoTokenLayout

    layout = VideoTokenLayout(prefix_len=8, latent_grid=(6, 2, 2))

    with pytest.raises(VdnConfigError, match="which rows are the prompt"):
        MiniMaxH3HybridGeometry.from_video_layout(layout, packed_total=64)
    with pytest.raises(VdnConfigError, match="without one"):
        MiniMaxH3HybridGeometry.from_video_layout(None, packed_total=64)


def test_the_one_tail_layout_is_readable_too():
    """The older ``prefix_len``/``latent_grid`` shape states the same geometry."""
    from vllm_omni.diffusion.attention.backends.abstract import VideoTokenLayout

    layout = VideoTokenLayout(prefix_len=8, latent_grid=(6, 2, 2), text_len=4)

    geometry = MiniMaxH3HybridGeometry.from_video_layout(layout, packed_total=64)

    assert (geometry.video_start, geometry.num_frames, geometry.frame_size) == (8, 6, (2, 2))
    assert geometry.used_len == 8 + 6 * 4


def test_the_forward_keyword_contract_is_unchanged():
    """The hybrid added no keyword and no packed-metadata key.

    ``MiniMaxH3DiTModel.forward`` refuses any kwarg it does not consume and
    TeaCache mirrors that set, so a new one would have to be added in two
    places to stay consistent. There is not one.
    """
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import _FORWARD_SUPPORTED_KWARGS

    branch, _ = _branch()

    assert "hybrid_geometry" not in _FORWARD_SUPPORTED_KWARGS
    assert set(branch.static_kwargs["packed_seq_params"]) == {
        "cu_seqlens_q",
        "max_seqlen_q",
        "num_requests",
        "vsa_prefix_segments",
    }


def test_the_teacache_mirror_already_carries_what_the_branch_needs():
    """The extractor reimplements the forward rather than calling it.

    It passed ``video_layout`` before the hybrid existed and still does, which
    is why a VDN checkpoint caches the residual its own attention produced
    rather than a dense one - and why this feature changed nothing there.
    """
    from vllm_omni.diffusion.cache.teacache.extractors import extract_minimax_h3_context

    source = inspect.getsource(extract_minimax_h3_context)

    assert "video_layout=video_layout" in source


def _od_config(model_config):
    return SimpleNamespace(
        model_config=model_config,
        tf_model_config={"attention_head_dim": HEAD_DIM, "num_attention_heads": 2, "hidden_size": 8},
    )


def test_the_server_flag_claims_a_checkpoint(tmp_path):
    root = write_release(tmp_path / "release")

    contract = resolve_vdn_serving(_od_config({"vdn": {"checkpoint": str(root)}}), {}, tmp_path)

    assert contract is not None
    assert contract.checkpoint.has_turbo
    assert (contract.config.chunk, contract.config.radius) == (5, 1)
    assert contract.config.linear_head_dim == HEAD_DIM


def test_a_packaged_release_may_declare_its_own_hybrid(tmp_path):
    """``vllm-omni serve <dir>`` then needs no flags, and the path is its own."""
    write_release(tmp_path / "vdn")

    assert resolve_vdn_serving(_od_config({}), {"vdn": {"checkpoint": "vdn"}}, tmp_path) is not None


def test_an_ordinary_h3_checkpoint_claims_nothing(tmp_path):
    """The dense path must not so much as look at a VDN file."""
    assert resolve_vdn_serving(_od_config({}), {}, tmp_path) is None
    assert resolve_vdn_serving(SimpleNamespace(), {}, tmp_path) is None


def _pipeline(tmp_path):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline._fasth3 = None
    pipeline._vdn = resolve_vdn_serving(
        _od_config({"vdn": {"checkpoint": str(write_release(tmp_path / "release"))}}), {}, tmp_path
    )
    pipeline._vdn_fusion = None
    pipeline.partition = "fl2va"
    pipeline.supported_tasks = frozenset({"t2va", "fl2va"})
    pipeline.default_video_shift = 12.0
    pipeline.default_audio_shift = 3.0
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


def test_a_dense_server_never_imports_the_hybrid():
    """``hybrid_config=None`` is the H3 this class has always been.

    The import lives inside the constructor branch, so a server without a VDN
    checkpoint loads none of the package - which is what makes the feature
    something a checkpoint switches on rather than something every H3 carries.
    """
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer

    source = inspect.getsource(minimax_h3_transformer)
    signature = inspect.signature(minimax_h3_transformer.MiniMaxH3Attention.__init__)

    assert signature.parameters["hybrid_config"].default is None
    module_level = source.split("class MiniMaxH3Rope")[0]
    assert "from .vdn.hybrid import" not in module_level
