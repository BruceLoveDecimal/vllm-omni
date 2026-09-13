# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn.functional as F

from tests.diffusion.models.solarwm.window_attention import SolarWMWindowAttention
from vllm_omni.diffusion.attention import layer
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend, SDPAImpl
from vllm_omni.diffusion.models.solarwm.camera import (
    apply_projective_transform,
    camera_projection,
    transform_relative_viewmats,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def attention(monkeypatch):
    # Select the real shared SDPA backend on CPU; no attention math is mocked.
    monkeypatch.setattr(layer, "get_attn_backend_for_role", lambda **kwargs: (SDPABackend, None))
    monkeypatch.setattr(SDPAImpl, "forward", SDPAImpl.forward_cuda)
    return SolarWMWindowAttention(2, 8)


@pytest.mark.parametrize("chunk", [None, 4, 5])
@pytest.mark.parametrize("transform", ["linear", "logd4"])
def test_projective_attention_matches_dense_oracle(attention, chunk, transform):
    generator = torch.Generator().manual_seed(42)
    q, k, v = [torch.randn(2, 12, 2, 8, generator=generator) for _ in range(3)]
    cameras = torch.eye(4).repeat(2, 3, 1, 1)
    cameras[..., :3, 3] = torch.randn(2, 3, 3, generator=generator)
    cameras[:, 1, :2, :2] = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    intrinsics = torch.eye(3).repeat(2, 3, 1, 1)
    intrinsics[..., 0, 0] = 1.7
    intrinsics[..., 1, 1] = 0.8
    intrinsics[..., :2, 2] = 0.5  # ignored by the released PRoPE convention

    # Independent dense homogeneous-matrix oracle: per-token matrices and a
    # full visibility mask, instead of the production tiled/sliced path.
    projection = cameras.clone()
    if transform == "logd4":
        translation = projection[..., :3, 3]
        norm = translation.norm(dim=-1, keepdim=True)
        projection[..., :3, 3] = translation * torch.log1p(norm) / (4 * norm)
    diagonal = torch.diag(torch.tensor([1.7, 0.8, 1.0, 1.0]))
    projection = (diagonal @ projection).repeat_interleave(4, dim=1)
    inverse = torch.linalg.inv(projection)

    def apply(x, matrix):
        return (matrix[:, :, None, None] @ x.reshape(2, 12, 2, 2, 4, 1)).reshape_as(x)

    tq = apply(q, projection.transpose(-1, -2)).transpose(1, 2)
    tk = apply(k, inverse).transpose(1, 2)
    tv = apply(v, inverse).transpose(1, 2)
    tokens = torch.arange(12)
    mask = None if chunk is None else tokens[:, None] // chunk >= tokens[None, :] // chunk
    expected = apply(F.scaled_dot_product_attention(tq, tk, tv, attn_mask=mask).transpose(1, 2), projection)
    actual = attention(
        q,
        k,
        v,
        viewmats=cameras,
        intrinsics=intrinsics,
        tokens_per_chunk=chunk,
        translation_transform=transform,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_camera_layout_and_zero_translation():
    cameras = torch.eye(4).repeat(1, 3, 1, 1)
    before = cameras.clone()
    compressed = transform_relative_viewmats(cameras, "logd4")
    torch.testing.assert_close(compressed, cameras)
    torch.testing.assert_close(cameras, before)
    projection, inverse = camera_projection(cameras, None)
    features = torch.arange(96, dtype=torch.float32).reshape(1, 6, 2, 8)
    torch.testing.assert_close(apply_projective_transform(features, projection), features)
    torch.testing.assert_close(projection @ inverse, cameras)
    torch.testing.assert_close(apply_projective_transform(features, projection.repeat_interleave(2, dim=1)), features)
    with pytest.raises(ValueError, match="multiple"):
        apply_projective_transform(features[:, :5], projection)
    with pytest.raises(ValueError, match="linear or logd4"):
        transform_relative_viewmats(cameras, "unknown")


def test_chunk_visibility_does_not_leak_future(attention):
    generator = torch.Generator().manual_seed(7)
    q, k, v = [torch.randn(1, 6, 2, 8, generator=generator) for _ in range(3)]
    kwargs = dict(
        viewmats=torch.eye(4).repeat(1, 6, 1, 1),
        intrinsics=torch.eye(3).repeat(1, 6, 1, 1),
        tokens_per_chunk=3,
    )
    baseline = attention(q, k, v, **kwargs)
    changed = v.clone()
    changed[:, 3:] += 100
    result = attention(q, k, changed, **kwargs)
    torch.testing.assert_close(result[:, :3], baseline[:, :3])
    assert not torch.allclose(result[:, 3:], baseline[:, 3:])
    with pytest.raises(ValueError, match="positive"):
        attention(q, k, v, **(kwargs | {"tokens_per_chunk": 0}))
