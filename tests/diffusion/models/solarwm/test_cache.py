# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.attention import layer
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend, SDPAImpl
from vllm_omni.diffusion.models.solarwm.transformer import SolarWMCache, SolarWMTransformer

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@torch.no_grad()
def test_request_cache_isolation_commit_and_eviction(monkeypatch):
    monkeypatch.setattr(layer, "get_attn_backend_for_role", lambda **kwargs: (SDPABackend, None))
    monkeypatch.setattr(SDPAImpl, "forward", SDPAImpl.forward_cuda)
    torch.manual_seed(17)
    model = SolarWMTransformer(
        dim=16,
        ffn_dim=24,
        num_heads=2,
        num_layers=2,
        in_dim=4,
        out_dim=4,
        freq_dim=8,
        text_dim=8,
        text_len=8,
        max_history_frames=3,
    ).eval()
    for param in model.parameters():
        torch.nn.init.normal_(param, std=0.05)
    x = torch.randn(1, 4, 3, 4, 4)
    t = torch.ones(1, 3)
    text = torch.randn(1, 8, 8)
    views = torch.eye(4).repeat(1, 3, 1, 1)
    ks = torch.eye(3).repeat(1, 3, 1, 1)
    cache = SolarWMCache()
    args = (x, t, text, views, ks)
    baseline = model(*args, cache=cache, start_frame=0)
    assert cache.next_frame == 0 and not cache.keys
    other = SolarWMCache()
    model(x * 2, t, text + 1, views, ks, cache=other, start_frame=0, commit=True)
    torch.testing.assert_close(model(*args, cache=cache, start_frame=0), baseline)
    for start in (0, 3, 6):
        model(*args, cache=cache, start_frame=start, commit=True)
    assert cache.next_frame == 9 and other.next_frame == 3
    assert cache.viewmats.shape[1] == 6  # one clean history chunk + current chunk
    assert all(k.shape[1] == 24 for k in cache.keys.values())
    assert all(k.shape[1] == 12 for k in other.keys.values())
    with pytest.raises(ValueError, match="Out-of-order"):
        model(*args, cache=cache, start_frame=0)
