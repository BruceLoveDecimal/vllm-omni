# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stepwise generation + session-owned streaming VAE decode, end to end.

Integration of the two RFC #6672 tracks: B1's single-request stepwise path
(one ``AsyncOmni.generate()`` streaming one latent chunk per AR block) feeds
B3's session-owned streaming decoder (``WanStreamingDecoder`` +
``StreamingDecodeState``), so one request streams *pixels* chunk by chunk on
the standard ``generate()`` surface — no tick wrapper, no extra entrypoint.

The decode oracle follows the streaming-decode PR's own control: the same
concatenated latents pushed through the same streaming path in a single call.
That isolates cross-chunk cache continuity from unrelated kernel-selection
effects (see ``streaming_decode.py``'s module docstring).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import torch

from tests.e2e.offline_inference.test_lingbot_world_v2_stepwise import (
    _FRAMES_PER_BLOCK,
    _IMAGE_PATH,
    _NUM_CHUNKS,
    _NUM_FRAMES,
    _TEMPORAL_COMPRESSION,
    MODEL,
    _stream_chunks,
)
from tests.helpers.mark import hardware_test

pytestmark = [
    pytest.mark.slow,
    pytest.mark.diffusion,
    pytest.mark.skipif(
        _IMAGE_PATH is None,
        reason="VLLM_OMNI_LINGBOT_WORLD_V2_IMAGE_PATH is required",
    ),
]


def _build_streaming_decoder(device: torch.device):
    from diffusers import AutoencoderKLWan

    from vllm_omni.experimental.ar_diffusion.streaming_decode import WanStreamingDecoder

    vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32)
    vae = vae.to(device).eval()
    return vae, WanStreamingDecoder(vae)


def _to_vae_latents(latent: torch.Tensor, vae, device: torch.device) -> torch.Tensor:
    """Invert the checkpoint's latent normalization, as the pipeline does."""
    shape = (1, -1, 1, 1, 1)
    mean = torch.as_tensor(vae.config.latents_mean, device=device, dtype=torch.float32).view(*shape)
    std = torch.as_tensor(vae.config.latents_std, device=device, dtype=torch.float32).view(*shape)
    return (latent.to(device=device, dtype=torch.float32) * std + mean).to(vae.dtype)


@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_stepwise_request_streams_pixels_through_session_decoder() -> None:
    """One generate() call -> per-chunk pixel frames with temporal continuity."""

    assert _IMAGE_PATH is not None
    chunks = asyncio.run(_stream_chunks(Path(_IMAGE_PATH).expanduser().resolve()))
    assert len(chunks) == _NUM_CHUNKS

    device = torch.device("cuda")
    vae, decoder = _build_streaming_decoder(device)

    request_id = chunks[0][1]["request_id"]
    state = decoder.new_decode_state(request_id)
    decoded: list[torch.Tensor] = []
    state_bytes: list[int] = []
    with torch.inference_mode():
        for latent, _, _ in chunks:
            decoded.append(decoder.decode_chunk(_to_vae_latents(latent, vae, device), state))
            state_bytes.append(state.nbytes())
    streamed = torch.cat(decoded, dim=2)

    # A causal decoder expands a session's opening latent frame to one raw
    # frame and every later one to the full temporal factor.
    assert decoded[0].shape[2] == 1 + (_FRAMES_PER_BLOCK - 1) * _TEMPORAL_COMPRESSION
    for chunk in decoded[1:]:
        assert chunk.shape[2] == _FRAMES_PER_BLOCK * _TEMPORAL_COMPRESSION
    assert streamed.shape[2] == _NUM_FRAMES
    assert torch.isfinite(streamed).all()
    assert float(streamed.abs().max()) <= 1.0

    # Resident decoder state is bounded by construction, not by session length.
    assert state_bytes[1] == state_bytes[-1]

    # Continuity control from the streaming-decode PR: chunked decode must be
    # bit-identical to the same session decoded in one call on the same path.
    whole = torch.cat([latent for latent, _, _ in chunks], dim=2)
    oracle_state = decoder.new_decode_state(request_id + "-oracle")
    with torch.inference_mode():
        oracle = decoder.decode_chunk(_to_vae_latents(whole, vae, device), oracle_state)
    assert torch.equal(streamed, oracle)
