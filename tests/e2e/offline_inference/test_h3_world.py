# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in real-weight H3-World smoke: adapter binding and changed controls."""

import os

import numpy as np
import pytest
from PIL import Image, ImageOps

from tests.helpers.mark import hardware_marks

_MODEL = os.environ.get("VLLM_TEST_MINIMAX_H3_FL2VA_MODEL")
_LORA = os.environ.get("VLLM_TEST_H3_WORLD_LORA")
_FRAME = os.environ.get("VLLM_TEST_H3_WORLD_FIRST_FRAME")
pytestmark = [
    pytest.mark.local_model,
    pytest.mark.diffusion,
    pytest.mark.cuda,
    *hardware_marks(res={"cuda": "H100"}, num_cards=2),
]


@pytest.mark.skipif(
    not (_MODEL and _LORA and _FRAME), reason="set H3 base, World LoRA and first-frame environment paths"
)
def test_h3_world_changed_controls_on_one_engine():
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.lora.request import LoRARequest
    from vllm_omni.lora.utils import stable_lora_int_id

    with Image.open(_FRAME) as source:
        image = ImageOps.fit(source.convert("RGB"), (832, 480), method=Image.Resampling.LANCZOS)
    engine = Omni(
        model=_MODEL,
        task_type="fl2va",
        num_gpus=2,
        tensor_parallel_size=2,
        text_encoder_tp_size=2,
        vae_patch_parallel_size=2,
        usp=1,
        ring=1,
        diffusion_attention_backend="FLASH_ATTN",
        enforce_eager=True,
        cache_backend="none",
        lora_path=_LORA,
        vae_parallel_mode="tile",
        vae_use_tiling=True,
    )
    videos = []
    try:
        for keys in (["W"], []):
            outputs = engine.generate(
                {
                    "prompt": "A man in a yellow floral shirt stands in a dim, multi-level concrete parking garage.",
                    "multi_modal_data": {"image": image},
                },
                OmniDiffusionSamplingParams(
                    height=480,
                    width=832,
                    num_frames=124,
                    fps=24,
                    seed=2,
                    num_inference_steps=50,
                    guidance_scale=1.0,
                    quality="lossless",
                    lora_request=LoRARequest("h3-world", stable_lora_int_id(_LORA), _LORA),
                    extra_args={"task": "fl2va", "h3_world_actions": {"keys": keys}},
                ),
                use_tqdm=False,
            )
            assert len(outputs) == 1
            video = np.asarray(outputs[0].images[0])
            assert video.shape == (124, 480, 832, 3)
            assert video.dtype == np.uint8 and video.std() > 1
            assert outputs[0].multimodal_output["audio_sample_rate"] == 32000
            videos.append(video)
    finally:
        engine.close()
    # Confirms the request consumes actions; visual control quality still
    # needs a paired comparison to official output and human assessment.
    assert not np.array_equal(videos[0], videos[1])
