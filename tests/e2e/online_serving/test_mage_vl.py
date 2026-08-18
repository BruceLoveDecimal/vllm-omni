# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""End-to-end online serving for Mage-VL image and frame-sampled video understanding.

Equivalent to running:
    vllm-omni serve "microsoft/Mage-VL" --omni \\
        --deploy-config vllm_omni/deploy/mage_vl.yaml --port 8077

then POSTing an image to ``/v1/chat/completions``.

Scope note: these assert serving behaviour (request succeeds, text comes back, greedy is
reproducible, concurrent requests do not cross-talk). Agreement with the HuggingFace
reference is a separate, stronger check that needs the reference model on the same box --
see ``tests/model_executor/models/mage_vl/parity/``. Images are passed as ``data:`` URIs
because CI has no network; the remote-URL form goes through the shared ``MediaConnector``
(with ``allowed_media_domains``) and is not model-specific.
"""

import base64
import os
from io import BytesIO

import pytest
from vllm.assets.image import ImageAsset

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "microsoft/Mage-VL"
STAGE_CONFIGS_PATH = get_deploy_config_path("ci/mage_vl.yaml")

test_params = [
    OmniServerParams(
        model=MODEL,
        stage_config_path=STAGE_CONFIGS_PATH,
        stage_init_timeout=300,
    ),
]


def _image_data_uri(name: str = "stop_sign") -> str:
    image = ImageAsset(name).pil_image.convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def _messages(prompt: str, image_uris: list[str]) -> list[dict]:
    content: list[dict] = [{"type": "image_url", "image_url": {"url": uri}} for uri in image_uris]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _greedy_request(messages: list[dict], model: str, **overrides) -> dict:
    request = {
        "model": model,
        "messages": messages,
        "modalities": ["text"],
        # Sampling params must ride in ``extra_body``: ``send_omni_request`` forwards only
        # a fixed set of top-level keys, so a top-level ``temperature`` is dropped without
        # a word and the server samples with the checkpoint's generation_config defaults --
        # which makes every "is the output stable" assertion below meaningless.
        "extra_body": {"temperature": 0.0, "max_tokens": 32},
    }
    request.update(overrides)
    return request


@pytest.mark.core_model
@pytest.mark.advanced_model
@hardware_test(res={"cuda": "L4"})
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_mage_vl_single_image(omni_server, openai_client) -> None:
    """The baseline case: one image in, non-empty text out."""
    request = _greedy_request(
        _messages("What is in this image?", [_image_data_uri()]),
        omni_server.model,
    )

    responses = openai_client.send_omni_request(request)

    assert responses, "no response returned"
    assert responses[0].text_content.strip(), "model returned empty text for an image prompt"


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "L4"})
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_mage_vl_greedy_is_reproducible(omni_server, openai_client) -> None:
    """Same image and prompt at temperature 0 must give the same text twice.

    This is what makes the parity drivers' token-exact comparison meaningful: if serving
    were non-deterministic at greedy, a matching run would prove nothing.
    """
    request = _greedy_request(
        _messages("What is in this image?", [_image_data_uri()]),
        omni_server.model,
    )

    first = openai_client.send_omni_request(request)[0].text_content
    second = openai_client.send_omni_request(request)[0].text_content

    assert first == second, f"greedy output drifted between runs: {first!r} vs {second!r}"


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "L4"})
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_mage_vl_multi_image_video_frames(omni_server, openai_client) -> None:
    """Video reaches this model as frames in the image tensors, one block per frame.

    Four distinct assets stand in for sampled frames -- the same shape the reference
    pipeline produces online, and the path that exercises per-image slicing of
    ``pixel_values`` / ``patch_positions``.
    """
    frames = [
        _image_data_uri("stop_sign"),
        _image_data_uri("cherry_blossom"),
        _image_data_uri("stop_sign"),
        _image_data_uri("cherry_blossom"),
    ]
    request = _greedy_request(
        _messages("Describe what happens across these frames.", frames),
        omni_server.model,
    )

    responses = openai_client.send_omni_request(request)

    assert responses, "no response returned"
    assert responses[0].text_content.strip(), "model returned empty text for a multi-frame prompt"


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "L4"})
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_mage_vl_concurrent_requests_do_not_cross_talk(omni_server, openai_client) -> None:
    """Four identical greedy requests in flight must all return, and all agree.

    Batched multimodal requests share the encoder cache; a slicing bug there shows up as
    one request receiving another's visual features, which at greedy means divergent text.
    """
    request = _greedy_request(
        _messages("What is in this image?", [_image_data_uri()]),
        omni_server.model,
    )

    responses = openai_client.send_omni_request(request, request_num=4, max_concurrency=4)

    assert len(responses) == 4, f"expected 4 responses, got {len(responses)}"
    texts = {response.text_content for response in responses}
    assert len(texts) == 1, f"concurrent requests disagreed, so features crossed: {texts}"
