# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.mage_flow import pipeline_mage_flow
from vllm_omni.diffusion.models.mage_flow.content_policy import FilterVerdict
from vllm_omni.diffusion.models.mage_flow.pipeline_mage_flow import (
    MageFlowPipeline,
    _MageFlowBatchItem,
    _resolve_mage_flow_output_type,
    get_mage_flow_post_process_func,
    get_mage_flow_pre_process_func,
)
from vllm_omni.errors import GuardrailViolationError, OmniClientError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("request_output_type", "config_output_type", "expected"),
    [
        ("tensor", "latent", "tensor"),
        (None, "latent", "latent"),
        (None, None, "pil"),
        ("numpy", "pil", None),
    ],
)
def test_output_type_resolution(
    request_output_type: str | None,
    config_output_type: str | None,
    expected: str | None,
) -> None:
    sampling = SimpleNamespace(output_type=request_output_type)
    config = SimpleNamespace(output_type=config_output_type)

    if expected is None:
        with pytest.raises(OmniClientError, match="output_type"):
            _resolve_mage_flow_output_type(sampling, config)
    else:
        assert _resolve_mage_flow_output_type(sampling, config) == expected


def test_postprocess_uses_configured_latent_default() -> None:
    postprocess = get_mage_flow_post_process_func(SimpleNamespace(output_type="latent"))
    latents = torch.randn(1, 4, 2, 2)

    assert (
        postprocess(
            latents,
            sampling_params=SimpleNamespace(output_type=None),
        )
        is latents
    )


def test_pad_token_sequences_returns_padding_mask_and_lengths() -> None:
    short = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    long = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)

    padded, mask, lengths = MageFlowPipeline._pad_token_sequences([short, long])

    assert padded.shape == (2, 4, 3)
    assert lengths == [2, 4]
    torch.testing.assert_close(padded[0, :2], short[0])
    assert not padded[0, 2:].count_nonzero()
    assert mask.tolist() == [
        [True, True, False, False],
        [True, True, True, True],
    ]


def test_checkpoint_metadata_reads_transformer_config_at_requested_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = object.__new__(MageFlowPipeline)
    pipeline.od_config = SimpleNamespace(
        model="microsoft/Mage-Flow-Turbo",
        revision="test-revision",
    )
    documents = {
        "model_index.json": {
            "_class_name": "MageFlowPipeline",
            "_text_encoder_path": "text_encoder",
            "_vae_source": "vae/diffusion_pytorch_model.safetensors",
        },
        "vae/config.json": {
            "_class_name": "MageVAE",
            "latent_channels": 128,
            "downsample_factor": 16,
            "sample_posterior": False,
        },
        "transformer/config.json": {
            "_class_name": "MageFlowTransformer2DModel",
            "hidden_size": 3072,
        },
    }
    calls = []

    def fake_get_hf_file_to_dict(filename, model, revision):
        calls.append((filename, model, revision))
        return documents[filename]

    monkeypatch.setattr(
        pipeline_mage_flow,
        "get_hf_file_to_dict",
        fake_get_hf_file_to_dict,
    )
    monkeypatch.setattr(pipeline_mage_flow, "file_exists", lambda *_args, **_kwargs: True)

    metadata = pipeline._load_checkpoint_metadata()

    assert metadata[-1] is documents["transformer/config.json"]
    assert calls == [
        (filename, pipeline.od_config.model, pipeline.od_config.revision)
        for filename in (
            "model_index.json",
            "vae/config.json",
            "transformer/config.json",
        )
    ]


def test_safety_rejection_uses_shared_client_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = object.__new__(MageFlowPipeline)
    pipeline.text_encoder = object()
    pipeline.processor = object()
    monkeypatch.setattr(
        pipeline_mage_flow,
        "screen_mage_flow_prompt",
        lambda *_args, **_kwargs: FilterVerdict(
            violates=True,
            categories=["violence"],
            reason="blocked",
        ),
    )
    sampling = SimpleNamespace(
        height=512,
        width=512,
        extra_args=None,
    )

    with pytest.raises(GuardrailViolationError, match="violence"):
        pipeline._prepare_batch_item(
            "unsafe prompt",
            sampling,
            do_cfg=False,
        )


@pytest.mark.parametrize(
    ("prompt", "sampling_overrides", "message"),
    [
        ({"prompt": 123}, {}, "string 'prompt'"),
        ("", {}, "must not be empty"),
        ("valid", {"height": 513}, "multiple of 16"),
        ("valid", {"generator": [torch.Generator()]}, "exactly one request generator"),
        ("valid", {"seed": 1 << 64}, "seed must be an integer"),
        ("valid", {"generator_device": "cuda:999"}, "must be 'cpu' or 'cuda'"),
        ("valid", {"latents": "invalid"}, "must be a torch.Tensor"),
        (
            "valid",
            {"extra_args": {"mage_enable_watermark": "yes"}},
            "must be a bool",
        ),
        (
            {"prompt": "valid", "multi_modal_data": []},
            {},
            "must be a dictionary",
        ),
    ],
)
def test_request_local_validation_runs_before_batching(
    prompt,
    sampling_overrides,
    message,
) -> None:
    sampling = {
        "height": 512,
        "width": 512,
        "extra_args": None,
    }
    sampling.update(sampling_overrides)
    request = SimpleNamespace(
        prompt=prompt,
        sampling_params=SimpleNamespace(**sampling),
    )

    with pytest.raises(ValueError, match=message):
        get_mage_flow_pre_process_func(SimpleNamespace())(request)


def test_request_batch_isolates_client_errors_without_hiding_runtime_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = MageFlowPipeline.__new__(MageFlowPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cuda")
    pipeline.od_config = SimpleNamespace(
        model="mage-flow",
        output_type=None,
    )
    pipeline.vae = SimpleNamespace(
        downsample_factor=1,
        latent_channels=1,
    )
    safe_item = _MageFlowBatchItem(
        height=1,
        width=2,
        prompt_embeds=torch.zeros(1, 1, 1),
        negative_prompt_embeds=None,
        target_latents=torch.ones(1, 2, 1),
        reference_latents=torch.empty(1, 0, 1),
        image_grids=[(1, 2)],
    )

    def prepare(_self, prompt_data, _sampling, *, do_cfg):
        assert not do_cfg
        if prompt_data == "unsafe":
            raise GuardrailViolationError("blocked request")
        return safe_item

    monkeypatch.setattr(MageFlowPipeline, "_prepare_batch_item", prepare)
    monkeypatch.setattr(
        MageFlowPipeline,
        "diffuse",
        lambda _self, **kwargs: kwargs["target_latents"],
    )
    sampling = SimpleNamespace(
        num_outputs_per_prompt=1,
        num_inference_steps=1,
        guidance_scale_provided=True,
        guidance_scale=1.0,
        output_type="latent",
        cfg_normalize=False,
    )
    request_batch = SimpleNamespace(
        num_reqs=2,
        prompts=["unsafe", "safe"],
        sampling_params_list=[sampling, sampling],
    )

    outputs = pipeline.forward(request_batch)

    assert len(outputs) == 2
    assert outputs[0].error == "blocked request"
    assert outputs[0].error_status_code == 400
    assert outputs[1].error is None
    assert outputs[1].output.shape == (1, 1, 1, 2)

    def crash(_self, _prompt_data, _sampling, *, do_cfg):
        assert not do_cfg
        raise RuntimeError("model execution failed")

    monkeypatch.setattr(MageFlowPipeline, "_prepare_batch_item", crash)
    request_batch.num_reqs = 1
    request_batch.prompts = ["broken"]
    request_batch.sampling_params_list = [sampling]
    with pytest.raises(RuntimeError, match="model execution failed"):
        pipeline.forward(request_batch)
