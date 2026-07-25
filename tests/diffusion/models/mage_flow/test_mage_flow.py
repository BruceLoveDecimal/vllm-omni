# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from PIL import Image

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.model_metadata import (
    MAGE_FLOW_MAX_INPUT_IMAGES,
    get_diffusion_model_metadata,
)
from vllm_omni.diffusion.models.internvla_a1.adapter_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
    Qwen3VLTextConfig,
    Qwen3VLTextModel,
)
from vllm_omni.diffusion.models.mage_flow.autoencoder_mage import MageVAE
from vllm_omni.diffusion.models.mage_flow.mage_flow_layers import (
    MageFlowEmbedRope,
    MageJointAttention,
    _request_isolated_forward,
    apply_rotary_emb_mage_flow,
)
from vllm_omni.diffusion.models.mage_flow.mage_flow_transformer import (
    MageFlowTransformer2DModel,
)
from vllm_omni.diffusion.models.mage_flow.pipeline_mage_flow import (
    MageFlowPipeline,
    make_mage_flow_request_scheduler,
    map_mage_flow_weight_name,
)
from vllm_omni.diffusion.models.mage_flow.prompt_utils import (
    MAGE_FLOW_EDIT_PROMPT_START_INDEX,
    MAGE_FLOW_PROMPT_START_INDEX,
    encode_mage_flow_edit_prompt,
    encode_mage_flow_prompt,
    format_mage_flow_edit_prompt,
    format_mage_flow_prompt,
    get_mage_flow_variant_defaults,
    normalize_mage_flow_reference_images,
    parse_mage_flow_extra_args,
    preprocess_mage_flow_reference_for_vae,
    validate_mage_flow_size,
    validate_reference_image_count,
)
from vllm_omni.diffusion.models.mage_flow.watermark import (
    make_initial_noise,
)
from vllm_omni.diffusion.registry import DiffusionModelRegistry
from vllm_omni.model_extras import get_extra_body_params

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_qwen3_vl_adapter_exposes_concrete_config_class() -> None:
    assert Qwen3VLForConditionalGeneration.config_class is Qwen3VLConfig
    assert Qwen3VLForConditionalGeneration._tied_weights_keys == {
        "lm_head.weight": "model.language_model.embed_tokens.weight",
    }


def test_qwen3_vl_text_adapter_forward() -> None:
    config = Qwen3VLTextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
    )
    model = Qwen3VLTextModel(config).eval()
    output = model(input_ids=torch.tensor([[1, 2, 3]]))
    assert output.last_hidden_state.shape == (1, 3, 16)


@pytest.mark.parametrize(
    ("model", "steps", "cfg"),
    [
        ("microsoft/Mage-Flow-Base", 30, 5.0),
        ("microsoft/Mage-Flow", 20, 5.0),
        ("microsoft/Mage-Flow-Turbo", 4, 1.0),
        ("microsoft/Mage-Flow-Edit-Base", 30, 5.0),
        ("microsoft/Mage-Flow-Edit", 30, 5.0),
        ("microsoft/Mage-Flow-Edit-Turbo", 4, 1.0),
    ],
)
def test_variant_defaults(model: str, steps: int, cfg: float) -> None:
    defaults = get_mage_flow_variant_defaults(model)
    assert defaults.num_inference_steps == steps
    assert defaults.guidance_scale == cfg


@pytest.mark.parametrize(
    ("height", "width"),
    [(1024, 1024), (512, 2048), (2048, 512), (768, 1024)],
)
def test_valid_sizes(height: int, width: int) -> None:
    validate_mage_flow_size(height, width)


@pytest.mark.parametrize(
    ("height", "width"),
    [(511, 1024), (512, 2064), (513, 1024), (512, 512 * 5)],
)
def test_invalid_sizes(height: int, width: int) -> None:
    with pytest.raises(ValueError):
        validate_mage_flow_size(height, width)


def test_extra_args_are_strict() -> None:
    parsed = parse_mage_flow_extra_args(
        {
            "mage_enable_safety_check": False,
            "mage_enable_watermark": True,
            "mage_vision_long_edge": 512,
            "unrelated_option": "allowed",
        }
    )
    assert parsed.enable_safety_check is False
    assert parsed.enable_watermark is True
    assert parsed.vision_long_edge == 512

    with pytest.raises(ValueError, match="Unknown Mage-Flow"):
        parse_mage_flow_extra_args({"mage_typo": True})
    with pytest.raises(ValueError, match="must be a bool"):
        parse_mage_flow_extra_args({"mage_enable_watermark": 1})
    with pytest.raises(ValueError, match="positive integer"):
        parse_mage_flow_extra_args({"mage_vision_long_edge": 0})


def test_reference_image_limit() -> None:
    validate_reference_image_count([object()])
    validate_reference_image_count([object()] * MAGE_FLOW_MAX_INPUT_IMAGES)
    with pytest.raises(ValueError):
        validate_reference_image_count([])
    with pytest.raises(ValueError):
        validate_reference_image_count([object()] * 4)


def test_reference_images_normalize_pil_numpy_and_tensor_in_order() -> None:
    images = normalize_mage_flow_reference_images(
        [
            Image.new("RGB", (5, 4), (255, 0, 0)),
            np.full((4, 5, 3), (0, 255, 0), dtype=np.uint8),
            torch.tensor([[[0]], [[0]], [[255]]], dtype=torch.uint8),
        ]
    )
    assert [image.mode for image in images] == ["RGB", "RGB", "RGB"]
    assert [image.getpixel((0, 0)) for image in images] == [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
    ]


def test_reference_image_normalization_rejects_empty_and_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="between 1 and 3"):
        normalize_mage_flow_reference_images([])
    with pytest.raises(TypeError, match="PIL images"):
        normalize_mage_flow_reference_images([object()])
    with pytest.raises(ValueError, match="reference tensors"):
        normalize_mage_flow_reference_images([torch.zeros(2, 3)])


def test_reference_vae_preprocessing_matches_official_range_and_shape() -> None:
    image = Image.new("RGB", (8, 4), (255, 127, 0))
    tensor = preprocess_mage_flow_reference_for_vae(
        image,
        height=16,
        width=32,
    )
    assert tensor.shape == (3, 16, 32)
    torch.testing.assert_close(tensor[:, 0, 0], torch.tensor([1.0, 127 / 127.5 - 1.0, -1.0]))


def test_rope_shape_and_application() -> None:
    rope = MageFlowEmbedRope(theta=10000, axes_dim=[2, 2, 4])
    frequencies = rope([(1, 2, 3)])
    assert frequencies.shape == (6, 4)

    hidden_states = torch.randn(1, 6, 2, 8)
    rotated = apply_rotary_emb_mage_flow(hidden_states, frequencies)
    assert rotated.shape == hidden_states.shape
    assert rotated.dtype == hidden_states.dtype


def test_batched_rope_application_preserves_per_sample_frequencies() -> None:
    rope = MageFlowEmbedRope(theta=10000, axes_dim=[2, 2, 4])
    first = rope([(1, 2, 3)])
    second = rope([(1, 3, 2)])
    hidden_states = torch.randn(2, 6, 1, 8)

    batched = apply_rotary_emb_mage_flow(
        hidden_states,
        torch.stack([first, second]),
    )
    first_result = apply_rotary_emb_mage_flow(hidden_states[:1], first)
    second_result = apply_rotary_emb_mage_flow(hidden_states[1:], second)

    torch.testing.assert_close(batched[:1], first_result)
    torch.testing.assert_close(batched[1:], second_result)


def test_rope_preserves_complex_phase_after_bfloat16_conversion() -> None:
    rope = MageFlowEmbedRope(theta=10000, axes_dim=[2, 2, 4])
    expected = rope([(1, 2, 3)])
    rope.to(dtype=torch.bfloat16)
    frequencies = rope([(1, 2, 3)])

    assert frequencies.is_complex()
    assert torch.count_nonzero(frequencies.imag)
    torch.testing.assert_close(frequencies, expected)


def test_rope_tables_ignore_ambient_construction_device() -> None:
    with torch.device("meta"):
        rope = MageFlowEmbedRope(theta=10000, axes_dim=[2, 2, 4])

    assert rope.pos_freqs.device.type == "cpu"
    assert rope.neg_freqs.device.type == "cpu"
    assert rope.pos_freqs.dtype == torch.complex64


def test_rope_uses_bounded_device_aware_grid_cache() -> None:
    rope = MageFlowEmbedRope(
        theta=10000,
        axes_dim=[2, 2, 4],
        max_positions=32,
    )
    rope._compute_grid.cache_clear()
    device = torch.device("cpu")

    first = rope._compute_grid(device, 1, 2, 3)
    second = rope._compute_grid(device, 1, 2, 3)
    assert second is first
    assert rope._compute_grid.cache_info().hits == 1

    for width in range(1, 18):
        rope._compute_grid(device, 1, 1, width)
    cache_info = rope._compute_grid.cache_info()
    assert cache_info.maxsize == 16
    assert cache_info.currsize == 16


def test_joint_attention_matches_pytorch_reference() -> None:
    torch.manual_seed(7)
    attention = MageJointAttention(
        query_dim=8,
        heads=1,
        dim_head=8,
        added_kv_proj_dim=8,
    )
    image = torch.randn(1, 6, 8)
    text = torch.randn(1, 3, 8)
    rope = MageFlowEmbedRope(theta=10000, axes_dim=[2, 2, 4])([(1, 2, 3)])

    image_output, text_output = attention(image, text, rope)
    image_q, image_k, image_v = attention._project(
        image,
        attention.to_q,
        attention.to_k,
        attention.to_v,
        attention.norm_q,
        attention.norm_k,
    )
    text_q, text_k, text_v = attention._project(
        text,
        attention.add_q_proj,
        attention.add_k_proj,
        attention.add_v_proj,
        attention.norm_added_q,
        attention.norm_added_k,
    )
    image_q = apply_rotary_emb_mage_flow(image_q, rope)
    image_k = apply_rotary_emb_mage_flow(image_k, rope)
    query = torch.cat([text_q, image_q], dim=1).transpose(1, 2)
    key = torch.cat([text_k, image_k], dim=1).transpose(1, 2)
    value = torch.cat([text_v, image_v], dim=1).transpose(1, 2)
    reference = F.scaled_dot_product_attention(
        query,
        key,
        value,
        scale=attention.dim_head**-0.5,
    ).transpose(1, 2)
    reference_text, reference_image = reference.split([3, 6], dim=1)
    reference_image = attention.to_out[0](reference_image.flatten(2, 3))
    reference_text = attention.to_add_out(reference_text.flatten(2, 3))

    torch.testing.assert_close(image_output, reference_image)
    torch.testing.assert_close(text_output, reference_text)


def test_joint_attention_uses_request_isolated_kernel_shapes_for_batch() -> None:
    class RecordingAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = []

        def forward(self, query, key, value, attn_metadata=None):
            assert attn_metadata is None
            self.calls.append(tuple(query.shape))
            return value

    attention = MageJointAttention(
        query_dim=8,
        heads=1,
        dim_head=8,
        added_kv_proj_dim=8,
    )
    recording_attention = RecordingAttention()
    attention.attention = recording_attention
    image = torch.randn(2, 6, 8)
    text = torch.randn(2, 4, 8)
    rope = MageFlowEmbedRope(theta=10000, axes_dim=[2, 2, 4])([(1, 2, 3)])
    image_mask = torch.tensor([[True] * 4 + [False] * 2, [True] * 6])
    text_mask = torch.tensor([[True] * 3 + [False], [True] * 2 + [False] * 2])

    image_output, text_output = attention(
        image,
        text,
        torch.stack([rope, rope]),
        image_attention_mask=image_mask,
        encoder_attention_mask=text_mask,
    )

    assert recording_attention.calls == [(1, 7, 1, 8), (1, 8, 1, 8)]
    assert image_output.shape == image.shape
    assert text_output.shape == text.shape
    assert not torch.count_nonzero(image_output[0, 4:])
    assert not torch.count_nonzero(text_output[0, 3:])
    assert not torch.count_nonzero(text_output[1, 2:])


def test_request_isolated_forward_uses_valid_standalone_shapes() -> None:
    class RecordingProjection(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = []

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            self.calls.append(tuple(hidden_states.shape))
            return torch.cat([hidden_states, hidden_states], dim=-1)

    projection = RecordingProjection()
    hidden_states = torch.randn(2, 5, 3)
    attention_mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, True],
        ]
    )

    output = _request_isolated_forward(
        projection,
        hidden_states,
        attention_mask,
    )

    assert projection.calls == [(1, 3, 3), (1, 5, 3)]
    assert output.shape == (2, 5, 6)
    torch.testing.assert_close(output[0, :3], hidden_states[0, :3].repeat(1, 2))
    assert not torch.count_nonzero(output[0, 3:])


def test_attention_mask_falls_back_to_sdpa_for_unsupported_backend() -> None:
    class UnsupportedBackend:
        @classmethod
        def supports_attention_mask(cls):
            return False

        @staticmethod
        def get_name():
            return "UNSUPPORTED"

    class UnexpectedImplementation:
        def forward(self, *args):
            raise AssertionError("configured backend must not receive a mask")

    class FallbackImplementation:
        def __init__(self):
            self.called = False

        def forward(self, query, key, value, metadata):
            self.called = True
            assert metadata.attn_mask.tolist() == [[True, False]]
            return query

    attention = Attention.__new__(Attention)
    torch.nn.Module.__init__(attention)
    attention.attn_backend = UnsupportedBackend
    attention.attention = UnexpectedImplementation()
    attention.sdpa_fallback = FallbackImplementation()
    attention.backend_pref = "unsupported"
    query = torch.randn(1, 2, 1, 8, dtype=torch.bfloat16)
    output = attention._run_local_attention(
        query,
        query,
        query,
        AttentionMetadata(
            attn_mask=torch.tensor([[True, False]])
        ),
    )

    assert attention.sdpa_fallback.called
    assert output is query


def test_tiny_transformer_forward_shape() -> None:
    model = MageFlowTransformer2DModel(
        in_channels=4,
        out_channels=4,
        context_in_dim=6,
        hidden_size=8,
        num_heads=1,
        depth=1,
        axes_dim=[2, 2, 4],
    )
    output = model(
        hidden_states=torch.randn(1, 6, 4),
        encoder_hidden_states=torch.randn(1, 5, 6),
        timestep=torch.tensor([0.5]),
        image_grid_hw=[(2, 3)],
    )[0]
    assert output.shape == (1, 6, 4)


def test_tiny_transformer_padded_batch_matches_dense_requests() -> None:
    torch.manual_seed(17)
    model = MageFlowTransformer2DModel(
        in_channels=4,
        out_channels=4,
        context_in_dim=6,
        hidden_size=8,
        num_heads=1,
        depth=1,
        axes_dim=[2, 2, 4],
    ).eval()
    first_image = torch.randn(1, 6, 4)
    second_image = torch.randn(1, 8, 4)
    first_text = torch.randn(1, 3, 6)
    second_text = torch.randn(1, 5, 6)

    first_output = model(
        hidden_states=first_image,
        encoder_hidden_states=first_text,
        timestep=torch.tensor([0.5]),
        image_grid_hw=[(2, 3)],
    )[0]
    second_output = model(
        hidden_states=second_image,
        encoder_hidden_states=second_text,
        timestep=torch.tensor([0.5]),
        image_grid_hw=[(2, 4)],
    )[0]

    padded_images = torch.zeros(2, 8, 4)
    padded_images[0, :6] = first_image
    padded_images[1] = second_image
    image_mask = torch.tensor(
        [[True] * 6 + [False] * 2, [True] * 8]
    )
    padded_text = torch.zeros(2, 5, 6)
    padded_text[0, :3] = first_text
    padded_text[1] = second_text
    text_mask = torch.tensor(
        [[True] * 3 + [False] * 2, [True] * 5]
    )
    batched_output = model(
        hidden_states=padded_images,
        encoder_hidden_states=padded_text,
        timestep=torch.tensor([0.5, 0.5]),
        image_grid_hw=[[(2, 3)], [(2, 4)]],
        image_attention_mask=image_mask,
        encoder_attention_mask=text_mask,
    )[0]

    torch.testing.assert_close(
        batched_output[0, :6],
        first_output[0],
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        batched_output[1],
        second_output[0],
        rtol=1e-5,
        atol=1e-6,
    )
    assert not torch.count_nonzero(batched_output[0, 6:])


def test_tiny_transformer_padding_cannot_leak_between_requests() -> None:
    torch.manual_seed(23)
    model = MageFlowTransformer2DModel(
        in_channels=4,
        out_channels=4,
        context_in_dim=6,
        hidden_size=8,
        num_heads=1,
        depth=1,
        axes_dim=[2, 2, 4],
    ).eval()
    images = torch.randn(2, 8, 4)
    text = torch.randn(2, 5, 6)
    image_mask = torch.tensor(
        [[True] * 6 + [False] * 2, [True] * 8]
    )
    text_mask = torch.tensor(
        [[True] * 3 + [False] * 2, [True] * 5]
    )
    kwargs = {
        "timestep": torch.tensor([0.25, 0.75]),
        "image_grid_hw": [[(2, 3)], [(2, 4)]],
        "image_attention_mask": image_mask,
        "encoder_attention_mask": text_mask,
    }
    baseline = model(
        hidden_states=images,
        encoder_hidden_states=text,
        **kwargs,
    )[0]
    images_with_poisoned_padding = images.clone()
    text_with_poisoned_padding = text.clone()
    images_with_poisoned_padding[0, 6:] = 1e4
    text_with_poisoned_padding[0, 3:] = -1e4
    poisoned = model(
        hidden_states=images_with_poisoned_padding,
        encoder_hidden_states=text_with_poisoned_padding,
        **kwargs,
    )[0]

    torch.testing.assert_close(poisoned[0, :6], baseline[0, :6])
    torch.testing.assert_close(poisoned[1], baseline[1])


def test_transformer_config_validation() -> None:
    with pytest.raises(ValueError, match=r"sum\(axes_dim\)"):
        MageFlowTransformer2DModel(
            in_channels=4,
            out_channels=4,
            context_in_dim=6,
            hidden_size=8,
            num_heads=1,
            depth=1,
            axes_dim=[2, 2, 2],
        )
    with pytest.raises(ValueError, match="schedule_mode"):
        MageFlowTransformer2DModel(
            in_channels=4,
            out_channels=4,
            context_in_dim=6,
            hidden_size=8,
            num_heads=1,
            depth=1,
            axes_dim=[2, 2, 4],
            schedule_mode="unknown",
        )
    with pytest.raises(ValueError, match="qkv_bias"):
        MageFlowTransformer2DModel(
            in_channels=4,
            out_channels=4,
            context_in_dim=6,
            hidden_size=8,
            num_heads=1,
            depth=1,
            axes_dim=[2, 2, 4],
            qkv_bias=False,
        )


def test_tiny_vae_encode_decode_shapes() -> None:
    vae = MageVAE(
        latent_channels=4,
        downsample_factor=4,
        sample_posterior=False,
        encoder_kwargs={
            "z_ch": 4,
            "hidden_size": 32,
            "num_blocks": 1,
            "patch_size": 4,
            "mlp_ratio": 2.0,
            "head_size": 32,
            "num_head_blocks": 1,
        },
        decoder_kwargs={
            "patch_size": 4,
            "hidden_size": 32,
            "hidden_size_x": 8,
            "mlp_ratio": 2.0,
            "num_blocks": 2,
            "num_cond_blocks": 1,
            "bottleneck_dim": 4,
        },
    )
    image = torch.randn(1, 3, 32, 32)
    latent = vae.encode(image)
    decoded = vae.decode(latent)
    assert latent.shape == (1, 4, 8, 8)
    assert decoded.shape == image.shape


def test_prompt_template_and_conditioning_slice() -> None:
    class FakeProcessor:
        def __call__(self, **kwargs):
            assert kwargs["text"] == [format_mage_flow_prompt("a fox")]
            length = MAGE_FLOW_PROMPT_START_INDEX + 4
            return {
                "input_ids": torch.arange(length).unsqueeze(0),
                "attention_mask": torch.ones(1, length, dtype=torch.long),
            }

    class FakeModel:
        def __call__(self, *, input_ids, **kwargs):
            hidden = input_ids[..., None].expand(-1, -1, 6).float()
            return type("Output", (), {"last_hidden_state": hidden})()

    encoder = type("Encoder", (), {"model": FakeModel()})()
    embeddings = encode_mage_flow_prompt(
        encoder,
        FakeProcessor(),
        "a fox",
        device=torch.device("cpu"),
    )
    assert embeddings.shape == (1, 4, 6)
    assert embeddings[0, 0, 0] == MAGE_FLOW_PROMPT_START_INDEX


def test_edit_prompt_preserves_reference_order_and_visual_inputs() -> None:
    references = [
        Image.new("RGB", (20, 10), (255, 0, 0)),
        Image.new("RGB", (10, 20), (0, 255, 0)),
    ]

    class FakeProcessor:
        def __call__(self, **kwargs):
            assert kwargs["text"] == [format_mage_flow_edit_prompt("combine them", 2)]
            assert [image.size for image in kwargs["images"]] == [(8, 4), (4, 8)]
            assert [image.getpixel((0, 0)) for image in kwargs["images"]] == [
                (255, 0, 0),
                (0, 255, 0),
            ]
            length = MAGE_FLOW_EDIT_PROMPT_START_INDEX + 5
            return {
                "input_ids": torch.arange(length).unsqueeze(0),
                "attention_mask": torch.ones(1, length, dtype=torch.long),
                "pixel_values": torch.ones(2, 3, 4, 4),
                "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]]),
                "mm_token_type_ids": torch.ones(1, length, dtype=torch.long),
            }

    class FakeModel:
        def __call__(
            self,
            *,
            input_ids,
            pixel_values,
            image_grid_thw,
            mm_token_type_ids,
            **kwargs,
        ):
            assert pixel_values.shape[0] == 2
            assert image_grid_thw.shape == (2, 3)
            assert mm_token_type_ids.shape == input_ids.shape
            hidden = input_ids[..., None].expand(-1, -1, 6).float()
            return type("Output", (), {"last_hidden_state": hidden})()

    encoder = type("Encoder", (), {"model": FakeModel()})()
    embeddings = encode_mage_flow_edit_prompt(
        encoder,
        FakeProcessor(),
        "combine them",
        references,
        device=torch.device("cpu"),
        vision_long_edge=8,
    )
    assert embeddings.shape == (1, 5, 6)
    assert embeddings[0, 0, 0] == MAGE_FLOW_EDIT_PROMPT_START_INDEX


def test_edit_diffusion_keeps_reference_latents_clean() -> None:
    class Progress:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def update(self):
            return None

    class FakePipeline:
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=6.0,
            use_dynamic_shifting=False,
        )
        device = torch.device("cpu")
        vae = type("VAE", (), {"downsample_factor": 16})()

        def __init__(self):
            self.combined_inputs = []

        def progress_bar(self, total):
            return Progress()

        def predict_noise_maybe_with_cfg(self, *, positive_kwargs, **kwargs):
            hidden_states = positive_kwargs["hidden_states"]
            self.combined_inputs.append(hidden_states.clone())
            return hidden_states

        def scheduler_step(self, noise_pred, timestep, latents, *, per_request_scheduler):
            return per_request_scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]

    pipeline = FakePipeline()
    target = torch.ones(1, 4, 2)
    references = torch.full((1, 8, 2), 7.0)
    output = MageFlowPipeline.diffuse_edit(
        pipeline,
        latents=target,
        reference_latents=references,
        num_reference_images=2,
        prompt_embeds=torch.zeros(1, 3, 4),
        negative_prompt_embeds=None,
        height=32,
        width=32,
        num_inference_steps=2,
        guidance_scale=1.0,
        cfg_normalize=False,
    )
    assert output.shape == target.shape
    assert len(pipeline.combined_inputs) == 2
    for combined in pipeline.combined_inputs:
        torch.testing.assert_close(combined[:, 4:], references)


def test_packed_cfg_uses_one_transformer_forward() -> None:
    class FakePipeline:
        def __init__(self):
            self.calls = 0

        def predict_noise(self, **kwargs):
            self.calls += 1
            prompt_mask = kwargs["encoder_attention_mask"][..., None]
            prompt_signal = (
                kwargs["encoder_hidden_states"] * prompt_mask
            ).sum(
                dim=(1, 2),
                keepdim=True,
            ) / (
                prompt_mask.sum(dim=(1, 2), keepdim=True)
                * kwargs["encoder_hidden_states"].shape[-1]
            )
            return kwargs["hidden_states"] + prompt_signal

        def combine_cfg_noise(
            self,
            positive,
            negative,
            guidance_scale,
            cfg_normalize,
        ):
            assert not cfg_normalize
            return negative + guidance_scale * (positive - negative)

    pipeline = FakePipeline()
    hidden_states = torch.randn(2, 4, 3)
    image_mask = torch.ones(2, 4, dtype=torch.bool)
    positive_text_mask = torch.ones(2, 3, dtype=torch.bool)
    negative_text_mask = torch.ones(2, 2, dtype=torch.bool)
    common = {
        "hidden_states": hidden_states,
        "timestep": torch.tensor([0.5, 0.5]),
        "image_grid_hw": [[(2, 2)], [(2, 2)]],
        "image_attention_mask": image_mask,
    }
    positive_prompt = torch.full((2, 3, 5), 2.0)
    negative_prompt = torch.full((2, 2, 5), -1.0)
    output = MageFlowPipeline._predict_noise_packed_cfg(
        pipeline,
        positive_kwargs={
            **common,
            "encoder_hidden_states": positive_prompt,
            "encoder_attention_mask": positive_text_mask,
        },
        negative_kwargs={
            **common,
            "encoder_hidden_states": negative_prompt,
            "encoder_attention_mask": negative_text_mask,
        },
        guidance_scale=5.0,
        cfg_normalize=False,
    )

    assert pipeline.calls == 1
    torch.testing.assert_close(output, hidden_states + 14.0)


def test_pad_token_sequences_preserves_lengths_and_masks() -> None:
    first = torch.full((1, 2, 3), 1.0)
    second = torch.full((1, 4, 3), 2.0)
    padded, mask, lengths = MageFlowPipeline._pad_token_sequences(
        [first, second]
    )

    assert padded.shape == (2, 4, 3)
    assert lengths == [2, 4]
    assert mask.tolist() == [
        [True, True, False, False],
        [True, True, True, True],
    ]
    assert not torch.count_nonzero(padded[0, 2:])


def test_request_scheduler_is_isolated_and_reproducible() -> None:
    base = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=6.0,
        use_dynamic_shifting=False,
    )
    first = make_mage_flow_request_scheduler(
        base,
        num_inference_steps=4,
        device=torch.device("cpu"),
    )
    second = make_mage_flow_request_scheduler(
        base,
        num_inference_steps=4,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(first.timesteps, second.timesteps)
    torch.testing.assert_close(first.sigmas, second.sigmas)
    assert first is not second
    assert base.step_index is None


def test_request_local_noise_is_deterministic() -> None:
    shape = (1, 4, 8, 8)
    first = make_initial_noise(
        shape,
        generator=torch.Generator().manual_seed(123),
        device=torch.device("cpu"),
        dtype=torch.float32,
        enable_watermark=True,
    )
    second = make_initial_noise(
        shape,
        generator=torch.Generator().manual_seed(123),
        device=torch.device("cpu"),
        dtype=torch.float32,
        enable_watermark=True,
    )
    different = make_initial_noise(
        shape,
        generator=torch.Generator().manual_seed(124),
        device=torch.device("cpu"),
        dtype=torch.float32,
        enable_watermark=True,
    )
    torch.testing.assert_close(first, second)
    assert not torch.equal(first, different)


def test_registry_and_metadata() -> None:
    model_class = DiffusionModelRegistry._try_load_model_cls("MageFlowPipeline")
    assert model_class is not None
    assert model_class.support_image_input
    assert model_class.supports_request_batch
    assert model_class.request_batch_ignored_sampling_param_fields == {
        "height",
        "width",
        "resolution",
    }
    metadata = get_diffusion_model_metadata("MageFlowPipeline")
    assert metadata.supports_multimodal_inputs
    assert metadata.max_multimodal_image_inputs == 3
    assert get_extra_body_params("MageFlowPipeline") == {
        "mage_enable_safety_check",
        "mage_enable_watermark",
        "mage_vision_long_edge",
    }


def test_official_weight_prefix_mapping() -> None:
    assert map_mage_flow_weight_name("vae.student.dconv_encoder.proj_out.weight") == "vae.dconv_encoder.proj_out.weight"
    assert map_mage_flow_weight_name("vae.pipeline.blocks.0.conv1.weight") == "vae.decoder_model.blocks.0.conv1.weight"
    assert map_mage_flow_weight_name("vae.pipeline.y_embedder.encoder.some.weight") is None
    assert map_mage_flow_weight_name("transformer.img_in.weight") == "transformer.img_in.weight"
