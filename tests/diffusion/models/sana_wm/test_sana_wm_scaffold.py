# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _write_config(path) -> None:
    path.write_text(
        "\n".join(
            [
                "model: SanaMSVideoCamCtrl_1600M_P1_D20",
                "image_size: 720",
                "mixed_precision: bf16",
                "fp32_attention: true",
                "attn_type: BidirectionalGDNTriton",
                "softmax_every_n: 4",
                "linear_head_dim: 112",
                "conv_kernel_size: 4",
                "ffn_type: GLUMBConvTemp",
                "pos_embed_type: wan_rope",
                "mlp_ratio: 3",
                "chunk_plucker_channels: 48",
                "chunk_plucker_post_attn_blocks: 20",
                "scheduler:",
                "  inference_flow_shift: 9.8",
                "  vis_sampler: flow_dpm-solver",
                "text_encoder:",
                "  chi_prompt:",
                "    - enhance prompt",
                "  y_norm_scale_factor: 0.01",
                "  model_max_length: 300",
            ]
        ),
        encoding="utf-8",
    )


def test_sana_wm_pipelines_registered() -> None:
    from vllm_omni.diffusion.registry import (
        _DIFFUSION_MODELS,
        _DIFFUSION_POST_PROCESS_FUNCS,
        _DIFFUSION_PRE_PROCESS_FUNCS,
    )

    assert _DIFFUSION_MODELS["SanaWmPipeline"] == ("sana_wm", "pipeline_sana_wm", "SanaWmPipeline")
    assert _DIFFUSION_MODELS["SanaWmTwoStagesPipeline"] == (
        "sana_wm",
        "pipeline_sana_wm_two_stages",
        "SanaWmTwoStagesPipeline",
    )
    assert _DIFFUSION_PRE_PROCESS_FUNCS["SanaWmPipeline"] == "get_sana_wm_pre_process_func"
    assert _DIFFUSION_PRE_PROCESS_FUNCS["SanaWmTwoStagesPipeline"] == "get_sana_wm_pre_process_func"
    assert _DIFFUSION_POST_PROCESS_FUNCS["SanaWmPipeline"] == "get_sana_wm_post_process_func"
    assert _DIFFUSION_POST_PROCESS_FUNCS["SanaWmTwoStagesPipeline"] == "get_sana_wm_post_process_func"


def test_sana_wm_exports_and_constants() -> None:
    from vllm_omni.diffusion.models.sana_wm import (
        SANA_WM_DEFAULT_NUM_FRAMES,
        SANA_WM_DISABLE_TRITON_GDN_ENV,
        SANA_WM_FORCE_CLI_ENV,
        SANA_WM_GDN_ERROR,
        SANA_WM_INPROCESS_REFINER_ARG,
        SANA_WM_INPROCESS_REFINER_STEPS_ARG,
        SANA_WM_MODEL_ID,
        SANA_WM_NATIVE_BACKEND_ERROR,
        SANA_WM_OFFICIAL_BACKEND_ERROR,
        SANA_WM_OFFICIAL_REPO_ENV,
        SANA_WM_OFFICIAL_SCRIPT,
        SANA_WM_OUTPUT_HEIGHT,
        SANA_WM_OUTPUT_WIDTH,
        SANA_WM_REFINER_CONNECTORS_WEIGHT_FILE,
        SANA_WM_REFINER_ROOT_ENV,
        SANA_WM_REFINER_TRANSFORMER_WEIGHT_FILE,
        SANA_WM_REQUIRE_TRITON_GDN_ENV,
        SANA_WM_STAGE1_PROMPT_CHANNELS,
        SANA_WM_STAGE1_DIT_FILE,
        SANA_WM_STAGE1_TEXT_ENCODER_ENV,
        SANA_WM_STAGE1_TEXT_ENCODER_FALLBACK_ID,
        SANA_WM_STAGE1_TEXT_ENCODER_ID,
        SANA_WM_TRANSFORMER_FORWARD_ERROR,
        SANA_WM_VAE_WEIGHT_FILE,
        BidirectionalGatedDeltaNetTriton,
        SanaWmCameraEmbedder,
        SanaWmConfig,
        SanaWmFlowDpmScheduler,
        SanaWmPipeline,
        SanaWmNativeRunResult,
        SanaWmTransformer3DModel,
        SanaWmTwoStagesPipeline,
        build_sana_wm_official_command,
        reference_bidirectional_gated_delta_net,
        shift_flow_timestep,
    )

    assert SANA_WM_MODEL_ID == "Efficient-Large-Model/SANA-WM_bidirectional"
    assert SANA_WM_STAGE1_DIT_FILE == "dit/sana_wm_1600m_720p.safetensors"
    assert SANA_WM_VAE_WEIGHT_FILE == "vae/diffusion_pytorch_model.safetensors"
    assert SANA_WM_REFINER_TRANSFORMER_WEIGHT_FILE == "refiner/transformer/diffusion_pytorch_model.safetensors"
    assert SANA_WM_REFINER_CONNECTORS_WEIGHT_FILE == "refiner/connectors/diffusion_pytorch_model.safetensors"
    assert SANA_WM_REFINER_ROOT_ENV == "VLLM_OMNI_SANA_WM_REFINER_ROOT"
    assert SANA_WM_STAGE1_TEXT_ENCODER_ID == "google/gemma-2-2b-it"
    assert SANA_WM_STAGE1_TEXT_ENCODER_FALLBACK_ID == "Efficient-Large-Model/gemma-2-2b-it"
    assert SANA_WM_STAGE1_TEXT_ENCODER_ENV == "VLLM_OMNI_SANA_WM_STAGE1_TEXT_ENCODER"
    assert SANA_WM_INPROCESS_REFINER_ARG == "sana_wm_inprocess_refiner"
    assert SANA_WM_INPROCESS_REFINER_STEPS_ARG == "sana_wm_inprocess_refiner_steps"
    assert SANA_WM_OUTPUT_HEIGHT == 704
    assert SANA_WM_OUTPUT_WIDTH == 1280
    assert SANA_WM_DEFAULT_NUM_FRAMES == 321
    assert SANA_WM_STAGE1_PROMPT_CHANNELS == 2304
    assert SANA_WM_FORCE_CLI_ENV == "VLLM_OMNI_SANA_WM_USE_OFFICIAL_CLI"
    assert SANA_WM_DISABLE_TRITON_GDN_ENV == "VLLM_OMNI_SANA_WM_DISABLE_TRITON_GDN"
    assert SANA_WM_REQUIRE_TRITON_GDN_ENV == "VLLM_OMNI_SANA_WM_REQUIRE_TRITON_GDN"
    assert "in-process native backend" in SANA_WM_NATIVE_BACKEND_ERROR
    assert SanaWmPipeline.support_image_input is True
    assert SanaWmPipeline._dit_modules == ["transformer"]
    assert SanaWmPipeline._encoder_modules == ["text_encoder", "camera_encoder"]
    assert SanaWmTwoStagesPipeline.support_image_input is True
    assert SanaWmTwoStagesPipeline._dit_modules == ["transformer"]
    assert SanaWmTwoStagesPipeline._encoder_modules == [
        "text_encoder",
        "camera_encoder",
        "refiner_text_encoder",
        "refiner_connectors",
    ]
    assert SanaWmTwoStagesPipeline._resident_modules == ["refiner_transformer"]
    assert SanaWmTransformer3DModel._repeated_blocks == ["blocks"]
    assert SanaWmConfig().num_blocks == 20
    assert SanaWmNativeRunResult.__name__ == "SanaWmNativeRunResult"
    assert "Gated DeltaNet" in SANA_WM_GDN_ERROR
    assert "forward is not implemented" in SANA_WM_TRANSFORMER_FORWARD_ERROR
    assert "official runner" in SANA_WM_OFFICIAL_BACKEND_ERROR
    assert SANA_WM_OFFICIAL_REPO_ENV == "VLLM_OMNI_SANA_WM_OFFICIAL_REPO"
    assert SANA_WM_OFFICIAL_SCRIPT == "inference_video_scripts/inference_sana_wm.py"
    gdn = BidirectionalGatedDeltaNetTriton()
    assert gdn.__class__.__name__ == "BidirectionalGatedDeltaNetTriton"
    assert gdn.triton_available is True
    assert SanaWmCameraEmbedder().__class__.__name__ == "SanaWmCameraEmbedder"
    assert SanaWmFlowDpmScheduler(num_inference_steps=1).num_inference_steps == 1
    assert callable(reference_bidirectional_gated_delta_net)
    assert callable(shift_flow_timestep)
    assert callable(build_sana_wm_official_command)


def test_sana_wm_config_parses_release_yaml(tmp_path) -> None:
    from vllm_omni.diffusion.models.sana_wm import SanaWmConfig

    config_path = tmp_path / "config.yaml"
    _write_config(config_path)

    config = SanaWmConfig.from_yaml(config_path)

    assert config.architecture_name == "SanaMSVideoCamCtrl_1600M_P1_D20"
    assert config.num_blocks == 20
    assert config.hidden_size == 2240
    assert config.mlp_ratio == 3.0
    assert config.attn_type == "BidirectionalGDNTriton"
    assert config.qk_norm is True
    assert config.cross_norm is True
    assert config.cam_attn_compress == 1
    assert config.patch_size == (1, 1, 1)
    assert config.t_kernel_size == 3
    assert config.k_conv_only is True
    assert config.scheduler_type == "flow_dpm-solver"
    assert config.inference_flow_shift == 9.8
    assert config.chi_prompt == ["enhance prompt"]
    assert config.y_norm_scale_factor == 0.01
    assert config.model_max_length == 300


def test_sana_wm_flow_shift_scheduler() -> None:
    import torch

    from vllm_omni.diffusion.models.sana_wm import SanaWmFlowDpmScheduler, shift_flow_timestep

    assert shift_flow_timestep(torch.tensor([1.0]), 9.8).item() == pytest.approx(1.0)
    assert shift_flow_timestep(torch.tensor([0.0]), 9.8).item() == pytest.approx(0.0)
    scheduler = SanaWmFlowDpmScheduler(num_inference_steps=2, shift=9.8)

    assert scheduler.timesteps(device=torch.device("cpu")).shape == (2,)
    assert scheduler.deltas(device=torch.device("cpu")).shape == (2,)


def test_sana_wm_transformer_native_smoke_forward_shape() -> None:
    import torch

    from vllm_omni.diffusion.models.sana_wm import SanaWmConfig, SanaWmTransformer3DModel

    config = SanaWmConfig(
        num_blocks=1,
        hidden_size=8,
        linear_head_dim=4,
        mlp_ratio=1,
        model_max_length=3,
        chunk_plucker_channels=6,
    )
    model = SanaWmTransformer3DModel(config=config)
    latents = torch.randn(1, 4, 2, 4, 4)
    prompt_embeds = torch.randn(1, 3, 6)

    output = model(latents, torch.tensor([1.0]), encoder_hidden_states=prompt_embeds)

    assert output.shape == latents.shape
    assert torch.isfinite(output).all()
    assert model.is_materialized is True


def test_sana_wm_transformer_marks_hybrid_softmax_blocks() -> None:
    from vllm_omni.diffusion.models.sana_wm import SanaWmConfig, SanaWmTransformer3DModel

    config = SanaWmConfig(
        num_blocks=4,
        hidden_size=8,
        linear_head_dim=4,
        mlp_ratio=1,
        softmax_every_n=2,
    )
    model = SanaWmTransformer3DModel(config=config, materialize=True)

    assert [block.attn.use_gdn for block in model.blocks] == [True, False, True, False]


def test_sana_wm_transformer_hybrid_softmax_forward_shape() -> None:
    import torch

    from vllm_omni.diffusion.models.sana_wm import SanaWmConfig, SanaWmTransformer3DModel

    config = SanaWmConfig(
        num_blocks=1,
        hidden_size=8,
        linear_head_dim=4,
        mlp_ratio=1,
        model_max_length=3,
        softmax_every_n=1,
    )
    model = SanaWmTransformer3DModel(config=config)
    latents = torch.randn(1, 4, 2, 4, 4)
    prompt_embeds = torch.randn(1, 3, 6)

    output = model(latents, torch.tensor([1.0]), encoder_hidden_states=prompt_embeds)

    assert output.shape == latents.shape
    assert torch.isfinite(output).all()


def test_sana_wm_transformer_uses_vllm_parallel_layers_when_available() -> None:
    from vllm_omni.diffusion.models.sana_wm import SanaWmConfig, SanaWmTransformer3DModel
    from vllm_omni.diffusion.models.sana_wm.sana_wm_transformer import (
        _vllm_attention_available,
        _vllm_parallel_layers_available,
    )

    config = SanaWmConfig(
        num_blocks=1,
        hidden_size=8,
        linear_head_dim=4,
        mlp_ratio=1,
        model_max_length=3,
    )
    model = SanaWmTransformer3DModel(config=config, materialize=True)
    block = model.blocks[0]

    assert model.use_vllm_parallel_layers is _vllm_parallel_layers_available()
    if _vllm_parallel_layers_available():
        # --- pre-existing main-attention sites ---
        assert block.attn.qkv.__class__.__name__ == "QKVParallelLinear"
        assert block.attn.proj.__class__.__name__ == "RowParallelLinear"
        assert block.attn.beta_proj.__class__.__name__ == "ColumnParallelLinear"
        assert block.attn.output_gate.__class__.__name__ == "ColumnParallelLinear"
        assert block.cross_attn.q_linear.__class__.__name__ == "ColumnParallelLinear"
        assert block.cross_attn.kv_linear.__class__.__name__ == "ColumnParallelLinear"
        assert block.cross_attn.proj.__class__.__name__ == "RowParallelLinear"
        assert model.y_embedder.y_proj.fc1.__class__.__name__ == "ColumnParallelLinear"
        assert model.y_embedder.y_proj.fc2.__class__.__name__ == "RowParallelLinear"
        assert model.t_embedder.mlp[0].__class__.__name__ == "ColumnParallelLinear"
        assert model.t_embedder.mlp[2].__class__.__name__ == "RowParallelLinear"
        # --- newly parallelised sites (5a completion) ---
        assert block.attn.q_proj_cam.__class__.__name__ == "ColumnParallelLinear"
        assert block.attn.k_proj_cam.__class__.__name__ == "ColumnParallelLinear"
        assert block.attn.v_proj_cam.__class__.__name__ == "ColumnParallelLinear"
        assert block.attn.out_proj_cam.__class__.__name__ == "RowParallelLinear"
        assert block.plucker_proj.__class__.__name__ == "ColumnParallelLinear"
        assert model.final_layer.linear.__class__.__name__ == "ColumnParallelLinear"
        assert model.t_block[1].__class__.__name__ == "ColumnParallelLinear"
        assert "RMSNorm" in model.attention_y_norm.__class__.__name__
    else:
        assert block.attn.qkv.__class__.__name__ == "Linear"
        assert block.cross_attn.kv_linear.__class__.__name__ == "Linear"
        # fallback for new sites
        assert block.attn.q_proj_cam.__class__.__name__ == "Linear"
        assert block.attn.out_proj_cam.__class__.__name__ == "Linear"
        assert block.plucker_proj.__class__.__name__ == "Linear"
        assert model.final_layer.linear.__class__.__name__ == "Linear"
        assert block.attn.q_proj_cam.in_features == config.hidden_size
        assert block.attn.out_proj_cam.out_features == config.hidden_size
    if _vllm_attention_available():
        assert block.cross_attn.softmax_attn.__class__.__name__ == "Attention"


def test_sana_wm_transformer_softmax_blocks_use_attention_layer_when_available() -> None:
    from vllm_omni.diffusion.models.sana_wm import SanaWmConfig, SanaWmTransformer3DModel
    from vllm_omni.diffusion.models.sana_wm.sana_wm_transformer import _vllm_attention_available

    config = SanaWmConfig(
        num_blocks=1,
        hidden_size=8,
        linear_head_dim=4,
        mlp_ratio=1,
        model_max_length=3,
        softmax_every_n=1,
    )
    model = SanaWmTransformer3DModel(config=config, materialize=True)
    block = model.blocks[0]

    assert block.attn.use_gdn is False
    if _vllm_attention_available():
        assert block.attn.softmax_attn.__class__.__name__ == "Attention"
        assert block.cross_attn.softmax_attn.__class__.__name__ == "Attention"
    else:
        assert block.attn.softmax_attn is None
        assert block.cross_attn.softmax_attn is None


def test_sana_wm_transformer_declares_sp_plan_when_available() -> None:
    from vllm_omni.diffusion.models.sana_wm import SanaWmTransformer3DModel
    from vllm_omni.diffusion.models.sana_wm.sana_wm_transformer import _sequence_parallel_plan_available

    if not _sequence_parallel_plan_available():
        assert SanaWmTransformer3DModel._sp_plan is None
        return

    assert SanaWmTransformer3DModel._sp_plan is not None
    assert "blocks.0" in SanaWmTransformer3DModel._sp_plan
    assert "final_layer" in SanaWmTransformer3DModel._sp_plan
    assert "hidden_states" in SanaWmTransformer3DModel._sp_plan["blocks.0"]
    assert "camera_hidden_states" in SanaWmTransformer3DModel._sp_plan["blocks.0"]


def test_sana_wm_pipeline_passes_quant_config_to_stage1_transformer() -> None:
    from vllm_omni.diffusion.models.sana_wm import SanaWmPipeline

    quant_config = object()
    pipeline = SanaWmPipeline(od_config=SimpleNamespace(model=None, quantization_config=quant_config))

    assert pipeline.quant_config is quant_config
    assert pipeline.transformer.quant_config is quant_config


def test_sana_wm_transformer_limits_plucker_post_blocks() -> None:
    from vllm_omni.diffusion.models.sana_wm import SanaWmConfig, SanaWmTransformer3DModel

    config = SanaWmConfig(
        num_blocks=3,
        hidden_size=8,
        linear_head_dim=4,
        mlp_ratio=1,
        chunk_plucker_post_attn_blocks=1,
    )
    model = SanaWmTransformer3DModel(config=config, materialize=True)

    assert [block.plucker_proj is not None for block in model.blocks] == [True, False, False]


def test_sana_wm_gdn_reference_forward_shape() -> None:
    import torch

    from vllm_omni.diffusion.models.sana_wm import BidirectionalGatedDeltaNetTriton

    op = BidirectionalGatedDeltaNetTriton()
    query = torch.rand(1, 2, 4, 6) + 0.1
    key = torch.rand(1, 2, 4, 6) + 0.1
    value = torch.rand(1, 2, 4, 6)
    beta = torch.full((1, 2, 3, 2), 0.5)
    decay = torch.full((1, 2, 3), 0.9)

    output = op(query, key, value, beta=beta, decay=decay, spatial_tokens=2)

    assert output.shape == query.shape
    assert torch.isfinite(output).all()


def test_sana_wm_download_patterns() -> None:
    from vllm_omni.diffusion.models.sana_wm import build_sana_wm_download_patterns

    stage1 = build_sana_wm_download_patterns(include_refiner=False)
    assert stage1 == (
        "config.yaml",
        "dit/sana_wm_1600m_720p.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    )

    full = build_sana_wm_download_patterns(include_refiner=True)
    assert "refiner/transformer/diffusion_pytorch_model.safetensors" in full
    assert "refiner/connectors/diffusion_pytorch_model.safetensors" in full
    assert "refiner/text_encoder/*" in full


def test_sana_wm_local_path_validation(tmp_path) -> None:
    from vllm_omni.diffusion.models.sana_wm import resolve_sana_wm_local_paths, validate_sana_wm_local_paths

    for path in [
        "config.yaml",
        "dit/sana_wm_1600m_720p.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
        "refiner/transformer/config.json",
        "refiner/transformer/diffusion_pytorch_model.safetensors",
        "refiner/connectors/config.json",
        "refiner/connectors/diffusion_pytorch_model.safetensors",
        "refiner/text_encoder/config.json",
        "refiner/text_encoder/model.safetensors.index.json",
    ]:
        file_path = tmp_path / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch()

    paths = resolve_sana_wm_local_paths(tmp_path)
    validate_sana_wm_local_paths(paths, include_refiner=True)
    assert paths.stage1_dit == tmp_path / "dit/sana_wm_1600m_720p.safetensors"


def test_sana_wm_local_path_validation_reports_missing_refiner(tmp_path) -> None:
    from vllm_omni.diffusion.models.sana_wm import resolve_sana_wm_local_paths, validate_sana_wm_local_paths

    for path in [
        "config.yaml",
        "dit/sana_wm_1600m_720p.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ]:
        file_path = tmp_path / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch()

    paths = resolve_sana_wm_local_paths(tmp_path)
    validate_sana_wm_local_paths(paths, include_refiner=False)
    with pytest.raises(FileNotFoundError, match="refiner/transformer"):
        validate_sana_wm_local_paths(paths, include_refiner=True)


def test_sana_wm_weight_mapping() -> None:
    from vllm_omni.diffusion.models.sana_wm import normalize_sana_wm_stage1_weight_name

    assert normalize_sana_wm_stage1_weight_name("pos_embed") == "transformer.pos_embed"
    assert normalize_sana_wm_stage1_weight_name("blocks.0.attn.A_log") == "transformer.blocks.0.attn.A_log"
    assert normalize_sana_wm_stage1_weight_name("blocks.0.scale_shift_table") == (
        "transformer.blocks.0.scale_shift_table"
    )
    assert (
        normalize_sana_wm_stage1_weight_name("blocks.3.plucker_post.weight")
        == "transformer.blocks.3.plucker_proj.weight"
    )
    assert normalize_sana_wm_stage1_weight_name("y_embedder.y_proj.fc1.weight") == (
        "transformer.y_embedder.y_proj.fc1.weight"
    )
    assert normalize_sana_wm_stage1_weight_name("x_embedder.proj.weight") == "transformer.x_embedder.proj.weight"
    assert normalize_sana_wm_stage1_weight_name("plucker_embedder.proj.weight") == (
        "transformer.plucker_embedder.proj.weight"
    )
    assert normalize_sana_wm_stage1_weight_name("raymap_embedder.proj.weight") == (
        "transformer.raymap_embedder.proj.weight"
    )
    assert normalize_sana_wm_stage1_weight_name("unknown.weight") is None


def test_sana_wm_pipeline_fails_fast_when_executed() -> None:
    from vllm_omni.diffusion.models.sana_wm import SANA_WM_SCAFFOLD_ERROR, SanaWmPipeline, SanaWmTwoStagesPipeline

    req = SimpleNamespace(
        prompts=[{"multi_modal_data": {"image": object()}, "sana_wm": {"action": "w-1"}}],
        sampling_params=SimpleNamespace(),
    )
    for cls in (SanaWmPipeline, SanaWmTwoStagesPipeline):
        pipeline = cls(od_config=None)
        with pytest.raises(NotImplementedError, match="native Stage-1"):
            pipeline(req)

    assert "Sana-WM native Stage-1 DiT" in SANA_WM_SCAFFOLD_ERROR
    assert "official runner" in SANA_WM_SCAFFOLD_ERROR


def test_sana_wm_pipeline_native_smoke_opt_in_runs_small_latents() -> None:
    from vllm_omni.diffusion.models.sana_wm import SanaWmConfig, SanaWmPipeline

    config = SanaWmConfig(
        num_blocks=1,
        hidden_size=8,
        linear_head_dim=4,
        mlp_ratio=1,
        model_max_length=3,
        chunk_plucker_channels=48,
    )
    pipeline = SanaWmPipeline(od_config=None)
    pipeline.sana_wm_config = config
    pipeline.transformer.config = config
    req = SimpleNamespace(
        prompts=[
            {
                "prompt": "native smoke",
                "multi_modal_data": {"image": object()},
                "sana_wm": {"action": "w-1", "num_frames": 1, "height": 64, "width": 64},
            }
        ],
        sampling_params=SimpleNamespace(
            height=64,
            width=64,
            num_frames=1,
            num_inference_steps=1,
            seed=0,
            extra_args={"sana_wm_native_smoke": True, "sana_wm_hash_prompt_smoke": True},
        ),
    )

    output = pipeline(req)

    assert output.output.shape == (1, 128, 1, 2, 2)
    assert output.custom_output["sana_wm_backend"] == "native_gdn_smoke"
    assert output.custom_output["sana_wm_output_space"] == "latent"
    assert output.custom_output["sana_wm_prompt_source"] == "hash_smoke"
    assert output.custom_output["sana_wm_chi_prompt_applied"] is False


def test_sana_wm_native_smoke_prompt_embeddings_are_deterministic() -> None:
    import torch

    from vllm_omni.diffusion.models.sana_wm import SanaWmConfig, SanaWmPipeline

    pipeline = SanaWmPipeline(od_config=None)
    pipeline.sana_wm_config = SanaWmConfig(model_max_length=2, chi_prompt=["enhance prompt"])

    first, source = pipeline._native_smoke_prompt_embeds(
        {"prompt": "drive forward"},
        device=torch.device("cpu"),
        dtype=torch.float32,
        allow_hash_fallback=True,
    )
    second, _ = pipeline._native_smoke_prompt_embeds(
        {"prompt": "drive forward"},
        device=torch.device("cpu"),
        dtype=torch.float32,
        allow_hash_fallback=True,
    )
    different, _ = pipeline._native_smoke_prompt_embeds(
        {"prompt": "turn left"},
        device=torch.device("cpu"),
        dtype=torch.float32,
        allow_hash_fallback=True,
    )

    assert source == "hash_smoke"
    assert torch.equal(first, second)
    assert not torch.equal(first, different)


def test_sana_wm_native_smoke_prompt_embeddings_can_use_real_encoder(monkeypatch) -> None:
    import sys
    import types

    import torch

    from vllm_omni.diffusion.models.sana_wm import SanaWmConfig, SanaWmPipeline

    class FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"

        def __call__(self, *args, **kwargs):
            class FakeBatch(dict):
                def to(self, device):
                    return self

            return FakeBatch(input_ids=torch.zeros(1, 2, dtype=torch.long))

    class FakeModel(torch.nn.Module):
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def forward(self, *args, **kwargs):
            return types.SimpleNamespace(hidden_states=[torch.ones(1, 2, 2304)])

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeTokenizer())
    fake_transformers.AutoModelForCausalLM = FakeModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    pipeline = SanaWmPipeline(od_config=None)
    pipeline.sana_wm_config = SanaWmConfig(model_max_length=2, chi_prompt=["enhance prompt"])

    embeds, source = pipeline._native_smoke_prompt_embeds(
        {"prompt": "drive forward"},
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert source == "gemma2"
    assert embeds.shape == (1, 2, 2304)


def test_sana_wm_official_backend_env_skips_stage1_weight_source(monkeypatch, tmp_path) -> None:
    from vllm_omni.diffusion.models.sana_wm import SANA_WM_OFFICIAL_REPO_ENV, SanaWmPipeline

    monkeypatch.setenv(SANA_WM_OFFICIAL_REPO_ENV, str(tmp_path))
    pipeline = SanaWmPipeline(
        od_config=SimpleNamespace(model="Efficient-Large-Model/SANA-WM_bidirectional", revision="main")
    )

    assert pipeline.use_official_backend is True
    assert pipeline.weights_sources == []
    assert list(pipeline.named_parameters()) == []
    assert pipeline.load_weights([("pos_embed", object())]) == set()


def test_sana_wm_official_backend_command_builds_release_cli(tmp_path) -> None:
    from vllm_omni.diffusion.models.sana_wm import build_sana_wm_official_command

    paths = SimpleNamespace(
        root=tmp_path,
        config=tmp_path / "config.yaml",
        stage1_dit=tmp_path / "dit/sana_wm_1600m_720p.safetensors",
        refiner_root=tmp_path / "refiner",
        refiner_text_encoder_dir=tmp_path / "refiner/text_encoder",
    )

    cmd = build_sana_wm_official_command(
        script=tmp_path / "Sana/inference_video_scripts/inference_sana_wm.py",
        image_path=tmp_path / "input.png",
        prompt_path=tmp_path / "prompt.txt",
        output_dir=tmp_path / "out",
        num_frames=17,
        release_paths=paths,
        include_refiner=False,
        action="w-4",
        translation_speed=0.055,
        rotation_speed_deg=1.2,
        python_executable="python",
    )

    assert cmd[:2] == ["python", str(tmp_path / "Sana/inference_video_scripts/inference_sana_wm.py")]
    assert "--action" in cmd
    assert "w-4" in cmd
    assert "--no_refiner" in cmd
    assert "--config" in cmd
    assert str(paths.config) in cmd
    assert "--model_path" in cmd
    assert str(paths.stage1_dit) in cmd

    refiner_cmd = build_sana_wm_official_command(
        script=tmp_path / "Sana/inference_video_scripts/inference_sana_wm.py",
        image_path=tmp_path / "input.png",
        prompt_path=tmp_path / "prompt.txt",
        output_dir=tmp_path / "out",
        num_frames=17,
        release_paths=paths,
        include_refiner=True,
        action="w-4",
        python_executable="python",
    )
    assert "--refiner_root" in refiner_cmd
    assert str(paths.root / "refiner") in refiner_cmd
    assert "--refiner_gemma_root" in refiner_cmd
    assert str(paths.refiner_text_encoder_dir) in refiner_cmd
    assert "--refiner_checkpoint" not in refiner_cmd


def test_sana_wm_refiner_root_override(tmp_path) -> None:
    from vllm_omni.diffusion.models.sana_wm import resolve_sana_wm_local_paths

    model_root = tmp_path / "model"
    refiner_root = tmp_path / "official-refiner"

    paths = resolve_sana_wm_local_paths(model_root, refiner_root=refiner_root)

    assert paths.root == model_root
    assert paths.refiner_root == refiner_root
    assert paths.refiner_transformer_config == refiner_root / "transformer/config.json"
    assert paths.refiner_connectors_weights == refiner_root / "connectors/diffusion_pytorch_model.safetensors"
    assert paths.refiner_text_encoder_dir == refiner_root / "text_encoder"


def test_sana_wm_action_rollout_and_plucker_shapes() -> None:
    from vllm_omni.diffusion.models.sana_wm import SanaWmCameraCondition
    from vllm_omni.diffusion.models.sana_wm.camera_control import action_string_to_c2w, build_plucker_condition

    c2w = action_string_to_c2w("w-16", translation_speed=0.055, rotation_speed_deg=1.2)
    assert c2w.shape == (17, 4, 4)
    assert c2w[0, 2, 3] == 0.0
    assert c2w[-1, 2, 3] > 0.8

    camera = build_plucker_condition(
        SanaWmCameraCondition(
            action="w-16",
            intrinsics={"fx": 1000.0, "fy": 1000.0, "cx": 640.0, "cy": 352.0},
            num_frames=17,
            height=704,
            width=1280,
        )
    )
    assert camera["raymap"].shape == (3, 20)
    assert camera["chunk_plucker"].shape == (48, 3, 22, 40)


def test_sana_wm_native_backend_script_resolution(tmp_path) -> None:
    from vllm_omni.diffusion.models.sana_wm import (
        find_sana_wm_native_script,
        require_sana_wm_native_script,
    )

    script = tmp_path / "inference_video_scripts/inference_sana_wm.py"
    script.parent.mkdir(parents=True)
    script.touch()

    assert find_sana_wm_native_script(tmp_path) == script
    assert require_sana_wm_native_script(tmp_path) == script


def test_sana_wm_stage1_weight_loads_with_remap() -> None:
    import torch

    from vllm_omni.diffusion.models.sana_wm import SanaWmPipeline

    pipeline = SanaWmPipeline(od_config=None)
    loaded = pipeline.load_weights(
        [
            ("pos_embed", torch.zeros(1, 2)),
            ("blocks.0.attn.A_log", torch.zeros(20)),
            ("plucker_embedder.proj.weight", torch.zeros(2, 2)),
        ]
    )

    assert "transformer.pos_embed" in loaded
    assert "transformer.blocks.0.attn.A_log" in loaded
    assert "transformer.plucker_embedder.proj.weight" in loaded
    assert pipeline.transformer.last_load_report.total_weights == 3
    assert pipeline.transformer.last_load_report.loaded_weights == 3
    assert pipeline.transformer.get_loaded_tensor("transformer.pos_embed").shape == (1, 2)


def test_sana_wm_stage1_weight_audit_materializes_cached_weights() -> None:
    import torch

    from vllm_omni.diffusion.models.sana_wm import SanaWmConfig, SanaWmTransformer3DModel

    config = SanaWmConfig(num_blocks=1, hidden_size=8, linear_head_dim=4, mlp_ratio=1, model_max_length=3)
    model = SanaWmTransformer3DModel(config=config)

    model.load_weights([("x_embedder.proj.bias", torch.ones(8))])
    model.materialize(latent_channels=4, prompt_channels=6)

    assert torch.equal(model.x_embedder.proj.bias, torch.ones_like(model.x_embedder.proj.bias))
    assert model.last_load_report.materialized_weights == 1
    assert model.last_load_report.unapplied_weights == ()


def test_sana_wm_stage1_weight_audit_rejects_unconsumed_remapped_keys() -> None:
    import torch

    from vllm_omni.diffusion.models.sana_wm import SanaWmConfig, SanaWmTransformer3DModel

    config = SanaWmConfig(num_blocks=1, hidden_size=8, linear_head_dim=4, mlp_ratio=1, model_max_length=3)
    model = SanaWmTransformer3DModel(config=config)
    model.load_weights([("blocks.0.attn.no_such_weight", torch.zeros(1))])

    with pytest.raises(ValueError, match="not consumed"):
        model.materialize(latent_channels=4, prompt_channels=6)

    assert model.last_load_report.unapplied_weights == ("transformer.blocks.0.attn.no_such_weight",)


def test_sana_wm_stage1_weight_loader_rejects_unmapped_keys() -> None:
    import torch

    from vllm_omni.diffusion.models.sana_wm import SanaWmPipeline

    pipeline = SanaWmPipeline(od_config=None)
    with pytest.raises(ValueError, match="unmapped"):
        pipeline.load_weights([("unexpected.weight", torch.zeros(1))])


def test_sana_wm_diffusion_pre_process_normalizes_prompts() -> None:
    from vllm_omni.diffusion.models.sana_wm import SANA_WM_DEFAULT_NUM_FRAMES, get_sana_wm_pre_process_func

    image = object()
    sampling_params = SimpleNamespace(height=None, width=None, num_frames=1)
    request = SimpleNamespace(
        sampling_params=sampling_params,
        prompts=[
            {
                "multi_modal_data": {"image": image},
                "sana_wm": {"action": "w-1", "translation_speed": 0.055, "rotation_speed_deg": 1.2},
            }
        ]
    )

    result = get_sana_wm_pre_process_func(SimpleNamespace())(request)
    payload = result.prompts[0]["additional_information"]["sana_wm"]

    assert result.prompts[0]["multi_modal_data"]["image"] is image
    assert payload["num_frames"] == SANA_WM_DEFAULT_NUM_FRAMES
    assert payload["action"] == "w-1"
    assert payload["translation_speed"] == 0.055
    assert payload["rotation_speed_deg"] == 1.2
    assert sampling_params.height == 704
    assert sampling_params.width == 1280
    assert sampling_params.num_frames == SANA_WM_DEFAULT_NUM_FRAMES


def test_sana_wm_registry_lazy_loads_classes() -> None:
    from vllm_omni.diffusion.registry import DiffusionModelRegistry

    for name in ("SanaWmPipeline", "SanaWmTwoStagesPipeline"):
        cls = DiffusionModelRegistry._try_load_model_cls(name)
        assert cls is not None
        assert cls.__name__ == name


def test_sana_wm_checkpoint_resolution_uses_od_config_model(tmp_path) -> None:
    from vllm_omni.diffusion.models.sana_wm import SanaWmPipeline

    for path in [
        "config.yaml",
        "dit/sana_wm_1600m_720p.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ]:
        file_path = tmp_path / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch()
    _write_config(tmp_path / "config.yaml")

    pipeline = SanaWmPipeline(od_config=SimpleNamespace(model=str(tmp_path), revision=None))
    paths = pipeline.resolve_checkpoint(include_refiner=False)
    assert paths.root == tmp_path
    assert pipeline.sana_wm_config is not None
    assert pipeline.sana_wm_config.num_blocks == 20


def test_sana_wm_layout_uses_default_diffusion_stage_config(tmp_path) -> None:
    from vllm_omni.entrypoints.utils import resolve_model_config_path

    (tmp_path / "dit").mkdir()
    (tmp_path / "config.yaml").write_text("model:\n  model: SanaMSVideoCamCtrl_1600M_P1_D20\n", encoding="utf-8")
    (tmp_path / "dit" / "sana_wm_1600m_720p.safetensors").touch()

    assert resolve_model_config_path(str(tmp_path)) is None


def test_sana_wm_pipeline_declares_stage1_weight_source() -> None:
    from vllm_omni.diffusion.models.sana_wm import SanaWmPipeline

    pipeline = SanaWmPipeline(
        od_config=SimpleNamespace(model="Efficient-Large-Model/SANA-WM_bidirectional", revision="main")
    )

    assert len(pipeline.weights_sources) == 1
    source = pipeline.weights_sources[0]
    assert source.subfolder == "dit"
    assert source.prefix == ""
    assert source.fall_back_to_pt is False
    assert source.allow_patterns_overrides == ["sana_wm_1600m_720p.safetensors"]


def test_sana_wm_two_stage_exposes_refiner_loader_surface() -> None:
    from vllm_omni.diffusion.models.sana_wm import SanaWmTwoStagesPipeline

    pipeline = SanaWmTwoStagesPipeline(od_config=None)

    assert callable(pipeline.ensure_refiner_components)
    assert pipeline.refiner_transformer is None
    assert pipeline.refiner_text_encoder is None
    assert pipeline.refiner_connectors is None
    assert pipeline.refiner_tokenizer is None


# ---------------------------------------------------------------------------
# TP-layer completeness tests (5a) — no GPU required
# ---------------------------------------------------------------------------


def test_sana_wm_transformer_all_new_sites_fallback_to_linear() -> None:
    """When use_vllm_parallel_layers=False every new parallel site is nn.Linear."""
    import torch

    from vllm_omni.diffusion.models.sana_wm import SanaWmConfig, SanaWmTransformer3DModel

    config = SanaWmConfig(
        num_blocks=2,
        hidden_size=8,
        linear_head_dim=4,
        mlp_ratio=1,
        model_max_length=3,
        chunk_plucker_post_attn_blocks=1,  # block 0 has plucker_proj, block 1 does not
    )
    model = SanaWmTransformer3DModel(config=config, materialize=True, use_vllm_parallel_layers=False)
    b0 = model.blocks[0]
    b1 = model.blocks[1]

    # camera attention projections
    assert isinstance(b0.attn.q_proj_cam, torch.nn.Linear), "q_proj_cam must be nn.Linear in fallback"
    assert isinstance(b0.attn.k_proj_cam, torch.nn.Linear), "k_proj_cam must be nn.Linear in fallback"
    assert isinstance(b0.attn.v_proj_cam, torch.nn.Linear), "v_proj_cam must be nn.Linear in fallback"
    assert isinstance(b0.attn.out_proj_cam, torch.nn.Linear), "out_proj_cam must be nn.Linear in fallback"
    # shapes match config
    assert b0.attn.q_proj_cam.in_features == config.hidden_size
    assert b0.attn.out_proj_cam.out_features == config.hidden_size

    # plucker_proj
    assert isinstance(b0.plucker_proj, torch.nn.Linear), "plucker_proj must be nn.Linear in fallback"
    assert b1.plucker_proj is None, "block beyond chunk_plucker_post_attn_blocks must have None"

    # final layer output head
    assert isinstance(model.final_layer.linear, torch.nn.Linear), "final_layer.linear must be nn.Linear in fallback"

    # t_block Sequential[1]
    assert isinstance(model.t_block[1], torch.nn.Linear), "t_block[1] must be nn.Linear in fallback"
    assert model.t_block[1].in_features == config.hidden_size
    assert model.t_block[1].out_features == 6 * config.hidden_size


def test_sana_wm_final_layer_shape_in_both_modes() -> None:
    """SanaWmFinalLayer produces the correct output tensor shape in both TP and fallback modes."""
    import torch

    from vllm_omni.diffusion.models.sana_wm.sana_wm_transformer import SanaWmFinalLayer

    hidden_size = 16
    patch_size = (1, 2, 2)
    out_channels = 8
    batch, seq = 2, 10
    expected_out_features = 1 * 2 * 2 * out_channels  # = 32

    for use_parallel in (True, False):
        layer = SanaWmFinalLayer(
            hidden_size, patch_size, out_channels, use_vllm_parallel_layers=use_parallel
        )
        hidden = torch.randn(batch, seq, hidden_size)
        t_embed = torch.randn(batch, hidden_size)
        out = layer(hidden, t_embed)
        assert out.shape == (batch, seq, expected_out_features), (
            f"use_vllm_parallel_layers={use_parallel}: got {out.shape}"
        )
        assert torch.isfinite(out).all()


def test_sana_wm_t_block_indexing_unwraps_parallel_output() -> None:
    """_linear_output correctly handles plain tensors and (tensor, None) tuples.

    This guards the t_block[1] indexing pattern introduced to support
    ColumnParallelLinear inside nn.Sequential.
    """
    import torch

    from vllm_omni.diffusion.models.sana_wm.sana_wm_transformer import _linear_output

    tensor = torch.randn(2, 8)
    # plain tensor passes through unchanged
    assert _linear_output(tensor) is tensor
    # (output, None) tuple — as ColumnParallelLinear returns — unwraps to output
    assert _linear_output((tensor, None)) is tensor
    # (output, bias) tuple also unwraps to output
    bias = torch.zeros(8)
    result = _linear_output((tensor, bias))
    assert result is tensor


def test_sana_wm_forward_shape_consistent_across_layer_modes() -> None:
    """Full forward produces the same output shape regardless of parallel-layer mode."""
    import torch

    from vllm_omni.diffusion.models.sana_wm import SanaWmConfig, SanaWmTransformer3DModel

    config = SanaWmConfig(
        num_blocks=2,
        hidden_size=8,
        linear_head_dim=4,
        mlp_ratio=1,
        model_max_length=3,
        chunk_plucker_channels=6,
        softmax_every_n=2,  # block 0 GDN, block 1 softmax — exercises both paths
    )
    latents = torch.randn(1, 4, 1, 4, 4)
    enc = torch.randn(1, 3, 6)
    plucker = torch.randn(1, 6, 1, 4, 4)

    for use_parallel in (True, False):
        model = SanaWmTransformer3DModel(config=config, materialize=True, use_vllm_parallel_layers=use_parallel)
        out = model(latents, torch.tensor([0.5]), encoder_hidden_states=enc, plucker=plucker)
        assert out.shape == latents.shape, (
            f"use_vllm_parallel_layers={use_parallel}: shape {out.shape} != {latents.shape}"
        )
        assert torch.isfinite(out).all(), f"non-finite output at use_vllm_parallel_layers={use_parallel}"
