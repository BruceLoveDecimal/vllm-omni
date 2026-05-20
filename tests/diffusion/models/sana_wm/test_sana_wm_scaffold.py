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
        SANA_WM_GDN_ERROR,
        SANA_WM_MODEL_ID,
        SANA_WM_OFFICIAL_BACKEND_ERROR,
        SANA_WM_OFFICIAL_REPO_ENV,
        SANA_WM_OFFICIAL_SCRIPT,
        SANA_WM_OUTPUT_HEIGHT,
        SANA_WM_OUTPUT_WIDTH,
        SANA_WM_REFINER_CONNECTORS_WEIGHT_FILE,
        SANA_WM_REFINER_TRANSFORMER_WEIGHT_FILE,
        SANA_WM_STAGE1_DIT_FILE,
        SANA_WM_STAGE1_TEXT_ENCODER_ID,
        SANA_WM_TRANSFORMER_FORWARD_ERROR,
        SANA_WM_VAE_WEIGHT_FILE,
        BidirectionalGatedDeltaNetTriton,
        SanaWmConfig,
        SanaWmPipeline,
        SanaWmTransformer3DModel,
        SanaWmTwoStagesPipeline,
        build_sana_wm_official_command,
    )

    assert SANA_WM_MODEL_ID == "Efficient-Large-Model/SANA-WM_bidirectional"
    assert SANA_WM_STAGE1_DIT_FILE == "dit/sana_wm_1600m_720p.safetensors"
    assert SANA_WM_VAE_WEIGHT_FILE == "vae/diffusion_pytorch_model.safetensors"
    assert SANA_WM_REFINER_TRANSFORMER_WEIGHT_FILE == "refiner/transformer/diffusion_pytorch_model.safetensors"
    assert SANA_WM_REFINER_CONNECTORS_WEIGHT_FILE == "refiner/connectors/diffusion_pytorch_model.safetensors"
    assert SANA_WM_STAGE1_TEXT_ENCODER_ID == "google/gemma-2-2b-it"
    assert SANA_WM_OUTPUT_HEIGHT == 704
    assert SANA_WM_OUTPUT_WIDTH == 1280
    assert SANA_WM_DEFAULT_NUM_FRAMES == 321
    assert SanaWmPipeline.support_image_input is True
    assert SanaWmPipeline._dit_modules == ["transformer"]
    assert SanaWmPipeline._encoder_modules == ["text_encoder"]
    assert SanaWmTwoStagesPipeline.support_image_input is True
    assert SanaWmTwoStagesPipeline._dit_modules == ["transformer"]
    assert SanaWmTwoStagesPipeline._encoder_modules == [
        "text_encoder",
        "refiner_text_encoder",
        "refiner_connectors",
    ]
    assert SanaWmTwoStagesPipeline._resident_modules == ["refiner_transformer"]
    assert SanaWmTransformer3DModel._repeated_blocks == ["blocks"]
    assert SanaWmConfig().num_blocks == 20
    assert "Gated DeltaNet" in SANA_WM_GDN_ERROR
    assert "forward is not implemented" in SANA_WM_TRANSFORMER_FORWARD_ERROR
    assert "official runner" in SANA_WM_OFFICIAL_BACKEND_ERROR
    assert SANA_WM_OFFICIAL_REPO_ENV == "VLLM_OMNI_SANA_WM_OFFICIAL_REPO"
    assert SANA_WM_OFFICIAL_SCRIPT == "inference_video_scripts/inference_sana_wm.py"
    assert BidirectionalGatedDeltaNetTriton().__class__.__name__ == "BidirectionalGatedDeltaNetTriton"
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
    assert config.scheduler_type == "flow_dpm-solver"
    assert config.inference_flow_shift == 9.8
    assert config.chi_prompt == ["enhance prompt"]
    assert config.y_norm_scale_factor == 0.01
    assert config.model_max_length == 300


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
    assert (
        normalize_sana_wm_stage1_weight_name("blocks.0.attn.A_log")
        == "transformer.blocks.0.attention.A_log"
    )
    assert (
        normalize_sana_wm_stage1_weight_name("blocks.3.plucker_post.weight")
        == "transformer.blocks.3.camera_post.weight"
    )
    assert normalize_sana_wm_stage1_weight_name("y_embedder.linear.weight") == (
        "transformer.text_embedder.linear.weight"
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


def test_sana_wm_official_backend_env_skips_stage1_weight_source(monkeypatch, tmp_path) -> None:
    from vllm_omni.diffusion.models.sana_wm import SANA_WM_OFFICIAL_REPO_ENV, SanaWmPipeline

    monkeypatch.setenv(SANA_WM_OFFICIAL_REPO_ENV, str(tmp_path))
    pipeline = SanaWmPipeline(
        od_config=SimpleNamespace(model="Efficient-Large-Model/SANA-WM_bidirectional", revision="main")
    )

    assert pipeline.use_official_backend is True
    assert pipeline.weights_sources == []
    assert pipeline.load_weights([("pos_embed", object())]) == set()


def test_sana_wm_official_backend_command_builds_release_cli(tmp_path) -> None:
    from vllm_omni.diffusion.models.sana_wm import build_sana_wm_official_command

    paths = SimpleNamespace(
        root=tmp_path,
        config=tmp_path / "config.yaml",
        stage1_dit=tmp_path / "dit/sana_wm_1600m_720p.safetensors",
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
    assert "transformer.blocks.0.attention.A_log" in loaded
    assert "camera_encoder.plucker.proj.weight" in loaded
    assert pipeline.transformer.last_load_report.total_weights == 3
    assert pipeline.transformer.last_load_report.loaded_weights == 3
    assert pipeline.transformer.get_loaded_tensor("transformer.pos_embed").shape == (1, 2)


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
