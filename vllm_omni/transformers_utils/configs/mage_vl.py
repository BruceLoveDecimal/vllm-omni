# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mage-VL configuration.

Vendored from ``microsoft/Mage-VL``'s ``configuration_mage_vl.py`` so the model can be
served without ``trust_remote_code``. The upstream file declares its fields through
``huggingface_hub.dataclasses.strict``, which pins it to the transformers release the
checkpoint shipped against; this copy uses a plain :class:`PretrainedConfig` so it stays
loadable across the transformers range vLLM-Omni supports.

Field names, defaults, and the ``__post_init__`` behaviour (resolving ``text_config``
through ``CONFIG_MAPPING`` and mirroring generation token ids to the top level) are kept
identical to the checkpoint's own class.
"""

from transformers import CONFIG_MAPPING
from transformers.configuration_utils import PretrainedConfig


class MageVLVisionConfig(PretrainedConfig):
    """Configuration for the Mage-ViT visual encoder."""

    model_type = "mage_vl_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        num_channels: int = 3,
        image_size: int = 448,
        patch_size: int = 14,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-6,
        layer_norm_type: str = "layer_norm",
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        rope_theta: float = 10000.0,
        use_head: bool = False,
        out_hidden_size: int = 1024,
        spatial_merge_size: int = 2,
        # Declared explicitly because Qwen2-VL-derived processing helpers read it off the
        # vision config; the released checkpoint ships temporal_patch_size=1.
        temporal_patch_size: int = 1,
        tokens_per_second: int = 1,
        frame_windows_size: int = 4,
        use_patch_position_encoding: bool = False,
        patch_position_encoding_type: str = "absolute",
        max_position_embeddings: int = 8192,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.layer_norm_type = layer_norm_type
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.rope_theta = rope_theta
        self.use_head = use_head
        self.out_hidden_size = out_hidden_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.tokens_per_second = tokens_per_second
        self.frame_windows_size = frame_windows_size
        self.use_patch_position_encoding = use_patch_position_encoding
        self.patch_position_encoding_type = patch_position_encoding_type
        self.max_position_embeddings = max_position_embeddings
        super().__init__(**kwargs)


class MageVLConfig(PretrainedConfig):
    """Top-level Mage-VL configuration: Mage-ViT vision tower + Qwen3 text decoder."""

    model_type = "mage_vl"
    sub_configs = {"vision_config": MageVLVisionConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id: int = 151655,
        video_token_id: int = 151656,
        vision_start_token_id: int = 151652,
        vision_end_token_id: int = 151653,
        tie_word_embeddings: bool = False,
        bos_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        pad_token_id: int | None = None,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = MageVLVisionConfig(**vision_config)
        elif vision_config is None:
            self.vision_config = MageVLVisionConfig()
        else:
            self.vision_config = vision_config

        # ``text_config`` is resolved dynamically by its own ``model_type`` (Qwen3 for
        # the released checkpoint) so a future Mage-VL variant with a different backbone
        # keeps loading without a config change here.
        if isinstance(text_config, dict):
            text_model_type = text_config.get("model_type", "qwen3")
            text_config = dict(text_config)
            text_config["model_type"] = text_model_type
            text_config_cls = CONFIG_MAPPING[text_model_type]
            self.text_config = text_config_cls(**text_config)
        elif text_config is None:
            self.text_config = CONFIG_MAPPING["qwen3"]()
        else:
            self.text_config = text_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id

        # Mirror generation token ids from ``text_config`` to the top level so callers
        # that read them from the outer config (generate, chat templates, vLLM) work.
        for tok_key, value in (
            ("bos_token_id", bos_token_id),
            ("eos_token_id", eos_token_id),
            ("pad_token_id", pad_token_id),
        ):
            if value is None:
                value = getattr(self.text_config, tok_key, None)
            kwargs.setdefault(tok_key, value)

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


__all__ = ["MageVLConfig", "MageVLVisionConfig"]
