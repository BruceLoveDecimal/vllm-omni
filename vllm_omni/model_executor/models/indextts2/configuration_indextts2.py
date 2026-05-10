# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for the IndexTTS2 single-stage MVP wrapper."""

from transformers.configuration_utils import PretrainedConfig


class IndexTTS2Config(PretrainedConfig):
    """Minimal HF config used by vLLM-Omni's single-stage IndexTTS2 wrapper.

    The upstream checkpoint is driven mostly by ``config.yaml`` and loads its
    own submodules inside ``indextts.infer_v2.IndexTTS2``. vLLM still needs a
    text-like config surface for model registration, dummy profiling, and the
    AR scheduler interface, so this class exposes conservative defaults.
    """

    model_type = "indextts2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.hidden_size: int = getattr(self, "hidden_size", 1280)
        self.num_hidden_layers: int = getattr(self, "num_hidden_layers", 24)
        self.num_attention_heads: int = getattr(self, "num_attention_heads", 20)
        self.num_key_value_heads: int = getattr(self, "num_key_value_heads", self.num_attention_heads)
        self.head_dim: int = getattr(self, "head_dim", self.hidden_size // self.num_attention_heads)
        self.vocab_size: int = getattr(self, "vocab_size", 12000)
        self.max_position_embeddings: int = getattr(self, "max_position_embeddings", 4096)
        self.intermediate_size: int = getattr(self, "intermediate_size", self.hidden_size * 4)

        self.audio_sample_rate: int = getattr(self, "audio_sample_rate", 22050)
        self.start_mel_token: int = getattr(self, "start_mel_token", 8192)
        self.stop_mel_token: int = getattr(self, "stop_mel_token", 8193)
        self.number_mel_codes: int = getattr(self, "number_mel_codes", 8194)

        # Optional upstream runtime knobs. These can be set in config.json or
        # via --hf-overrides for local checkpoints that only ship config.yaml.
        self.indextts2_config_path: str | None = getattr(self, "indextts2_config_path", None)
        self.use_cuda_kernel: bool | None = getattr(self, "use_cuda_kernel", None)
        self.use_deepspeed: bool = getattr(self, "use_deepspeed", False)
        self.use_accel: bool = getattr(self, "use_accel", False)
        self.use_torch_compile: bool = getattr(self, "use_torch_compile", False)

        self.speculative_config = None

    def get_text_config(self, **kwargs):
        """Return self so vLLM uses the wrapper's top-level config."""
        return self
