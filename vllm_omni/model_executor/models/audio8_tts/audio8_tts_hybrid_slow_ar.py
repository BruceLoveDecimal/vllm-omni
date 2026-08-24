# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Audio8 TTS Preview 0.1b -- hybrid Slow AR model (Stage 0).

Same DualAR pipeline as the 0.6b (:mod:`audio8_tts_slow_ar`) with two
substitutions, which are exactly the three hooks this module overrides:

* **Backbone.** ``slow_backbone="falcon_h1"``: a hybrid Mamba2 + attention
  Falcon-H1, run on vLLM's upstream ``FalconH1Model``. Attention KV lives in
  paged blocks while the SSM state lives in fixed per-request slots, so the
  class declares ``IsHybrid``/``HasInnerState`` and vLLM's hybrid cache
  coordinator manages both.
* **Head.** The checkpoint ships a compact ``semantic_output`` of
  ``[codebook_size + 1, dim]`` (4096 semantic codes + EOS) instead of a
  full-vocabulary ``lm_head``.

Everything else -- multi-codebook input embedding, Repetition-Aware sampling,
the Fast AR, voice cloning, the delta streaming contract -- is inherited
unchanged.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.falcon_h1 import FalconH1ForCausalLM, FalconH1Model
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .audio8_tts_slow_ar import (
    Audio8TTSSlowARForConditionalGeneration,
    _remap_audio8_tts_weights,
)
from .configuration_audio8_tts import Audio8TTSHybridSlowARConfig

if TYPE_CHECKING:
    from vllm.model_executor.models.interfaces import MambaStateCopyFunc

logger = init_logger(__name__)


def _translate_slow_weight(name: str, tensor: torch.Tensor) -> tuple[str, torch.Tensor]:
    """Map one ``slow.*`` checkpoint tensor onto vLLM's Falcon-H1 names.

    The checkpoint's slow branch is a verbatim ``transformers.FalconH1Model``
    state dict behind a ``slow.`` prefix, so this reproduces the two
    substitutions in ``FalconH1ForCausalLM.hf_to_vllm_mapper``; ``q/k/v_proj``
    and ``gate/up_proj`` are then fused by the inherited stacked-parameter loop.
    """
    suffix = name[len("slow.") :]
    if ".mamba." in suffix:
        # vLLM nests MambaMixer2 one level deeper (FalconH1SSMDecoderLayer).
        suffix = suffix.replace(".mamba.", ".mamba.mamba.", 1)
        suffix = suffix.replace(".A_log", ".A")
    return f"model.{suffix}", tensor


def _remap_audio8_tts_hybrid_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    fast_q_size: int,
    fast_kv_size: int,
) -> Iterable[tuple[str, torch.Tensor]]:
    """Rename 0.1b checkpoint tensors to vLLM names.

    ``slow.*`` goes to the Falcon-H1 backbone; everything else (Fast AR,
    ``codebook_embeddings``, ``semantic_output``) is shaped exactly as on the
    0.6b and reuses that remapper.
    """
    for name, tensor in weights:
        if name.startswith("slow."):
            yield _translate_slow_weight(name, tensor)
            continue
        yield from _remap_audio8_tts_weights(
            ((name, tensor),),
            # The 0.1b has no unprefixed ``layers.`` block, so the Slow AR
            # split sizes are never consulted; only the Fast AR ones are.
            q_size=0,
            kv_size=0,
            fast_q_size=fast_q_size,
            fast_kv_size=fast_kv_size,
        )


class Audio8TTSHybridSlowARForConditionalGeneration(
    Audio8TTSSlowARForConditionalGeneration,
    HasInnerState,
    IsHybrid,
):
    """Stage 0 for the 0.1b: Falcon-H1 hybrid backbone + compact semantic head."""

    # ---------------- hybrid state cache (vLLM calls these on the class) ------
    #
    # vLLM passes the *top-level* VllmConfig, whose ``hf_config`` is the arktts
    # config rather than a FalconH1Config, so the shape variant reads the Slow
    # AR sub-config instead of delegating. Same split as
    # ``nemotron_voicechat_thinker.py``.

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig) -> tuple[torch.dtype, torch.dtype]:
        return FalconH1ForCausalLM.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig) -> tuple[tuple[int, int], tuple[int, int, int]]:
        from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator

        hf_config = vllm_config.model_config.hf_config.get_text_config()
        intermediate_size = (
            int(hf_config.mamba_expand * hf_config.hidden_size)
            if hf_config.mamba_d_ssm is None
            else int(hf_config.mamba_d_ssm)
        )
        return MambaStateShapeCalculator.mamba2_state_shape(
            intermediate_size=intermediate_size,
            tp_world_size=vllm_config.parallel_config.tensor_parallel_size,
            n_groups=hf_config.mamba_n_groups,
            num_heads=hf_config.mamba_n_heads,
            head_dim=hf_config.mamba_d_head,
            state_size=hf_config.mamba_d_state,
            conv_kernel=hf_config.mamba_d_conv,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return FalconH1ForCausalLM.get_mamba_state_copy_func()

    # ---------------- backbone / head hooks ----------------

    def _build_backbone(self, vllm_config: VllmConfig, prefix: str) -> None:
        """Build ``FalconH1Model`` against the Slow AR sub-config.

        ``FalconH1Model`` reads ``vllm_config.model_config.hf_config`` directly
        (unlike ``Qwen2Model``, which calls ``get_text_config()``), so the
        sub-config is swapped in on a shallow copy -- the same pattern as
        ``higgs_audio_v3_talker`` and ``moss_tts_talker``.
        """
        text_config = self.text_config
        if not isinstance(text_config, Audio8TTSHybridSlowARConfig):
            raise ValueError(
                "Audio8 TTS hybrid Slow AR requires a Falcon-H1 Slow AR sub-config "
                f"(slow_backbone='falcon_h1'); got {type(text_config).__name__}."
            )

        backbone_vllm_config = copy.copy(vllm_config)
        backbone_model_config = copy.copy(vllm_config.model_config)
        backbone_model_config.hf_config = text_config
        backbone_vllm_config.model_config = backbone_model_config

        self.model = FalconH1Model(
            vllm_config=backbone_vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        # Falcon-H1 RoPE is NeoX-style with theta 1e11, which is what both the
        # checkpoint and vLLM's default already use -- no restyling, unlike the
        # 0.6b's interleaved RoPE.

    def _build_output_head(self, vllm_config: VllmConfig, prefix: str) -> None:
        """Compact ``[codebook_size + 1, dim]`` head; no full-vocabulary lm_head.

        The reference applies no scaling here -- ``lm_head_multiplier`` belongs
        to the Falcon-H1 ``lm_head`` this checkpoint does not ship -- so the
        ``LogitsProcessor`` keeps the default scale of 1.0.
        """
        self._compact_vocab_size = self._codebook_size + 1
        if get_pp_group().is_last_rank:
            self.semantic_output = ParallelLMHead(
                self._compact_vocab_size,
                self.text_config.hidden_size,
                bias=False,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "semantic_output"),
            )
        else:
            self.semantic_output = PPMissingLayer()
        self.logits_processor = LogitsProcessor(self._compact_vocab_size)

    def _remap_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Iterable[tuple[str, torch.Tensor]]:
        fast_config = self.fast_ar_config
        return _remap_audio8_tts_hybrid_weights(
            weights,
            fast_q_size=fast_config.num_attention_heads * fast_config.head_dim,
            fast_kv_size=fast_config.num_key_value_heads * fast_config.head_dim,
        )

    # ---------------- logits ----------------

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> torch.Tensor | None:
        """Scatter the compact head onto full-vocabulary logits.

        The inherited sampler applies ``allowed_token_ids_mask`` and vLLM's
        logits processors in vocabulary space, and falls back to vLLM's own
        sampler for request mixes RAS does not model; both require real token
        ids. Widening here (~70k columns, still less than the 0.6b's ~156k)
        keeps that contract intact instead of forking the sampler.
        """
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None
        compact = self.logits_processor(self.semantic_output, hidden_states)
        if compact is None:
            return None

        vocab = int(self.text_config.vocab_size)
        logits = compact.new_full((compact.shape[0], vocab), float("-inf"))
        logits[:, self._semantic_begin_id : self._semantic_end_id + 1] = compact[:, : self._num_semantic_ids]
        logits[:, self._eos_token_id] = compact[:, self._num_semantic_ids]
        return logits


__all__ = ["Audio8TTSHybridSlowARForConditionalGeneration"]
