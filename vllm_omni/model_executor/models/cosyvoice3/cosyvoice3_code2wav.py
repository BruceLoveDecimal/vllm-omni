# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CosyVoice3 Code2Wav Stage - Converts speech tokens to audio waveforms.

This module contains the code2wav (token-to-waveform) stage which uses:
1. DiT (Diffusion Transformer) with optimized attention backends
2. CFM (Conditional Flow Matching) for mel spectrogram generation
3. HiFiGAN vocoder for waveform synthesis
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from vllm.logger import init_logger

from vllm_omni.diffusion.models.cosyvoice3_audio.cosyvoice3_dit import DiT
from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.cfm import (
    CausalConditionalCFM,
    CausalMaskedDiffWithDiT,
)
from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.hifigan import (
    CausalConvRNNF0Predictor,
    CausalHiFTGenerator,
)
from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.layers import PreLookaheadLayer
from vllm_omni.transformers_utils.configs.cosyvoice3 import CosyVoice3Config

logger = init_logger(__name__)


# Above this mel length the batched vocoder loses to the per-request loop, so
# long rows fall back to one-at-a-time decoding. Batching pays off while the
# conv trunk is launch-bound (short rows) and stops paying once the trunk
# saturates memory bandwidth on its own, where padding and the larger working
# set only add cost.
#
# Measured on RTX PRO 6000 with f0 supplied by the streaming cache, speedup of
# batch over loop at batch 2/4/8:
#     768  frames  1.29x / 1.78x / 1.46x
#     1280 frames  1.02x / 1.29x / 1.11x
#     1408 frames  0.97x / 1.16x / 0.85x
#     2048 frames  1.09x / 0.67x / 0.72x
#     4096 frames  0.90x / 0.87x / 0.77x
# 1280 is the last length where no batch size regresses; it is ~26s of audio at
# the 50Hz mel rate, well beyond a typical request.
MAX_BATCHED_VOCODER_FRAMES = 1280


def plan_vocoder_buckets(lengths: list[int], waste_threshold: float = 1.3) -> list[list[int]]:
    """Group variable-length rows into padded batches with bounded pad waste.

    Returns buckets of indices into ``lengths``. Rows in a bucket are padded
    to the bucket max; a row is only added while padded volume stays within
    ``waste_threshold`` times the real volume, so one short row never rides
    along with much longer ones at full padded cost.
    """
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    buckets: list[list[int]] = []
    current: list[int] = []
    current_sum = 0
    for i in order:
        if current and lengths[current[0]] * (len(current) + 1) > waste_threshold * (current_sum + lengths[i]):
            buckets.append(current)
            current = []
            current_sum = 0
        current.append(i)
        current_sum += lengths[i]
    if current:
        buckets.append(current)
    return buckets


class CosyVoice3Code2Wav(nn.Module):
    """CosyVoice3 Code2Wav stage for token-to-waveform conversion.

    This class encapsulates:
    - Flow matching decoder with DiT backbone (using diffusion attention)
    - HiFiGAN vocoder for mel-to-waveform conversion
    """

    def __init__(self, config: CosyVoice3Config):
        super().__init__()
        self.config = config

        # Build flow matching components
        pre_lookahead_layer = PreLookaheadLayer(**config.flow["pre_lookahead_layer"])

        decoder_cfg = config.flow["decoder"]
        cfm_params = DictConfig(decoder_cfg["cfm_params"])

        # DiT estimator using diffusion attention (Flash/Sage/SDPA backends)
        estimator = DiT(**decoder_cfg["estimator"])

        decoder = CausalConditionalCFM(
            in_channels=decoder_cfg["in_channels"],
            estimator=estimator,
            cfm_params=cfm_params,
            n_spks=decoder_cfg["n_spks"],
            spk_emb_dim=decoder_cfg["spk_emb_dim"],
        )

        self.flow_model = CausalMaskedDiffWithDiT(
            input_size=config.flow["input_size"],
            output_size=config.flow["output_size"],
            spk_embed_dim=config.flow["spk_embed_dim"],
            output_type=config.flow["output_type"],
            vocab_size=config.flow["vocab_size"],
            input_frame_rate=config.flow["input_frame_rate"],
            only_mask_loss=config.flow["only_mask_loss"],
            token_mel_ratio=config.flow["token_mel_ratio"],
            pre_lookahead_len=config.flow["pre_lookahead_len"],
            pre_lookahead_layer=pre_lookahead_layer,
            decoder=decoder,
        )

        # Build HiFiGAN vocoder
        f0_predictor = CausalConvRNNF0Predictor(
            num_class=config.hift["f0_predictor"]["num_class"],
            in_channels=config.hift["f0_predictor"]["in_channels"],
            cond_channels=config.hift["f0_predictor"]["cond_channels"],
        )

        self.hift = CausalHiFTGenerator(
            in_channels=config.hift["in_channels"],
            base_channels=config.hift["base_channels"],
            nb_harmonics=config.hift["nb_harmonics"],
            sampling_rate=config.hift["sampling_rate"],
            nsf_alpha=config.hift["nsf_alpha"],
            nsf_sigma=config.hift["nsf_sigma"],
            nsf_voiced_threshold=config.hift["nsf_voiced_threshold"],
            upsample_rates=config.hift["upsample_rates"],
            upsample_kernel_sizes=config.hift["upsample_kernel_sizes"],
            istft_params=config.hift["istft_params"],
            resblock_kernel_sizes=config.hift["resblock_kernel_sizes"],
            resblock_dilation_sizes=config.hift["resblock_dilation_sizes"],
            source_resblock_kernel_sizes=config.hift["source_resblock_kernel_sizes"],
            source_resblock_dilation_sizes=config.hift["source_resblock_dilation_sizes"],
            lrelu_slope=config.hift["lrelu_slope"],
            audio_limit=config.hift["audio_limit"],
            conv_pre_look_right=config.hift["conv_pre_look_right"],
            f0_predictor=f0_predictor,
        )
        # Run hift in float32 to avoid dtype mismatches in internal ops
        self.hift = self.hift.float()

        # Streaming/chunking parameters
        self.token_overlap_len = 20
        self.mel_overlap_len = int(self.token_overlap_len / self.flow_model.input_frame_rate * 22050 / 256)
        self.mel_window = np.hamming(2 * self.mel_overlap_len)
        self.mel_cache_len = 20
        self.source_cache_len = int(self.mel_cache_len * 256)
        self.speech_window = np.hamming(2 * self.source_cache_len)

    @property
    def input_frame_rate(self) -> int:
        """Input frame rate from flow model."""
        return self.flow_model.input_frame_rate

    @property
    def token_mel_ratio(self) -> int:
        """Token to mel ratio."""
        return self.flow_model.token_mel_ratio

    @property
    def output_size(self) -> int:
        """Output mel dimension."""
        return self.flow_model.output_size

    @property
    def input_embedding(self) -> nn.Embedding:
        """Token embedding layer."""
        return self.flow_model.input_embedding

    @property
    def pre_lookahead_layer(self) -> nn.Module:
        """Pre-lookahead layer."""
        return self.flow_model.pre_lookahead_layer

    @property
    def decoder(self) -> nn.Module:
        """Flow matching decoder."""
        return self.flow_model.decoder

    @property
    def spk_embed_affine_layer(self) -> nn.Linear:
        """Speaker embedding affine layer."""
        return self.flow_model.spk_embed_affine_layer

    @torch.inference_mode()
    def _forward_mel(
        self,
        token: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        n_timesteps: int = 10,
        token_offset_tokens: int = 0,
        streaming: bool = True,
        finalize: bool = False,
    ) -> torch.Tensor:
        """Generate mel features via the upstream flow-model inference path."""
        flow_weight = next(self.flow_model.parameters())
        device = flow_weight.device
        dtype = flow_weight.dtype

        token = token.to(device=device, dtype=torch.int32)
        prompt_token = prompt_token.to(device=device, dtype=torch.int32)
        prompt_feat = prompt_feat.to(device=device, dtype=dtype)
        embedding = embedding.to(device=device, dtype=dtype)
        token_len = torch.tensor([token.shape[1]], device=device, dtype=torch.int32)
        prompt_token_len = torch.tensor([prompt_token.shape[1]], device=device, dtype=torch.int32)
        prompt_feat_len = torch.tensor([prompt_feat.shape[1]], device=device, dtype=torch.int32)

        feat, _ = self.flow_model.inference(
            token=token,
            token_len=token_len,
            prompt_token=prompt_token,
            prompt_token_len=prompt_token_len,
            prompt_feat=prompt_feat,
            prompt_feat_len=prompt_feat_len,
            embedding=embedding,
            streaming=streaming,
            finalize=finalize,
            n_timesteps=n_timesteps,
        )

        trim_mel = max(0, int(token_offset_tokens)) * int(self.token_mel_ratio)
        if trim_mel > 0:
            feat = feat[:, :, trim_mel:]

        return feat

    @torch.inference_mode()
    def prepare_streaming_mel(
        self,
        token: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        *,
        cache_state: dict[str, torch.Tensor] | None = None,
        n_timesteps: int = 10,
        token_offset_tokens: int = 0,
        finalize: bool = False,
    ) -> tuple[torch.Tensor, int, torch.Tensor]:
        """Run flow for one streaming chunk and extend the cumulative mel.

        Returns the cumulative mel (on the HiFT device/dtype), the
        emitted-speech offset carried in ``cache_state``, and the f0 for that
        mel — extended from the cached prefix instead of recomputed, since the
        predictor's receptive field is finite. Vocoding is left to
        ``vocode_batch`` so requests scheduled in the same step share one
        batched HiFT pass; ``emit_streaming`` then slices the new suffix.
        """
        feat = self._forward_mel(
            token=token,
            prompt_token=prompt_token,
            prompt_feat=prompt_feat,
            embedding=embedding,
            n_timesteps=n_timesteps,
            token_offset_tokens=token_offset_tokens,
            streaming=True,
            finalize=finalize,
        )
        hift_weight = self.hift.m_source.l_linear.weight
        chunk_mel = feat.to(device=hift_weight.device, dtype=hift_weight.dtype)

        cached_mel = None if not cache_state else cache_state.get("mel")
        speech_offset_obj = None if not cache_state else cache_state.get("speech_offset")
        try:
            speech_offset = int(speech_offset_obj) if speech_offset_obj is not None else 0
        except (TypeError, ValueError):
            speech_offset = 0

        if isinstance(cached_mel, torch.Tensor) and cached_mel.numel() > 0:
            cached_mel = cached_mel.to(device=chunk_mel.device, dtype=chunk_mel.dtype)
            tts_mel = torch.cat([cached_mel, chunk_mel], dim=-1) if chunk_mel.numel() > 0 else cached_mel
        else:
            tts_mel = chunk_mel

        cached_f0 = None if not cache_state else cache_state.get("f0")
        if not isinstance(cached_f0, torch.Tensor):
            cached_f0 = None
        f0 = self.hift.predict_f0(tts_mel, finalize=finalize, cached_f0=cached_f0)

        return tts_mel, speech_offset, f0

    @torch.inference_mode()
    def prepare_mel(
        self,
        token: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        n_timesteps: int = 10,
        token_offset_tokens: int = 0,
    ) -> torch.Tensor:
        """Run flow for a full (non-streaming) request; vocoding is batched separately."""
        feat = self._forward_mel(
            token=token,
            prompt_token=prompt_token,
            prompt_feat=prompt_feat,
            embedding=embedding,
            n_timesteps=n_timesteps,
            token_offset_tokens=token_offset_tokens,
            streaming=False,
            finalize=True,
        )
        hift_weight = self.hift.m_source.l_linear.weight
        return feat.to(device=hift_weight.device, dtype=hift_weight.dtype)

    @torch.inference_mode()
    def vocode_batch(
        self,
        mels: list[torch.Tensor],
        finalize_flags: list[bool],
        f0s: list[torch.Tensor | None] | None = None,
    ) -> list[torch.Tensor]:
        """Vocode per-request mels, batching rows that share a finalize flag.

        Rows are grouped by ``finalize`` (it changes HiFT's look-right
        handling) and length-bucketed to bound padding waste; single-row
        buckets keep today's ``hift.inference`` path bit-for-bit. Rows longer
        than ``MAX_BATCHED_VOCODER_FRAMES`` skip batching entirely. ``f0s``
        carries f0 already predicted by ``prepare_streaming_mel``; rows without
        one fall back to predicting inside HiFT.
        """
        results: list[torch.Tensor | None] = [None] * len(mels)
        row_f0 = f0s if f0s is not None else [None] * len(mels)
        groups: dict[bool, list[int]] = {}
        for i, mel in enumerate(mels):
            if mel.shape[-1] == 0:
                results[i] = mel.new_zeros((mel.shape[0], 0))
            elif mel.shape[-1] > MAX_BATCHED_VOCODER_FRAMES:
                results[i] = self.hift.inference(speech_feat=mel, finalize=bool(finalize_flags[i]), f0=row_f0[i])[0]
            else:
                groups.setdefault(bool(finalize_flags[i]), []).append(i)

        for finalize, idxs in groups.items():
            for bucket in plan_vocoder_buckets([int(mels[i].shape[-1]) for i in idxs]):
                group = [idxs[j] for j in bucket]
                if len(group) == 1:
                    tts_speech, _ = self.hift.inference(
                        speech_feat=mels[group[0]], finalize=finalize, f0=row_f0[group[0]]
                    )
                    results[group[0]] = tts_speech
                else:
                    # All-or-nothing: a mixed bucket falls back to HiFT's single
                    # batched CPU prediction rather than splitting the pass.
                    group_f0 = [row_f0[i] for i in group] if all(row_f0[i] is not None for i in group) else None
                    speeches = self.hift.inference_batch([mels[i] for i in group], finalize, f0s=group_f0)
                    for i, tts_speech in zip(group, speeches):
                        results[i] = tts_speech
        return results

    def emit_streaming(
        self,
        tts_speech: torch.Tensor,
        tts_mel: torch.Tensor,
        speech_offset: int,
        finalize: bool,
        f0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """Emit the newly grown speech suffix and build the next cache state."""
        tts_speech = tts_speech.reshape(tts_speech.shape[0], -1)
        speech_offset = max(0, min(speech_offset, int(tts_speech.shape[-1])))
        emitted_speech = tts_speech[:, speech_offset:]

        if finalize:
            return emitted_speech.reshape(emitted_speech.shape[0], 1, -1), None

        new_state = {
            "mel": tts_mel.detach().cpu().contiguous(),
            "speech_offset": int(tts_speech.shape[-1]),
        }
        if f0 is not None:
            new_state["f0"] = f0.detach().cpu().contiguous()
        return emitted_speech.reshape(emitted_speech.shape[0], 1, -1), new_state

    @torch.inference_mode()
    def forward_streaming(
        self,
        token: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        *,
        cache_state: dict[str, torch.Tensor] | None = None,
        n_timesteps: int = 10,
        token_offset_tokens: int = 0,
        finalize: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """Decode streaming audio using cumulative mel + emitted-speech offset.

        This mirrors upstream CosyVoice3 streaming semantics more closely than
        waveform-domain overlap-add: keep a cumulative mel history per request,
        re-run causal HiFT on the history, and emit only the newly grown speech
        suffix. That preserves causal look-right handling without double
        trimming or duplicated overlap at chunk boundaries.

        Single-request convenience wrapper over ``prepare_streaming_mel`` /
        ``vocode_batch`` / ``emit_streaming``.
        """
        tts_mel, speech_offset, f0 = self.prepare_streaming_mel(
            token=token,
            prompt_token=prompt_token,
            prompt_feat=prompt_feat,
            embedding=embedding,
            cache_state=cache_state,
            n_timesteps=n_timesteps,
            token_offset_tokens=token_offset_tokens,
            finalize=finalize,
        )
        tts_speech = self.vocode_batch([tts_mel], [finalize], [f0])[0]
        return self.emit_streaming(tts_speech, tts_mel, speech_offset, finalize, f0)

    @torch.inference_mode()
    def forward(
        self,
        token: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        n_timesteps: int = 10,
        token_offset_tokens: int = 0,
    ) -> torch.Tensor:
        """Generate audio waveform from speech tokens."""
        tts_mel = self.prepare_mel(
            token=token,
            prompt_token=prompt_token,
            prompt_feat=prompt_feat,
            embedding=embedding,
            n_timesteps=n_timesteps,
            token_offset_tokens=token_offset_tokens,
        )
        return self.vocode_batch([tts_mel], [True])[0]

    def load_weights(self, model_dir: str, device: torch.device) -> None:
        """Load flow.pt and hift.pt weights.

        Args:
            model_dir: Model directory containing flow.pt and hift.pt
            device: Device to load weights to
        """
        import os

        # Load flow weights
        flow_path = os.path.join(model_dir, "flow.pt")
        self.flow_model.load_state_dict(torch.load(flow_path, map_location=device), strict=True)
        self.flow_model.to(device).eval()
        logger.info(f"Loaded flow weights from {flow_path}")

        # Load hift weights
        hift_path = os.path.join(model_dir, "hift.pt")
        hift_state_dict = {
            k.replace("generator.", ""): v for k, v in torch.load(hift_path, map_location=device).items()
        }
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(device).eval()
        logger.info(f"Loaded hift weights from {hift_path}")
