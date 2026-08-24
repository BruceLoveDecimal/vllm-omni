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

from dataclasses import dataclass

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


# Mel frames of history the conv trunk needs before a windowed decode matches
# a full replay. Measured on the production conv topology: 32 frames already
# reaches the float32 round-off floor (8.0e-08, -141.9 dBFS) and 96 or 240
# frames change nothing, while 16 frames is audibly wrong at -49.8 dBFS. 48
# keeps a margin over the measured requirement at negligible cost.
VOCODER_LEFT_CONTEXT_FRAMES = 48


@dataclass
class _StreamWindow:
    """Bounded decoding state for one streaming request.

    Frame counters are absolute indices into the utterance, which is what makes
    the sliding window auditable: ``mel``/``f0`` cover
    ``[ctx_start, ctx_start + mel.shape[-1])``, ``source`` covers
    ``[ctx_start, src_end)``, and everything below ``emitted_frames`` has
    already been handed to the caller.
    """

    mel: torch.Tensor
    f0: torch.Tensor
    source: torch.Tensor
    phase: torch.Tensor | None
    ctx_start: int
    src_end: int
    emitted_frames: int

    @classmethod
    def from_cache(
        cls, cache_state: dict | None, *, device: torch.device, dtype: torch.dtype, channels: int
    ) -> _StreamWindow:
        """Rebuild the window from a cache entry, or start a fresh utterance."""
        state = cache_state.get("window") if cache_state else None
        if not isinstance(state, cls):
            return cls(
                mel=torch.zeros((1, channels, 0), device=device, dtype=dtype),
                f0=torch.zeros((1, 0), device=device, dtype=dtype),
                source=torch.zeros((1, 1, 0), device=device, dtype=dtype),
                phase=None,
                ctx_start=0,
                src_end=0,
                emitted_frames=0,
            )
        return cls(
            mel=state.mel.to(device=device, dtype=dtype),
            f0=state.f0.to(device=device, dtype=dtype),
            source=state.source.to(device=device, dtype=dtype),
            phase=None if state.phase is None else state.phase.to(device=device),
            ctx_start=state.ctx_start,
            src_end=state.src_end,
            emitted_frames=state.emitted_frames,
        )

    def slide(self, samples_per_frame: int, left_context: int) -> _StreamWindow:
        """Drop history beyond ``left_context`` frames.

        The state stays on its device: the window is ~150KB per request, and
        round-tripping it through the CPU costs a sync per tensor per chunk —
        at 8 streams that is 24 device transfers per scheduler step, which
        measured as a large share of per-chunk wall time.
        """
        end = self.ctx_start + self.mel.shape[-1]
        new_ctx_start = max(0, end - left_context)
        drop = new_ctx_start - self.ctx_start
        # Only frames below ``src_end`` are final; the rest were predicted with
        # a zero look-right and get redone next chunk.
        keep_f0 = self.src_end - self.ctx_start
        return _StreamWindow(
            mel=self.mel[:, :, drop:].detach().contiguous(),
            f0=self.f0[:, drop:keep_f0].detach().contiguous(),
            source=self.source[:, :, drop * samples_per_frame :].detach().contiguous(),
            phase=None if self.phase is None else self.phase.detach(),
            ctx_start=new_ctx_start,
            src_end=self.src_end,
            emitted_frames=self.emitted_frames,
        )


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

    def __init__(self, config: CosyVoice3Config, flow_graph_config: dict | None = None):
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
            flow_graph_config=flow_graph_config,
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
    ) -> tuple[torch.Tensor, _StreamWindow]:
        """Run flow for one streaming chunk and build the vocoder window.

        Returns the bounded mel window (on the HiFT device/dtype) and the
        window's decoding state. Only ``VOCODER_LEFT_CONTEXT_FRAMES`` frames of
        history are kept, so per-chunk cost is O(window) rather than O(history);
        the f0, the excitation and the source generator's phase are carried
        alongside so the window still decodes to what a full replay would give.
        Vocoding is left to ``vocode_batch`` so requests scheduled in the same
        step share one batched HiFT pass, then ``emit_streaming`` slices out
        the newly complete audio.
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
        device, dtype = hift_weight.device, hift_weight.dtype
        chunk_mel = feat.to(device=device, dtype=dtype)

        prev = _StreamWindow.from_cache(cache_state, device=device, dtype=dtype, channels=chunk_mel.shape[1])
        spf = self.hift.source_samples_per_frame
        look_right = self.hift.f0_predictor.condnet[0].causal_padding

        mel = torch.cat([prev.mel, chunk_mel], dim=-1) if prev.mel.numel() else chunk_mel
        end = prev.ctx_start + mel.shape[-1]

        f0 = self.hift.predict_f0(mel, finalize=True, cached_f0=prev.f0)
        # Mid-stream the trailing frames have no look-right yet; leave them for
        # the next chunk instead of freezing their provisional values.
        target = end if finalize else end - look_right
        new_f0 = f0[:, prev.src_end - prev.ctx_start : target - prev.ctx_start]
        new_source, phase = self.hift.synthesize_source(
            new_f0, sample_offset=prev.src_end * spf, phase_carry=prev.phase
        )
        source = torch.cat([prev.source, new_source], dim=-1) if prev.source.numel() else new_source

        window = _StreamWindow(
            mel=mel,
            f0=f0,
            source=source,
            phase=phase,
            ctx_start=prev.ctx_start,
            src_end=target,
            emitted_frames=prev.emitted_frames,
        )
        return mel, window

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
        sources: list[torch.Tensor | None] | None = None,
    ) -> list[torch.Tensor]:
        """Vocode per-request mels, batching the streaming windows.

        Batching is a streaming-only optimization, and rows announce
        themselves by carrying an excitation in ``sources`` (the streaming
        window must, since its phase continues from the previous chunk).
        Those rows are bounded near ``VOCODER_LEFT_CONTEXT_FRAMES`` — the
        regime where one batched trunk pass beats a per-row loop by 1.8-4.6x
        at batch 2-8 and the shape set is small enough for CUDA-graph replay.

        Offline whole-utterance rows carry no excitation and keep the original
        per-row ``inference`` path: measured on RTX PRO 6000 with the GPU f0
        and device-side conv caches in place, the per-row loop matches or
        beats a padded batch from ~512 mel frames up (0.90x at 640, 0.76x at
        1024 for batch 8), and a fixed-round split (Qwen3-Omni style) measured
        worse still (0.72-0.81x, with error growing past 30s). Keeping those
        rows out of the batch also keeps their arbitrary lengths out of the
        lazy trunk-graph cache.

        Streaming rows are grouped by ``finalize`` (it changes HiFT's
        look-right handling) and length-bucketed to bound padding waste.
        """
        results: list[torch.Tensor | None] = [None] * len(mels)
        row_f0 = f0s if f0s is not None else [None] * len(mels)
        row_src = sources if sources is not None else [None] * len(mels)
        groups: dict[bool, list[int]] = {}
        for i, mel in enumerate(mels):
            if mel.shape[-1] == 0:
                results[i] = mel.new_zeros((mel.shape[0], 0))
            elif row_src[i] is None:
                results[i] = self.hift.inference(speech_feat=mel, finalize=bool(finalize_flags[i]), f0=row_f0[i])[0]
            else:
                groups.setdefault(bool(finalize_flags[i]), []).append(i)

        for finalize, idxs in groups.items():
            for bucket in plan_vocoder_buckets([int(mels[i].shape[-1]) for i in idxs]):
                group = [idxs[j] for j in bucket]
                if len(group) == 1 and self.hift.trunk_graph is None:
                    # Without graph replay the single-row path stays on
                    # ``inference``, bit-identical to the pre-batching code.
                    # With it, single rows go through ``inference_batch`` too,
                    # so a lone stream still gets the captured trunk.
                    tts_speech, _ = self.hift.inference(
                        speech_feat=mels[group[0]],
                        finalize=finalize,
                        f0=row_f0[group[0]],
                        source=row_src[group[0]],
                    )
                    results[group[0]] = tts_speech
                else:
                    speeches = self.hift.inference_batch(
                        [mels[i] for i in group],
                        finalize,
                        f0s=[row_f0[i] for i in group],
                        sources=[row_src[i] for i in group],
                    )
                    for i, tts_speech in zip(group, speeches):
                        results[i] = tts_speech
        return results

    def emit_streaming(
        self,
        tts_speech: torch.Tensor,
        window: _StreamWindow,
        finalize: bool,
    ) -> tuple[torch.Tensor, dict[str, object] | None]:
        """Emit the audio this window newly completed, and carry the state on.

        The window decodes ``[ctx_start, src_end)``, but mid-stream HiFT
        withholds ``conv_pre``'s look-right frames plus one for the istft tail,
        so only frames below ``covered`` are final. Everything from
        ``emitted_frames`` up to there is new.
        """
        spf = self.hift.source_samples_per_frame
        tts_speech = tts_speech.reshape(tts_speech.shape[0], -1)
        covered = window.src_end if finalize else window.src_end - self.hift.conv_pre_look_right - 1

        start = max(0, (window.emitted_frames - window.ctx_start) * spf)
        stop = max(start, min((covered - window.ctx_start) * spf, int(tts_speech.shape[-1])))
        emitted_speech = tts_speech[:, start:stop]

        if finalize:
            return emitted_speech.reshape(emitted_speech.shape[0], 1, -1), None

        window.emitted_frames = covered
        new_state = {"window": window.slide(spf, VOCODER_LEFT_CONTEXT_FRAMES)}
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
        """Decode streaming audio from a bounded mel window.

        Preserves upstream CosyVoice3 streaming semantics — causal look-right
        handling, no waveform-domain overlap-add — but keeps only the receptive
        field's worth of history instead of the whole utterance, so per-chunk
        cost stops growing with the audio produced so far.

        Single-request convenience wrapper over ``prepare_streaming_mel`` /
        ``vocode_batch`` / ``emit_streaming``.
        """
        mel, window = self.prepare_streaming_mel(
            token=token,
            prompt_token=prompt_token,
            prompt_feat=prompt_feat,
            embedding=embedding,
            cache_state=cache_state,
            n_timesteps=n_timesteps,
            token_offset_tokens=token_offset_tokens,
            finalize=finalize,
        )
        tts_speech = self.vocode_batch([mel], [finalize], [window.f0], [window.source])[0]
        return self.emit_streaming(tts_speech, window, finalize)

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

        # Bounded streaming windows give the conv trunk a small fixed shape
        # set, so its ~50 launch-bound kernels can replay as CUDA graphs.
        # The wrapper falls back to eager per shape, so this is safe to turn
        # on whenever the vocoder lives on a CUDA device.
        if device.type == "cuda":
            from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.trunk_graph import HiFTTrunkGraph

            hift = self.hift
            self.hift.trunk_graph = HiFTTrunkGraph(lambda mel, s_stft: hift._decode_trunk(hift.conv_pre(mel), s_stft))
            logger.info("CosyVoice3: HiFT trunk CUDA-graph replay enabled")
