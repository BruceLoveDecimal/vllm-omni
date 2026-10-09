# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adopted from https://github.com/FunAudioLLM/CosyVoice/tree/main/cosyvoice/flow
"""Conditional Flow Matching (CFM) classes for audio generation."""

from abc import ABC
from contextlib import nullcontext

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.nn import functional as F
from vllm.logger import init_logger

from vllm_omni.model_executor.models.cosyvoice3.runtime import cosyvoice3_batch_flow_profile
from vllm_omni.model_executor.models.cosyvoice3.utils import build_dit_attention_mask, make_pad_mask

logger = init_logger(__name__)


class BASECFM(torch.nn.Module, ABC):
    def __init__(
        self,
        n_feats,
        cfm_params,
        n_spks=1,
        spk_emb_dim=128,
    ):
        super().__init__()
        self.n_feats = n_feats
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.solver = cfm_params.solver
        if hasattr(cfm_params, "sigma_min"):
            self.sigma_min = cfm_params.sigma_min
        else:
            self.sigma_min = 1e-4

        self.estimator = None


class ConditionalCFM(BASECFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: torch.nn.Module = None):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        in_channels = in_channels + (spk_emb_dim if n_spks > 0 else 0)
        # Just change the architecture of the estimator here
        self.estimator = estimator

    @torch.inference_mode()
    def forward(
        self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, prompt_len=0, cache=torch.zeros(1, 80, 0, 2)
    ):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond (Optional[Any], optional): Not used but kept for future purposes

        Returns:
            sample (torch.Tensor): generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """

        with cosyvoice3_batch_flow_profile("cosyvoice3_cfm_noise_cache"):
            z = torch.randn_like(mu).to(mu.device).to(mu.dtype) * temperature
            cache_size = cache.shape[2]
            # fix prompt and overlap part mu and z
            if cache_size != 0:
                z[:, :, :cache_size] = cache[:, :, :, 0]
                mu[:, :, :cache_size] = cache[:, :, :, 1]
            z_cache = torch.concat([z[:, :, :prompt_len], z[:, :, -34:]], dim=2)
            mu_cache = torch.concat([mu[:, :, :prompt_len], mu[:, :, -34:]], dim=2)
            cache = torch.stack([z_cache, mu_cache], dim=-1)

        with cosyvoice3_batch_flow_profile("cosyvoice3_cfm_t_span"):
            t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
            if self.t_scheduler == "cosine":
                t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        with cosyvoice3_batch_flow_profile(f"cosyvoice3_cfm_euler_{max(1, int(n_timesteps))}_steps"):
            return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond), cache

    def solve_euler(self, x, t_span, mu, mask, spks, cond, streaming: bool = False):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond (Optional[Any], optional): Not used but kept for future purposes
            streaming: forwarded to the PyTorch DiT estimator (chunk attention).
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)

        sol = []

        # Do not use concat, it may cause memory format changed and trt infer with wrong results!
        # NOTE when flow run in amp mode, x.dtype is float32, which cause nan in trt fp16
        # inference, so set dtype=spks.dtype.  The batch is doubled for CFG:
        # first B rows are conditioned, second B rows are unconditional.
        batch_size = int(x.size(0))
        estimator_batch = 2 * batch_size
        estimator_dtype = spks.dtype if spks is not None else x.dtype
        with cosyvoice3_batch_flow_profile("cosyvoice3_cfm_cfg_allocate_2b"):
            x_in = torch.zeros([estimator_batch, 80, x.size(2)], device=x.device, dtype=estimator_dtype)
            mask_in = torch.zeros([estimator_batch, 1, x.size(2)], device=x.device, dtype=estimator_dtype)
            mu_in = torch.zeros([estimator_batch, 80, x.size(2)], device=x.device, dtype=estimator_dtype)
            t_in = torch.zeros([estimator_batch], device=x.device, dtype=estimator_dtype)
            spks_in = torch.zeros([estimator_batch, 80], device=x.device, dtype=estimator_dtype)
            cond_in = torch.zeros([estimator_batch, 80, x.size(2)], device=x.device, dtype=estimator_dtype)
        # mask/mu/spks/cond do not change during one Euler solve. Stage
        # them once; only x and t need refreshing on every estimator call.
        with cosyvoice3_batch_flow_profile("cosyvoice3_cfm_cfg_prepare_static_2b"):
            mask_in[:batch_size] = mask
            mask_in[batch_size:] = mask
            mu_in[:batch_size] = mu
            if spks is not None:
                spks_in[:batch_size] = spks
            if cond is not None:
                cond_in[:batch_size] = cond
            # The TensorRT attention map follows mask_in, so it is also the
            # same on every Euler step: build it once per solve rather than
            # rebuilding (and host-syncing to compare) on each one.
            attn_mask = None
            if not isinstance(self.estimator, torch.nn.Module):
                attn_mask = self._trt_attention_mask(mask_in, streaming)

        estimator_session = nullcontext(None)
        if not isinstance(self.estimator, torch.nn.Module):
            session_factory = getattr(self.estimator, "estimation_session", None)
            if session_factory is not None:
                session_inputs = (x_in, mask_in, mu_in, t_in, spks_in, cond_in)
                if attn_mask is not None:
                    session_inputs += (attn_mask,)
                estimator_session = session_factory(*session_inputs)

        with estimator_session as trt_session:
            for step in range(1, len(t_span)):
                # Classifier-Free Guidance inference introduced in VoiceBox
                with cosyvoice3_batch_flow_profile("cosyvoice3_cfm_cfg_prepare_dynamic_2b"):
                    x_in[:batch_size] = x
                    x_in[batch_size:] = x
                    t_in[:] = t
                with cosyvoice3_batch_flow_profile("cosyvoice3_cfm_forward_estimator"):
                    dphi_dt = self.forward_estimator(
                        x_in,
                        mask_in,
                        mu_in,
                        t_in,
                        spks_in,
                        cond_in,
                        estimator_session=trt_session,
                        streaming=streaming,
                        attn_mask=attn_mask,
                    )
                with cosyvoice3_batch_flow_profile("cosyvoice3_cfm_cfg_combine"):
                    dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [batch_size, batch_size], dim=0)
                    dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
                with cosyvoice3_batch_flow_profile("cosyvoice3_cfm_euler_update"):
                    x = x + dt * dphi_dt
                    t = t + dt
                sol.append(x)
                if step < len(t_span) - 1:
                    dt = t_span[step + 1] - t

        return sol[-1].float()

    def _forward_estimator_trt_once(self, x, mask, mu, t, spks, cond, attn_mask=None):
        """Execute one TensorRT estimator call for a profile-supported shape."""
        io_dtype = getattr(self.estimator, "io_dtype", x.dtype)
        [estimator, stream], trt_engine = self.estimator.acquire_estimator(
            int(x.shape[0]),
            int(x.shape[2]),
        )
        caller_stream = torch.cuda.current_stream(x.device)
        stream.wait_stream(caller_stream)
        try:
            with torch.cuda.stream(stream):
                x_e = x.to(io_dtype).contiguous()
                mask_e = mask.to(io_dtype).contiguous()
                mu_e = mu.to(io_dtype).contiguous()
                t_e = t.to(io_dtype).contiguous()
                spks_e = spks.to(io_dtype).contiguous()
                cond_e = cond.to(io_dtype).contiguous()
                out_e = torch.empty_like(x_e, dtype=self.estimator.out_dtype)
                inputs = {
                    "x": x_e,
                    "mask": mask_e,
                    "mu": mu_e,
                    "t": t_e,
                    "spks": spks_e,
                    "cond": cond_e,
                }
                if attn_mask is not None:
                    inputs["attn_mask"] = attn_mask
                # Bind only what the engine declares: an exporter prunes
                # inputs the graph never reads.
                declared = self.estimator.input_names
                inputs = {name: tensor for name, tensor in inputs.items() if name in declared}
                for name, tensor in inputs.items():
                    if not estimator.set_input_shape(name, tuple(tensor.shape)):
                        raise RuntimeError(
                            f"TensorRT flow estimator rejected shape {tuple(tensor.shape)} for input "
                            f"'{name}' (outside the engine's optimization profile)"
                        )
                    estimator.set_tensor_address(name, tensor.data_ptr())
                estimator.set_tensor_address("estimator_out", out_e.data_ptr())
                # ``assert`` would vanish under ``python -O`` and leave a
                # zero-filled output; check explicitly.
                if not estimator.execute_async_v3(stream.cuda_stream):
                    raise RuntimeError("TensorRT flow estimator failed to enqueue (execute_async_v3 returned False)")
                for tensor in (*inputs.values(), out_e):
                    if tensor.is_cuda:
                        tensor.record_stream(stream)
            caller_stream.wait_stream(stream)
            if out_e.is_cuda:
                out_e.record_stream(caller_stream)
            return out_e.to(x.dtype)
        finally:
            self.estimator.release_estimator(estimator, stream)

    def forward_estimator(
        self, x, mask, mu, t, spks, cond, estimator_session=None, streaming: bool = False, attn_mask=None
    ):
        """One estimator call. ``attn_mask`` is the TensorRT engine's query-key
        map; ``solve_euler`` builds it once per solve, and it is built here only
        when a caller passes none."""
        if isinstance(self.estimator, torch.nn.Module):
            # The DiT builds its own chunk map from ``streaming``; the flag is
            # only passed when set, so the default call stays the plain one.
            if streaming:
                return self.estimator(x, mask, mu, t, spks, cond, streaming=True)
            return self.estimator(x, mask, mu, t, spks, cond)
        if estimator_session is not None:
            return estimator_session.run(x, t)

        if attn_mask is None:
            attn_mask = self._trt_attention_mask(mask, streaming)
        batch_size = int(x.shape[0])
        sequence_length = int(x.shape[2])
        supports_shape = getattr(self.estimator, "supports_estimator_shape", None)
        if batch_size > 2 and supports_shape is not None and not supports_shape(batch_size, sequence_length):
            if batch_size % 2 != 0:
                raise RuntimeError(f"TensorRT CFG estimator batch must be even, got {batch_size}")

            # Dynamic profile 1 intentionally covers the common batched-flow
            # window only. If a long cumulative stream (or an unusually large
            # request batch) falls outside that profile, preserve TensorRT and
            # exact CFG semantics by serializing request pairs through profile 0.
            # CFG layout is [B conditioned rows | B unconditional rows].
            requests = batch_size // 2
            conditioned = []
            unconditional = []
            for index in range(requests):
                pair = (
                    slice(index, index + 1),
                    slice(index + requests, index + requests + 1),
                )

                def pair_rows(tensor):
                    return torch.cat((tensor[pair[0]], tensor[pair[1]]), dim=0)

                pair_out = self._forward_estimator_trt_once(
                    pair_rows(x),
                    pair_rows(mask),
                    pair_rows(mu),
                    pair_rows(t),
                    pair_rows(spks),
                    pair_rows(cond),
                    None if attn_mask is None else pair_rows(attn_mask),
                )
                conditioned.append(pair_out[:1])
                unconditional.append(pair_out[1:])

            return torch.cat((*conditioned, *unconditional), dim=0)

        return self._forward_estimator_trt_once(x, mask, mu, t, spks, cond, attn_mask)

    def _trt_attention_mask(self, mask: torch.Tensor, streaming: bool) -> torch.Tensor | None:
        """The query-key map for a chunk-mask TensorRT engine, or None.

        A legacy engine (no ``attn_mask`` input) was traced with full
        attention and cannot honour ``streaming``; say so once rather than
        silently diverging from upstream's streaming semantics. The map is
        step-invariant within a solve, so ``solve_euler`` calls this once;
        there is no cache, whose content check would sync the host per step.
        """
        estimator = self.estimator
        if not estimator.supports_attn_mask:
            if streaming:
                logger.warning_once(
                    "The TensorRT flow estimator has no attn_mask input, so streaming chunks run with full "
                    "attention instead of upstream's chunk-causal mask. Rebuild it with "
                    "build_chunk_mask_flow_estimator_trt to align with upstream."
                )
            return None
        return build_dit_attention_mask(
            mask.bool(), streaming=streaming, static_chunk_size=estimator.static_chunk_size
        ).contiguous()


# Upstream CosyVoice draws the flow's initial noise from one fixed buffer,
# ``torch.randn([1, 80, 50 * 300])`` under seed 0, and slices it by mel
# position, so the noise at a given position is the same on every call. A
# streaming decode regenerates its left context each chunk; with fixed noise
# that context comes out the same as when it was emitted, which is what keeps
# chunk boundaries consistent, and the same seed reproduces the same audio.
_FIXED_NOISE_SEED = 0
_FIXED_NOISE_CHANNELS = 80
_FIXED_NOISE_FRAMES = 50 * 300


class CausalConditionalCFM(ConditionalCFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: torch.nn.Module = None):
        super().__init__(in_channels, cfm_params, n_spks, spk_emb_dim, estimator)
        # Upstream's seeded draw without touching the global RNG. On the CPU so
        # the values match whatever device the model is built under.
        generator = torch.Generator(device="cpu").manual_seed(_FIXED_NOISE_SEED)
        noise = torch.randn([1, _FIXED_NOISE_CHANNELS, _FIXED_NOISE_FRAMES], generator=generator, device="cpu")
        self.register_buffer("rand_noise", noise, persistent=False)

    def fixed_noise(self, mu, prompt_len=0, noise_offset=None, temperature: float = 1.0):
        """Initial noise indexed by absolute mel position.

        Positions ``[0, prompt_len)`` are the prompt and always map to the
        start of the buffer. ``prompt_len`` is one int, or one per batch row
        when the rows carry references of different lengths. Positions after the prompt map to
        ``prompt_len + noise_offset + j``, where ``noise_offset`` (one int, or
        one per batch row) is the absolute mel index of the first post-prompt
        frame in the stream. A bounded left context therefore reuses exactly
        the noise its frames were first generated with. Positions past the
        buffer wrap around.
        """
        batch, channels, length = mu.shape
        if self.rand_noise.device != mu.device:
            self.rand_noise = self.rand_noise.to(mu.device)
        noise = self.rand_noise[0]
        if channels != noise.shape[0]:
            raise ValueError(f"fixed noise has {noise.shape[0]} channels, mu has {channels}")
        if noise_offset is None:
            noise_offset = torch.zeros(batch, dtype=torch.long, device=mu.device)
        else:
            noise_offset = torch.as_tensor(noise_offset, dtype=torch.long, device=mu.device).reshape(-1)
            if noise_offset.numel() == 1:
                noise_offset = noise_offset.expand(batch)
        prompt_len = torch.as_tensor(prompt_len, dtype=torch.long, device=mu.device).reshape(-1)
        if prompt_len.numel() == 1:
            prompt_len = prompt_len.expand(batch)
        prompt_len = prompt_len.clamp(min=0, max=length).unsqueeze(1)
        positions = torch.arange(length, device=mu.device).unsqueeze(0).expand(batch, length)
        shifted = positions + noise_offset.clamp(min=0).unsqueeze(1)
        index = torch.where(positions < prompt_len, positions, shifted) % noise.shape[1]
        # Gather first so only the used frames are cast, not the whole buffer.
        return noise[:, index].permute(1, 0, 2).to(mu.dtype) * temperature

    @torch.inference_mode()
    def forward(
        self,
        mu,
        mask,
        n_timesteps,
        temperature=1.0,
        spks=None,
        cond=None,
        streaming: bool = False,
        prompt_len=0,
        noise_offset=None,
    ):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond (Optional[Any], optional): Not used but kept for future purposes
            streaming: forwarded to the DiT estimator for chunk attention.

        Returns:
            sample (torch.Tensor): generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """

        with cosyvoice3_batch_flow_profile("cosyvoice3_cfm_noise_cache"):
            z = self.fixed_noise(mu, prompt_len=prompt_len, noise_offset=noise_offset, temperature=temperature)

        with cosyvoice3_batch_flow_profile("cosyvoice3_cfm_t_span"):
            t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)

            if self.t_scheduler == "cosine":
                t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        with cosyvoice3_batch_flow_profile(f"cosyvoice3_cfm_euler_{max(1, int(n_timesteps))}_steps"):
            return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond, streaming=streaming), None


class CausalMaskedDiffWithDiT(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        output_type: str = "mel",
        vocab_size: int = 4096,
        input_frame_rate: int = 50,
        only_mask_loss: bool = True,
        token_mel_ratio: int = 2,
        pre_lookahead_len: int = 3,
        pre_lookahead_layer: torch.nn.Module = None,
        decoder: torch.nn.Module = None,
        decoder_conf: dict = {
            "in_channels": 240,
            "out_channel": 80,
            "spk_emb_dim": 80,
            "n_spks": 1,
            "cfm_params": DictConfig(
                {
                    "sigma_min": 1e-06,
                    "solver": "euler",
                    "t_scheduler": "cosine",
                    "training_cfg_rate": 0.2,
                    "inference_cfg_rate": 0.7,
                    "reg_loss_type": "l1",
                }
            ),
            "decoder_params": {
                "channels": [256, 256],
                "dropout": 0.0,
                "attention_head_dim": 64,
                "n_blocks": 4,
                "num_mid_blocks": 12,
                "num_heads": 8,
                "act_fn": "gelu",
            },
        },
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logger.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.pre_lookahead_len = pre_lookahead_len
        self.pre_lookahead_layer = pre_lookahead_layer
        self.decoder = decoder
        self.only_mask_loss = only_mask_loss
        self.token_mel_ratio = token_mel_ratio

    @torch.inference_mode()
    def inference(
        self,
        token,
        token_len,
        prompt_token,
        prompt_token_len,
        prompt_feat,
        prompt_feat_len,
        embedding,
        streaming: bool = True,
        finalize: bool = False,
        n_timesteps: int = 10,
        noise_offset=None,
    ):
        """``noise_offset``: absolute mel index (int, or one per row) of the first
        frame after the prompt, so a bounded left context keeps the noise it
        was first generated with. Defaults to 0, the unbounded stream start."""
        with cosyvoice3_batch_flow_profile("cosyvoice3_cfm_speaker_embedding"):
            embedding = F.normalize(embedding, dim=1)
            embedding = self.spk_embed_affine_layer(embedding)

        batch_size = int(token.shape[0])
        ratio = int(self.token_mel_ratio)
        lookahead = 0 if finalize else int(self.pre_lookahead_len)
        # One host sync for the row geometry; the loops below need it anyway,
        # and the scalar path used to sync on ``int(prompt_feat.shape[1])``.
        prompt_lens = [int(v) for v in prompt_token_len.reshape(-1).tolist()]
        target_lens = [int(v) for v in token_len.reshape(-1).tolist()]
        prompt_mel_lens = [int(v) for v in prompt_feat_len.reshape(-1).tolist()]
        combined_lens = [p + t for p, t in zip(prompt_lens, target_lens, strict=True)]
        # Rows that share a reference length keep the single-tensor path, so a
        # uniform batch (every batch before mixed references) is unchanged.
        uniform_prompts = len(set(prompt_lens)) <= 1 and len(set(prompt_mel_lens)) <= 1
        uniform_combined = len(set(combined_lens)) <= 1

        with cosyvoice3_batch_flow_profile("cosyvoice3_cfm_token_embedding_lookahead"):
            # concat text and prompt_text
            codec_token_len = token_len
            total_token_len = prompt_token_len + codec_token_len
            if uniform_prompts:
                token = torch.concat([prompt_token, token], dim=1)
            else:
                # Each row is prompt-then-generated and padded at the tail, so
                # a row's own tokens start where its own reference ends.
                token = torch.nn.utils.rnn.pad_sequence(
                    [
                        torch.concat([prompt_token[i, : prompt_lens[i]], token[i, : target_lens[i]]])
                        for i in range(batch_size)
                    ],
                    batch_first=True,
                )
            mask = (~make_pad_mask(total_token_len, max_len=token.shape[1])).unsqueeze(-1).to(embedding)
            token = self.input_embedding(torch.clamp(token, min=0)) * mask
            # text encode
            if finalize is True:
                h = self.pre_lookahead_layer(token)
            elif uniform_combined:
                h = self.pre_lookahead_layer(
                    token[:, : -self.pre_lookahead_len], context=token[:, -self.pre_lookahead_len :]
                )
            else:
                # The rows are left-aligned and padded to the longest, so
                # ``token[:, -lookahead:]`` would feed a short row its padding
                # as lookahead context instead of its own last tokens.
                bodies = [max(length - lookahead, 0) for length in combined_lens]
                max_body = max(bodies)
                rows = []
                for i, length in enumerate(combined_lens):
                    row = self.pre_lookahead_layer(
                        token[i : i + 1, : bodies[i]],
                        context=token[i : i + 1, length - lookahead : length],
                    )
                    if row.shape[1] < max_body:
                        row = F.pad(row, (0, 0, 0, max_body - int(row.shape[1])))
                    rows.append(row)
                h = torch.concat(rows, dim=0)

        with cosyvoice3_batch_flow_profile("cosyvoice3_cfm_repeat_to_mel_axis"):
            h = h.repeat_interleave(ratio, dim=1)

        with cosyvoice3_batch_flow_profile("cosyvoice3_cfm_cond_prompt_mel"):
            mel_width = int(h.shape[1])

            # get conditions
            conds = torch.zeros([batch_size, mel_width, self.output_size], device=token.device).to(h.dtype)
            if uniform_prompts:
                conds[:, : prompt_mel_lens[0]] = prompt_feat[:, : prompt_mel_lens[0]]
            else:
                for i, prompt_mel in enumerate(prompt_mel_lens):
                    conds[i, :prompt_mel] = prompt_feat[i, :prompt_mel]
            conds = conds.transpose(1, 2)

            valid_h_lens = torch.clamp(total_token_len.to(torch.long) - lookahead, min=0)
            mel_lens = torch.clamp(valid_h_lens * ratio, max=mel_width)
            mask = (~make_pad_mask(mel_lens, max_len=mel_width)).to(h)

        prompt_mel_arg = (
            prompt_mel_lens[0]
            if uniform_prompts
            else torch.tensor(prompt_mel_lens, dtype=torch.long, device=token.device)
        )
        feat, _ = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=max(1, int(n_timesteps)),
            streaming=streaming,
            prompt_len=prompt_mel_arg,
            noise_offset=noise_offset,
        )

        with cosyvoice3_batch_flow_profile("cosyvoice3_cfm_crop_prompt_mel"):
            if uniform_prompts:
                feat = feat[:, :, prompt_mel_lens[0] :]
            else:
                # Drop each row's own prompt region, keeping the rows
                # left-aligned at the first generated frame.
                widths = [mel_width - prompt_mel for prompt_mel in prompt_mel_lens]
                max_width = max(widths)
                feat = torch.concat(
                    [
                        F.pad(feat[i : i + 1, :, prompt_mel_lens[i] :], (0, max_width - widths[i]))
                        for i in range(batch_size)
                    ],
                    dim=0,
                )
        return feat.float(), None
