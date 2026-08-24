# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adopted from https://github.com/FunAudioLLM/CosyVoice/tree/main/cosyvoice/flow
"""Conditional Flow Matching (CFM) classes for audio generation."""

from abc import ABC

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.nn import functional as F
from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm_omni.model_executor.models.cosyvoice3.utils import make_pad_mask
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

# Graphs are cached per exact estimator input shape (no padding: the DiT
# applies its mask only to the block *outputs*, not inside attention, so
# padded frames would leak into the softmax and change the math). Streaming
# chunks reuse a handful of shapes per voice, so a small cache covers steady
# state; bounding the resent left context shrinks that set further.
_FLOW_CUDA_GRAPH_MAX_ENTRIES = 64
# Skip capture below this much free device memory so a saturated device
# degrades to eager instead of capturing itself into an OOM.
_MIN_CAPTURE_FREE_BYTES = 1 << 30  # 1 GiB


class _EstimatorGraphEntry:
    """One captured estimator forward: the graph plus its static I/O."""

    __slots__ = ("graph", "inputs", "out")

    def __init__(self, graph, inputs, out):
        self.graph = graph
        self.inputs = inputs
        self.out = out


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
    def __init__(
        self,
        in_channels,
        cfm_params,
        n_spks=1,
        spk_emb_dim=64,
        estimator: torch.nn.Module = None,
        flow_graph_config: dict | None = None,
    ):
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
        cfg = dict(flow_graph_config or {})
        self._estimator_graph_enabled = bool(cfg.get("enabled", False))
        self._estimator_graph_max = max(1, int(cfg.get("max_graphs", _FLOW_CUDA_GRAPH_MAX_ENTRIES)))
        self._estimator_graph_min_free_bytes = int(cfg.get("min_free_bytes", _MIN_CAPTURE_FREE_BYTES))
        self._estimator_graphs: dict[tuple, _EstimatorGraphEntry] = {}
        self._estimator_graph_stats = {"calls": 0, "hits": 0, "captures": 0, "flushes": 0, "eager": 0}

    def flow_graph_stats(self) -> dict[str, int]:
        """Bounded cumulative telemetry for the Euler graph cache."""
        return {**self._estimator_graph_stats, "cache_size": len(self._estimator_graphs)}

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

        z = torch.randn_like(mu).to(mu.device).to(mu.dtype) * temperature
        cache_size = cache.shape[2]
        # fix prompt and overlap part mu and z
        if cache_size != 0:
            z[:, :, :cache_size] = cache[:, :, :, 0]
            mu[:, :, :cache_size] = cache[:, :, :, 1]
        z_cache = torch.concat([z[:, :, :prompt_len], z[:, :, -34:]], dim=2)
        mu_cache = torch.concat([mu[:, :, :prompt_len], mu[:, :, -34:]], dim=2)
        cache = torch.stack([z_cache, mu_cache], dim=-1)

        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond), cache

    def solve_euler(self, x, t_span, mu, mask, spks, cond):
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
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)

        sol = []

        # Do not use concat, it may cause memory format changed and trt infer with wrong results!
        # NOTE when flow run in amp mode, x.dtype is float32, which cause nan in trt fp16
        # inference, so set dtype=spks.dtype
        x_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=spks.dtype)
        mask_in = torch.zeros([2, 1, x.size(2)], device=x.device, dtype=spks.dtype)
        mu_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=spks.dtype)
        t_in = torch.zeros([2], device=x.device, dtype=spks.dtype)
        spks_in = torch.zeros([2, 80], device=x.device, dtype=spks.dtype)
        cond_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=spks.dtype)
        for step in range(1, len(t_span)):
            # Classifier-Free Guidance inference introduced in VoiceBox
            x_in[:] = x
            mask_in[:] = mask
            mu_in[0] = mu
            t_in[:] = t.unsqueeze(0)
            spks_in[0] = spks
            cond_in[0] = cond
            dphi_dt = self.forward_estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
            dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1].float()

    def _estimator_graph_usable(self, x) -> bool:
        # CUDA graphs cover only the in-process torch estimator; the TRT
        # estimator manages its own streams/host syncs and cannot be captured
        # here. Capturing while an outer capture is already recording would
        # nest graphs, so defer to eager there too.
        return (
            self._estimator_graph_enabled
            and isinstance(self.estimator, torch.nn.Module)
            and not self.estimator.training
            and x.is_cuda
            and not torch.cuda.is_current_stream_capturing()
        )

    def _flush_estimator_graphs(self) -> None:
        """Retire every captured graph at once.

        Never evict a single graph while its peers stay live. torch records
        every capture on one process-wide side stream, and cuBLAS keeps one
        workspace per (handle, stream) first allocated *during* a capture, so
        it lives inside a graph memory pool while every later graph bakes in
        the same address. Tearing one graph down releases that pool and leaves
        the survivors replaying against reclaimed memory -- an illegal memory
        access at the next replay (issue #6457, MiniCPM-o hit this first).
        Dropping the whole generation keeps the cache bounded without ever
        leaving a live graph behind a dead one.
        """
        if not self._estimator_graphs:
            return
        device = next(iter(self._estimator_graphs.values())).inputs[0].device
        torch.accelerator.synchronize(device)
        for entry in self._estimator_graphs.values():
            entry.graph.reset()
        self._estimator_graphs.clear()
        clear_cublas_workspaces = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
        if clear_cublas_workspaces is not None:
            clear_cublas_workspaces()
        torch.accelerator.synchronize(device)
        torch.accelerator.empty_cache()
        self._estimator_graph_stats["flushes"] += 1
        logger.info("Flow estimator graph cache flushed; stats=%s", self.flow_graph_stats())

    def _disable_estimator_graphs(self, reason: str, key: tuple) -> None:
        # A failed capture can leave the capture stream current and the
        # allocator still routing into the graph pool, so there is no safe way
        # to keep capturing afterwards.
        logger.warning(
            "Disabling flow estimator CUDA graphs (%s) for shape=%s; using eager", reason, key, exc_info=True
        )
        self._estimator_graph_enabled = False
        self._flush_estimator_graphs()

    def _graphed_estimator(self, inputs: tuple) -> torch.Tensor | None:
        """Replay one estimator forward from a per-shape CUDA graph.

        Returns None when no graph can serve this call (cache full, capture
        failed, or memory too tight), in which case the caller runs eager.

        Capture is attempted on the first miss, with no recurrence or
        process-lifetime heuristic, because the graph covers a single DiT
        forward: the very call that captures it replays it for the remaining
        ``n_timesteps - 1`` Euler steps, so it repays itself within that one
        call. This is what MiniCPM-o's CFMGraphWrapper does.
        """
        key = tuple((tuple(t.shape), t.dtype) for t in inputs)
        entry = self._estimator_graphs.get(key)
        if entry is None:
            if len(self._estimator_graphs) >= self._estimator_graph_max:
                self._flush_estimator_graphs()
                return None
            free_bytes, _total = current_omni_platform.get_device_memory(inputs[0].device)
            if free_bytes < self._estimator_graph_min_free_bytes:
                return None
            try:
                entry = self._capture_estimator_graph(inputs)
            except Exception:
                self._disable_estimator_graphs("capture failed", key)
                return None
            self._estimator_graphs[key] = entry
            self._estimator_graph_stats["captures"] += 1
            logger.info(
                "Captured flow estimator CUDA graph for mel_len=%d; stats=%s",
                inputs[0].shape[-1],
                self.flow_graph_stats(),
            )
        else:
            self._estimator_graph_stats["hits"] += 1

        for static, current in zip(entry.inputs, inputs, strict=True):
            static.copy_(current)
        entry.graph.replay()
        # The static output is overwritten by the next replay of any graph
        # sharing the pool, so hand the caller its own copy.
        return entry.out.clone()

    def _capture_estimator_graph(self, inputs: tuple) -> _EstimatorGraphEntry:
        static_inputs = tuple(t.clone() for t in inputs)

        # Warm up on a side stream, the canonical capture recipe: it keeps the
        # warmup's allocations off the stream that is about to record.
        device = static_inputs[0].device
        current_stream = torch.cuda.current_stream(device)
        warmup_stream = torch.cuda.Stream(device=device)
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                warmup_out = self.estimator(*static_inputs)
        current_stream.wait_stream(warmup_stream)
        del warmup_out

        graph = torch.cuda.CUDAGraph()
        # Share the platform's global graph pool, as the other hand-rolled
        # graph wrappers in this repo do.
        with torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
            static_out = self.estimator(*static_inputs)
        return _EstimatorGraphEntry(graph, static_inputs, static_out)

    def forward_estimator(self, x, mask, mu, t, spks, cond):
        if isinstance(self.estimator, torch.nn.Module):
            self._estimator_graph_stats["calls"] += 1
            if self._estimator_graph_usable(x):
                out = self._graphed_estimator((x, mask, mu, t, spks, cond))
                if out is not None:
                    return out
            self._estimator_graph_stats["eager"] += 1
            return self.estimator(x, mask, mu, t, spks, cond)
        else:
            # TensorRT estimator: bind raw device pointers. The flow runs in
            # fp32 but the engine may have fp16 I/O (strongly-typed fp16 engine),
            # so cast inputs/output to the engine's dtype at the boundary. Keep
            # references to the cast buffers alive until execute completes (a bare
            # ``.contiguous().data_ptr()`` could free the temp -> dangling ptr).
            io_dtype = getattr(self.estimator, "io_dtype", x.dtype)
            [estimator, stream], trt_engine = self.estimator.acquire_estimator()
            # NOTE need to synchronize when switching stream
            torch.cuda.current_stream().synchronize()
            with stream:
                x_e = x.to(io_dtype).contiguous()
                mask_e = mask.to(io_dtype).contiguous()
                mu_e = mu.to(io_dtype).contiguous()
                t_e = t.to(io_dtype).contiguous()
                spks_e = spks.to(io_dtype).contiguous()
                cond_e = cond.to(io_dtype).contiguous()
                out_e = torch.empty_like(x_e)
                estimator.set_input_shape("x", (2, 80, x_e.size(2)))
                estimator.set_input_shape("mask", (2, 1, x_e.size(2)))
                estimator.set_input_shape("mu", (2, 80, x_e.size(2)))
                estimator.set_input_shape("t", (2,))
                estimator.set_input_shape("spks", (2, 80))
                estimator.set_input_shape("cond", (2, 80, x_e.size(2)))
                data_ptrs = [
                    x_e.data_ptr(),
                    mask_e.data_ptr(),
                    mu_e.data_ptr(),
                    t_e.data_ptr(),
                    spks_e.data_ptr(),
                    cond_e.data_ptr(),
                    out_e.data_ptr(),
                ]
                for i, j in enumerate(data_ptrs):
                    estimator.set_tensor_address(trt_engine.get_tensor_name(i), j)
                # run trt engine
                assert estimator.execute_async_v3(torch.cuda.current_stream().cuda_stream) is True
                torch.cuda.current_stream().synchronize()
            self.estimator.release_estimator(estimator, stream)
            return out_e.to(x.dtype)


class CausalConditionalCFM(ConditionalCFM):
    def __init__(
        self,
        in_channels,
        cfm_params,
        n_spks=1,
        spk_emb_dim=64,
        estimator: torch.nn.Module = None,
        flow_graph_config: dict | None = None,
    ):
        super().__init__(in_channels, cfm_params, n_spks, spk_emb_dim, estimator, flow_graph_config)

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, streaming: bool = False):
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

        z = (
            torch.randn(
                (mu.size(0), mu.size(1), mu.size(2)),
                device=mu.device,
                dtype=mu.dtype,
            )
            * temperature
        )

        # fix prompt and overlap part mu and z
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)

        if self.t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond), None


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
    ):
        assert token.shape[0] == 1
        # xvec projection

        embedding = F.normalize(embedding, dim=1)

        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask
        # text encode
        if finalize is True:
            h = self.pre_lookahead_layer(token)
        else:
            h = self.pre_lookahead_layer(
                token[:, : -self.pre_lookahead_len], context=token[:, -self.pre_lookahead_len :]
            )
        h = h.repeat_interleave(self.token_mel_ratio, dim=1)

        mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]

        # get conditions
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device).to(h.dtype)

        conds[:, :mel_len1] = prompt_feat

        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)
        feat, _ = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=max(1, int(n_timesteps)),
            streaming=streaming,
        )

        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat.float(), None
