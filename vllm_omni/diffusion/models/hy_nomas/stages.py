# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Two-stage Hy-Nomas with ordered, request-owned incremental decoding."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import nn
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.transformers_utils.config import get_hf_file_to_dict

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.media import (
    DiffusionMediaOutput,
    VideoMediaOutput,
    VideoTensorEncoding,
    VideoTensorLayout,
    VideoTensorSpec,
    VideoValueRange,
)
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.hy_nomas.decoder import PixelUMMDecoder, PixelUMMStreamingDecodeSession
from vllm_omni.diffusion.models.hy_nomas.pipeline import HyNomasPipeline
from vllm_omni.diffusion.models.hy_nomas.pipeline import (
    get_hy_nomas_post_process_func as get_hy_nomas_post_process_func,
)
from vllm_omni.diffusion.models.hy_nomas.pipeline import (
    get_hy_nomas_pre_process_func as get_hy_nomas_pre_process_func,
)
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.segment import SEGMENT_KEY
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState

PAYLOAD_KEY = "hy_nomas_latent_segments"


class HyNomasDenoisePipeline(HyNomasPipeline):
    """Text encoder and DiT only; preserve the original KV commit/RNG order."""

    _decode_in_process: ClassVar[bool] = False
    _vae_modules: ClassVar[list[str]] = []
    # Yield between denoising steps so the process can receive controls and aborts.
    supports_chunk_step_grouping: ClassVar[bool] = False

    def post_decode(self, state: StepRequestState) -> DiffusionOutput:
        result = super().post_decode(state)
        latent = result.output
        assert isinstance(latent, torch.Tensor)
        segment = {
            "version": 1,
            "index": result.chunk_index,
            "total": result.total_chunks,
            "latent_offset": self._session(state).frame_offset - latent.shape[2],
            "pixel_frames": state.extra["pixels"][result.chunk_index],
            "finished": result.finished,
            # Full-request handoff holds latents on the host, not the DiT GPU.
            "latents": latent.detach().cpu(),
        }
        return DiffusionOutput(
            custom_output={
                PAYLOAD_KEY: [segment],
                SEGMENT_KEY: {"index": result.chunk_index, "finished": result.finished},
            },
            chunk_index=result.chunk_index,
            total_chunks=result.total_chunks,
            finished=result.finished,
        )

    def _combine_chunk_outputs(self, chunks: list[DiffusionOutput]) -> DiffusionOutput:
        return DiffusionOutput(custom_output={PAYLOAD_KEY: [c.custom_output[PAYLOAD_KEY][0] for c in chunks]})


def validate_latent_segments(
    segments: object,
    *,
    index: int = 0,
    offset: int = 0,
    total: int | None = None,
    geometry=None,
    dtype=None,
) -> list[dict]:
    """Reject incomplete, reordered or inconsistent payloads before decoding."""
    if not isinstance(segments, list) or not segments:
        raise ValueError("Hy-Nomas decode requires a nonempty latent segment list.")
    total = len(segments) if total is None else total
    for index, segment in enumerate(segments, start=index):
        if not isinstance(segment, dict):
            raise ValueError("Hy-Nomas latent segments must be dictionaries.")
        expected = {"version": 1, "index": index, "total": total, "latent_offset": offset}
        if any(type(segment.get(key)) is not int or segment[key] != value for key, value in expected.items()):
            raise ValueError("Invalid Hy-Nomas segment version, order, total or latent offset.")
        if segment.get("finished") is not (index == total - 1):
            raise ValueError("Only the last Hy-Nomas segment must be marked finished.")
        latent = segment.get("latents")
        if (
            not isinstance(latent, torch.Tensor)
            or latent.ndim != 5
            or latent.shape[:2] != (1, 32)
            or min(latent.shape[2:]) <= 0
            or not latent.is_floating_point()
        ):
            raise ValueError("Hy-Nomas latents must be floating [1,32,T,H,W] tensors.")
        if geometry is None:
            geometry, dtype = latent.shape[3:], latent.dtype
        if latent.shape[3:] != geometry or latent.dtype != dtype:
            raise ValueError("Hy-Nomas segment geometry and dtype must remain constant.")
        pixels = segment.get("pixel_frames")
        if type(pixels) is not int or not (latent.shape[2] - 1) * 4 < pixels <= latent.shape[2] * 4:
            raise ValueError("Invalid Hy-Nomas segment visible frame count.")
        offset += latent.shape[2]
    return segments


class HyNomasDecodePipeline(nn.Module, SupportsComponentDiscovery):
    """Decoder-only worker with request-local overlap state and no DiT/encoder."""

    dummy_run_num_frames: ClassVar[int] = 0
    _dit_modules: ClassVar[list[str]] = []
    _encoder_modules: ClassVar[list[str]] = []
    _vae_modules: ClassVar[list[str]] = ["decoder"]

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        if od_config.step_execution or od_config.quantization_config is not None:
            raise ValueError("Hy-Nomas decode requires unquantized, full-request execution.")
        parallel = od_config.parallel_config
        if parallel.tensor_parallel_size != 1 or parallel.sequence_parallel_size != 1:
            raise ValueError("Hy-Nomas decoding requires TP=1 and SP=1.")
        self.od_config = od_config
        config = get_hf_file_to_dict("decoder/config.json", od_config.model, revision=od_config.revision)
        if config is None:
            raise ValueError("Hy-Nomas requires decoder/config.json.")
        self.decoder = PixelUMMDecoder(**{k: v for k, v in config.items() if not k.startswith("_")})
        self.device = get_local_device()
        self._decode_sessions: dict[str, _DecodeState] = {}
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model, subfolder="decoder", prefix="decoder.", revision=od_config.revision
            )
        ]

    @torch.no_grad()
    def forward(self, batch: DiffusionRequestBatch) -> DiffusionOutput:
        if len(batch.requests) != 1:
            raise ValueError("Hy-Nomas decoding supports one request per batch.")
        request = batch.requests[0]
        if request.sampling_params.output_type == "latent":
            raise ValueError("The Hy-Nomas decode stage produces video, not latent output.")
        prompt = request.prompt
        if not isinstance(prompt, dict):
            raise ValueError("Hy-Nomas decode requires a stage payload.")
        sampling = request.sampling_params
        typed_video = (sampling.output_type or "np") == "np"
        # Retain legacy CPU output for other presentations. The shared runner
        # bounds conversion temporaries and owns uint8 reduction/SHM policy.
        to_cpu = not (
            typed_video
            and self.od_config.video_output_transport.enable_device_postprocess
            and not sampling.enable_frame_interpolation
        )
        if SEGMENT_KEY in prompt:
            output = self._decode_segment(request.request_id, prompt, to_cpu=to_cpu)
        else:
            segments = validate_latent_segments(prompt.get(PAYLOAD_KEY))
            # A local session cannot leak into a later request or survive an error.
            session = self.decoder.create_streaming_decode_session(to_cpu=to_cpu, retain_history=False)
            chunks = []
            for segment in segments:
                latent = segment["latents"].to(device=self.device, dtype=self.decoder.dtype)
                chunk = session.push(latent, pixel_frames=segment["pixel_frames"])
                if chunk.numel():
                    chunks.append(chunk)
            tail = session.finish()
            if tail.numel():
                chunks.append(tail)
            output = DiffusionOutput(output=torch.cat(chunks, dim=2))
        if typed_video and output.output is not None:
            output.media = DiffusionMediaOutput(
                video=VideoMediaOutput(
                    tensor=output.output,
                    spec=VideoTensorSpec(
                        layout=VideoTensorLayout.BCTHW,
                        encoding=VideoTensorEncoding.NORMALIZED_FLOAT,
                        value_range=VideoValueRange.NEGATIVE_ONE_TO_ONE,
                        preserve_input_dtype=True,
                    ),
                )
            )
            output.output = None
        return output

    def release_stream_request(self, request_id: str) -> None:
        self._decode_sessions.pop(request_id, None)

    def _decode_segment(self, request_id: str, prompt: dict, *, to_cpu: bool = True) -> DiffusionOutput:
        state = self._decode_sessions.get(request_id)
        try:
            segments = prompt.get(PAYLOAD_KEY)
            if not isinstance(segments, list) or len(segments) != 1:
                raise ValueError("Streaming decode requires exactly one segment.")
            segment = segments[0]
            if not isinstance(segment, dict):
                raise ValueError("Streaming segment must be a dictionary.")
            total = segment.get("total")
            if type(total) is not int or total <= 0:
                raise ValueError("Streaming segment total must be positive.")
            validate_latent_segments(
                segments,
                index=state.index if state else 0,
                offset=state.offset if state else 0,
                total=state.total if state else total,
                geometry=state.geometry if state else None,
                dtype=state.dtype if state else None,
            )
            marker = {"index": segment["index"], "finished": segment["finished"]}
            if prompt[SEGMENT_KEY] != marker:
                raise ValueError("Segment envelope does not match latent metadata.")
            if state is None:
                if self._decode_sessions:
                    raise ValueError("Hy-Nomas streaming decode supports one active session.")
                state = _DecodeState(
                    self.decoder.create_streaming_decode_session(to_cpu=to_cpu, retain_history=False),
                    total,
                    segment["latents"].shape[3:],
                    segment["latents"].dtype,
                )
                self._decode_sessions[request_id] = state
            latent = segment["latents"].to(device=self.device, dtype=self.decoder.dtype)
            chunks = [state.session.push(latent, pixel_frames=segment["pixel_frames"])]
            state.index += 1
            state.offset += latent.shape[2]
            if segment["finished"]:
                chunks.append(state.session.finish())
                self.release_stream_request(request_id)
            chunks = [chunk for chunk in chunks if chunk.numel()]
            output = torch.cat(chunks, dim=2) if chunks else None
            # Each invocation is terminal to the request scheduler. The envelope
            # carries the parent rollout lifetime to the orchestrator.
            return DiffusionOutput(
                output=output,
                finished=True,
                chunk_index=segment["index"],
                total_chunks=total,
                custom_output={SEGMENT_KEY: {**marker, "ack": True}},
                started_event_ids=prompt.get("diffusion_segment_events", {}).get("started_event_ids", []),
                active_event_ids=prompt.get("diffusion_segment_events", {}).get("active_event_ids", []),
                completed_event_ids=prompt.get("diffusion_segment_events", {}).get("completed_event_ids", []),
            )
        except BaseException:
            self.release_stream_request(request_id)
            raise

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights)


@dataclass
class _DecodeState:
    session: PixelUMMStreamingDecodeSession
    total: int
    geometry: torch.Size
    dtype: torch.dtype
    index: int = 0
    offset: int = 0
