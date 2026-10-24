# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Segment-boundary prompt updates preserving masks and action boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from vllm_omni.diffusion.interaction.modality_handlers.base import InteractionHandler
from vllm_omni.diffusion.interaction.types import InteractionChunkMetadata, InteractionPayload, InteractionSession
from vllm_omni.diffusion.worker.utils import StepRequestState


@dataclass
class HyNomasPromptSession(InteractionSession):
    pending: tuple[str, str] | None = None
    prompt: str | None = None


class HyNomasPromptHandler(InteractionHandler):
    """Last-write-wins text applied as a whole segment, without embedding lerp.

    Token masks and action ranges change with the prompt; blending embeddings
    using the generic prompt handler would retain the preceding segment's mask.
    Encoding happens in prepare_next_chunk on the inference worker.
    """

    modality: ClassVar[str] = "prompt"

    def validate_payload(
        self,
        state: StepRequestState,
        *,
        event_id: str,
        payload: InteractionPayload,
        transition_chunks: int | None,
    ) -> None:
        if not event_id or not isinstance(payload.get("prompt"), str) or not payload["prompt"].strip():
            raise ValueError("Hy-Nomas prompt interaction needs an event_id and nonempty prompt.")
        if transition_chunks not in (None, 0):
            raise ValueError("Hy-Nomas applies prompts at segment boundaries; use transition_chunks=0.")

    def enqueue(
        self,
        state: StepRequestState,
        *,
        event_id: str,
        received_at: float,
        payload: InteractionPayload,
        transition_chunks: int | None,
    ) -> None:
        self.validate_payload(state, event_id=event_id, payload=payload, transition_chunks=transition_chunks)
        session = state.interaction_sessions.setdefault("prompt", HyNomasPromptSession())
        assert isinstance(session, HyNomasPromptSession)
        prompt = payload["prompt"]
        assert isinstance(prompt, str)
        with session.lock:
            session.pending = (event_id, prompt)

    def apply_at_chunk_boundary(
        self,
        state: StepRequestState,
        *,
        boundary_at: float,
        chunk_index: int | None = None,
        num_media_frames: int | None = None,
        fps: float | None = None,
        num_latent_frames: int | None = None,
    ) -> InteractionChunkMetadata | None:
        session = state.interaction_sessions.get("prompt")
        assert isinstance(session, HyNomasPromptSession)
        with session.lock:
            session.last_boundary_at = boundary_at
            if session.pending is None:
                return None
            event_id, session.prompt = session.pending
            session.pending = None
        return InteractionChunkMetadata(
            started_event_ids=[event_id],
            active_event_ids=[event_id],
            completed_event_ids=[event_id],
        )
