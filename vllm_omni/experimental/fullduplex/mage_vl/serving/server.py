# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A WebSocket duplex endpoint for Mage-VL, owned by the model package.

Why this is not the shared ``/v1/realtime?duplex=1`` path: that path's input layer
(``openai/realtime_input.py``) only accepts PCM audio buffers and text content parts --
there is no image or video append in it, and no per-model hook to add one -- and the
``ServingRuntimeAdapter`` protocol it loads is PCM-shaped down to
``PcmAppendBuffer`` and ``interrupted_tts_prefix``. Serving a video-in/text-out model
there means teaching the shared transport a **generic** input-modality append, which the
package's own rule says to defer: land the capability in the model package first, promote
it when a second model needs it. PersonaPlex made the same call and ships
``personaplex/serving/``; this mirrors it.

The wire protocol is the core duplex protocol verbatim
(``vllm_omni/experimental/fullduplex/core/protocol.py``), so nothing new was invented:

    client -> ``{"type": "input.append", "modality": "video", "data": "<data: URI>"}``
              ``{"type": "input.append", "modality": "text", "data": "what is he doing?"}``
              ``{"type": "response.cancel"}``   (barge-in)
              ``{"type": "close"}``
    server -> ``response.created`` / ``response.delta`` / ``response.done`` /
              ``response.cancelled`` / ``error``, plus the gate's ``response.listen`` and
              ``response.speak`` decision events.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Imported at module scope on purpose: this module uses postponed annotations, and FastAPI
# resolves a handler's annotations against module globals. With ``WebSocket`` imported
# inside the factory the name would not resolve, and the route would quietly treat the
# socket as a request parameter and fail at connect time.
from fastapi import APIRouter, FastAPI, WebSocket

from vllm_omni.experimental.fullduplex.core import protocol as ev
from vllm_omni.experimental.fullduplex.core.runtime import DuplexRuntime
from vllm_omni.experimental.fullduplex.core.session import DuplexSession, DuplexSessionConfig
from vllm_omni.experimental.fullduplex.mage_vl.adapter import MageVLDuplexAdapter
from vllm_omni.experimental.fullduplex.mage_vl.policy import MageVLStreamConfig

# A factory rather than an instance: the gate keeps recurrent state, so every session
# needs its own scorer. Sharing one would blend two viewers' histories.
AdapterFactory = Callable[[Callable[[dict[str, Any]], Any]], MageVLDuplexAdapter]


@dataclass(frozen=True)
class MageVLServingConfig:
    """Session-lifecycle limits, independent of what the model does."""

    max_sessions: int = 1
    idle_timeout_s: float = 300.0
    stream_config: MageVLStreamConfig | None = None

    def __post_init__(self) -> None:
        if self.max_sessions < 1:
            raise ValueError(f"max_sessions must be >= 1, got {self.max_sessions}")
        if self.idle_timeout_s <= 0:
            raise ValueError(f"idle_timeout_s must be > 0, got {self.idle_timeout_s}")


class SessionsFull(RuntimeError):
    """The server is already serving as many streams as it admits."""


class MageVLDuplexServer:
    """Session admission and the socket-to-runtime pump.

    Capacity is a hard limit rather than a queue: a video stream that is admitted late has
    already missed the footage it would have been answering about, so refusing is more
    honest than accepting and lagging.
    """

    def __init__(
        self,
        adapter_factory: AdapterFactory,
        config: MageVLServingConfig | None = None,
    ) -> None:
        self._adapter_factory = adapter_factory
        self.config = config or MageVLServingConfig()
        self._active: dict[str, float] = {}
        self._lock = asyncio.Lock()

    @property
    def active_sessions(self) -> int:
        return len(self._active)

    async def _admit(self, session_id: str) -> None:
        async with self._lock:
            self._evict_idle()
            if len(self._active) >= self.config.max_sessions:
                raise SessionsFull(
                    f"all {self.config.max_sessions} session slot(s) are in use"
                )
            self._active[session_id] = time.monotonic()

    async def _release(self, session_id: str) -> None:
        async with self._lock:
            self._active.pop(session_id, None)

    def _evict_idle(self) -> None:
        """Drop slots whose stream went quiet: a dropped client never sends ``close``."""
        deadline = time.monotonic() - self.config.idle_timeout_s
        for session_id, last_seen in list(self._active.items()):
            if last_seen < deadline:
                del self._active[session_id]

    def _touch(self, session_id: str) -> None:
        if session_id in self._active:
            self._active[session_id] = time.monotonic()

    async def run_session(self, socket: Any, session_id: str) -> None:
        """Drive one accepted WebSocket to completion."""
        try:
            await self._admit(session_id)
        except SessionsFull as full:
            await socket.send_json(ev.error(str(full)))
            await socket.close()
            return

        session = DuplexSession(
            session_id=session_id,
            config=DuplexSessionConfig(
                input_modalities=("video", "text"),
                output_modalities=("text",),
                proactive=True,
            ),
        )

        async def emit(event: dict[str, Any]) -> None:
            await socket.send_json(event)

        adapter = self._adapter_factory(emit)
        if self.config.stream_config is not None:
            adapter.policy.config = self.config.stream_config
        runtime = DuplexRuntime(session, adapter)

        try:
            await runtime.run(self._events(socket, session_id), emit)
        except Exception as failure:  # a broken socket must not take the server down
            with contextlib.suppress(Exception):
                await socket.send_json(ev.error(f"session failed: {failure}"))
        finally:
            await self._release(session_id)
            with contextlib.suppress(Exception):
                await socket.close()

    async def _events(self, socket: Any, session_id: str):
        """Client frames as core duplex events, ending on ``close`` or disconnect."""
        while True:
            try:
                message = await asyncio.wait_for(
                    socket.receive_json(), timeout=self.config.idle_timeout_s
                )
            except TimeoutError:
                yield {"type": ev.CLOSE}
                return
            except Exception:  # disconnect: end the stream, do not crash the pump
                yield {"type": ev.CLOSE}
                return

            self._touch(session_id)
            if not isinstance(message, dict) or "type" not in message:
                await socket.send_json(ev.error("each frame must be an object with a type"))
                continue

            yield message
            if message.get("type") == ev.CLOSE:
                return


def create_router(server: MageVLDuplexServer, *, path: str = "/v1/duplex/mage-vl"):
    """A router the OpenAI api_server can mount, or an app can carry on its own."""
    router = APIRouter()

    @router.websocket(path)
    async def duplex(websocket: WebSocket) -> None:  # pragma: no cover - thin binding
        await websocket.accept()
        session_id = websocket.query_params.get("session_id") or f"mage-{id(websocket):x}"
        await server.run_session(websocket, session_id)

    return router


def create_app(server: MageVLDuplexServer, *, path: str = "/v1/duplex/mage-vl"):
    app = FastAPI(title="Mage-VL duplex")
    app.include_router(create_router(server, path=path))
    return app


__all__ = [
    "AdapterFactory",
    "MageVLDuplexServer",
    "MageVLServingConfig",
    "SessionsFull",
    "create_app",
    "create_router",
]
