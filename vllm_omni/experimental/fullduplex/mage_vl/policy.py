# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""When Mage-VL should speak, and what it looks at when it does.

The model watches a video stream in fixed-length segments. Its cognition gate scores each
segment with ``p_speak``; this module turns that score into a decision and keeps the
sliding window of segments a response is built from.

The policy is deliberately separate from the adapter: the adapter owns the duplex
lifecycle, this owns the model's judgement about it, which is the part worth testing on
its own and tuning per session.
"""

from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass
from typing import Any


class Decision(enum.Enum):
    LISTEN = "listen"
    SPEAK = "speak"


@dataclass(frozen=True)
class MageVLStreamConfig:
    """Session-level knobs, defaulting to the reference pipeline's settings."""

    speak_threshold: float = 0.5
    segment_seconds: float = 8.0
    # How many recent segments a response is built from. The gate's own state spans the
    # whole session; the prompt cannot, so this bounds what the decoder re-reads.
    window_segments: int = 8
    # Segments to stay quiet for after answering. Without it the gate, which stays
    # confident for as long as the interesting thing is on screen, would re-fire on the
    # very next segment and talk over its own answer.
    cooldown_segments: int = 1

    def __post_init__(self) -> None:
        if not 0.0 <= self.speak_threshold <= 1.0:
            raise ValueError(f"speak_threshold must be in [0, 1], got {self.speak_threshold}")
        if self.window_segments < 1:
            raise ValueError(f"window_segments must be >= 1, got {self.window_segments}")
        if self.cooldown_segments < 0:
            raise ValueError(f"cooldown_segments must be >= 0, got {self.cooldown_segments}")


class MageVLGatePolicy:
    """Per-session gate state: the score history, the cooldown, and the prompt window."""

    def __init__(self, config: MageVLStreamConfig | None = None) -> None:
        self.config = config or MageVLStreamConfig()
        self._window: deque[Any] = deque(maxlen=self.config.window_segments)
        self._cooldown = 0
        self._pending = Decision.LISTEN
        self.last_score: float | None = None
        self.segments_seen = 0

    def observe(self, segment: Any, p_speak: float) -> Decision:
        """Fold one scored segment into the window and decide."""
        self._window.append(segment)
        self.segments_seen += 1
        self.last_score = p_speak

        if self._cooldown > 0:
            self._cooldown -= 1
            self._pending = Decision.LISTEN
        else:
            self._pending = (
                Decision.SPEAK if p_speak >= self.config.speak_threshold else Decision.LISTEN
            )
        return self._pending

    @property
    def decision(self) -> Decision:
        return self._pending

    def window(self) -> list[Any]:
        """The segments a response should be built from, oldest first."""
        return list(self._window)

    def note_response_started(self) -> None:
        """Called when a response actually begins: arm the cooldown, disarm the trigger."""
        self._cooldown = self.config.cooldown_segments
        self._pending = Decision.LISTEN

    def note_interrupted(self) -> None:
        """Barge-in: drop the standing trigger but keep the window and the cooldown.

        The window is what the user is interrupting *about*, so throwing it away would
        make the next answer blind to the thing that prompted the interruption.
        """
        self._pending = Decision.LISTEN

    def reset(self) -> None:
        self._window.clear()
        self._cooldown = 0
        self._pending = Decision.LISTEN
        self.last_score = None
        self.segments_seen = 0


__all__ = ["Decision", "MageVLGatePolicy", "MageVLStreamConfig"]
