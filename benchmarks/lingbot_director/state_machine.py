# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Deterministic Director state: a sensor reports state, this picks the prompt."""

from __future__ import annotations

CONFIRM_CHUNKS = 2
TRANSITION_LIMIT = 12
RECOVER_LIMIT = 6
RECOVER_ATTEMPTS = 2


class DirectorStateMachine:
    def __init__(self, execute_prompt: str, hold_prompt: str) -> None:
        self.execute_prompt = execute_prompt
        self.hold_prompt = hold_prompt
        self.state = "TRANSITION"
        self.prompt = execute_prompt
        self.streak = 0
        self.chunks_in_state = 0
        self.recover_attempts = 0

    def hold(self) -> None:
        """The world was restarted from a frame that already shows the goal."""
        self._enter("HOLD")

    def _enter(self, state: str) -> None:
        self.state = state
        self.streak = 0
        self.chunks_in_state = 0
        if state == "HOLD":
            self.prompt = self.hold_prompt
            self.recover_attempts = 0
        elif state == "RECOVER":
            self.prompt = self.execute_prompt

    def observe(self, achieved: bool) -> None:
        """Advance one chunk."""
        if self.state == "FAILED":
            return
        self.chunks_in_state += 1
        if self.state == "HOLD":
            if achieved:
                self.streak = 0
            else:
                self.streak += 1
            if self.streak >= CONFIRM_CHUNKS:
                self._enter("RECOVER")
            return
        if achieved:
            self.streak += 1
        else:
            self.streak = 0
        if self.streak >= CONFIRM_CHUNKS:
            self._enter("HOLD")
        elif self.state == "TRANSITION" and self.chunks_in_state >= TRANSITION_LIMIT:
            self._enter("FAILED")
        elif self.state == "RECOVER" and self.chunks_in_state >= RECOVER_LIMIT:
            self.recover_attempts += 1
            self.chunks_in_state = 0
            if self.recover_attempts >= RECOVER_ATTEMPTS:
                self._enter("FAILED")
