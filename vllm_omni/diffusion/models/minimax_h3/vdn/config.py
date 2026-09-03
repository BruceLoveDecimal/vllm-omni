# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""What a VDN-H3 checkpoint says the architecture is, and where its rows sit.

Two objects, deliberately separate. :class:`MiniMaxH3HybridAttentionConfig` is
the checkpoint's own architecture statement (VDN's ``model_spec.json``
``transforms[0].config``), fixed for the life of the server. VDN allows several
values per field because it also trains; the released checkpoint uses exactly
one combination, so anything else is refused here rather than implemented
untested - a second delta rule or a different anchor mode would be a branch
nobody ever runs and nobody would notice breaking.

:class:`MiniMaxH3HybridGeometry` is the per-request packed-row layout the
branch needs and H3's shared ``VideoTokenLayout`` does not carry: the linear
branch seeds both of its scans from the PROMPT rows, which is not the same set
as "everything the softmax keeps dense" (that also holds the soundtrack). Plain
ints throughout, so reading it never forces a device-to-host sync.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class VdnConfigError(ValueError):
    """A VDN checkpoint or request states something this server cannot serve."""


# The single combination the released checkpoints (stage-b-step-2000 and
# stage-dmd-step-250) were trained and published with.
SUPPORTED_DELTA_RULE = "vdn_solve"
SUPPORTED_BRIDGE = "alpha"
SUPPORTED_ANCHOR_FRAMES = "both"
SUPPORTED_SHORT_CONV_TARGETS = ("k", "v")
HYBRID_TRANSFORM_TYPE = "hybrid_attention"
HYBRID_TRANSFORM_VERSION = 2
SPEC_FORMAT_VERSION = 2

WINDOW_IMPLS = ("auto", "grouped", "varlen")

_TRANSFORM_KEYS = frozenset({"anchor_frames", "enable_softmax_gate", "softmax_attention", "linear_attention"})
_SOFTMAX_KEYS = frozenset({"chunk", "radius"})
_LINEAR_KEYS = frozenset({"a_fp32", "bridge", "delta_rule", "enable_text_state", "linear_head_dim", "short_conv"})


@dataclass(frozen=True)
class MiniMaxH3HybridAttentionConfig:
    """The hybrid architecture, plus the runtime knobs that do not enter a spec.

    ``chunk``/``radius`` are the window: frame ``t`` belongs to chunk
    ``t // chunk`` and attends whole chunks ``[c - radius, c + radius]``. The
    VAE codes every ``chunk`` latent frames independently, so a window that
    stops mid-chunk shows a query a fragment of a unit that was never coded as
    separable; alignment, not width, is what the chunk mode buys.
    """

    # --- the checkpoint's architecture statement ---
    chunk: int
    radius: int
    anchor_frames: str
    enable_softmax_gate: bool
    delta_rule: str
    bridge: str
    linear_head_dim: int
    short_conv_targets: tuple[str, ...]
    enable_text_state: bool
    a_fp32: bool

    # --- runtime, never part of the checkpoint ---
    #: Window groups whose K/V is gathered in one call. The gather is the
    #: branch's memory peak (~0.5 GiB per group at 768p/TP1), so this trades
    #: transient memory against kernel launches.
    window_group_batch: int = 4
    #: ``auto`` picks the packed-varlen path when the resolved attention backend
    #: isolates multi-document cu_seqlens and the grouped path otherwise.
    window_impl: str = "auto"
    #: Ablation only: ``False`` runs the window without the linear branch, which
    #: is NOT the released model and only exists to attribute a speedup.
    linear_attention_enabled: bool = True

    @property
    def full_cover_frames(self) -> int:
        """Frame counts at or below which the window IS full attention.

        A clip whose every frame sees every chunk has nothing outside the
        window, so the linear branch would double-count and the hybrid
        degenerates to the dense teacher (VDN's ``full_cover``).
        """
        return self.chunk * (2 * self.radius + 1)

    def covers_all_frames(self, num_frames: int) -> bool:
        return num_frames <= self.full_cover_frames

    @classmethod
    def from_transform_config(
        cls,
        config: Mapping[str, Any],
        *,
        attention_head_dim: int,
        window_group_batch: int = 4,
        window_impl: str = "auto",
        linear_attention_enabled: bool = True,
    ) -> MiniMaxH3HybridAttentionConfig:
        """Read and check VDN's ``transforms[0].config``.

        Every field is refused rather than defaulted: a spec that omits one is
        not the released architecture, and guessing would serve a model whose
        arithmetic silently disagrees with the weights it just loaded.
        """
        _refuse_unknown(config, _TRANSFORM_KEYS, "transform config")
        softmax = _require_mapping(config, "softmax_attention")
        linear = _require_mapping(config, "linear_attention")
        _refuse_unknown(softmax, _SOFTMAX_KEYS, "softmax_attention")
        _refuse_unknown(linear, _LINEAR_KEYS, "linear_attention")

        anchor_frames = _require(config, "anchor_frames", str)
        if anchor_frames != SUPPORTED_ANCHOR_FRAMES:
            raise VdnConfigError(
                f"anchor_frames={anchor_frames!r}: this server serves {SUPPORTED_ANCHOR_FRAMES!r} only. "
                "Only that mode makes the softmax window and the linear branch an exact partition, "
                "which is what lets the branch drop the two anchor frames."
            )
        delta_rule = _require(linear, "delta_rule", str)
        if delta_rule != SUPPORTED_DELTA_RULE:
            raise VdnConfigError(
                f"delta_rule={delta_rule!r}: this server serves {SUPPORTED_DELTA_RULE!r} only "
                "(the exact Cholesky solve the released checkpoints were trained with)"
            )
        bridge = _require(linear, "bridge", str)
        if bridge != SUPPORTED_BRIDGE:
            raise VdnConfigError(f"bridge={bridge!r}: this server serves {SUPPORTED_BRIDGE!r} only")
        short_conv = _require_mapping(linear, "short_conv")
        targets = tuple(short_conv.get("targets") or ())
        if targets != SUPPORTED_SHORT_CONV_TARGETS:
            raise VdnConfigError(
                f"short_conv.targets={list(targets)}: this server serves {list(SUPPORTED_SHORT_CONV_TARGETS)} only"
            )
        if not _require(linear, "a_fp32", bool):
            raise VdnConfigError(
                "a_fp32=false reproduces a pre-fix bf16 statistic whose asymmetry pushes the smallest "
                "eigenvalue of I+A below 1, so the Cholesky the delta rule depends on can fail outright"
            )
        if not _require(linear, "enable_text_state", bool):
            raise VdnConfigError("enable_text_state=false is not the released architecture")
        if not _require(config, "enable_softmax_gate", bool):
            raise VdnConfigError("enable_softmax_gate=false is not the released architecture")

        linear_head_dim = _require(linear, "linear_head_dim", int)
        if linear_head_dim != attention_head_dim:
            raise VdnConfigError(
                f"linear_head_dim={linear_head_dim} != attention head_dim={attention_head_dim}; "
                "the branch shares the softmax branch's QKV projections, so the two must agree"
            )
        chunk = _require(softmax, "chunk", int)
        radius = _require(softmax, "radius", int)
        if chunk <= 0:
            raise VdnConfigError(f"softmax_attention.chunk must be positive, got {chunk}")
        if radius < 0:
            raise VdnConfigError(f"softmax_attention.radius must be non-negative, got {radius}")

        if window_group_batch < 1:
            raise VdnConfigError(f"window_group_batch must be >= 1, got {window_group_batch}")
        if window_impl not in WINDOW_IMPLS:
            raise VdnConfigError(f"window_impl={window_impl!r} not in {list(WINDOW_IMPLS)}")

        return cls(
            chunk=chunk,
            radius=radius,
            anchor_frames=anchor_frames,
            enable_softmax_gate=True,
            delta_rule=delta_rule,
            bridge=bridge,
            linear_head_dim=linear_head_dim,
            short_conv_targets=targets,
            enable_text_state=True,
            a_fp32=True,
            window_group_batch=int(window_group_batch),
            window_impl=window_impl,
            linear_attention_enabled=bool(linear_attention_enabled),
        )


@dataclass(frozen=True)
class MiniMaxH3HybridGeometry:
    """Where each kind of row sits in one packed T2VA sequence.

    The T2VA packing is ``[text | audio | video | pad]`` (see
    ``packed_sequence.py``), so the branch's three row sets are contiguous
    ranges rather than index tensors. ``used_len`` excludes the alignment
    padding, which belongs to no branch and whose output rows stay zero.
    """

    seq_len: int
    used_len: int
    text_start: int
    text_len: int
    video_start: int
    num_frames: int
    frame_height: int
    frame_width: int

    def __post_init__(self) -> None:
        if self.text_len <= 0:
            raise VdnConfigError("the linear branch seeds its scans from the prompt; the layout carries no text rows")
        if self.num_frames <= 0 or self.frame_height <= 0 or self.frame_width <= 0:
            raise VdnConfigError(
                f"invalid video geometry: frames={self.num_frames}, grid={self.frame_height}x{self.frame_width}"
            )
        if self.text_start != 0:
            raise VdnConfigError(f"T2VA packs text first; got text_start={self.text_start}")
        if self.video_start < self.text_len:
            raise VdnConfigError(
                f"video rows overlap the prompt: video_start={self.video_start}, text_len={self.text_len}"
            )
        if self.video_end != self.used_len:
            raise VdnConfigError(
                "the hybrid branch expects the target video to be the last content block of the packed "
                f"sequence: video ends at {self.video_end}, content ends at {self.used_len}"
            )
        if self.used_len > self.seq_len:
            raise VdnConfigError(f"used rows {self.used_len} exceed the packed sequence {self.seq_len}")

    @property
    def tokens_per_frame(self) -> int:
        return self.frame_height * self.frame_width

    @property
    def video_end(self) -> int:
        return self.video_start + self.num_frames * self.tokens_per_frame

    @property
    def frame_size(self) -> tuple[int, int]:
        return self.frame_height, self.frame_width

    @property
    def num_video_rows(self) -> int:
        return self.num_frames * self.tokens_per_frame

    def frame_rows(self, frame: int) -> tuple[int, int]:
        """The half-open packed-row range of one latent frame."""
        start = self.video_start + frame * self.tokens_per_frame
        return start, start + self.tokens_per_frame


def _require(config: Mapping[str, Any], key: str, kind: type) -> Any:
    if key not in config:
        raise VdnConfigError(f"the VDN transform config omits {key!r}; a spec states every resolved value")
    value = config[key]
    # bool is an int subclass, so an int field must not silently accept True.
    if kind is int and isinstance(value, bool):
        raise VdnConfigError(f"{key} must be an integer, got {value!r}")
    if not isinstance(value, kind):
        raise VdnConfigError(f"{key} must be {kind.__name__}, got {value!r}")
    return value


def _require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise VdnConfigError(f"the VDN transform config omits the {key!r} section")
    return value


def _refuse_unknown(config: Mapping[str, Any], known: frozenset[str], where: str) -> None:
    unknown = sorted(set(config) - known)
    if unknown:
        raise VdnConfigError(
            f"unknown {where} keys {unknown}; this server reads the released architecture only, "
            "and an unread key would change what the checkpoint means without changing what runs"
        )


__all__ = [
    "HYBRID_TRANSFORM_TYPE",
    "HYBRID_TRANSFORM_VERSION",
    "SPEC_FORMAT_VERSION",
    "WINDOW_IMPLS",
    "MiniMaxH3HybridAttentionConfig",
    "MiniMaxH3HybridGeometry",
    "VdnConfigError",
]
