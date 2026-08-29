# MiniMax-H3 Sliding-Window Long Video Generation

> **Status:** proposal (spec) for
> [vllm-project/vllm-omni#6737](https://github.com/vllm-project/vllm-omni/issues/6737).
> Nothing in this document is implemented yet; file references point at the
> code the proposal builds on.

## Table of Contents

1. [Overview](#overview)
2. [Goals and Non-Goals](#goals-and-non-goals)
3. [API Surface](#api-surface)
4. [Window Planning](#window-planning)
5. [Overlap Conditioning](#overlap-conditioning)
6. [Orchestration Loop](#orchestration-loop)
7. [Caching Considerations](#caching-considerations)
8. [Stitching](#stitching)
9. [Determinism](#determinism)
10. [Error Handling and Limits](#error-handling-and-limits)
11. [Prior Art in This Codebase](#prior-art-in-this-codebase)
12. [Files to Change](#files-to-change)
13. [Testing Plan](#testing-plan)
14. [Milestones](#milestones)
15. [Open Questions](#open-questions)

## Overview

MiniMax-H3 natively generates joint video+audio clips of 4–15 seconds at a
fixed 24 FPS. The pipeline enforces this contract today via
`MINIMAX_H3_MIN_OUTPUT_SECONDS = 4.0` and
`MINIMAX_H3_MAX_OUTPUT_SECONDS = 15.0` in
`vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py`, and any
request above 15 s is rejected with an `OmniClientError`.

This proposal adds **sliding-window generation with overlapped
conditioning**: a request for `duration > 15 s` is decomposed into N
windows, each within the model's native 4–15 s contract. Every window after
the first is conditioned on the tail latent frames (video **and** audio) of
the previous window, so motion, identity, and sound continue across the
seam. The stitched result is returned as one clip, exactly as if the model
had generated it in a single pass.

Requests with `duration <= 15 s` are untouched: they keep the native
single-pass path, which remains the quality- and cost-optimal mode.

## Goals and Non-Goals

**Goals**

- Accept `duration_seconds` up to a configurable windowed maximum
  (default **60 s**) for `t2va` and `fl2va`.
- Each window individually satisfies the model's native contract
  (4–15 s, 24 FPS, aligned frame counts).
- Temporal continuity across seams via overlapped latent conditioning for
  both the video and the audio stream.
- No pixel-space VAE round trip for conditioning: the tail **latents** of
  window *i* condition window *i+1* directly.
- Single response object; callers cannot tell (other than by latency) that
  windowing happened.

**Non-Goals (v1)**

- `ref2va` long-form support. Its reference-block packing is orthogonal and
  can be layered on later.
- Cross-window transformer KV caching (see
  [Caching Considerations](#caching-considerations); listed as follow-up).
- Sophisticated blending (optical-flow warping, latent interpolation). v1
  ships a hard cut for video and a short crossfade for audio.
- Streaming partial windows back to the client.

## API Surface

Windowing is triggered implicitly by duration, per the issue:

```json
{
  "extra_params": {
    "target": { "duration_seconds": 30.0 }
  }
}
```

- `4 <= duration <= 15` → native single-pass path (unchanged).
- `15 < duration <= MINIMAX_H3_MAX_WINDOWED_OUTPUT_SECONDS` → sliding-window
  path.
- Outside `[4, windowed max]` → `OmniClientError`, with the error message
  updated to name both limits.

New optional knobs under `extra_args` (all with safe defaults, validated
like the existing `target` fields in `_resolve_shape`):

| Key | Default | Meaning |
|-----|---------|---------|
| `window_overlap_seconds` | `1.0` | Duration of the conditioning overlap between adjacent windows. Clamped to `[0.5, 3.0]`, then aligned to the latent grid. |
| `max_window_seconds` | `15.0` | Upper bound the planner may use per window. Must be within `[8.0, 15.0]`; lowering it trades quality for smaller per-window memory. |

The equivalent `num_frames` request path (`sampling.num_frames`) follows the
same rule: `num_frames / 24 > 15` triggers windowing.

Constants added next to the existing ones in `pipeline_minimax_h3.py`:

```python
MINIMAX_H3_MAX_WINDOWED_OUTPUT_SECONDS = 60.0
MINIMAX_H3_DEFAULT_WINDOW_OVERLAP_SECONDS = 1.0
```

## Window Planning

A new pure-Python planner, `vllm_omni/diffusion/models/minimax_h3/
sliding_window.py`, computes the window layout. Working in **frames** (not
seconds) avoids drift; all counts are aligned with
`minimax_h3_align_frame_count` (`time_request.py:112`) so each window's
`latent_t`/`audio_t` are valid for the shape planner.

Definitions, all in frames:

- `D` — requested total frames (`round(duration * 24)`, then aligned).
- `O` — overlap frames (`round(window_overlap_seconds * 24)`, aligned to
  the temporal latent stride so the overlap is a whole number of latent
  frames).
- `Lmax` / `Lmin` — max/min frames per window (from
  `max_window_seconds` / `MINIMAX_H3_MIN_OUTPUT_SECONDS`).

Each window *i* generates `L_i` frames; windows after the first contribute
`L_i - O` net new frames, because their first `O` frames re-cover the tail
of the previous window and are dropped at stitch time:

```
D = L_1 + Σ_{i=2..N} (L_i - O)
```

The planner:

1. Computes the minimal window count
   `N = 1 + ceil((D - Lmax) / (Lmax - O))`.
2. Distributes length evenly: every window gets
   `L ≈ (D + (N - 1) * O) / N`, rounded per-window to the frame-alignment
   grid, with the rounding remainder absorbed by the last window.
3. Verifies every `L_i ∈ [Lmin, Lmax]`. Even distribution makes this hold
   for all `D` in range: it is exactly the guard against the degenerate
   `15 s + 1 s` split — **16 s is never planned as 15+1**; it becomes two
   windows of ~8.5 s each (net 8.5 + 7.5 with a 1 s overlap).
4. Emits an immutable `WindowPlan`:

```python
@dataclass(frozen=True)
class WindowSpec:
    index: int
    num_frames: int          # L_i, aligned
    net_frames: int          # L_i, or L_i - overlap_frames for i > 0
    overlap_frames: int      # 0 for the first window
    latent_t: int            # from MINIMAX_H3_SHAPE_PLANNER
    audio_t: int
    overlap_latent_t: int    # latent frames pinned as history condition
    overlap_audio_t: int

@dataclass(frozen=True)
class WindowPlan:
    windows: tuple[WindowSpec, ...]
    total_frames: int        # Σ net_frames == D
```

Worked examples (24 FPS, `O = 24` frames = 1 s, before grid rounding):

| Request | N | Per-window length | Net contribution |
|---------|---|-------------------|------------------|
| 16 s | 2 | ~8.5 s each | 8.5 s + 7.5 s |
| 30 s | 3 | ~10.7 s each | 10.7 + 9.7 + 9.7 s |
| 60 s | 5 | ~12.8 s each | 12.8 + 11.8 × 4 s |

## Overlap Conditioning

This is the core model-side change, and it is a **generalization of the
existing FL2VA keyframe mechanism** rather than a new one.

Today (`packed_sequence.py:116`, `denoise_loop.py`):

- FL2VA reserves *cond rows* in the packed sequence
  (`MINIMAX_H3_IMGVID_COND_ID`), one frame-row block per keyframe, with
  `keyframe_frame_indices ∈ {[0], [-1], [0, -1]}` mapping each block to a
  video frame position.
- Cond rows carry clean latents (`keyframe_cond_rows`), are excluded from
  the Euler update by `update_mask`, and are pinned at the condition
  noise-aug timestep (`MINIMAX_H3_IMGVID_COND_TIMESTEP = 0.999`; audio refs
  use `MINIMAX_H3_AUDIO_REF_COND_TIMESTEP = 1.0`).

Proposed generalization — a **history block**:

1. `minimax_h3_packed_sequence` accepts
   `history_latent_t: int = 0` and `history_audio_t: int = 0`. When set,
   the first `history_latent_t` video latent frames and the first
   `history_audio_t` audio latent rows of the *target* region are marked as
   condition rows: `update_mask` is `False` there and their timestep rows
   are pinned exactly like keyframe cond rows. Unlike FL2VA keyframes they
   occupy the target grid positions themselves (in-place conditioning, the
   LongCat/Wan-style layout) instead of separate cond blocks, so RoPE/g-grid
   coordinates need no changes.
2. `diffuse()` accepts optional `history_video_latent` /
   `history_audio_latent` tensors. `_initial_noise` output is overwritten
   at the history positions with these clean latents before the loop
   starts, mirroring how `keyframe_cond_rows` anchor cond rows in
   `denoise_loop.py` (`video_rows[~update] = cond_anchor`).
3. The Euler-ancestral update in `denoise_loop.py` already skips
   non-update rows, so no scheduler change is needed beyond passing the
   extended masks.
4. Noise augmentation of the history block reuses
   `minimax_h3_imgvid_cond_noise_aug_rows` / `minimax_h3_audio_cond_noise_aug_rows`
   (`condition_noise.py`) so history conditioning matches the training-time
   conditioning distribution of the FL2VA partition.

FL2VA long requests compose naturally: window 1 keeps its first-frame
keyframe; a `[0, -1]` (first *and* last image) request pins the last-frame
keyframe on the **final** window instead, so the last image constrains the
end of the full clip.

**Latents are reused directly.** Window *i*'s denoised
`(video_latent, audio_latent)` are kept on device; the planner's
`overlap_latent_t` / `overlap_audio_t` tail slices become the next window's
history tensors. There is deliberately **no decode→re-encode round trip**
(LongCat-Video-Avatar re-encodes decoded pixels in `_generate_avc`,
`pipeline_longcat_video_avatar.py:1605`; we skip that both for speed and to
avoid VAE round-trip drift accumulating across windows).

## Orchestration Loop

Lives entirely inside `MiniMaxH3Pipeline.forward` /
a new `_forward_windowed` helper — no scheduler, engine, or stage changes:

```
plan = plan_windows(...)                      # sliding_window.py
history = None
video_chunks, audio_chunks = [], []
for spec in plan.windows:
    video_lat, audio_lat = self.diffuse(..., history=history, seed=seed_i)
    history = tail_slices(video_lat, audio_lat, spec_next)
    video, audio = self.decode(video_lat, audio_lat, ...)   # per window
    video_chunks.append(drop_overlap(video, spec))
    audio_chunks.append(drop_overlap(audio, spec))
return stitch(video_chunks, audio_chunks)
```

Notes:

- Text encoding, reference preparation, and shape resolution run **once**
  before the loop; only `diffuse`/`decode` repeat.
- Decoding per window keeps peak VAE memory identical to today's
  single-pass path; only the two tail latent slices persist across
  iterations.
- Progress/timeout: per-request wall time scales ~linearly with N. The
  worker timeout raised in #4845 (large broadcast payloads) already covers
  long outputs; the windowed path must report
  `stage_durations` per window for observability.

## Caching Considerations

- **Cache-DiT** (step-wise feature caching) operates within one denoising
  loop. Each window is an independent loop, so the Cache-DiT context is
  **reset at window boundaries**; no cross-window state is valid.
- **Clean-latent KV cache (follow-up, not v1):** because history rows are
  clean and pinned at a fixed timestep, their K/V are identical at every
  denoising step. LongCat-Video-Avatar exploits this
  (`_cache_clean_latents`, `pipeline_longcat_video_avatar.py:755`):
  compute the condition tokens' K/V once per window, drop them from the
  per-step input, and attend to the cache. For MiniMax-H3 this requires
  `kv_cache_dict` plumbing through `minimax_h3_transformer.py` and the
  packed-sequence attention path, so it is specced as a separate
  performance milestone. v1 keeps history rows in the sequence every step
  (the FL2VA keyframe cost model, extended to `O` frames).

## Stitching

- **Video:** hard cut. Window *i+1*'s first `overlap_frames` decoded frames
  are dropped; frames are concatenated. This is the issue's stated v1
  baseline.
- **Audio:** a hard cut can click. The overlap region is decoded by both
  windows, so v1 applies a short (~20 ms) equal-power crossfade centered on
  the seam inside the overlap, at negligible cost.
- Output assembly reuses `_minimax_h3_post_process` unchanged: one video
  tensor, one waveform, `fps=24`, `audio_sample_rate=32000`.

## Determinism

Per-window seeds derive from the request seed the same way multi-output
seeds do (`_minimax_h3_output_seeds`): `seed_i = derive(seed, window_index)`.
A `duration <= 15` request is bit-identical to today's output; a windowed
request is reproducible for a fixed `(seed, plan)`.

## Error Handling and Limits

- `duration > 60 s` (windowed max) → `OmniClientError` naming both limits.
- `window_overlap_seconds` / `max_window_seconds` outside their ranges, or
  a combination the planner cannot satisfy (e.g. overlap ≥ max window) →
  `OmniClientError` at validation time, before any GPU work.
- `ref2va` with `duration > 15 s` → explicit "not yet supported for
  windowed generation" error (not silent truncation).
- Planner failure is impossible for in-range inputs by construction
  (property-tested; see below).

## Prior Art in This Codebase

| Implementation | Pattern | What we take |
|---|---|---|
| LongCat-Video-Avatar (`longcat_video/pipeline_longcat_video_avatar.py`) | Segment loop, tail-frame conditioning, clean-latent KV cache | Loop shape; KV-cache design for the follow-up milestone. We diverge on latent reuse (no pixel round trip). |
| MiniMax-H3 FL2VA keyframes (`minimax_h3/packed_sequence.py`, `denoise_loop.py`) | Cond rows pinned at condition timestep, excluded from update | Extended into the multi-frame history block. |
| Wan 2.2 S2V | Motion-buffer chunking | Precedent for in-place history conditioning on the target grid. |
| MiniMax Music 3 | Fixed windows, half overlap | Precedent for overlap-and-drop stitching of audio. |

## Files to Change

| File | Change |
|---|---|
| `vllm_omni/diffusion/models/minimax_h3/sliding_window.py` | **New.** `WindowSpec`, `WindowPlan`, `plan_windows`, tail-slice/stitch helpers. Pure, GPU-free. |
| `pipeline_minimax_h3.py` | New constants; `_resolve_shape` accepts long durations and returns/plumbs a `WindowPlan`; `_forward_windowed` loop; audio crossfade. |
| `packed_sequence.py` | `history_latent_t` / `history_audio_t` in `minimax_h3_packed_sequence`; extended `update_mask`. |
| `denoise_loop.py` | Accept history anchors alongside `keyframe_cond_rows`; pin history timesteps. |
| `condition_noise.py` | Reuse noise-aug helpers for history rows (signature widening only, if any). |
| `docs/models/…` (MiniMax-H3 user guide) | Document the new limit, knobs, and quality caveat: single-pass ≤ 15 s remains the best-quality mode. |
| `tests/…` (see below) | Planner unit tests, packed-sequence layout tests, GPU e2e. |

## Testing Plan

Following the repo's CI levels (`vllm-omni-test` conventions):

- **L1 (CPU, per-PR):**
  - Property tests for `plan_windows`: for all durations in
    `(15, 60]` s at 0.1 s granularity × overlaps in `[0.5, 3]` s — every
    `L_i ∈ [Lmin, Lmax]`, `Σ net == D`, alignment holds. Explicit cases:
    16 s (the 15+1 trap), 30 s, 60 s, and boundary 15.0/15.04 s.
  - `minimax_h3_packed_sequence` layout tests with history blocks:
    `update_mask`, ids, and slice arithmetic for `history_latent_t > 0`.
  - Validation tests: error messages for >60 s, bad knobs, ref2va long.
- **L2/L3 (GPU):**
  - Single-window equivalence: `duration=12 s` produces bit-identical
    output before/after the change.
  - Two-window smoke (`duration=16 s`, reduced steps): shape checks —
    `frames == plan.total_frames`, audio length matches, seam index has no
    duplicated frames.
- **L4 (nightly):** 30 s FL2VA and t2va runs with seam-quality inspection
  (frame-difference spike at seam below threshold; audio crossfade has no
  discontinuity above threshold).

## Milestones

1. **M1 — Planner + validation** (CPU-only, mergeable alone): constants,
   `sliding_window.py`, `_resolve_shape` plumbing, L1 tests. Long requests
   still rejected at the end of validation behind a feature flag.
2. **M2 — History conditioning:** packed-sequence/denoise-loop extension +
   layout tests; single-window equivalence test.
3. **M3 — Windowed orchestration:** `_forward_windowed`, stitching,
   seeds, docs; enable the path; GPU smoke tests.
4. **M4 (follow-up) — Clean-latent KV cache:** transformer `kv_cache_dict`
   plumbing, LongCat-style per-window condition caching, benchmark.

## Open Questions

1. Should the windowed max be 60 s or configurable per deployment
   (engine arg vs. constant)? v1 proposes a constant; a server-level
   override is a one-line follow-up.
2. In-place history conditioning assumes the FL2VA partition tolerates
   multi-frame clean anchors at the sequence head. If early experiments
   show drift, fall back to the separate-cond-block layout (closer to how
   keyframes attach today) at the cost of RoPE grid bookkeeping.
3. Whether `fl2va` `[0, -1]` long requests should *interpolate* (last image
   pinned on final window, as specced) or reject; specced behavior matches
   user intent but needs a quality check at 30 s.
