# NemotronVoiceChat Code2Wav (Vocoder) Streaming Optimization

> **Status:** proposal / spec. Nothing in this document is implemented yet.
> It targets the Stage-2 RVQ-VAE codec decode of the NemotronVoiceChat
> pipeline introduced in
> [#5842](https://github.com/vllm-project/vllm-omni/pull/5842) and does not
> change the thinker or talker stages.

## Summary

In async-chunk streaming mode, `NemotronVoiceChatCode2Wav` follows NeMo's
`decode_one_audio_step` semantics with an **unbounded window**: every chunk
ships the cumulative prompt-trimmed code history over the connector, and the
stage re-decodes the entire prefix through the RVQ-VAE decoder, slicing off
only the new tail samples
(`vllm_omni/model_executor/models/nemotron_voicechat/nemotron_voicechat_code2wav.py`).

For a stream of `T` codec frames at chunk cadence `c` (default
`codec_chunk_frames: 13`, one frame = 80 ms, so ~1 s per chunk), total decoded
frames are `~T²/(2c)` instead of `T`, and per-chunk decode latency grows
linearly with stream position. A 60 s response (750 frames) decodes ~22k
frames — a ~29× compute overhead — and the final chunk decodes the whole 60 s
in one call. Once per-chunk decode time exceeds the ~1 s chunk cadence, the
stream falls behind real time: this is a correctness cliff, not just an
efficiency problem. Connector transport is likewise cumulative
(`O(T²)` bytes over the stream).

The vendored NeMo codec already contains the machinery to fix this:
`CausalConv1dCache` and the `cache=`/`flush=` parameters on
`RVQVAEModel.decode`
(`nemo_vendored/ear_tts_vae_codec.py`) — vendored in #5842 but never
instantiated by any caller (NeMo upstream does not use it in
`decode_one_audio_step` either). NeMo's own escape hatch, the
`number_prev_tokens` bounded window, was also dropped when the vLLM-Omni
producer was written.

This spec proposes a phased optimization:

| Phase | Change | Per-chunk cost | Risk |
|-------|--------|----------------|------|
| P0 | Bounded left-context window (producer-side only) | `O(W + c)` constant | Low — NeMo-sanctioned (`number_prev_tokens`) |
| P1 | True incremental decode via `CausalConv1dCache` | `O(c)` constant, no redundancy | Medium — no upstream precedent, needs equivalence proof |
| P2 | Cross-request batching + optional CUDA Graph | amortized | Medium — multi-session serving only |
| P3 | Micro-optimizations (fused dequantize, sync removal, precision eval) | — | Low, incremental |

P0 alone removes the real-time cliff. P1 enables fixed chunk shapes, which
P2 needs. Phases are independently landable and each keeps a fallback to the
previous behavior.

## Background: current implementation

Data flow in streaming mode (`nemotron_labs_voicechat_streaming.yaml`):

1. `talker2code2wav_async_chunk`
   (`vllm_omni/model_executor/stage_input_processors/nemotron_voicechat.py`)
   fires once ≥ `codec_chunk_frames` new frames have accumulated. It ships the
   **full** prompt-trimmed code history `[frames, 31]` (int64, CPU) with
   `meta.left_context_size = frames already emitted`.
2. `NemotronVoiceChatCode2Wav.forward` validates the whole payload, moves it
   to GPU, decodes the **whole** prefix in fp32 with TF32 disabled
   (batch=1, per-request Python loop), then emits
   `wav[left_context_size * 1764:]` and copies it to CPU.

Reference points in the MiniCPM-o 4.5 code2wav stage
(`vllm_omni/model_executor/models/minicpmo_4_5/{minicpmo_4_5_code2wav.py,batched_token2wav.py}`),
which already implements the target architecture for a different vocoder:
per-request persistent decode state with bounded caches, exact-shape batch
buckets, HiFT/CFM CUDA Graph capture, and request lifecycle cleanup via
`on_requests_finished`.

Decoder structure (`nemo_vendored/ear_tts_vae_codec.py`): dequantize
(31 codebook-embedding sums) → `Latent2Wav`: per-stage
`ConvTranspose1d(kernel=stride)` upsampling (frame-local, no overlap) +
causal `ConvNeXt1d` blocks (left context `kernel_size − 1` per block) → 1×1
conv → ISTFT overlap-add. Every cross-frame dependency is causal and
bounded, which is what makes both P0 and P1 exact.

## Goals

- Constant per-chunk decode cost and transport size, independent of stream
  position; sustained real-time decode for arbitrarily long streams.
- Preserve the existing quality bar: the non-streaming path stays
  bit-identical; the streaming path must be **no worse** than today's
  verified baseline (99.98 % samples bit-equal to the batch decode, boundary
  |diff| ≤ 3.3e-3, identical Whisper transcript — see #5842).
- Keep the out-of-range-code hard error (numeric-drift tripwire) intact.
- Every phase is opt-in via config with the current behavior as fallback.

## Non-goals

- Thinker/talker performance (tracked separately; the talker's serial
  EAR-TTS compute currently dominates TTFP).
- Online/duplex serving, function-call handling, NPU support (follow-ups;
  P1/P2 are prerequisites for a future NPU graph port).
- Codec weight quantization.

## Design

### P0 — Bounded left-context window (producer-side)

NeMo's `decode_one_audio_step` accepts `number_prev_tokens` and truncates the
history to the last `N` frames before decoding
(`nemo_vendored/duplex_ear_tts.py`). vLLM-Omni mirrored the unbounded
default. P0 adopts the bounded form.

**Change:** `talker2code2wav_async_chunk` gains a connector-config knob
`codec_left_context_frames` (int, `0` = unbounded = current behavior).
When `W > 0`, instead of the full history the producer ships only the last
`min(W, frames_sent) + new_frames` frames and sets
`meta.left_context_size = min(W, frames_sent)`.

**The stage code needs no change**: it already decodes the payload and slices
`left_context_size * wav_per_frame`. The whole phase is a producer-side diff
plus config plumbing. Early chunks with `frames_sent < W` ship everything —
identical to today.

**Window sizing.** The tail samples are exact iff `W` covers the decoder's
left receptive field. In latent frames:

```
W_min = ceil( Σ_i  num_blocks · (kernel_size − 1) / Π_{j≤i} rate_j )
        + ceil( istft_overlap_frames / Π_j rate_j )
```

computed from the codec config at startup (with the class defaults —
`kernel_size=7`, `num_blocks=3`, `rates=(8,8,8)` — this is ≈ 3 latent
frames; the actual checkpoint config decides). The stage logs `W_min` at
load time and the producer validates `W ≥ W_min` (reject smaller values
rather than silently degrading audio). Default: `codec_left_context_frames:
8` in `nemotron_labs_voicechat_streaming.yaml` (≥ 2× margin over the
computed minimum for the shipped checkpoint).

Zero-padding at the truncated window start cannot reach the emitted tail
when `W ≥ W_min`, so the emitted samples are mathematically identical to the
unbounded decode. cuDNN may select different conv algorithms for different
input lengths, so bit-equality is asserted with a tiny float tolerance (see
acceptance criteria).

### P1 — Incremental decode with `CausalConv1dCache`

Remove the redundant window recompute entirely.

**Stage:**

- `NemotronVoiceChatCode2Wav` gains `requires_request_ids = True`, a
  per-request state dict `request_id → CausalConv1dCache`, and an
  `on_requests_finished` hook that drops state (mirroring
  `MiniCPMO45Code2Wav`).
- Per chunk: decode **only the new frames** with
  `audio_codec.decode(codes_new, lens, cache=state, flush=is_last_chunk)`.
  The cache carries every `ConvNeXt1d` causal tail and the ISTFT
  overlap-add spec cache; `flush=True` on the terminal chunk emits the held
  right-edge padding.
- Chunk-sequencing guards (chunk_seq monotonicity, missing-state,
  duplicate-request) copied from the MiniCPM stage's `_parse_item`
  discipline; a hard error on gaps, never silent resync.

**Producer:** a new mode (`code2wav_incremental: true` in connector extra)
ships only the new `[c, 31]` frames with `meta.incremental = True` and
`meta.chunk_seq`; `meta.left_context_size` is no longer used in this mode.
Transport becomes `O(c)` per chunk; codes are additionally narrowed to
int16 on the wire (codebook size 1024) and widened back in the stage.

**Exactness.** The decoder is fully causal and the cache mechanism holds
back `half_wav_padding` right-edge samples per chunk, emitting them with the
next chunk, so the **concatenated** stream equals the one-shot decode of the
full stack — stronger than today's prefix-redecode, which is only 99.98 %
bit-equal (right-edge overlap-add normalization differs per chunk). Two
compat notes:

- Per-chunk emitted sample counts shift by ≤ `half_wav_padding · hop`
  samples relative to today (`new_frames · 1764` exactly); consumers must
  not assume the per-chunk count, only the total. The e2e assertions compare
  concatenated audio, which is unaffected.
- Total sample count at stream end is identical to the one-shot decode.

**Dummy/profile runs** (`get_dummy_runtime_additional_information`) never
touch the state dict.

### P2 — Cross-request batching and CUDA Graph (multi-session serving)

With P1, every steady-state chunk has the same shape `[c, 31]`, so
same-shape requests batch trivially:

- Exact-shape bucketing by `(num_new_frames, is_last_chunk, cache-shape
  signature)`, stacking per-request `CausalConv1dCache` tensors (they are
  plain `[B, C, pad]` tensors; add `stack/split` helpers analogous to
  `_stack_flow_cache`/`_split_flow_cache` in `batched_token2wav.py`).
- Optional CUDA Graph capture of the decode for fixed `(B, c)` shapes,
  following the `HiFTGraphWrapper` pattern (static input/output buffers,
  capture at init for configured batch sizes, lazy capture bounded by
  `max_graphs`). Knobs mirror the MiniCPM naming:
  `enable_code2wav_graph`, `code2wav_graph_capture_batch_sizes`.
- The current deploy yamls are `max_num_seqs: 1`; P2 only pays off with a
  multi-session config and is gated behind it. Batch math must remain
  bit-equal to the sequential per-request decode.

### P3 — Micro-optimizations

Independent, individually benchmarked:

1. **Fused dequantize:** `PreTrainedProbabilisticVQ.decode` runs 31
   sequential `F.embedding + add` kernels; stack `mus_list` into a
   `[Q, V, C]` buffer at load and replace with one gather + `sum(dim)`.
2. **Cached ISTFT window envelope:** `spec_to_wav` recomputes the window
   envelope and runs `assert (envelope > 1e-11).all()` (a GPU sync) every
   call; the envelope depends only on `(T, n_fft, hop)` — cache it per
   length and pre-validate once.
3. **Remove per-chunk host syncs:** keep decoded audio on-device in
   `multimodal_outputs` (the output processor already handles transfer, as
   in the MiniCPM stage) and avoid `int(wav_len…)` item syncs.
4. **Precision evaluation (experiment, not a default change):** the decode
   runs fp32 with TF32 explicitly disabled — the slowest conv path on
   H100. bf16 is documented as collapsing reconstruction; evaluate TF32
   and fp16 on the conv trunk only (dequantize stays fp32), gated behind
   the existing WER/bit-equality harness. Any default change requires the
   acceptance bar below and a recipe note.
5. **Lower first-audio cadence:** with P1's constant per-chunk cost,
   `codec_chunk_frames` can drop from 13 (~1 s) toward 4–5 (~350 ms) to cut
   TTFP; document the new tradeoff in the recipe.

## Config surface

All knobs live in the connector `extra` block of the deploy yaml, matching
existing conventions:

```yaml
connectors:
  connector_of_shared_memory:
    name: SharedMemoryConnector
    extra:
      codec_chunk_frames: 13          # existing
      codec_left_context_frames: 8    # P0; 0 = unbounded (current behavior)
      code2wav_incremental: false     # P1; overrides left-context mode
      enable_code2wav_graph: false            # P2
      code2wav_graph_capture_batch_sizes: [1] # P2
```

Defaults keep today's behavior; `nemotron_labs_voicechat_streaming.yaml`
flips P0 (and later P1) on once accepted. The non-streaming yaml is never
touched.

## Acceptance criteria

### Global invariants (every phase)

| # | Criterion | How verified |
|---|-----------|--------------|
| G1 | Non-streaming (sync full-payload) path is byte-identical: existing parity fixture still produces 196/196 tokens and a bit-identical WAV. | existing e2e parity test, unchanged |
| G2 | Out-of-range codes still raise (never clamp), including in windowed/incremental/batched modes. | L1 unit test |
| G3 | No behavior change with new knobs at their defaults (`git diff`-level: existing tests pass unmodified). | full nemotron CPU suite + e2e |
| G4 | No state leak: after N streaming requests finish or abort, stage-held per-request state is empty. | L1 lifecycle test |

### P0 — bounded window

| # | Criterion | Threshold |
|---|-----------|-----------|
| A1 | Unit equivalence: for random valid code stacks (lengths 1–200 frames, several seeds), windowed tail decode vs unbounded tail decode. | max abs diff ≤ 1e-6 (fp32; allowance for conv-algorithm variation only) |
| A2 | `W < W_min` is rejected at startup with an actionable error naming `W_min`. | L1 unit test |
| A3 | E2e streaming quality on the reference fixture is **not worse than the current baseline**: ≥ 99.98 % samples bit-equal to the batch decode, boundary max abs diff ≤ 3.3e-3, identical Whisper transcript, identical total sample count. | GPU e2e |
| A4 | Per-chunk decode latency is flat: on a synthetic ≥ 60 s stream (≥ 750 frames), decode-time(chunk N) / decode-time(chunk 2) ≤ 1.2 for all N. Today this ratio grows linearly (~29× at N=58). | benchmark script committed with the PR |
| A5 | Per-chunk connector payload is bounded by `(W + c_max) · 31 · dtype` bytes, independent of stream position. | L1 unit test on producer output |

### P1 — incremental cache

| # | Criterion | Threshold |
|---|-----------|-----------|
| B1 | Concatenated streaming output vs one-shot decode of the full stack: max abs diff ≤ 1e-6 and identical total sample count (target: bit-equal; strictly tighter than today's 99.98 %). | L1 unit (random stacks) + GPU e2e |
| B2 | Chunk-sequencing violations (gap, replay, unknown request mid-stream) hard-error with structured messages. | L1 unit tests |
| B3 | Per-chunk decode cost ≤ P0's (measured on the A4 benchmark); transport per chunk is `O(c)`. | benchmark |
| B4 | Abort/finish cleanup: `on_requests_finished` drops cache; interleaved concurrent streams never cross-contaminate (two-stream unit test with different code content). | L1 unit tests |
| B5 | Whisper transcript on the reference fixture identical to the batch WAV. | GPU e2e |

### P2 — batching / CUDA Graph

| # | Criterion | Threshold |
|---|-----------|-----------|
| C1 | Batched decode output per row is bit-equal to the same requests decoded sequentially at batch=1. | L1 unit test |
| C2 | Graph replay output equals eager decode. | max abs diff ≤ 1e-6, GPU test |
| C3 | Throughput: ≥ 2× aggregate decoded-frames/s at B=4 vs 4 sequential B=1 streams on the benchmark; report the measured table in the PR (no silent regressions at B=1). | benchmark |
| C4 | Graph memory bounded: capture set limited to configured shapes; unseen shape falls back to eager, never errors. | L1 + GPU test |

### P3 — micro-optimizations

Each item lands separately with: (a) B1-level equivalence on its scope
(bit-equal for 1–3, tolerance + WER bar for the precision experiment: WER
delta = 0 on the fixture set and ≥ 99.9 % samples within 1e-3 of fp32), and
(b) a before/after microbenchmark in the PR description. Item 5 additionally
reports TTFP before/after.

## Test & CI plan

- **L1 (CPU, `tests/model_executor/`):** windowed-vs-unbounded equivalence
  on a randomly-initialized small codec config; `W_min` derivation and
  rejection; producer payload shape/bounds; chunk-sequencing and lifecycle
  tests; fused-dequantize equivalence. Wired into `test-ready.yml`.
- **L3 (GPU, `tests/e2e/offline_inference/`):** extend the existing
  nemotron_voicechat streaming e2e with the P0/P1 modes (parametrized),
  asserting A3/B1/B5. Wired into `test-merge.yml`.
- **Benchmark:** a small script under `tests/dfx/perf/` producing the
  per-chunk latency curve (A4/B3/C3) from a synthetic long code stream —
  runnable standalone on one GPU; numbers quoted in each PR.

## Rollout & risks

- Each phase is a separate PR against the current behavior as default-off;
  the streaming yaml flips a phase on only after its acceptance table is
  green on H100.
- P0 risk: mis-derived `W_min` for an unusual codec config → mitigated by
  the load-time derivation + rejection test and the tail-equivalence unit
  test running on the actual config math.
- P1 risk: cache semantics drift from upstream NeMo (no precedent to diff
  against) → mitigated by B1's strict one-shot equivalence, which is the
  ground truth regardless of NeMo's implementation choices.
- P2 is inert for the shipped single-session yamls and cannot regress them
  (G3).
