# NemotronVoiceChat Code2Wav (Vocoder) Streaming Optimization

> **Status:** proposal / spec, revision 3. Revision 3 adds an informative
> pipeline-level roadmap and a duplex frame-budget analysis (last section)
> and records P2's dependency on the in-flight PRs; the normative scope
> (the vocoder phases) is unchanged. Revision 2 rewrote against the in-flight
> upstream work: [#6089](https://github.com/vllm-project/vllm-omni/pull/6089)
> (native full-duplex serving, approved, pending conflict resolution) and
> [#6354](https://github.com/vllm-project/vllm-omni/pull/6354) (draft,
> stacked on #6089) already implement the per-request
> `CausalConv1dCache` incremental codec decode that revision 1 of this spec
> proposed as its Phase 1. This revision drops everything those PRs cover
> and specs only the remaining gaps. It targets the Stage-2 RVQ-VAE codec
> decode introduced in
> [#5842](https://github.com/vllm-project/vllm-omni/pull/5842) and does not
> change the thinker or talker stages.

## Summary

### Where the codec stands after the in-flight PRs

`NemotronVoiceChatCode2Wav` decodes talker code stacks with the vendored
NeMo RVQ-VAE codec
(`vllm_omni/model_executor/models/nemotron_voicechat/nemotron_voicechat_code2wav.py`).
As merged (#5842), async-chunk streaming re-decodes the whole cumulative
prefix per chunk — `O(T²)` codec work over a `T`-frame stream and per-chunk
latency growing linearly with stream position.

The in-flight PRs already fix the compute half of that:

- **#6089** (duplex serving path): per-segment payloads marked
  `meta.codec_streaming` carry only new frames; the stage keeps one
  `CausalConv1dCache` per `meta.request_id` and drops it in
  `on_requests_finished`.
- **#6354** (offline async-chunk path, draft): opt-in
  `hf_overrides.use_incremental_codec_cache` (default **off**); the
  connector payload stays cumulative, but the stage slices
  `codes[left_context_frames:]` and decodes only the new frames against the
  per-request cache (`requires_request_ids` when enabled). The PR notes this
  is the codec-cache path NVIDIA's own serving stack runs
  (`S2S_USE_CODEC_CACHE`).

### What this spec still covers

| Phase | Gap left by #6089/#6354 | Risk |
|-------|-------------------------|------|
| P1 | Complete the incremental path: terminal `flush`, incremental transport (`O(c)` bytes + `O(c)` validation per chunk), equivalence acceptance, then default-on | Low–medium |
| P2 | Cross-request batching + optional CUDA Graph capture for the codec | Medium — multi-session serving only |
| P3 | Micro-optimizations (fused dequantize, cached ISTFT envelope, host-sync removal, gated precision experiments, lower TTFP cadence) | Low, incremental |

Revision 1's P0 (bounded left-context window, NeMo `number_prev_tokens`
semantics) is **dropped**: it was the low-risk alternative to an incremental
cache that did not exist yet; with the cache path implemented in both
in-flight PRs it has no remaining value.

All work here lands as follow-ups **on top of** #6354/#6089 — coordinate
with those branches, do not re-implement in parallel.

## Background: current implementation (head of #6354)

Data flow in offline streaming mode with `use_incremental_codec_cache`:

1. `talker2code2wav_async_chunk`
   (`vllm_omni/model_executor/stage_input_processors/nemotron_voicechat.py`)
   fires once ≥ `codec_chunk_frames` new frames have accumulated (default
   13 frames ≈ 1 s; one frame = 80 ms) and ships the **full**
   prompt-trimmed code history `[frames, 31]` (int64, CPU) with
   `meta.left_context_size = frames already emitted`.
2. `NemotronVoiceChatCode2Wav.forward` validates the **whole** cumulative
   payload, slices off the already-emitted prefix, and decodes only the new
   frames with the per-request `CausalConv1dCache` (fp32, TF32 disabled,
   batch=1 per-request Python loop). Duplex serving payloads
   (`meta.codec_streaming`) skip the slice — they already carry only new
   frames.
3. Neither path ever passes `flush=True` to `RVQVAEModel.decode`.

Reference points in the MiniCPM-o 4.5 code2wav stage
(`vllm_omni/model_executor/models/minicpmo_4_5/{minicpmo_4_5_code2wav.py,batched_token2wav.py}`):
exact-shape batch buckets, HiFT/CFM CUDA Graph capture, bounded per-request
caches — the target architecture for P2.

Decoder structure (`nemo_vendored/ear_tts_vae_codec.py`): dequantize
(31 codebook-embedding sums) → `Latent2Wav`: frame-local
`ConvTranspose1d(kernel=stride)` upsampling + causal `ConvNeXt1d` blocks →
1×1 conv → ISTFT overlap-add. Fully causal; the `CausalConv1dCache`
carries the conv tails and the ISTFT spec cache, and holds back
`half_wav_padding` right-edge samples per chunk that only `flush=True`
emits.

## Goals

- Incremental decode becomes complete (no dropped tail samples), cheap end
  to end (`O(c)` compute **and** transport **and** validation per chunk),
  provably equivalent to the one-shot decode, and the default.
- Preserve the quality bar: the non-streaming path stays bit-identical; the
  prefix-redecode fallback stays available and untouched.
- Keep the out-of-range-code hard error (numeric-drift tripwire) intact.
- Every change is opt-in until its acceptance table is green.

## Non-goals

- Anything #6089/#6354 already implement: the cache mechanism itself, duplex
  data plane, native talker, bf16 thinker, CFG.
- Thinker/talker performance, barge-in, tool execution, multi-session
  session policy (tracked in those PRs' own limitation lists).
- NPU support (follow-up; P1/P2 are prerequisites for a graph port).
- Codec weight quantization.

## Design

### P1 — Complete the incremental decode, then make it the default

Three defects/gaps remain in the in-flight implementation, plus the
default flip:

**1. Terminal flush.** Neither PR passes `flush=True` on the final chunk,
so the ISTFT spec cache's held-back right-edge samples
(`half_wav_padding · hop` per stream) are never emitted: the concatenated
incremental stream is **shorter than** the one-shot decode, and the caches
in the vendored `CausalConv1dCache` are deleted only by request-finish
cleanup rather than drained. Fix: thread the producer's `meta.finished`
into the decode call (`flush=is_last_chunk`); the duplex path
(`codec_streaming`) needs the equivalent end-of-segment signal from the
data plane. Acceptance requires exact total-sample-count equality with the
one-shot decode.

**2. Incremental transport and validation.** #6354's offline path still
ships the full cumulative history every chunk (`O(T²)` bytes over the
stream) and `validate_code_stack` re-validates the whole prefix each time
(`O(T)` per chunk). Under a connector-config flag
(`code2wav_incremental_transport: true`, valid only with the codec cache
enabled), `talker2code2wav_async_chunk` ships **only the new frames** with
`meta.chunk_seq`, and codes narrow to int16 on the wire (codebook size
1024), widened back in the stage. The stage gains chunk-sequencing guards
(monotonic `chunk_seq`, missing-state, duplicate-request → structured hard
errors, mirroring `MiniCPMO45Code2Wav._parse_item` discipline), which the
cumulative+slice path never needed because every payload was
self-contained.

**3. Default-on.** Once 1–2 pass their acceptance tables, flip
`use_incremental_codec_cache` (and the incremental transport) to default-on
in the streaming/duplex yamls, keeping the prefix-redecode fallback behind
the same knob. NVIDIA's serving stack already defaults to the codec-cache
path, and the acceptance bar here is strictly tighter than the merged
baseline (bit-equal vs 99.98 % bit-equal), so default-on is justified by
evidence rather than precedent alone.

**Compat note (unchanged from rev 1):** with flush, per-chunk emitted
sample counts shift by ≤ `half_wav_padding · hop` samples relative to the
prefix-redecode path (`new_frames · 1764` exactly); consumers must only
assume the concatenated total, which becomes exactly the one-shot decode's.

### P2 — Cross-request batching and CUDA Graph (multi-session serving)

Untouched by both PRs: the stage still decodes requests one at a time in a
Python loop, while #6354's multi-session duplex (4 concurrent sessions in
its probe results) multiplies concurrent codec work. With P1, every
steady-state chunk has the same shape `[c, 31]`, so same-shape requests
batch trivially:

- Exact-shape bucketing by `(num_new_frames, is_last_chunk, cache-shape
  signature)`, stacking per-request `CausalConv1dCache` tensors (plain
  `[B, C, pad]` tensors; add `stack/split` helpers analogous to
  `_stack_flow_cache`/`_split_flow_cache` in `batched_token2wav.py`).
- Optional CUDA Graph capture of the decode for fixed `(B, c)` shapes,
  following the `HiFTGraphWrapper` pattern (static buffers, capture at init
  for configured batch sizes, lazy capture bounded by `max_graphs`). Knobs
  mirror the MiniCPM naming: `enable_code2wav_graph`,
  `code2wav_graph_capture_batch_sizes`.
- Batch math must remain bit-equal to the sequential per-request decode;
  unseen shapes fall back to eager.

**Dependency on the in-flight PRs.** P2's *mechanism* requires the
per-request cache path to exist — exact-shape buckets only form when
chunks have a fixed shape, and the objects being stacked are the
`CausalConv1dCache` instances those PRs create. Either PR satisfies this:
#6089 covers duplex payloads (`meta.codec_streaming`), #6354 adds the
offline async-chunk knob. P2's *value*, however, depends on #6354
specifically: it is the only branch that runs multi-session serving (its
probe drives 4 concurrent sessions), while merged main and #6089's default
deployment are single-session (`max_num_seqs: 1`) — there is nothing to
batch without it. Development can proceed stacked on the #6354 head;
upstream merge waits for the base to land. P3 items 1–4 are the
dependency-free lane: they touch only code already on main.

### P3 — Micro-optimizations

Independent, individually benchmarked; none are touched by the in-flight
PRs:

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
5. **Lower first-audio cadence:** with constant per-chunk cost,
   `codec_chunk_frames` can drop from 13 (~1 s) toward 4–5 (~350 ms) to cut
   offline TTFP further than #6354's 408 ms duplex figure; document the new
   tradeoff in the recipe.

## Config surface

Aligned with the knobs #6354 introduces; additions in the connector `extra`
block:

```yaml
# hf_overrides (existing in #6354): use_incremental_codec_cache: true
connectors:
  connector_of_shared_memory:
    name: SharedMemoryConnector
    extra:
      codec_chunk_frames: 13               # existing
      code2wav_incremental_transport: false  # P1.2; requires the codec cache
      enable_code2wav_graph: false             # P2
      code2wav_graph_capture_batch_sizes: [1]  # P2
```

Defaults keep the behavior at the head of #6354 until P1.3 flips them in
the streaming/duplex yamls. The non-streaming yaml is never touched.

## Acceptance criteria

### Global invariants (every phase)

| # | Criterion | How verified |
|---|-----------|--------------|
| G1 | Non-streaming (sync full-payload) path is byte-identical: existing parity fixture still produces 196/196 tokens and a bit-identical WAV. | existing e2e parity test, unchanged |
| G2 | Out-of-range codes still raise (never clamp), in incremental-transport and batched modes included. | L1 unit test |
| G3 | No behavior change with new knobs at their defaults (existing tests pass unmodified), until the deliberate P1.3 default flip. | full nemotron CPU suite + e2e |
| G4 | No state leak: after N streaming requests finish or abort, stage-held per-request caches are empty. | L1 lifecycle test (extends #6354's) |

### P1 — completing incremental decode

| # | Criterion | Threshold |
|---|-----------|-----------|
| B1 | With flush: concatenated incremental stream vs one-shot decode of the full stack, for random valid code stacks (lengths 1–200 frames, several seeds): identical total sample count and max abs diff ≤ 1e-6 (fp32; allowance for conv-algorithm variation only). Strictly tighter than the merged baseline's 99.98 %. | L1 unit + GPU e2e |
| B2 | Incremental transport: per-chunk payload ≤ `c_max · 31 · 2` bytes (int16), independent of stream position; per-chunk validation cost `O(c)`. | L1 unit on producer output |
| B3 | Chunk-sequencing violations (gap, replay, unknown request mid-stream) hard-error with structured messages; cumulative fallback path unaffected. | L1 unit tests |
| B4 | Interleaved concurrent streams never cross-contaminate caches (two-stream unit test with different code content). | L1 unit test |
| B5 | Whisper transcript on the reference fixture identical to the batch WAV; per-chunk decode latency flat: decode-time(chunk N) / decode-time(chunk 2) ≤ 1.2 on a ≥ 60 s synthetic stream. | GPU e2e + benchmark |
| B6 | Default flip: streaming/duplex e2e suites green with incremental default-on; prefix-redecode fallback still selectable and bit-equal to its pre-flip behavior. | GPU e2e, both modes parametrized |

### P2 — batching / CUDA Graph

| # | Criterion | Threshold |
|---|-----------|-----------|
| C1 | Batched decode output per row is bit-equal to the same requests decoded sequentially at batch=1. | L1 unit test |
| C2 | Graph replay output equals eager decode. | max abs diff ≤ 1e-6, GPU test |
| C3 | Throughput: ≥ 2× aggregate decoded-frames/s at B=4 vs 4 sequential B=1 streams on the benchmark; report the measured table in the PR (no regression at B=1). | benchmark |
| C4 | Graph memory bounded: capture set limited to configured shapes; unseen shape falls back to eager, never errors. | L1 + GPU test |

### P3 — micro-optimizations

Each item lands separately with: (a) B1-level equivalence on its scope
(bit-equal for 1–3; tolerance + WER bar for the precision experiment: WER
delta = 0 on the fixture set and ≥ 99.9 % samples within 1e-3 of fp32), and
(b) a before/after microbenchmark in the PR description. Item 5 additionally
reports TTFP before/after.

## Test & CI plan

- **L1 (CPU, `tests/model_executor/`):** flush equivalence on a
  randomly-initialized small codec config; incremental-transport payload
  shape/bounds; chunk-sequencing and lifecycle tests (extending #6354's
  `test_nemotron_voicechat.py` additions); fused-dequantize equivalence.
  Wired into `test-ready.yml`.
- **L3 (GPU, `tests/e2e/offline_inference/`):** extend the nemotron
  streaming e2e with incremental-transport and default-on modes
  (parametrized), asserting B1/B5/B6. Wired into `test-merge.yml`.
- **Benchmark:** a small script under `tests/dfx/perf/` producing the
  per-chunk latency curve (B5/C3) from a synthetic long code stream —
  runnable standalone on one GPU; numbers quoted in each PR.

## Rollout & risks

- **Sequencing on the in-flight PRs is the main risk.** P1 items patch code
  that exists only on #6354/#6089 heads; land them as follow-up commits or
  stacked PRs on whichever merges first (#6089 is approved and only
  conflict-blocked; #6354 is a draft). If #6354's offline opt-in is
  reshaped during review, P1.2/P1.3 rebase onto the final knob names.
- P1 flush risk: end-of-segment semantics differ between offline
  (`meta.finished`) and duplex (segment boundaries in the data plane);
  B1's total-sample-count equality is the guard on both.
- P2 is inert for single-session deployments and cannot regress them (G3).
- Each phase remains a separate PR with the current behavior as default-off
  until its acceptance table is green on H100.

## Pipeline roadmap beyond the vocoder (informative)

> This section is informative context, not part of this spec's normative
> scope (which remains the vocoder phases above). It records where the rest
> of the pipeline stands after #6354 and why the duplex deployment — the
> model's core use case — is a deadline problem rather than a throughput
> problem, so that vocoder work is prioritized by its contribution to that
> deadline.

### Where the time goes after #6354

Per-stage warm timings from the #6354 recipe (196-frame / 15.7 s fixture,
1× H100, all stages on one GPU):

| config | thinker | talker | code2wav | total | RTF |
|---|---|---|---|---|---|
| parity (merged default) | fp32 eager, 3.35 s | fp32 eager, 6.7 s | 0.06 s | 10.3 s | 0.66 |
| fast | bf16 + vLLM graphs, 1.8 s | fp32 + step graph, 2.5 s | 0.06 s | 4.4 s | 0.28 |
| fast_streaming | same, stages overlap | same | incremental | **5.2 s** | 0.33 |
| native talker (experimental) | same | fp32 paged KV + MoG graph, 1.7 s | 0.06 s | 3.7 s | 0.23 |

Two readings matter. First, offline, the codec is 0.06 s — negligible; the
vocoder phases above are motivated by streaming real-time behavior and
multi-session cost, not by the offline share. Second, streaming overlap is
currently a *net loss* offline (5.2 s vs 4.4 s): three engines contending
on one GPU with `async_scheduling: false` — evidence that scheduling, not
kernel math, is a first-class cost.

### Duplex is a deadline problem, not a throughput problem

The duplex contract is a hard frame clock: consume one 80 ms audio frame
and produce one 80 ms audio frame, every 80 ms, indefinitely. The metric
that decides viability is the **per-frame wall time distribution against
the 80 ms deadline** — averages and offline RTF say nothing (RTF 0.23
offline coexists with a missed frame clock).

#6354's own native-talker numbers show the current margin is negative:
median inter-packet gap ~81 ms (already past the budget), p95 ~110 ms,
"RTF ~1.01". The session stays audible only because the first chunk banks
~400 ms of client-side buffer, which then drains ~1 ms per median frame
and ~30 ms per p95 spike: long responses arithmetically end in underruns,
and every transient (a capture-miss eager talker step costs 25–34 ms wall
for ~9 ms of GPU work — a third of the budget in one spike) lands on the
user. A further consequence: at ~1.0 RTF per session, multi-session duplex
capacity is zero regardless of batching.

Per-frame budget composition (one tick): perception step (cache-aware
conformer, streaming state landed in #6089) + thinker decode step (bf16 +
vLLM graphs) + talker step (~8.7–11 ms fp32 graph; +8.5 ms with CFG) +
codec share + data-plane/host overhead. The gap between summed GPU work
and the ~81 ms median wall strongly suggests host-side scheduling and
transport dominate the tick — which is the first thing to verify.

### Ranked levers

- **R0 — Instrument the tick.** Per-stage timeline within each 80 ms frame,
  deadline-miss accounting, p99 and underrun counters over a long soak.
  Jitter cannot be fixed unattributed; everything below is re-ranked by
  this data.
- **R1 — Talker bf16.** The largest per-frame GPU term (and 46 % of the
  offline total) is still fp32; the thinker's bf16 move bought ~2× and the
  same lever is unplayed here. Guarded by the code2wav out-of-range
  tripwire and the paired-ASR harness #6354 established. Also halves talker
  weights — memory headroom for multi-session.
- **R2 — Host-path slimming per tick.** Per-frame producer/scheduler
  overhead, cumulative payload re-ship (the thinker→talker timeline has the
  same O(T²) shape as the codes; codes covered by P1.2), `.cpu()` syncs
  (P3.3), and codec burst smoothing: with the incremental cache, per-frame
  decode (`codec_chunk_frames: 1`) becomes affordable and converts the
  every-N-frames codec burst into a flat ~0.3 ms/frame cost — directly a
  jitter lever, and a vocoder deliverable of P1.
- **R3 — Contention and capture control.** CUDA stream priorities /
  async scheduling / stage placement for the three engines sharing one
  GPU (the 5.2 s vs 4.4 s regression is the smoking gun); pre-capture all
  bucket shapes so no live session ever hits the 25–34 ms eager-fallback
  spike.
- **R4 — CFG batch-doubling.** When guidance is enabled, fold the
  unconditional stream into the native batch instead of mirroring a second
  HF StaticCache backbone (+8.5 ms/frame today, and extra sessions fall
  back to eager under CFG).
- **R5 — Multi-session.** Only meaningful once R1–R3 push per-session
  p99 well under the frame budget; then P2 (codec batching) plus batched
  talker stepping set the sessions-per-GPU slope.

### Proposed duplex SLOs (to ratify with maintainers)

| # | Metric | Target |
|---|--------|--------|
| D1 | Per-frame wall time, p99 over a ≥ 30 min soak | ≤ 72 ms (0.9 × frame budget) |
| D2 | Audio underruns with a ≤ 400 ms client buffer, per 30 min | 0 |
| D3 | TTFP (session start → first audio packet) | ≤ 500 ms |
| D4 | Session capacity: max N sessions each meeting D1–D2 | reported per config |

The current native head misses D1 at the median (81 ms), so these are
targets for the R-series, not a description of today; they give the
vocoder phases their acceptance context — P1's per-frame smoothing and P3's
sync removal are judged by their D1/D2 contribution, not offline seconds.
