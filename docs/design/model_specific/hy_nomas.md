# Hy-Nomas integration contract

The intended user is a realtime world-model client using `/v1/realtime/video`.
The first acceptance criterion is numerical agreement with native Hy-Nomas;
latency optimization follows correctness. This is the contributor's stated
priority, not a measured capability.

## Frozen references

- Tencent-Hunyuan/Hy-Nomas: `911cea6458a89c2663513ff7099561ec190dd491`.
- Conversion/reference runtime jiaxiangc/diffusers:
  `12722ef11acf8e335f96cfd4faea17de78cd8909`.
- Checkpoint tencent/Hy-Nomas-Diffusers:
  `a1ae04d11ae4ea176d1f76de634c115944f9882d`.
- Framework baseline: `1b6c6c50cf7aaaff82d579e41072a2fe0057d212`.

## Ownership and baseline

The model package owns DiT, camera conditioning, prompt action masks, PixelUMM,
and segment state. There is no runtime import from the reference fork. Standard
Transformers loads Qwen2.5-VL's text model and tokenizer. The existing modular
index reader resolves the checkpoint; no custom model-name discovery is needed.

One world per request, BF16 weights, CFG=1, Euler flow matching (8 steps,
shift 5), and a separate noisy cache commit at sigma 0.206897 define the baseline.
The text encoder uses hidden state -3 and crops the Qwen system/user prefix.
Normalization and modulation rounding must follow the reference explicitly.

Video latents have 32 channels and spatial/temporal ratios 32/4. The default
frame size is 704x1280, at 24 fps, with 30 latent frames per segment. The final
segment can be shorter. Requests retain their entire initial noise timeline;
cache-write draws must not interleave with initial noise creation. PixelUMM
uses its own deterministic absolute-frame noise seeded with 42.

Historical KV contains both video and text, with action-text visibility masks.
The default keeps bounded full history owned by the model and reserves it
through the AR runner's model-owned memory budget. The optional
`model_config.hy_nomas_max_cache_segments` positive integer follows the native
whole-segment eviction policy: denoising sees retained history; eviction happens
immediately before the next cache commit. Absolute frame positions continue
advancing. This policy is not numerically equivalent to full-history generation. It does not
claim paged-KV reuse, sliding-window equivalence, INT8, FP8, SP, or TP support.
Reset, completion, abort, and eviction must release this state. Limits are
validated before generation, rather than silently discarding historical KV.

Prompt updates replace a complete segment condition, including its token mask
and action boundaries; embedding interpolation is not supported. Camera events
use the existing camera timeline and model-local coordinate conversion.
Only stable PixelUMM prefixes may be appended to realtime media; finish emits
the remaining tail. Provisional previews cannot be appended because later
segments can revise them.

## Validation boundary

Initial transfer and isolated checks used a GPU-less instance with 2 GiB RAM.
A later GPU validation on 2026-10-25 used one RTX PRO 6000 Blackwell Server
Edition (97,887 MiB), CUDA 13.0, PyTorch 2.13.0, vLLM 0.31.0, Diffusers 0.40.0
and Transformers 5.14.1. The results below qualify only their stated workloads;
smoke success does not establish full native numerical parity or production
suitability.

## Experimental execution

The supplied deployment requests two 120-frame segments at 704x1280. The
original concatenating baseline reserved about 46 GiB for history before
weights and workspaces. The preallocated implementation reserves about 23 GiB
for mixed KV at this geometry, plus conditioning and decoder state. This is
an admission budget, not a claim that measured peak GPU memory halves.
There is no validated hardware minimum yet. The original 24 GB recipe relies on INT8, FP8 and broader offload policies
that remain unsupported here.

The supplied deployment pins TORCH_SDPA for its explicit precision baseline.
After the checkpoint is complete and an adequately sized GPU is available:

```bash
vllm serve /root/autodl-tmp/models/tencent/Hy-Nomas-Diffusers \
  --omni \
  --deploy-config vllm_omni/deploy/hy_nomas_stepwise.yaml \
  --port 8091
```

Reuse the client in
[`examples/online_serving/streaming_video_generation`](../../../examples/online_serving/streaming_video_generation/README.md).
No new protocol or model-specific Python example is introduced. Session-start
geometry must match the deployment. Prompts are strings, including serialized
structured Hy-Nomas JSON. `extra_args.segment_prompts` can supply one complete
prompt per generated segment for offline alignment; without it the initial
prompt is repeated until a live prompt event replaces it. There is no automatic
prompt enhancement or implicit rewriting of structured prompts.

Live prompt events use `transition_chunks=0`; their token masks and action
ranges are re-encoded atomically for the next segment. Live camera events use
the existing Unity timeline, resampled to the native `4k+1` poses before the
transformer repeats the first pose for temporal packing. Native normalized camera input can instead be
supplied through `extra_args.camera_extrinsics` with shape
`[1, num_frames, 4, 4]`; these inputs cannot be combined with live camera events.
The extrinsics-attention architecture does not consume intrinsics.

This implementation supports one bounded stepwise rollout per request, not
cross-request tick continuation, rollback, or unbounded sessions. Nonfinal
segments must cover at least one decoder window to emit stable video. Final
segments may be shorter and always flush the withheld tail. The sampler creates
all rollout noise before cache-write draws, using a CPU generator by default;
explicit caller generators remain supported. Reference comparisons must use
the same total noise capacity, segment lengths, normalized poses, text, seed,
steps, and checkpoint dtype. The native FA2/Magi/FA3 attention combination is
not qualified by the SDPA-only released-weight comparisons below.

## Local verification

Focused tests are collected by the existing `Simple · Diffusion Models Test`
CPU glob in `.buildkite/cuda/test-ready.yml`; no extra CI job is required.
Run in a supported environment with matching vLLM, PyTorch, Transformers and
Diffusers installed:

```bash
pytest tests/diffusion/models/test_hy_nomas_transformer.py \
  tests/diffusion/models/test_hy_nomas_decoder.py \
  tests/diffusion/models/test_hy_nomas_pipeline.py \
  tests/diffusion/models/test_hy_nomas_controls.py \
  tests/config/test_hy_nomas_stepwise_deploy.py \
  -m 'core_model and cpu' --run-level=core_model
```

During development, the local Intel macOS environment instead used an isolated
PyTorch 2.2.2 CPU harness: vLLM imports and shared Attention were replaced with
minimal infrastructure and real PyTorch SDPA. The 26 focused model tests passed
in that harness. The source arithmetic was tested
against the fixed reference. This is not a normal repository test run or a
serving test. Tiny random-weight transformer comparisons gave maximum absolute
differences of 2.384e-7 in FP32 and 0.00390625 in BF16; tiny PixelUMM FP32/BF16
comparisons were exact. Camera normalization agreed with the native NumPy/SciPy
implementation within 1e-7. Full-weight accuracy and generated video inspection
remain required before advertising model support.

The checkpoint index was also checked against complete models on the meta
device: all 2268 transformer keys and 476 decoder keys matched, with no missing
or extra parameters. This checks naming and architecture coverage, not tensor
contents, successful weight loading, or released-weight numerical parity.

A normal CPU pytest run was attempted on the supplied remote instance using
its existing vLLM 0.31 environment and isolated pytest dependencies. The process
received SIGKILL (exit 137) before collection produced results. Cgroup OOM
counters remained zero, so the cause is unconfirmed. This attempt provided no
normal-environment test pass. The later GPU instance successfully ran the
normal-environment focused tests described below.

## GPU validation on 2026-10-25

The focused suite passed 40 CPU cases and 4 FP32/BF16 CUDA component cases.
CUDA component tests execute the real shared TORCH_SDPA layer without backend
mocks. Existing Buildkite `Diffusion · Model Test` collects the CUDA test file;
the CPU and Other jobs collect the remaining focused cases.

```bash
pytest tests/diffusion/models/test_hy_nomas_gpu.py -q
pytest tests/diffusion/models/test_hy_nomas*.py \
  tests/config/test_hy_nomas_stepwise_deploy.py \
  -m 'core_model and (cpu or cuda)' --run-level=core_model
```

Requirements: compatible vLLM/PyTorch/Diffusers/Transformers/SciPy environment,
CUDA for GPU cases and BF16-capable hardware for BF16 cases. The focused tests
use tiny components and do not download model weights.

Released-weight smoke used the strict shared loader, BF16, 704x1280, 24 FPS,
8 steps, CFG=1 and seed 42. A one-segment offline run produced 120 frames with
full MP4 decoding and a 59.18 GiB measured PyTorch peak. Its final log event
initially failed due to a duplicate elapsed field after the video and successful
frame-count result had been written; the external test harness was corrected.

The existing WebSocket client and AR engine completed two segments as 84 + 156
frames plus the muxer trailer. The final MP4 fully decoded to 240 frames at
24 FPS (10 seconds), 1280x704, with strictly increasing PTS. A CUDA allocation
OOM warning appeared during this run before it successfully completed; retain
that limit when assessing the default backend and memory headroom. The first
frame was visually inspected as a coherent forest/fox image. This checks basic
content and transport, not perceptual parity or long-video stability.

## Image conditioning and long-duration boundary

The frozen public native `hymm/pipeline/causal_ar.py` inference entry point and
the converted modular pipeline expose text/camera continuation. They do not
provide a released image-conditioned inference contract. Training-only
`initial_latent` paths do not establish the released checkpoint's I2V behavior.
The current pipeline has no image encoder or image-conditioning capability.
Do not treat an inserted cover frame or an image caption as image conditioning.

At 24 FPS, 60/180/300 seconds require 12/36/60 five-second segments. Full BF16
mixed KV history alone is approximately 137/411/684 GiB at the configured
maximum text length. The bounded-history option limits KV retention, while
`hy_nomas_max_segments` still bounds total rollout length and noise capacity.
Minutes-long generation requires separate qualification of that history policy
and incremental video encoding. It is not qualified by the two-segment smoke.

### Full-weight DiT SDPA comparison

The released 54-layer BF16 transformer was compared with the frozen converted
reference using PyTorch native SDPA on both sides. Inputs were the same Qwen
embeddings (maximum length 1280), seed-42 CPU noise timeline with 31 latent
frames, 704x1280 geometry, 8 Euler steps/shift 5, a 30-latent-frame segment and
one-latent-frame continuation. The continuation used identity camera input.
All 16 checked predictions and updated latent tensors, both final latent
segments, and the checked first-layer committed K matched exactly in this row.
This does not qualify the original optional FA2/Magi/FA3 kernel combination.

The test exposed a real numerical defect in the initial port: removing masked
K/V tokens changed CUDA SDPA's BF16 layout and reduction order. The old first
prediction had relative L2 error 0.00984, which accumulated through independent
denoising. The corrected shared-attention path keeps complete sequences and a
broadcast key mask. The regression checks the actual Q/K/V objects and mask
layout, in addition to mathematical visibility. No quadratic mask is required.

### Full-weight PixelUMM SDPA comparison

Using the same saved generated BF16 latent as both inputs, all released decoder
weights (476 keys) were compared at 704x1280 with lengths 1, 10 and 11 latent
frames. The corresponding raw RGB tensors contained 4, 40 and 44 frames. Every
value matched exactly, including per-frame maxima at overlapping-window edges;
max/mean absolute error and RMSE were zero for all three cases. Both sides used
the configured local neighborhoods and native SDPA arithmetic. The measured
peak for the sequential comparison was 18.25 GiB.

The frozen reference unnecessarily constructs a Flex block mask even when its
native SDPA processor ignores it. That construction hit a PyTorch 2.13 symbolic
shape compiler error. The external comparison harness used the reference's
unchanged integer layout function on CPU and transferred the permutation to
CUDA, omitting only the unused Flex mask. Original neighborhood lists and RGB
arithmetic were retained. This workaround is oracle-only and introduces no
external reference dependency in the production model.

### Qualification matrix

All full-weight rows use the frozen checkpoint, BF16, one RTX PRO 6000, no
quantization/offload/distributed layout, and 704x1280 geometry. Unlisted feature
combinations remain `not tested`.

| Workload and execution | Backend / history | State | Evidence scope |
| --- | --- | --- | --- |
| Tiny CPU + CUDA component tests | SDPA / bounded state | `validated` | 44 cases; no released weights in unit tests |
| Full-weight DiT, offline, two segments (30+1 latent), 8 steps | Native SDPA vs TORCH_SDPA / full mixed KV | `validated` | Exact checked predictions, trajectory latents and first-layer committed K |
| Full-weight PixelUMM, offline, 1/10/11 latent frames | Native SDPA vs TORCH_SDPA / configured overlapping windows | `validated` | Exact raw RGB, including edge frames |
| Original FA2/Magi/FA3 reference runtime | Original kernels / full KV | `not tested` | SDPA equality does not establish this row |
| Initial two-segment WebSocket smoke | Platform default / full mixed KV | `limited` | 240 decoded frames and valid PTS; allocation OOM warning occurred |
| Three-segment WebSocket smoke, 360 frames | TORCH_SDPA / one retained full segment, 10-second pings | `limited` | Full MP4 decode and PTS checked; not a minutes-long qualification |
| Released image-conditioned inference | No public image input contract | `unsupported` | No encoder or image capability in this integration |
| Six requested minutes-long videos | Input contract pending | `not tested` | Not started as an unqualified I2V workload |

The mask-preserving precision path can exceed the realtime endpoint's existing
60-second stall interval between media chunks. An initial three-segment test
without application pings was stopped after its first 84 frames and is not a
successful full-video result. The shared client now accepts the optional
`--ping-interval 10` keepalive using the existing `session.ping` protocol;
server timeout defaults and ping semantics are unchanged. An external overall
runtime bound is still needed for unattended tests.

The corrected precision path completed a three-segment WebSocket smoke with
`hy_nomas_max_cache_segments=1` and `--ping-interval 10`. Media chunks contained
84 + 120 + 156 frames, followed by a trailer. The remuxed MP4 fully decoded to
360 frames at 24 FPS (15 seconds), 1280x704, with strictly increasing PTS. The
client observed about 223 seconds from session start to the trailer for this
single run; this is not a throughput benchmark. This row qualifies three
segments with whole-segment eviction and keepalive, not minutes-long stability.

The additional six shared-client keepalive tests passed in the same remote
package environment. Four existing server stall/ping contract tests also
passed. The model suite remains 44 cases; these are separate coverage groups.

## Acceleration milestones

The requested priorities are generation latency and single-GPU memory for the
realtime video integration. Freeze `1f428910efa3d82124e576e218bc402b06a6610a`
as the comparison baseline. Keep eight steps, the separate noisy cache-write
forward, full token/mask layout, and the selected history policy unchanged.

| Milestone | Scope | Acceptance |
| --- | --- | --- |
| M0 | Released-weight stage timing and memory baseline | Fixed geometry, inputs, seed, backend, hardware and revisions |
| M1 | Bounded exact conditioning reuse | Repeated prompt encoding and per-segment RoPE reuse; exact outputs and lifecycle coverage |
| M2 | Preallocated mixed KV | Full masks, uncommitted tail, whole-segment eviction; exact predictions, latents and RGB; measured memory |
| M3 | Single-GPU attention/decoder kernels, optional encoder offload, compile and graphs as separate experiments | Explicit numerical tolerance, memory/latency tradeoffs and individual timing evidence |
| M4 | Prototype streaming DiT-to-decoder stages if kernel gains are insufficient | Interaction ownership, bounded latent queue, backpressure, cancellation and decoder finalization; measured first-chunk and steady-state latency |
| M5 | Ulysses or independent decoder-window parallelism when multiple GPUs are available | Cross-frame/action boundaries, SP parity and scaling measurements |

M1 currently caches only the most recent `(prompt, max_length)` encoding per
session and the current segment's FP32 video RoPE tables. Prompt changes replace
the entry; action windows are recomputed with segment length and FPS. RoPE uses
the original CPU trigonometry on continuation segments. RoPE is released after cache-write, before pixel decoding. Reset/close drops
the session-owned references. Camera projections and decoder tables/noise remain
future M1 work; no unbounded dictionary of prompts or segments is retained.

M2 grows each layer's buffers on the first forward of each larger segment
prefix, bounded by the request's maximum simultaneous tokens. Later denoising
steps reuse this allocation; future segments are not allocated in advance. Attention reads a prefix with unchanged K/V order
and visibility mask. Denoising overwrites the scratch tail without advancing
committed length. The cache-write forward alone commits the tail and its action
visibility mask. Full history needs at most the request's total tokens. Bounded
history needs up to `retained_segments + 1` segments because eviction still occurs
only before cache-write. Growth and compaction use one layer's one tensor as temporary scratch, and
the admission budget includes the larger of these nonoverlapping bounds. Dictionary caches remain
supported by standalone transformer callers and provide the comparison oracle.

### Corrections to the earlier acceleration proposal

- Removing invisible historical tokens is **not** an exact optimization of this
  baseline: the earlier released-weight experiment already demonstrated a BF16
  SDPA regression. The current `_attend` has no `nonzero`/gather path to remove.
  Treat compaction/unpadding as a separate numerical experiment.
- A preallocated capacity is not measured live-token usage. The initial experiment that allocated the full request capacity increased
  first-segment allocation; per-segment growth avoids that regression.
  Report allocated peak, reserved peak and admission budget separately.
- `30 * 880 = 26400` is divisible by eight; `3300` tokens per rank cuts through
  frames. Camera/action boundary handling is required even without token padding.
  Head-sharded historical KV also requires explicit ownership around all-to-all;
  enabling a shared sequence-parallel plan alone does not establish that layout.
- Prompt-result caching does not release the text encoder's resident weights.
  The frozen baseline rejected CPU offload. The optional text-only module
  policy described below uses the shared offloader and requires its own
  prompt-change latency qualification.
- CUDA graphs require stable addresses and capture-safe operations in addition
  to stable shapes. Removing padding does not by itself make the pipeline safe
  to capture. Do not couple its acceptance to the eager exact-cache changes.
- Pinned asynchronous D2H requires lifetime/event ownership and overlapping work
  to improve latency. Moving quantization onto the GPU requires a separate
  postprocessing equivalence check.
- Decoder-window parallelism must preserve noise, window inputs, blend order and
  stream-tail semantics. Same-model GPUs alone do not prove bitwise equality.

The [PyTorch SDPA documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
explains backend-dependent numerical differences; mathematical equivalence is
not sufficient evidence for the exact-output acceptance tier.

### Optional text-encoder offload

The explicit shared policy below allows the Qwen text encoder to move to CPU
before DiT executes. DiT and PixelUMM remain GPU resident; the streaming decoder
is never added to the offload hook set. Repeated prompt hits skip encoder
execution and its H2D transfer. A changed prompt brings the encoder back to the
GPU, so prompt-change memory and latency must be budgeted separately.

```json
{"mode": "module", "components": ["text_encoder"]}
```

Pass this mapping as `diffusion_offload_config` (CLI:
`--diffusion-offload-config '{"mode":"module","components":["text_encoder"]}'`).
The normal model runner installs the existing shared offload backend. Direct
loader-based experiments must call `enable_offload_backend` explicitly, as for
other shared-offloader integrations. The supplied precision deployment remains
resident by default. Legacy `enable_cpu_offload`, DiT selection and layerwise
policies remain rejected because they select unqualified component lifecycles.
No quantization, step reduction or decoder-window changes are involved.

### Acceleration validation on 2026-10-25

Compared the frozen baseline against this implementation on one RTX PRO 6000
Blackwell (97,887 MiB), driver 595.71.05, PyTorch 2.13.0+cu130, vLLM 0.31.0 and
Diffusers 0.40.0. Both processes used the same local released checkpoint files,
BF16, TORCH_SDPA, seed 42, eight steps, 704x1280 and two 30-latent-frame segments.
Camera inputs were identity poses and the prompt repeated across segments.
Each variant ran two rollouts in its loaded process. These are exploratory
measurements, not a statistical performance qualification.

Timed segments sum condition preparation, denoising/scheduler, cache-write and
decode/D2H. They exclude weight loading, output hashing, media encoding and
network/engine overhead. The first and repeated rollout are shown separately
because the first offload incurs substantial host-allocation/transfer setup.

| Variant | First segment, first/repeated rollout (s) | Second segment mean (s) | Peak allocated (GiB) | Peak reserved (GiB) |
| --- | --- | --- | --- | --- |
| Frozen baseline | 52.83 / 52.08 | 82.90 | 70.69 | 93.78 |
| Segment KV growth + prompt/RoPE reuse, resident weights | 52.60 / 51.87 | 82.46 | 70.69 | 93.78 |
| Same changes + text-encoder module offload | 60.60 / 52.65 | 82.56 | 57.50 | 85.24 |

The resident changes do **not** establish a significant latency speedup or an
actual peak-memory reduction. Their mixed-KV admission estimate falls from
45.62 to 22.92 GiB including one-tensor growth scratch, independently of the
measured whole-process peak. The resident candidate emitted a CUDA allocation
failure/retry warning and completed; do not hide this allocator-headroom limit.
No such warning was observed in the text-offload runs. Allocated peak falls
18.7% with offload, at the cost of about eight seconds on the first transition
in this experiment. A repeated rollout's first segment was about 0.6 seconds
slower than the baseline. Reserved memory still exceeded 80 GiB, so this does
not establish an 80 GB deployment minimum. Use the documented client keepalive
for long intervals between media chunks; this experiment was offline.

All 32 predicted-flow tensors, four final latent tensors and four raw RGB
chunks per candidate matched the baseline's SHA-256 byte hashes. This covers
240 output frames per rollout, emitted as 84 + 156 frames. A second workload
used 10 + 10 + 1 latent frames, one retained segment, structured action groups,
a changing camera trajectory and a final prompt replacement. All 24 predicted
flows, three latents and three raw RGB chunks matched between the baseline and
the offloaded candidate (84 frames total). The encoder was on CPU after every
segment, including after the prompt replacement. Prompt-change preparation in
that short-tail case increased from 0.073 to 0.338 seconds; its denoising also
included the subsequent offload transition. This is not a bound for prompt
updates with larger histories.

The final focused suite passed 52 cases (48 CPU and four CUDA FP32/BF16 cases).
The CUDA cases compare concatenating and reserved KV with the real shared SDPA
backend. Tests cover noncommitting writes, capacity rejection, whole-segment
eviction timing, short tails, action/camera conditions, prompt replacement,
session cleanup and offload-policy rejection. Diff-scoped pre-commit checks,
including mypy, SPDX, test marks and markdown lint, passed. WebSocket serving,
multicard scaling, other attention backends and minutes-long rollouts were not
requalified by this acceleration work.

### M3: compressed decoder K/V and native GQA

PixelUMM now passes its original eight K/V heads to shared attention instead
of expanding them to 32 heads before gathering each spatial neighborhood.
The shared SDPA implementation checks fused GQA support and otherwise expands
K/V locally. No new attention backend or dependency is introduced. This uses
the existing [PyTorch GQA interface](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html);
kernel availability and numerical results remain hardware/version dependent.

On the same RTX PRO 6000 Blackwell and PyTorch 2.13.0+cu130, released BF16
decoder weights and saved generated latents at 704 x 1280 were decoded with
three implementations in one process. Each row below is the median of three
warm calls, after one untimed-for-comparison call. Timing includes GPU decode
with synchronization, excludes CPU copying/comparison, and uses the unchanged
temporal-window, spatial-neighborhood and noise settings.

| Latent frames | Original expanded K/V | Expand after neighborhood gathering | Native GQA | GQA reduction |
| --- | --- | --- | --- | --- |
| 1 | 0.221 s | 0.232 s | 0.213 s | 3.6% |
| 10 | 2.624 s | 2.566 s | 2.419 s | 7.8% |
| 11 (overlapping windows) | 5.286 s | 5.168 s | 4.866 s | 7.9% |

All three lengths produced byte-identical BF16 RGB tensors for both candidates.
Peak allocated memory was unchanged at 13.508, 17.823 and 18.247 GiB respectively:
this removes intermediate K/V copies but does not lower the full decoder's
peak in this workload. The native-GQA implementation was retained; the
window-local expansion experiment was discarded. These component measurements
do not establish an equivalent percentage improvement in end-to-end latency.

The focused decoder/GPU/shared-SDPA suite passed 13 cases, including real CUDA
FP32/BF16 global and spatial attention against expanded K/V, streaming overlap,
and the shared backend's fused-GQA selection and expansion fallback. Exact
released-weight parity above is specific to the measured BF16 platform; it is
not a cross-backend bitwise guarantee.

The two-segment, 240-frame offline pipeline benchmark was then repeated twice
with text-encoder offload enabled, comparing the previous offloaded candidate
against this GQA change. On the second rollout in each loaded process:

| Metric | Expanded K/V | Compressed K/V / GQA |
| --- | --- | --- |
| First segment decode + D2H | 13.751 s | 12.676 s |
| Second segment decode + D2H | 16.589 s | 15.429 s |
| First segment total timed compute | 52.651 s | 51.569 s |
| Second segment total timed compute | 82.540 s | 81.375 s |
| Peak allocated GPU memory | 57.496 GiB | 57.496 GiB |
| Peak reserved GPU memory | 85.244 GiB | 85.244 GiB |

The combined timed compute decreased by 1.7% in this comparison. All 32
predicted-flow tensors, four final latents and four raw RGB chunks across both
rollouts matched the previous candidate's byte hashes. The first rollout was
also faster, but includes variable initial offload/allocator effects. These
are two rollouts per configuration, not a statistical latency qualification;
the timings exclude loading, hashing, muxing and network transport. No
additional peak-memory improvement was measured from GQA.

### M3: reuse of HY1.5 DiT components

The image and text MLPs in all 54 main blocks now use the same shared
`FeedForward` class as `HunyuanVideo15Transformer3DModel` (defined in the Flux
module). This reuses its column-parallel projection, approximate GELU and
row-parallel projection instead of maintaining a separate GELU MLP. Released
checkpoint names `img_mlp/txt_mlp.fc1/fc2` are mapped to `net.0.proj/net.2`;
missing, duplicate and unexpected checkpoint tensors still fail loading.
The text token refiner retains its distinct SiLU MLP. TP/SP and quantization
remain unsupported by the Nomas pipeline: reusing parallel layer classes
does not qualify distributed execution. Direct component harnesses must set
up single-rank vLLM parallel state, as they do for HY1.5.

The following additional HY1.5 paths were tried but **not retained** in the
default implementation:

| Candidate | Observation on the same BF16 Blackwell setup | Decision |
| --- | --- | --- |
| Packed `QKVParallelLinear` for image/text | First full DiT prediction relative L2 0.010446, RMSE 0.014430, max absolute error 0.140625 versus separate projections | Retain separate Q/K/V for the precision baseline |
| vLLM RMSNorm with `vllm_c` IR dispatch | Representative operator input relative L2 0.002821 versus diffusers RMSNorm | Retain reference normalization |
| Shared fused RoPE with BF16 tables | Operator relative L2 0.002396; FP32 tables with BF16 inputs are rejected by the fused kernel | Retain FP32 tables and reference arithmetic |
| Shared fused RoPE with FP32 inputs and tables, cast afterward | Operator relative L2 0.0000100; still not byte-identical | Keep as a future numerical-tolerance experiment |

The full DiT comparison uses the released weights, actual encoded benchmark
prompt, seed-42 first noisy segment and identity camera inputs at 704 x 1280.
Its reference prediction reproduces the previous benchmark hash. The packed
QKV plus shared MLP candidate has the same first-prediction hash as the
packed-QKV-only experiment. Its initial full rollout also changes latent and
RGB hashes. The longer QKV timing run was stopped after detecting this mismatch;
no completed latency qualification is claimed for packed QKV.

The normalization/RoPE operator probes use seeded random BF16 inputs of shape
`[1, 26400, 16, 128]`, the released first image-Q normalization weight, and
continuation RoPE tables. Their errors are operator-level evidence, not final
video-quality measurements. vLLM RMSNorm's native fallback was byte-identical
in that probe; testing only that fallback would have missed the fused-kernel
precision change. No new custom kernel, approximation flag or accuracy
tolerance has been silently enabled.

The retained shared-MLP version passed 16 focused CPU/CUDA tests and two
released-weight 240-frame rollouts with the existing GQA decoder and
text-encoder offload. All 32 predictions, four final latents and four raw RGB
chunks matched the preceding GQA candidate's byte hashes. In the second
rollout, segment compute times were 51.591 / 81.314 seconds versus
51.569 / 81.375 seconds previously. Peak allocated/reserved memory remained
57.496 / 85.244 GiB. This establishes component reuse with unchanged outputs
on the measured workload; it does **not** establish a latency or memory gain.
The direct harness initializes single-rank NCCL/vLLM state for the shared
parallel linear layers. Diff-scoped pre-commit checks passed. The CPU loader
regression checks both image/text MLP outputs against the original checkpoint
names and rejects missing or duplicate mapped weights.

### Pure deduplication of HY1.5 components

The first reuse tier preserves the reference arithmetic and checkpoint contract:

- `img_in` uses `HunyuanVideo15PatchEmbed((1, 1, 1), ...)`; the Nomas caller
  retains `.contiguous()` so downstream GEMMs see the original memory layout.
- Shared `LayerNorm` replaces the local class. Token-refiner norms retain
  input dtype; double-stream and final-layer norms set `keep_fp32_output=True`
  for subsequent modulation. The option defaults to false and is keyword-only;
  enabling it bypasses MindIE-SD on NPU in favor of native FP32 normalization.
- `hunyuan_video/qwen_prompt.py` owns chat tokenization, user-marker discovery
  and hidden-state extraction. Tokenization stays separate from encoding so
  Nomas action boundaries reuse the same token IDs. HY1.5 retains its fixed
  crop of 108, empty-prompt space substitution, encoder fields, default cache
  behavior and final dtype cast. Nomas retains dynamic cropping and
  `use_cache=False`; its action timing logic remains model-specific.
- `WeightsMapper` declares the MLP name mapping. Loading still rejects missing,
  duplicate and unexpected checkpoint keys, including unmapped internal names.

No AdaLN replacement, timestep/RoPE rewrite, prompt-interaction redesign,
packed QKV or fused normalization is included in this tier. This is code reuse,
not a claim of lower latency or memory use. HY1.5 I2V retains the existing
`format_text_input` import location.

Validation on the same RTX PRO 6000 Blackwell / PyTorch 2.13 setup, BF16,
TORCH_SDPA, seed 42 and eight denoising steps:

- Two 240-frame, 704 x 1280 rollouts (30 + 30 latent frames) match the retained
  shared-MLP baseline: all 32 flow predictions, four latents and four raw RGB
  chunks have identical SHA-256 hashes.
- The 10 + 10 + 1 latent-frame workload, with one cached history segment,
  action grouping, moving camera and a final prompt replacement, matches the
  recorded control baseline: all 24 predictions, three latents and three RGB
  chunks are byte-identical.
- Focused Nomas/HY1.5, shared normalization and neighboring CPU suites pass
  1184 tests (eight skipped, 50 non-CPU tests deselected). Six Nomas CUDA
  component tests also pass. Tests cover all 2268 transformer checkpoint keys,
  strict loading failures, FP32-output NPU fallback, system-prompt equality,
  the real local Qwen tokenizer's crop of 108, and HY1.5 encoding arguments.
- Diff-scoped pre-commit hooks pass except three existing mypy diagnostics
  about `LayerNorm`'s multiple-inheritance constructor, reproduced on the
  unmodified baseline. No new mypy diagnostics remain.

Artifacts: `reuse-main.json`, `reuse-contract.json`, `reuse-summary.json`,
`reuse-tests.log`, `reuse-gpu-tests.log`, and `reuse-source-manifest.json`.
These tests do not qualify full HY1.5 video generation or NPU hardware execution;
NPU fallback is tested with a mocked fused-kernel entry point.

### M4: two-stage model boundary and realtime transport

A first split would keep conditioning and the autoregressive DiT in stage 0,
and move the stateful streaming PixelUMM decoder into stage 1. Transfer latent
segments with request ID, segment index, absolute frame offset, valid frame
count and final-segment status. A bounded queue must apply backpressure before
the producer creates more segments; the decoder must retain overlap state
and flush exactly once. Cancellation must release both stages and queued
latents, including during a partially emitted chunk.

The initial implementation provides `HyNomasDenoisePipeline` (text encoder
and DiT only) and `HyNomasDecodePipeline` (PixelUMM only). Select the explicit
`hy_nomas_two_stage.yaml` deployment for the experimental full-request topology.
The existing single-stage topology and realtime deployment remain unchanged.
The two-stage deployment uses the standard diffusion engine on both stages,
`async_chunk: false`, and the generic `diffusion_stage_handoff` processor.
It does **not** implement realtime chunk transport or interactive updates.

Stage 0 commits KV in the original order, copies completed latents to CPU,
and emits the complete segment list at request completion. Each entry carries
version 1, index, total segment count, absolute latent offset, valid pixel-frame
count, a final flag and the latent tensor. Stage 1 validates the whole list
before decoding, rejecting gaps, reordering, inconsistent geometry/dtype and
invalid frame counts. It transfers segments to its GPU one at a time, uses a
single request-local streaming decode session, and flushes once. Decoder state
is not stored on the pipeline across requests. Stage 0's existing `finally`
cleanup still releases its request KV state after success or failure.

This first topology buffers the full latent request on the host and the final
video on the decoder host; it is bounded by the configured producer segment
limit, not a cross-stage backpressure protocol. The deployment places both
workers on GPU 0 as an experimental single-card configuration. Independent
worker/allocator overhead and real service execution require qualification;
model-level parity alone does not establish that deployment's memory minimum.

Initial validation on the same RTX PRO 6000 Blackwell used two independently
loaded pipeline objects, the production Diffusers loader and generic handoff,
in one process with single-rank NCCL/vLLM state. BF16 / TORCH_SDPA, 704 x 1280,
seed 42, two 30-latent segments, eight steps and identity camera conditioning
produced 240 RGB frames. All 16 denoising predictions, two final latent tensors
and RGB slices corresponding to the previous 84 + 156 frame output matched
the single-stage baseline's byte hashes. The denoise object had no decoder;
the decode object had neither transformer nor text encoder. The test does not
cover multiprocessing/IPC, HTTP/WebSocket serving, concurrent requests or
mid-generation interactions, and makes no latency/memory improvement claim.
Forty existing/new pipeline, payload and configuration cases passed, followed
by two added tests for producer failure cleanup and real tiny-decoder overlap
with partial visible frames. Diff-scoped pre-commit checks passed.

The current orchestrator sends interactions explicitly to stage 0. Keeping the
DiT there is compatible with that routing, but arbitrary stage placement needs
an explicit interaction-owner contract and acknowledgement of the segment at
which an update took effect. Merely adding a decoder entry to deployment YAML
does not implement the streaming handoff or lifecycle.

On one GPU, separate stages still contend for the same compute and may retain
more activations concurrently. Stage separation alone is not evidence of
lower latency or memory. Qualification must compare first media-chunk latency,
steady-state chunk interval, interaction-to-visible-output delay and peak
allocated/reserved memory against the single-stage baseline.

#### Incremental realtime deployment

Select `vllm_omni/deploy/hy_nomas_two_stage_realtime.yaml` for the incremental
path. Stage 0 uses AR-Diffusion step execution and emits one CPU latent segment
at a time; stage 1 uses request execution and retains a PixelUMM streaming
session across those invocations. Both stage processes run on GPU 0. The
full-request deployment remains available separately.

Playback feedback/pacing (`streaming_pacing` or `streaming_buffer_seconds`)
continues to require a single-stage diffusion topology after rebasing onto main
`36027f8f9` (which includes #8721). It is rejected for both two-stage deployments.
The decoder acknowledgment bounds segment handoff only; it does not pace
production against the client's playback position.

The internal `diffusion_segment` envelope implements stop-and-wait flow control:

1. The step scheduler suspends a nonterminal producer request as soon as its
   segment is emitted, before scheduling another denoising wave.
2. The orchestrator forwards that segment through the generic diffusion bridge.
   The decoder validates the sequence, offsets, frame count, dtype and geometry
   against its retained session before pushing the latent.
3. A completed decoder invocation acknowledges the same segment index. Its
   engine invocation is terminal, while the orchestrator keeps the parent
   rollout active until the final segment. Only that acknowledgment restores
   the producer's credit. Duplicate or out-of-order segments/acks fail the request.
4. The producer retains the upstream runner's `pending_chunk_boundary` flag
   while waiting for the acknowledgment. Its next scheduled wave applies queued
   interactions and prepares the next chunk, in that order. Controls received
   while segment N is decoded therefore affect segment N+1. An interaction
   alone cannot restore segment credit.
5. The last segment flushes PixelUMM once and releases the decoder session.
   No acknowledgment is needed to resume an already completed producer.

There is at most one unacknowledged latent segment per request. This bounds
stage handoff, not the downstream WebSocket playback buffer. The initial
implementation supports exactly two stages and one decoder replica; it does
not add general multi-stage or session-affine replica scheduling. DiT stays in
stage 0, so the existing interaction routing remains applicable. Interaction
metadata accompanies the latent payload and returns with the decoded video.
The producer yields between denoising steps instead of grouping an entire
segment into one worker call, so control RPCs and aborts can run before the
next chunk boundary. For multiple stages, the frontend requires exactly two
diffusion stages, a step-execution streaming producer, and a decoder configured
with `streaming_stage_input` consuming stage 0. Encoder-first and full-request
multi-stage topologies reject interactions at ingress.
Controls received before worker admission are kept on the queued request
(up to 32 events), then enqueued into its interaction coordinator on admission.
Cancellation drops those controls with the request. Event metadata on media
chunks remains the indication of when an accepted control actually took effect.

Cancellation and stage errors use the existing all-stage abort path. The
producer retires step/KV state even while waiting for acknowledgment. The
decoder's abort hook drops retained history even between individual decode
invocations. Cleanup runs at serialized worker boundaries; cancellation does
not preempt an already running CUDA kernel. Late outputs are discarded after
the orchestrator removes the parent request.

The deployment enables `release_cache_between_segments` in both workers.
Separate processes cannot reuse one another's unused CUDA allocator blocks,
so each worker releases unused cache at a segment handoff. Live model weights,
DiT KV and PixelUMM overlap history remain allocated. This is a memory/latency
tradeoff for a shared GPU, not a general recommendation for separate GPUs.

The first multi-process WebSocket run on the RTX PRO 6000 Blackwell returned
84 + 156 media frames and a trailer, producing 240 frames at 1280 x 704 / 24 fps.
With eight denoising steps, BF16 / TORCH_SDPA and seed 42, first media arrived at
67.74 seconds and the trailer at 159.55 seconds. This run preceded allocator
cache release and reached approximately 96,924 MiB across the two CUDA worker
processes. These are serving observations, not a speedup comparison against
the earlier single-process benchmark.

The subsequent service run enabled segment cache release and per-step control
RPC boundaries. It passed three sequential WebSocket cases in the same server:

- Cancel during denoising, before any media output.
- Cancel immediately after the first 84-frame media payload.
- Immediately submit a recovery request and a same-text prompt event while
  admission is pending; receive its queued acknowledgment and its
  started/active/completed event IDs on segment 1, then finish all 240 frames.

The recovery video decoded at 1280 x 704 with strictly increasing PTS. Its
first binary media payload arrived at 63.401 seconds and the trailer/done at
155.053 seconds, including the preceding cancellation/admission overlap.
Sampling `nvidia-smi` every 0.5 seconds across the test sequence observed a
maximum sum of 88,932 MiB (86.85 GiB) for the two CUDA worker processes. This
includes process/context and allocator occupancy; it is not a PyTorch peak
allocated-memory measurement or a controlled single-stage comparison.

Validation artifacts are `hy-nomas-stream-runtime.json`,
`hy-nomas-stream-admission-runtime.log` and
`hy-nomas-stream-admission-server.log` in the remote validation directory.
The focused model, scheduler, RPC, cancellation, orchestration and entrypoint
suite passed **399 tests**. Diff-scoped formatting, lint, SPDX and test-marker
checks passed. The mypy hook still reports 59 diagnostics also reproduced on
the branch base for the same files; normalized comparison found no new
mypy diagnostics. Concurrent multi-client serving, multiple decoder replicas,
separate-GPU performance and three-stage routing remain unqualified.

#### Rebase and interaction-boundary regression validation

The branch was rebased onto main `36027f8f9`, including playback backpressure
from #8721. Validation on 2026-10-26 uses the same RTX PRO 6000 Blackwell with
PyTorch 2.13.0+cu130 and the repository CI's vLLM 0.32.0 wheel pinned to
`5df63a7203cdb9328d824a27b682e77c7b9928be`, in a separate environment. The old
vLLM 0.31.0 environment lacks APIs imported by current main and cannot collect
the affected entrypoint/runner tests.

The focused model, scheduler, interaction-wait, step-runner, RPC, cancellation,
orchestration, configuration and video-serving CPU suite passes 772 tests
(18 non-CPU cases deselected). Six CUDA component tests also pass, including
BF16 execution with the shared attention/linear dispatch. Replacing the runner
and entrypoint with their
pre-fix versions makes four regressions fail: prompt and camera delivery in
an acknowledgment wait, and interaction rejection for two/three-stage
full-request or encoder-first topologies. They pass with the fixes.

The real two-process WebSocket deployment passes three sequential cases:

1. Reject a paced request at ingress with the single-stage requirement.
2. Cancel after the decoder starts segment 0, then submit a recovery request.
3. During recovery segment 0's decode, submit a changed prompt and camera target;
   finish all 240 frames and verify both events start in segment 1.

External validation-only import wrappers record decoder entry/exit and the
producer's prepared prompt key without editing production sources. The recovery
controls were sent 0.024 seconds into a 12.702-second decoder window. The next
chunk was prepared after decoder exit and used the replacement prompt. Segment
0 has no interaction IDs; segment 1 carries both started/completed event IDs.
The resulting 1280 x 704 / 24 fps video contains 84 + 156 frames with strictly
increasing PTS. These observations validate timing and lifecycle semantics;
they do not establish a speedup or memory improvement.

Artifacts are `hy-nomas-review-runtime.json`, `hy-nomas-review-trace.jsonl`,
`hy-nomas-review-server.log`, `hy-nomas-review-all-tests.log` and
`hy-nomas-review-red-tests.log`. The external probe lives outside the checkout.
Diff-scoped hooks have no new mypy diagnostics: the runner property override
and two existing step-pipeline test diagnostics are also present on main.
