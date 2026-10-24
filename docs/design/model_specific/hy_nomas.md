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

The supplied deployment requests two 120-frame segments at 704x1280. Its full
history reservation is conservative (about 46 GiB before model weights and
workspaces). An 80 GB GPU is insufficient for this resident configuration;
there is no validated hardware minimum yet. The original 24 GB recipe relies
on INT8, FP8 and offload that this baseline deliberately rejects.

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
|---|---|---|---|
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
