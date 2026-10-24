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
The initial implementation keeps bounded full history owned by the model and
reserves it through the AR runner's model-owned memory budget. It does not
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

The supplied remote instance currently exposes no GPU and only 2 GiB RAM.
Weight transfer is independent of inference qualification. Tiny CPU parity,
source inspection, and checkpoint checks cannot establish released-weight
E2E parity, live WebSocket playback, latency, or production suitability.

## Experimental execution

The supplied deployment requests two 120-frame segments at 704x1280. Its full
history reservation is conservative (about 46 GiB before model weights and
workspaces). An 80 GB GPU is insufficient for this resident configuration;
there is no validated hardware minimum yet. The original 24 GB recipe relies
on INT8, FP8 and offload that this baseline deliberately rejects.

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
not yet reproduced by a released-weight test of the shared attention backend.

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
counters remained zero, so the cause is unconfirmed. This attempt provides no
normal-environment test pass; the 2 GiB instance remains download-only for now.
