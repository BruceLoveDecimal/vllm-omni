# LingBot World stepwise request alignment

> **Status:** proposed implementation contract.
> This document specifies how LingBot World v2 realtime generation should use
> the same request, step, and chunk surface as Helios-style streaming, while
> keeping AR-Diffusion paged KV ownership on `ARDiffusionEngine`.

Related in-tree contracts:

- [Realtime AR-Diffusion sessions](realtime_ar_diffusion.md)
- [AR-Diffusion pipeline capability](../ar_diffusion_pipeline_capability.md)
- [Diffusion continuous batching / step execution](diffusion_continuous_batching.md)

## 1. Goal

One `AsyncOmni.generate()` call is one interactive rollout.

Today, LingBot realtime wraps `AsyncOmni.generate()` so that **one call
produces one AR block** (three latent frames). An external session manager
loops those calls. Helios instead keeps **one request alive across many
chunks**: `prepare_encode` once, then `denoise_step` / `step_scheduler` /
`post_decode` at chunk boundaries, with `streaming_output=True`.

After this change, LingBot realtime uses that Helios-shaped path:

```text
one generate(request_id)
  -> prepare_encode()
  -> repeat
       4 x (denoise_step + step_scheduler)     # one AR block
       post_decode() yields one latent chunk
  -> request completes (or the caller aborts)
```

Paged KV stays on `ARDiffusionModelRunner`. The pipeline is not migrated onto
the generic `DiffusionModelRunner`.

## 2. Non-goals

Do not include these in the first landing:

- Mid-request camera or prompt interactions through `submit_interaction`.
  First landing may consume a **pre-baked** camera/prompt script on the
  request. Live WASD and prompt updates are a follow-up on this path.
- Public HTTP / WebSocket `/v1/realtime/video`. Serving can attach only after
  stepwise chunks exist on `AsyncOmni.generate()`.
- Stateful streaming VAE decode. Realtime output remains `output_type="latent"`.
- Removing the 117 raw-frame / 10-tick image-condition horizon.
  First landing uses `total_chunks <= 10`.
- Replacing `ARDiffusionEngine` or merging LingBot session memory with other
  world-model session managers.
- Enabling stepwise on DreamZero or other AR-Diffusion pipelines.
- Deleting `ARDiffusionSessionManager` / the tick example. Keep them until a
  later deprecation pass.
- `max_num_seqs > 1` or request-batch AR execution.

## 3. Time units

Three clocks must not be collapsed:

| Unit | Helios today | LingBot today | This alignment |
| --- | --- | --- | --- |
| Denoise step | One noise prediction | One of four DMD steps inside `forward()` | `denoise_step()` = one DMD step |
| Chunk | One streamed output boundary | One AR block (3 latent frames), called a tick | `post_decode()` boundary = one AR block |
| Request | One `generate()` | `session_id` spanning many `generate()` calls | One `generate()` = the whole rollout |

Do not implement “one `denoise_step()` runs the whole AR block.” That would
compile, but it would not match Helios chunk-boundary decode, streaming
output, or later interaction timing.

The fifth transformer call in `_generate_block()` (clean-x0 `update_cache=True`
plus `state.commit_paged_context`) is **not** a denoise step. It belongs in
`post_decode()`.

Constants:

- `LINGBOT_DMD_TIMESTEPS = (1000, 750, 500, 250)` so `chunk_num_steps = 4`.
- `transformer.config.num_frames_per_block` is 3.
- Horizon: `(_MAX_RAW_FRAMES - 1) // vae_scale_factor_temporal + 1` latent
  frames, currently 31, so `max total_chunks = 10`.

`StepRequestState` already encodes this:

- `chunk_denoise_completed` when `step_in_chunk >= chunk_num_steps`
- `request_denoise_completed` when `chunk_index >= total_chunks`

## 4. Identity

| Field | Tick path today | Stepwise path |
| --- | --- | --- |
| `request_id` | New id per AR block | Stable for the whole rollout |
| `session_id` | Persistent world across generate calls | Same value as `request_id` for this landing |
| `chunk_index` | Contiguous per session epoch | Contiguous on the one request, starting at 0 |
| `event_id` | Per control/prompt update | Unused in the first landing (empty applied ids) |

Output metadata keeps the existing envelope so observers do not need a new
key:

```json
{
  "metadata": {
    "ar_diffusion": {
      "session_id": "<request_id>",
      "request_id": "<request_id>",
      "chunk_index": 0,
      "applied_event_ids": []
    }
  }
}
```

`ARDiffusionTickRequest` is **not** required on the stepwise path. Camera and
prompt for the first landing are request-scoped (sampling `extra_args` and the
standard prompt), not per-tick snapshots.

The tick control plane (`experimental.ar_diffusion.session` /
`consumer`) stays working for the old example. Do not route the new path
through `next_chunk()`.

## 5. Design

### 5.1 Keep bind-as-context, bind every step

`SupportsARDiffusionPipeline.bind_ar_diffusion_state` already says the
pipeline must not retain KV state after the context exits. Do not change that
to a request-long bind.

Scratch and committed pages live on `ARDiffusionKVState` in
`ARDiffusionModelRunner._sessions`, not on the pipeline. Unbinding only
clears `pipeline._ar_diffusion_kv_state`. The next stepwise call re-binds the
same session object, so uncommitted scratch survives across the four DMD
steps of a chunk.

```text
execute_stepwise(scheduler_output)
  session_id = request_id            # first landing
  state = get_or_create_session(session_id)
  with pipeline.bind_ar_diffusion_state(session_id, state):
      inherited _execute_stepwise(...)   # prepare_encode / denoise / post_decode
```

Load-time rejection of `step_execution=True` exists because the inherited
stepwise entrypoint would skip that bind. Replace the rejection with this
wrapper. Keep rejecting `max_num_seqs > 1` and request-batch execution.

On any exception during stepwise execution, release the session the same way
`execute_model()` does (`reason="forward_exception"`). After
`request_denoise_completed` (or abort), close the session: the request *is*
the world, so KV must not outlive it.

### 5.2 Pipeline: split `forward()`, share the math

`LingBotWorldCausalDMDPipeline` implements `SupportsStepExecution`.

`forward()` remains the offline / request-mode path. Extract the DMD probe
and KV commit from `_generate_block()` into helpers used by both `forward()`
and the stepwise methods. Do not fork the x0 / re-noise / commit math.

| Method | Responsibility |
| --- | --- |
| `prepare_encode(state)` | Parse the single request. Encode the prompt. Build the full-horizon image condition once and store it on `state.extra`. Initialize generator, `total_chunks`, `chunk_num_steps=4`, `chunk_index=0`. Call `_prepare_next_chunk(state)`. |
| `_prepare_next_chunk(state)` | Slice image condition for `chunk_index`. Integrate this block’s camera from the request-scoped trajectory or action script, carrying `camera_tail` / `camera_pitch` in `state.extra`. Sample the 3-frame noise latent. Install the four-step DMD schedule on `state.timesteps`. Reset `step_in_chunk = 0`. |
| `denoise_step(input_batch, states=...)` | Single request only. One transformer forward with `commit_current=False`, `update_cache=False`. Return flow prediction. |
| `step_scheduler(state, noise_pred)` | `x0 = latents - sigma * flow`. Intermediate steps re-noise at the next sigma; the last step keeps x0. Advance `step_in_chunk` and `step_index`. |
| `post_decode(state)` | Clean-x0 cache write, `commit_paged_context`. Persist generator / camera tail / prompt on `state.extra`. Yield latent `DiffusionOutput` plus `ar_diffusion` metadata. If the request is not finished, `_prepare_next_chunk`. |

`state.extra` holds LingBot-private tensors (image condition, folded camera,
cross-attention caches, generator state, camera tail). Do not promote them
onto `StepRequestState` unless they become cross-model.

Prompt encoding and the first-chunk image condition happen once in
`prepare_encode`. Later chunks only slice that condition. A prompt change
inside one request is out of scope for this landing.

### 5.3 Runner and scheduler

`ARDiffusionModelRunner.execute_stepwise` binds KV, then delegates to
`DiffusionModelRunner._execute_stepwise`. Do not reimplement the step loop.

Config for the new path:

- `step_execution=True`
- `streaming_output=True` (auto-enables step execution if unset)
- `max_num_seqs=1`
- engine remains `ARDiffusionEngine`
- `output_type="latent"`

Helios already requires `max_num_seqs=1` in step mode. LingBot matches that.

`pipeline.supports_step_execution = True` only on LingBot. Other
`SupportsARDiffusionPipeline` implementations keep the load-time reject.

### 5.4 Offline request-mode must not regress

Request-mode `forward()` (no tick, no stepwise) still generates a contiguous
latent sequence and optionally VAE-decodes. Existing one-block offline smoke
must keep passing.

Warmup (`ar_diffusion_warmup_requests`) stays on `execute_model()`. Do not
require stepwise for load-time graph capture.

## 6. Wiring

### 6.1 Files to change

| File | Change |
| --- | --- |
| `vllm_omni/diffusion/models/lingbot_world/pipeline.py` | `supports_step_execution = True`. Add the four step methods and `_prepare_next_chunk`. Split `_generate_block` helpers. Keep `forward()`. |
| `vllm_omni/experimental/ar_diffusion/runner.py` | Allow `step_execution` when the pipeline supports it. Implement `execute_stepwise` with per-call bind, fail-closed release, close on request completion. Session id = request id. |
| `vllm_omni/experimental/ar_diffusion/capability.py` | Document that bind still lasts one runner invocation, including one stepwise invocation. Do not require the pipeline to hold KV across steps. |
| `tests/diffusion/ar_diffusion/test_capability_runner.py` | LingBot-like pipeline with `supports_step_execution` may load with `step_execution=True`. Incapable AR pipelines still raise. Inherited entrypoint tests cover the new wrapper. |
| `tests/diffusion/models/lingbot_world/test_pipeline_lingbot_world.py` | Fake-transformer stepwise contract: four probes then one commit per chunk; N chunks; camera tail continuity; metadata; failure does not leave a bound state. |
| `tests/e2e/offline_inference/test_lingbot_world_v2.py` | Keep the existing request-mode one-block smoke. Add a separate stepwise-vs-tick GPU case only if it can share the same runner fixture without mixing engine modes. Prefer a dedicated test module if engine flags conflict. |
| `examples/offline_inference/diffusion/lingbot_world_v2_realtime.py` | Leave the tick loop. Add a sibling example that issues **one** `generate()` with `step_execution` / `streaming_output` and writes per-chunk latents. |
| `recipes/Robbyant/LingBot-World-2.0.md` | Document the stepwise offline entry. Do not claim HTTP serving. |
| `docs/design/feature/realtime_ar_diffusion.md` | Note that stepwise request identity is an additional mode, not a replacement of the tick contract yet. |

Do not change `vllm_omni/diffusion/worker/diffusion_model_runner.py` unless a
generic hook is strictly required. The AR runner should wrap the existing
stepwise implementation.

### 6.2 Call path

```text
AsyncOmni.generate(prompt, sampling_params)
  OmniDiffusionConfig.step_execution = True
  OmniDiffusionConfig.streaming_output = True
  ARDiffusionEngine / StepScheduler
    scheduled_new_reqs -> ARDiffusionModelRunner.execute_stepwise
      bind_ar_diffusion_state(request_id, kv_state)
      pipeline.prepare_encode(state)                 # first invocation only
      pipeline.denoise_step(...)
      pipeline.step_scheduler(...)
      if chunk_denoise_completed:
          pipeline.post_decode(state) -> DiffusionOutput (latent chunk)
      unbind
    ... until request_denoise_completed
  caller iterates streaming OmniRequestOutput
```

Request-scoped camera script (first landing), for example in
`sampling_params.extra_args`:

```json
{
  "camera_action_script": [
    [["w"], ["w"], ["w"]],
    [["a"], [], []],
    [[], [], []]
  ]
}
```

That is one list per chunk, three latent-frame action lists per chunk. The
pipeline integrates them in `_prepare_next_chunk` using the existing
`integrate_lingbot_camera_actions` helper. A full pose/intrinsics trajectory
on the request remains valid for offline-style replay.

### 6.3 Implementation order

1. Extract DMD probe + KV commit helpers from `_generate_block()`. Prove
   request-mode `forward()` still matches the fake-transformer traces in
   `test_fixed_dmd_transition_and_cache_commit_trace`.
2. Implement stepwise methods on the pipeline. CPU tests with the same fake
   transformer: N chunks × 4 probes + 1 commit.
3. AR runner: lift the `step_execution` load reject for capable pipelines;
   wrap `execute_stepwise`. CPU tests for bind lifetime and fail-closed
   release. DreamZero-shaped pipelines still reject stepwise at load.
4. Example + recipe for one `generate()` over N latent chunks.
5. GPU oracle: stepwise N chunks vs N tick `generate()` calls.

## 7. Acceptance

### 7.1 CPU contract (required to merge)

From `tests/`:

```bash
pytest -q \
  diffusion/models/lingbot_world \
  diffusion/ar_diffusion \
  -m "core_model and cpu"
```

Must cover:

1. Stepwise progress: `chunk_num_steps == 4`; `post_decode` only when
   `chunk_denoise_completed`; `chunk_index` increases by one per yield;
   request finishes at `total_chunks`.
2. Shared math: fake transformer call counts equal the tick path
   (4 non-commit forwards + 1 commit per chunk).
3. Metadata: each yielded `DiffusionOutput` has `ar_diffusion.chunk_index`
   in `0 .. N-1` and `session_id == request_id`.
4. Camera script: pose tail from chunk `i` is the initial pose of chunk
   `i+1` (same as today’s typed-action tick test).
5. Bind lifetime: pipeline KV pointer is set during a stepwise call and
   cleared after; runner session object remains until request completion or
   failure.
6. Fail-closed: a transformer exception in the middle of a chunk releases
   runner KV and pipeline session state; a later stepwise call for that
   `request_id` does not resume the same pages.
7. Isolation: AR pipelines without `supports_step_execution` still fail at
   load if `step_execution=True`.
8. Offline `forward()` traces in the existing LingBot pipeline tests remain
   green.

### 7.2 GPU E2E (required to claim the path works)

Hardware: one NVIDIA GPU with enough free memory for the 14B causal-fast
checkpoint at 464×832 or 480×832, `gpu_memory_fraction` around 0.6, TP=1.
H100 80GB class is the intended smoke tier; larger H-series cards are also
valid.

Checkpoint: `robbyant/lingbot-world-v2-14b-causal-fast-diffusers`.

Oracle: existing in-process tick example,
`examples/offline_inference/diffusion/lingbot_world_v2_realtime.py`, run for
**N = 3** chunks (enough for cold start plus two continuations). Optionally
repeat N = 7 later to cross the sink + recent window; that is not the first
gate.

Candidate: one `AsyncOmni.generate()` with `step_execution=True`,
`streaming_output=True`, same image, prompt, seed 42, and the same camera
script.

Pass when:

| Check | Criterion |
| --- | --- |
| Chunk count | Candidate yields N outputs |
| Shape | Each latent is finite `[1, 16, 3, H_lat, W_lat]` |
| Metadata | `chunk_index == 0..N-1`, `session_id == request_id` |
| Numeric | Per-chunk latents match the tick oracle (`torch.equal`, or `allclose` only if a documented BF16 kernel difference exists; do not loosen a real RNG-order bug) |
| Concat | `cat` of N chunks has the same shape as concatenating the tick payloads |
| Request-mode regression | `tests/e2e/offline_inference/test_lingbot_world_v2.py` still passes (9-frame decode smoke) |
| Failure | Injected stepwise error releases KV; the process does not stream a later chunk for that request |

Existing offline e2e (assets via env):

```bash
cd tests
pytest -s -v e2e/offline_inference/test_lingbot_world_v2.py \
  -m "slow and diffusion"
```

Requires `VLLM_OMNI_LINGBOT_WORLD_V2_IMAGE_PATH` and
`VLLM_OMNI_LINGBOT_WORLD_V2_ACTION_DIR`.

Do not treat the following as acceptance for this landing:

- `/v1/realtime/video` or any WebSocket session
- Pixel-space PSNR / SSIM against Helios streaming tests
- Live mid-request camera updates
- TP=2, CUDA graph quality, or FPS targets
- Official long-horizon video quality beyond 10 chunks

### 7.3 Done when

CPU tests in §7.1 pass, the GPU oracle in §7.2 passes for N = 3, and the
offline 9-frame request-mode smoke still passes. Serving, live interaction,
and VAE streaming are follow-ups that consume this path; they are not part
of this contract.
