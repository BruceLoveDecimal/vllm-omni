# Realtime AR-Diffusion sessions

> **Status:** experimental. This document describes the internal execution
> contract implemented under `vllm_omni.experimental.ar_diffusion`. It does not
> define a public HTTP or WebSocket schema.

## Scope

Realtime AR-Diffusion serves world models that generate one autoregressive
latent block per request while retaining model state across requests. The
generic runtime owns identity, event ordering, session lifecycle, routing, and
KV allocation. A model adapter owns model-specific controls and conditioning.

LingBot World v2 is the first single-KV-branch integration. DreamZero uses the
same runner capability with two logical CFG branches.

## Identities

Four identifiers have distinct lifetimes:

| Identifier | Lifetime | Owner | Contract |
| --- | --- | --- | --- |
| `session_id` | One persistent world | Session manager | Selects runner KV and model-owned state. |
| `event_id` | One control/prompt update | Transport adapter | Unique and monotonically increasing within a session, including across reset. |
| `chunk_index` | One committed AR block position | Session | Contiguous from zero; reset starts again at zero. |
| `request_id` | One chunk execution | Session | Correlates the tick with its returned metadata. |

`AsyncOmni` keeps its normal UUID-suffixed engine routing ID. The tick
`request_id` remains inside the immutable AR-Diffusion snapshot and the
standard output metadata, while caller-visible `OmniRequestOutput.request_id`
continues to use AsyncOmni's existing external-ID contract. Protocol identity
therefore does not require a private entrypoint override.

## Request lifecycle

1. `ARDiffusionSession.accept_event()` validates and queues an immutable
   `ARDiffusionSessionEvent`. Prompt and opaque control tracks may be updated in
   the same event.
2. `ARDiffusionSession.next_chunk()` serializes chunk execution and calls
   `_build_tick_locked()`. The resulting `ARDiffusionTickRequest` is a snapshot
   of the current prompt, reduced controls, ordered event IDs, and chunk index.
3. `ARDiffusionOmniTickConsumer.execute_tick()` copies the stage sampling
   parameters, stores the typed tick in
   `sampling_params.extra_args["ar_diffusion_tick"]`, constructs the normal Omni
   prompt through its `prompt_provider`, and invokes `AsyncOmni.generate()`.
4. `AsyncOmni` allocates its normal unique engine routing ID and submits one
   `OmniDiffusionRequest` through the orchestrator.
5. `ARDiffusionModelRunner._request_session()` parses the typed tick.
   `_get_or_create_session()` obtains runner-owned paged KV, and
   `bind_ar_diffusion_state()` exposes it to the pipeline only for the duration
   of `execute_model()`.
6. The model pipeline validates the tick and generates exactly one latent
   block. LingBot interprets camera controls in
   `LingBotCameraControlReducer` and
   `LingBotWorldCausalDMDPipeline.forward()`; the generic runtime does not parse
   camera or action schemas.
7. The pipeline returns one `DiffusionOutput` with the generated payload and an
   `ARDiffusionChunkMetadata` mapping under `metadata["ar_diffusion"]`.
   `AsyncOmni` exposes it as
   `OmniRequestOutput.multimodal_output["metadata"]["ar_diffusion"]`.
8. The consumer parses the standard envelope. The session commits its reducer
   state, prompt, controls, event IDs, and next chunk index only when returned
   metadata exactly equals the submitted tick snapshot.

The metadata envelope is:

```json
{
  "metadata": {
    "ar_diffusion": {
      "session_id": "world-7",
      "request_id": "ar-world-7-3-...",
      "chunk_index": 3,
      "applied_event_ids": [10, 11]
    }
  }
}
```

## Transaction and failure semantics

Accepted events are not acknowledged as applied until the model output and
metadata are validated. Reducer `prepare()` is speculative; reducer `commit()`
and the session snapshot form one logical commit. A model, metadata, or reducer
failure leaves the chunk index and queued events uncommitted and transitions the
session to `FAILED`.

After a tick failure, worker state is closed because a forward may have
partially modified or already released KV. The same chunk is not retried in
place. A caller must explicitly reset the session, starting again at chunk
zero, or close it.

Explicit close follows this state machine:

```text
ACTIVE or FAILED
       |
       v
    CLOSING ---- worker cleanup succeeds ----> CLOSED
       |
       +------- worker cleanup fails --------> CLEANUP_FAILED
                                                   |
                                                   +-- close retry --> CLOSING
```

`CLEANUP_FAILED` is a local tombstone. The session manager retains it, rejects
creation of the same `session_id`, and rejects reset. Only an idempotent close
retry that succeeds removes the session. Transport disconnect uses the same
path.

## KV ownership and capacity

`ARDiffusionKVCacheSpec` is the pipeline-owned immutable geometry contract. It
declares:

- TP-local self-attention geometry, AR block size, sink, and recent window;
- logical KV branches and their worker-local slots;
- fixed-length cross-attention caches;
- scratch tokens required by an uncommitted forward;
- persistent model-owned CUDA bytes per resident session; and
- the requested resident-session capacity.

The runner computes:

```text
required(capacity) =
    scratch
  + managed_self_attention_pages(capacity)
  + capacity * cross_attention_bytes_per_session
  + capacity * model_owned_state_bytes_per_session
```

`gpu_memory_fraction` is a soft expansion budget:

```text
configured_budget = available_device_bytes * gpu_memory_fraction
effective_budget = max(configured_budget, required(1))
```

If `required(1)` exceeds actual available device memory, initialization fails.
Otherwise the runner always admits one viable session and uses the configured
fraction to select additional resident sessions up to the model-declared cap.
All model-owned reservations are deducted before the paged self-attention pool
is allocated.

LingBot advances a session-owned causal Wan VAE encoder for each condition
block: the opening pixel frame is followed by four zero pixel frames per later
latent frame. It retains one current condition block, with separate committed
and in-flight encoder histories so a failed block cannot overwrite committed
context. The state reservation includes both encoder histories, derived from
the encoder's convolution input grids, and the streaming decoder cache. These
caches are released through the same close/reset hooks. Condition encoding
currently requires the unpatched, non-tiled Wan encoder path; the existing
decoder's tiling fallback remains independent. Real-weight GPU parity with a
full-horizon encode still requires validation because the pointwise posterior
projection now runs per block.

DreamZero reports the measured 603 MiB per-session Wan VAE causal-convolution
state.

## Concurrency and routing limits

The current runner executes one request at a time:

- `max_num_seqs` must be one;
- request batching and diffusion step execution are rejected;
- one AR block is generated per engine request; and
- an AR-Diffusion stage must have exactly one replica.

The single-replica restriction is intentional. Multi-replica support requires
session-affine routing so every tick, reset, and close reaches the worker that
owns the session. Random or round-robin routing is not correct.

Multiple resident sessions may share one runner. Capacity selection accounts
for all persistent pools, and the runner performs LRU eviction through the same
model lifecycle hook used by explicit close.

## Model boundary

The generic runtime transports controls as opaque
`ARDiffusionControlInput(track, schema, data)` values. Model adapters validate
and reduce those values. This keeps target/velocity, key state, pose,
intrinsics, and other model-specific conventions out of the session protocol.

For LingBot World v2:

- `lingbot_world/actions.py` parses key-state and trajectory controls;
- `lingbot_world/camera.py` loads or constructs camera trajectories and Plücker embeddings;
- `lingbot_world/transformer.py` implements checkpoint-compatible causal attention; and
- `lingbot_world/pipeline.py` constructs conditioning, owns small non-KV session state, and
  produces one block plus the standard metadata envelope.

## Ulysses sequence parallelism

LingBot supports pure Ulysses sequence parallelism for both direct and
AR-Diffusion execution. With Ulysses degree greater than one, hidden tokens,
camera features, token-expanded timestep modulation, and RoPE tables are sharded
together. Without SP, timestep modulation retains the frame-broadcast path.
Self-attention performs the sequence-to-head all-to-all before reading or writing
paged KV, while static text K/V uses the same local head shard. Text K/V shards
own compact storage; cross-attention exchanges query/output layouts to use these
shards. This retains the shared cache geometry and reduces text K/V storage at
the cost of two all-to-all calls per layer.

The output head projects local tokens before gathering the flow values. For
the 14B model this reduces the gathered width from 5120 to 64; frame modulation
uses each shard's global token offset, including shards that split a frame. Only
`ulysses_mode="strict"` is supported; `advanced_uaa`, Ring, and AllGather-KV modes
remain unsupported for this model.

## Non-goals

This contract does not currently provide:

- a public HTTP or WebSocket request schema (the structured-interaction
  frontend is tracked separately in
  [#5527](https://github.com/vllm-project/vllm-omni/pull/5527));
- camera/action semantics shared by every world model;
- session migration or replication across workers;
- retry of an ambiguous partially executed chunk;
- cross-stage KV transfer; or
- stateful streaming VAE decode.

Those features can be layered on top without adding LingBot-specific identity
or lifecycle fields to the generic runtime.

## Required regression coverage

CPU contract tests cover:

- separation of tick correlation IDs from internal engine routing IDs;
- event ordering, metadata equality, and reducer fail-closed commit;
- close/disconnect failure tombstones and cleanup retry;
- capacity one under the default fraction for shipped LingBot geometry;
- capacity expansion under a tuned fraction;
- DreamZero model-owned state accounting;
- runner session reuse, reset, close, LRU eviction, and forward cleanup; and
- LingBot model registration, imports, camera/action reduction, and one-block
  output metadata.

GPU validation must additionally exercise real model loading, action input,
prompt switching, rolling-window replacement, reset, close, finite output
latents, and exact metadata/request identity.

## Experimental video memory policies

LingBot accepts an opt-in `model_config.video_memory` mapping. The initial
Deep Sink policy uses `window_frames: 18`, `sink_frames: 9`, and
`rebase_sink: true`. Units are latent frames; the window includes the current
chunk. Omitting the mapping preserves the existing paged/direct paths.

The eager policy keeps canonical clean keys at their original temporal RoPE
positions and rotates only the temporal channels of attention views. Values
and spatial channels are unchanged. Every noisy forward is read-only; all
layers and the output projection must succeed before clean history is
published. The runner owns session policy storage and releases it on close,
reset, eviction, and failure. Request-mode policy storage is request-local.
Its conservative device-memory reservation includes canonical rows, pending
rows, replacement rows, attention views, and position metadata. The legacy
pool uses minimum geometry and receives no duplicate self-KV writes.

This initial path requires single-rank eager attention and clean commits;
first-block sharing and last-noisy-step KV reuse are rejected. CPU tests check
rotation against complex arithmetic, repeated denoising, full-transformer
legacy parity with rebasing disabled, and transactional publication. These are
numerical and lifecycle checks, not evidence of checkpoint quality or GPU
latency improvements.

### World retrieval

Set `retrieval_frames` and a positive `bank_bytes` in `video_memory` to reserve
an attention-token budget for retrieved chunks and a separate per-session CPU
bank byte limit. For example, `window_frames: 18`, `sink_frames: 3`,
`retrieval_frames: 9`, `bank_bytes: 8589934592`, `rebase_sink: false` gives a
WorldKV-style split with three recent and three current frames. This example
is a configuration, not a validated quality preset.

Each successfully committed non-sink chunk is copied once to the CPU bank.
Entries include canonical K/V, original token positions, and an absolute C2W
pose. Oldest bank entries are evicted to meet the byte limit; an oversized
entry is skipped without discarding existing memory. Rank by normalized
squared translation distance plus rotation geodesic distance, breaking ties
by insertion order. Exclude sink and recent chunks, then present retrieved
chunks in chronological order. All layers use the same chunk IDs. Chunk
midpoint poses preserve a valid rotation matrix; do not average rotation
matrices or use normalized framewise control deltas for retrieval.

Request, tick, and stepwise paths pass latent-aligned absolute poses through
the DMD runner. `rebase_retrieval` defaults to false; enabling it changes only
the temporal phase of the retrieved attention view. Noisy probes cannot write
to the bank. Bank transfers currently run synchronously on the eager worker;
GPU latency and asynchronous prefetch remain separate validation/optimization
work. CPU tests cover analytical pose distances, exact attention assembly,
byte eviction, missing/invalid poses, and failed publication.

### World and participative compression

`world_keep_ratio` defaults to `1.0` (no bank compression). `0.25` preserves
all anchor-frame tokens and one quarter of pooled non-anchor tokens. A
three-frame chunk therefore occupies 1.5 frame-equivalents, subject to integer
rounding. Keys, values and original positions use the same per-layer indices.
`world_score: pairwise_mean` implements the paper's mean pairwise cosine score
without constructing a quadratic matrix; `centroid` selects the released
implementation's raw-anchor-centroid score. These are distinct algorithms.
Compression occurs once at bank insertion, before CPU transfer. Retrieval
budgets count actual compressed tokens, allowing more historical chunks in the
same attention budget. Bank byte accounting includes position/pose metadata.

Set `pc_budget_frames` below `window_frames` to enable Participative Compression;
`pc_recent_frames` protects the most recent frames including the current chunk.
For example, a standalone PC experiment can use window/sink/budget/recent
values `21/10/16/4`. The sink and recent regions must fit inside the budget.
At window saturation, the first forward of a chunk scores intermediate keys
using recent clean queries plus the current queries, aggregated as sum-QK
logits. Only the selected token indices remain resident across that chunk's
remaining noisy forwards and clean commit; selected K/V views are temporary.
Canonical history remains intact for rollback until successful publication. Clean query history is bounded by
the recent budget. Publication remains transactional. This query-history
convention is explicit; it is not a claim of bitwise parity with the upstream
research script's mutable denoising-query buffer.

With `rebase_sink: true`, PC also rebases selected intermediate keys into a
compact virtual timeline before the recent region. Canonical positions remain
unchanged. CPU tests compare scoring with full pairwise cosine/QK matrices,
check low-precision selection, stable ties, exact retained storage size, and
repeated compression over many windows. Real-weight quality remains unmeasured.

### Combined policy and CausVid integration

For a combined LingBot experiment, add the following mapping under a stage's
`model_config` in a copy of the single-GPU eager deployment:

```yaml
video_memory:
  window_frames: 18
  sink_frames: 3
  retrieval_frames: 3
  bank_bytes: 1073741824
  world_keep_ratio: 0.25
  world_score: pairwise_mean
  pc_budget_frames: 12
  pc_recent_frames: 3
  rebase_sink: true
  rebase_retrieval: true
```

The PC budget includes the reserved retrieval budget. Compression is triggered
when resident history plus the current chunk fills `window_frames -
retrieval_frames`; sink, recent (including current), and retrieval reservations
must fit inside `pc_budget_frames`. Stable absolute token IDs survive both
compression policies. Each layer removes only bank tokens already present in
its canonical PC history before chunk ranking and budgeting. All layers use
the same retrieved chunk IDs, but may retrieve different token subsets.
Budgeting uses the largest remaining subset across layers. Tokens discarded
by this step's PC selection become eligible on the next chunk.
When enabled, temporal correction packs retrieved tokens into consecutive
virtual frames before PC intermediate/recent tokens, and shifts the sink
before that region. Original bank/history tensors never receive these phases.

CausVid accepts the same mapping for Deep Sink and PC, without pose retrieval.
Use `causvid_wan_self_forcing.yaml` as the starting point, set `devices: "0"`,
`pipeline_parallel_size: 1`, keep `enforce_eager: true`, `step_replicas: 1`,
`max_batch_size: 1`, and remove `kv_history_chunks: 6` from the default sampling
arguments (or set it to `0`). A configuration for three-frame chunks is:

```yaml
video_memory:
  window_frames: 18
  sink_frames: 3
  pc_budget_frames: 12
  pc_recent_frames: 3
  rebase_sink: true
```

`WanCell` keeps a request-local transaction open across the real transformer
layers. Only the clean step publishes history after the output head succeeds.
The Self-Forcing timetable still runs the clean step, but declares zero legacy
NoisyKV history readers. The small legacy version pool remains executor-owned;
the policy has no duplicate writes into it. Policy storage is reserved before
sizing that pool, and is released on completion or failure. CausVid reserves
against the largest tokens-per-frame geometry allowed by its fixed chunk-token
count. Latest-KV/step replicas, layer/tensor/sequence parallelism, CUDA graphs,
and CausVid pose retrieval are rejected for this initial implementation.

These configurations are experimental starting points. CPU validation covers
component math and integration, including a 100-frame combined rollout, dense
attention oracles, real cell scheduling/attention hooks, failure rollback,
session/branch reset, and memory-budget boundaries. It does not establish
real-weight video quality, GPU memory peaks, throughput, or realtime deadlines.
Synchronous bank transfers and eager tensor assembly may add latency. Validate
those separately before selecting a production preset.

Implementation references: [WorldKV](https://arxiv.org/html/2605.22718v1) and
[Deep Forcing](https://arxiv.org/html/2512.05081v1). This adapts their memory
policies to the repository's inference lifecycle; it does not reproduce their
training procedures or imply that existing checkpoints have been retrained.

CPU reproduction (from the repository root; PyTorch CPU, pytest,
pytest-asyncio, diffusers and their dependencies; no model download or GPU):

```bash
python -m pytest --confcutdir=tests/diffusion -o addopts='' \
  tests/diffusion/ar_diffusion/test_video_*.py \
  tests/diffusion/models/lingbot_world/test_lingbot_world_attention.py -q
```

CI-equivalent selection (requires the normal repository test environment,
including vLLM and pytest-xdist for the root fixtures):

```bash
python -m pytest tests/diffusion/ar_diffusion/test_video_*.py \
  tests/diffusion/models/lingbot_world/test_lingbot_world_attention.py \
  -m 'core_model and cpu' --run-level core_model -q
```

The new `ar_diffusion/test_video_*.py` modules are included by the existing
Buildkite ready CPU diffusion lane. Import-light tests replace external runtime
and codec dependencies with fixture-scoped stubs; they execute the actual
policy, LingBot transformer, CausVid attention hooks/cell, NoisyKV allocation,
and session owner code. Full checkpoint/codec/serving integration is separate.
