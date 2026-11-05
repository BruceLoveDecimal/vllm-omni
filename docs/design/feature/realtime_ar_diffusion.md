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

## Experimental Deep Forcing for LingBot World

LingBot accepts an opt-in `model_config.deep_forcing` mapping for the two
mechanisms of [Deep Forcing](https://arxiv.org/html/2512.05081v1): Deep Sink
temporal RoPE rebasing and Participative Compression (PC). Omitting the mapping
preserves the native LingBot sink and paged/direct cache paths. All units are
latent frames.

LingBot already retains a deep attention sink natively: 9 of its 18 window
frames, inside the 40–60% range the paper recommends. Deep Forcing therefore
does not add a second sink-retention control. The mapping inherits the model's
`sink_size` and sliding window, rejects `sink_frames` and `window_frames`, and
only enables what the paper adds on top of sink retention:

| Mechanism | Key | Effect |
| --- | --- | --- |
| Deep Sink rebasing | `rebase_sink: true` | Rotate sink keys' temporal phase to sit right before attended history |
| Participative Compression | `pc_budget_frames: N` | When the window would evict, keep sink, recent and the top-scoring history within `N` frames |
| PC rebasing | `rebase_pc: true` | Pack the PC-selected history into consecutive frames before the recent region |

A mapping that enables neither `rebase_sink` nor `pc_budget_frames` is rejected:
native sink attention needs no mapping. The paper's Self-Forcing setting keeps
a 10-frame sink, a 16-frame budget and 4 recent frames in a 21-frame window.
Scaled to LingBot's 9-frame sink and 18-frame window:

```yaml
model_config:
  deep_forcing:
    rebase_sink: true
    pc_budget_frames: 15  # sink 9 + recent 4 + 2 frames of selected history
    pc_recent_frames: 4   # default; includes the 3-frame current chunk
    rebase_pc: true
```

Configuration is validated at model load against the 3-frame chunk:
`sink + pc_recent_frames <= pc_budget_frames < window`, the recent region must
cover the current chunk, and the window must fit the sink and one chunk.

### Deep Sink

Keys keep their original temporal RoPE phase in storage; each attention view
rotates only the temporal channels of the sink keys. The shifted sink ends
right before the earliest attended non-sink frame: the recent tail for the
sliding window, the packed PC history with `rebase_pc`, or the oldest retained
PC token without it. The shifted sink therefore never shares a temporal phase
with attended history. Before the window first fills there is no gap and no
shift. The paper's Eq. 5 overlaps the sink's last frame with the tail's first
frame by one frame, and the released code keeps the gap present at the first
eviction (one chunk); these placements differ by a constant of one to three
frames.

### Participative Compression

PC runs exactly where the native window would evict a chunk: history plus the
current chunk no longer fit in the window. The full window is used before that.
Candidates are history frames outside the sink and the protected recent frames
(`pc_recent_frames` including the current chunk). They are ranked by the sum
of raw QK logits over the recent queries and all heads, independently per
layer. The recent queries are the clean queries of the latest committed frames
plus the current chunk's queries at its first denoising step, as in the
released code. The selection is made once per chunk, reused for the remaining
denoising steps and the clean commit, and kept only on successful publication.
With LingBot's geometry and the example above, attention spans 18 frames when
the window is full and 15 right after compression.

`rebase_pc` packs the selected tokens into consecutive virtual frames ending at
the recent region. The released code instead shifts the whole selected block
by one uniform delta per chunk; the paper does not pin down the target
positions.

Only single-rank, single-sequence eager attention with clean-x0 KV commits is
supported. First-block sharing and last-denoise-step KV reuse are rejected. The
session reservation includes history, pending commits, query history, frozen
selection indices and layer-local rotation and scoring scratch; a minimum
legacy pool remains allocated for runtime geometry. CPU tests check rotation
against complex arithmetic and LingBot's RoPE layout, dense attention oracles
for every flag combination, phase separation of the shifted sink, PC timing,
selection reuse and transactional rollback, and real LingBot attention and
transformer integration. They are not evidence of video quality or GPU latency.
This branch does not include WorldKV retrieval or the CausVid integration.
