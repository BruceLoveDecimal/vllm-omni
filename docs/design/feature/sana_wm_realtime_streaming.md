# SANA-WM Stage-1 realtime streaming (distilled, output-only)

> **Status:** design spec, not yet implemented. Target branch `feat/sana_wm_realtime`.
> Scope is the union of what earlier discussion called "PR-1" (chunk-causal
> Stage-1 model + self-forcing sampler) and "PR-3" (serving), delivered as one
> PR in the shape of LingBot-World #6844: stepwise execution + the existing
> `WS /v1/realtime/video` transport. No streaming VAE decoder, no interactive
> input, no refiner.

## 1. Goals and non-goals

### Goals

1. Serve the distilled chunk-causal SANA-WM Stage-1 model so that a single
   `session.start` on `WS /v1/realtime/video` streams fragmented MP4 chunks as
   each 3-latent-frame block is generated, instead of waiting for the whole
   clip.
2. Reuse the step-execution framework (`SupportsStepExecution`,
   `DiffusionEngine.step_streaming()`, `OmniStreamingVideoOutputHandler`)
   without touching the runner, the engine, or the WebSocket handler.
3. Keep the existing bidirectional `SanaWmPipeline` byte-identical in request
   mode; the streaming path is a sibling pipeline that shares model code.

### Non-goals (explicit follow-ups, to be listed in the PR description)

- Stateful streaming VAE decode (LTX-2 causal conv cache). Chunks are decoded
  independently with a one-latent-frame overlap; see §8.
- Mid-request camera interaction. The camera trajectory is fixed at
  `session.start`. `session.interaction` (prompt update) is rejected in v1.
- LTX-2 refiner stage.
- `experimental/ar_diffusion` tick protocol, cross-request sessions, paged KV,
  session capacity planning. One request is one session; all causal state
  lives in `StepRequestState.extra`.
- Chunk-wise Triton GDN kernel. The current GDN is a PyTorch recurrence
  (`sana_wm_transformer.py:130`, no Triton in the module); the causal variant
  stays PyTorch. Kernel work is a perf follow-up.
- Non-distilled chunk-causal teacher (`chunk_flow_euler`, CFG>1). Only the
  distilled `self_forcing` student with `cfg_scale=1.0` is served.
- Tensor parallelism and CFG parallelism for the streaming pipeline
  (`cfg_parallel_size>1` is rejected; TP>1 is not validated in v1).

## 2. Upstream facts this spec relies on

Verified by reading NVlabs/Sana (`inference_video_scripts/wm/*`,
`diffusion/model/nets/sana_gdn_camctrl_blocks.py`,
`diffusion/model/nets/sana_multi_scale_video_camctrl.py`) and the sglang
SANA-WM PR (#27531):

| Fact | Source |
| --- | --- |
| Streaming model is a distinct class `SanaMSVideoCamCtrlStreaming_1600M_P1_D20` (depth 20, hidden 2240, patch (1,1,1)); it is not selected by `attn_type`. | `sana_multi_scale_video_camctrl.py` |
| `forward_long(x, timestep, y, start_f, end_f, frame_index, kv_cache, save_kv_cache, chunk_plucker, ...)`; `kv_cache` is a list with one entry per block; `frame_index` supports discontinuous layouts (sink + sliding window). | same |
| `pos_embed_type="casual_wan_rope"`: RoPE is built for the absolute latent-frame range `(start_f, end_f)` or an explicit `frame_index`. | same |
| GDN blocks: chunk-causal = bidirectional scan **inside** a chunk, forward recurrence state carried across chunks, backward recurrence isolated at the chunk boundary. Cached variant `CachedChunkCausalGDNUCPESinglePathLiteLA` reads/writes the main GDN state, the camera-branch GDN state (`_SLOT_CAM`) and the camera short-conv state (`_SLOT_CAM_AUX`). | `sana_gdn_camctrl_blocks.py` |
| Softmax blocks: `CachedSoftmaxUCPESinglePathLiteLA` concatenates cached post-RoPE main K/V and cached post-UCPE camera K/V in front of the current chunk and runs full (non-chunk-masked) SDPA. No in-class eviction; the caller trims. | same |
| Temporal short conv (`conv_k`, `conv_{q,k,v}_cam`) is chunk-causal when chunk boundaries exist. FFN uses `ChunkGLUMBConvTemp` (chunk-causal temporal conv). | same |
| Sampler: `sampling_algo=self_forcing`, `denoising_step_list="1000,960,889,727,0"` (must end with 0), `cfg_scale=1.0`, `num_frame_per_block=3`, `num_cached_blocks=2` (`-1` keeps all), `sink_token` on by default (`--no_sink_token`). `save_kv_cache` only on the final clean forward. Between steps the predicted x0 is re-noised to the next timestep. | `inference_sana_wm_streaming.py`, `inference_sana_wm.py` |
| Chunk schedule comes from `SelfForcingFlowEulerCamCtrl.create_autoregressive_segments(latent_T)`; the 3-stage orchestrator treats chunk 0 as covering latent frames `[0, 4)` (conditioning frame + 3 generated) and later chunks as 3 frames. `_snap_num_frames` enforces `F = 8 * 3 * k + 1`. | `streaming_pipeline.py`, `inference_sana_wm_streaming.py` |
| Checkpoint: `hf://Efficient-Large-Model/SANA-WM_streaming/sana_dit/model.pt`, config `sana_wm_streaming_1600m_720p.yaml` under `configs/sana_streaming/`. | `inference_sana_wm_streaming.py` |

**Not yet verified, must be read before implementing §5** (listed in §14 as
implementation prerequisites): exact `kv_cache` slot layout and tensor shapes
of `CachedChunkCausalGDN` / `CachedChunkCausalSoftmaxAttn` base classes, how
`num_cached_blocks` + `sink_token` are turned into `frame_index`, the exact
x0/re-noise formula in `SelfForcingFlowEulerCamCtrl`, and whether the final
cache-write forward is a separate call at `t=0` or folded into the last step.
The spec below is written so that each of these is an isolated, testable
decision.

## 3. Reuse map

Nothing in this table is modified.

| Layer | Reused as-is | Where |
| --- | --- | --- |
| Transport | `WS /v1/realtime/video`, `OmniStreamingVideoOutputHandler` (session start/stop/ping, stall clock, fMP4 muxing, chunk metadata, `image_reference` → `multi_modal_data.image`, `extra_params` → `sampling_params.extra_args`) | `entrypoints/openai/api_server.py:1692`, `serving_video_output_stream.py` |
| Engine | `--diffusion-streaming-output` → `streaming_output=True` → runner forces `step_execution=True`; `DiffusionEngine.step_streaming()`; `_stream_metadata_from_diffusion_output` (`generation_chunk_index`) | `cli/serve.py:869`, `async_omni_engine.py:1179`, `diffusion_model_runner.py:324`, `diffusion_engine.py:427`, `output_formatter.py:194` |
| Runner | Stepwise loop: `prepare_encode` once, then `denoise_step` → `step_scheduler` → `post_decode` when `chunk_denoise_completed`; `finished` from `request_denoise_completed`; per-request error isolation | `diffusion_model_runner.py:1000`, `:1168-1230`, `worker/utils.py:54` |
| Pipeline contract | `SupportsStepExecution`, `StepRequestState` (`chunk_index`, `step_in_chunk`, `total_chunks`, `chunk_num_steps`, `extra`) | `models/interface.py:49`, `worker/utils.py:54-160` |
| Reference impl | Helios: request-scoped history in `extra`, `_prepare_next_chunk`, per-chunk independent `vae.decode`, `DiffusionOutput(chunk_index=..., total_chunks=..., finished=...)` | `models/helios/pipeline_helios.py:286-925` |
| SANA-WM | checkpoint resolve/prefetch, text encoder load + fallback, `_native_prompt_embeds`, first-frame preprocess/encode, LTX-2 latent (de)normalisation, `FlowMatchEulerDiscreteScheduler`, camera DSL → c2w → Plücker/raymap, request validation, output envelope, pre-process hook | `models/sana_wm/pipeline_sana_wm.py`, `camera_control.py`, `request.py`, `ltx2/ltx2_latents.py` |
| Transformer | patch embed, text/timestep embedders, TP RMSNorm, cross-attention, AdaLN modulation (per-frame path), final layer, weight loader, UCPE geometry prep | `sana_wm_transformer.py`, `ucpe.py` |
| Tests | mock-pipeline streaming harness; WS test client | `tests/diffusion/test_diffusion_streaming_output.py`, `tests/helpers/client.py:1818` |

## 4. End-to-end flow

```text
client ──session.start──▶ OmniStreamingVideoOutputHandler
   (prompt, image_reference, width/height/num_frames/fps, extra_params.sana_wm)
        │  _build_prompt_and_sampling_params  (unchanged)
        ▼
AsyncOmni.generate ──▶ DiffusionEngine.step_streaming
        │  get_sana_wm_pre_process_func  (unchanged; normalises payload)
        ▼
DiffusionModelRunner (step mode, streaming_output=True)
   prepare_encode(state)                       once
   ┌─ for chunk k in 0..total_chunks-1
   │    for step s in 0..3:  denoise_step ─▶ step_scheduler
   │    post_decode(state)  ─▶ DiffusionOutput(chunk_index=k, finished=k==last)
   └─ (post_decode advances chunk_index and prepares chunk k+1)
        ▼
output_formatter ─▶ multimodal_output["video"], metadata.stream.generation_chunk_index
        ▼
handler: encoder.encode(video) ─▶ video.chunk_metadata + binary m4s frame
```

Per chunk the model runs 4 denoising forwards with the cache read-only plus one
clean forward at `t=0` with `save_kv_cache=True` (5 forwards, §6).

## 5. Model layer: `sana_wm_transformer.py`

### 5.1 Config (`config.py`)

Add fields, all read from the converted `transformer/config.json`:

```python
attn_type: str = "BidirectionalGDNTriton"        # existing
streaming: bool = False                          # NEW: chunk-causal cached forward available
pos_embed_type: str = "wan_rope"                 # existing; streaming ckpt sets "casual_wan_rope"
ffn_type: str = "GLUMBConvTemp"                  # existing; streaming ckpt sets "ChunkGLUMBConvTemp"
num_frame_per_block: int = 3                     # NEW
num_cached_blocks: int = 2                       # NEW (-1 = unbounded)
sink_token: bool = True                          # NEW
denoising_step_list: tuple[int, ...] = (1000, 960, 889, 727, 0)   # NEW
```

`streaming=True` is what `SanaWmStreamingPipeline` requires at load time; the
bidirectional pipeline ignores it. Field names mirror the upstream YAML so the
conversion script is a rename-free copy.

### 5.2 Cache object

```python
@dataclass
class SanaWmBlockCache:            # one per transformer block, TP-local shapes
    # GDN blocks
    gdn_state_kv: Tensor | None    # [B, H, D, D]    main-branch delta-rule state
    gdn_state_z:  Tensor | None    # [B, H, D, 1]    main-branch denominator state
    cam_state_kv: Tensor | None    # [B, Hc, Dc, Dc] camera-branch numerator state (skip_z)
    conv_k_tail:  Tensor | None    # [B, S, C, k-1]  last k-1 frames feeding conv_k
    conv_cam_tail: dict[str, Tensor]   # same for conv_{q,k,v}_cam
    # softmax blocks
    k_main: Tensor | None          # [B, H, F_cached*S, D]  post-RoPE keys (absolute positions)
    v_main: Tensor | None
    k_cam:  Tensor | None          # post-UCPE camera keys/values
    v_cam:  Tensor | None
    # FFN
    ffn_tconv_tail: Tensor | None  # last t_kernel_size-1 frames feeding the FFN temporal conv
    # bookkeeping
    cached_frame_index: Tensor     # absolute latent frame ids of the cached softmax frames

@dataclass
class SanaWmStreamingCache:
    blocks: list[SanaWmBlockCache]          # len == num_blocks
    frames_committed: int                   # absolute latent frames written so far
    def trim(self, *, sink_token: bool, num_cached_blocks: int, block_frames: int) -> None
```

`trim()` implements `num_cached_blocks`: keep frame 0 when `sink_token` plus
the last `num_cached_blocks * block_frames` frames of `k_main/v_main/k_cam/v_cam`
and `cached_frame_index`. GDN states and conv tails are fixed-size and never
trimmed (the recurrence already decays). `frames_committed` is what
`start_f` is derived from.

`SanaWmStreamingCache.new(config, batch, latent_hw, device, dtype)` allocates
empty caches; `bytes()` reports the resident size for logging.

### 5.3 GDN recurrence with state carry (`_delta_scan`, `_bidirectional_delta_scan`)

`_delta_scan(..., initial_state_kv=None, initial_state_z=None, return_state=False)`.
When `initial_state_*` is given it replaces the zero init at
`sana_wm_transformer.py:175`; when `return_state` is set the final
`(state_kv, state_z)` is returned alongside the outputs.

`_bidirectional_delta_scan(..., initial_state=None, return_state=False)`:
forward direction is seeded from `initial_state` and reports its final state;
backward direction is unchanged (it only ever sees the current chunk, so the
existing `_flip_and_shift` zero/1.0 padding is exactly the boundary isolation
upstream describes). `reference_bidirectional_gated_delta_net` gains the same
two kwargs and passes them through.

The camera branch scan (`skip_z=True`, `_forward_ucpe_*`) carries
`cam_state_kv` the same way.

### 5.4 Temporal short convs

`_bidirectional_temporal_short_conv(hidden, conv, spatial_shape, tail=None, return_tail=False)`:
when `tail` is provided, prepend it to the frame axis before the forward pass
and drop the first `k-1` output frames; the backward pass stays chunk-local.
Return the new tail (last `k-1` input frames) when asked. Applies to
`conv_k` and `conv_{q,k,v}_cam` in both the GDN and softmax attention paths.

### 5.5 Softmax hybrid blocks (blocks 3, 7, 11, 15, 19)

`_forward_softmax_raw` / `_forward_softmax_cam_branch` gain
`cache: SanaWmBlockCache | None` and `save: bool`:

- main branch: apply RoPE to the chunk's Q/K using the chunk's absolute
  `frame_index`, then `k = cat([cache.k_main, k])`, `v = cat([cache.v_main, v])`
  and run the existing non-causal attention (`causal=False` at
  `sana_wm_transformer.py:806`) over cached+current keys. Queries are only the
  current chunk. When `save`, overwrite `cache.k_main/v_main` with the
  concatenation and let `trim()` evict.
- camera branch: same with post-`cam_geometry.apply_kv` keys/values
  (`sana_wm_transformer.py:1172`) so cached camera K/V are already in the
  UCPE canonical frame; `cam_geometry` for the chunk is built from the chunk's
  raymap slice only.

### 5.6 RoPE

`SanaWmWanRotaryPosEmbed.forward(spatial_shape, device, *, frame_index=None)`.
With `frame_index` (1-D long tensor of absolute latent frame ids, length
`F_chunk`) the temporal frequency table is gathered by those ids instead of
`arange(F)`. The bidirectional path passes nothing and is unchanged.

### 5.7 FFN temporal conv

`SanaWmMbConvFfn` (`sana_wm_transformer.py:1468`) currently pads
symmetrically (`t_kernel_size // 2`). Add a `causal_tail` path: when the block
runs in streaming mode, pad only on the left with the cached tail (or zeros for
chunk 0) and keep the last `t_kernel_size - 1` frames as the new tail. This is
the `ChunkGLUMBConvTemp` behaviour; the checkpoint weights are identical, only
padding differs, so `ffn_type` selects the path at runtime.

### 5.8 Model entry point

```python
def forward_streaming(
    self,
    hidden_states: Tensor,          # [B, C, F_chunk, H, W]
    timestep: Tensor,               # (B, 1, F_chunk) per-frame, fp32
    *,
    frame_index: Tensor,            # absolute latent frame ids for this chunk
    cache: SanaWmStreamingCache,
    save_cache: bool,
    encoder_hidden_states, encoder_attention_mask,
    plucker, raymap, spatial_raymap,   # already sliced to this chunk
) -> Tensor
```

Shares everything with `forward()` except: RoPE takes `frame_index`; every
block receives `cache.blocks[i]` and `save_cache`; `cam_geometry` is built per
chunk. `forward()` stays untouched so the bidirectional pipeline remains
byte-identical (the PR must prove this with the existing
`tests/model_tests/diffusion/test_alignment.py` run).

## 6. Sampler: `sana_wm/self_forcing.py`

```python
@dataclass(frozen=True)
class SanaWmSelfForcingSchedule:
    timesteps: tuple[int, ...]        # (1000, 960, 889, 727) — denoising_step_list[:-1]
    @property
    def sigmas(self): return tuple(t / 1000.0 for t in self.timesteps)
    @classmethod
    def from_config(cls, config: SanaWmConfig, override: Sequence[int] | None) -> ...
```

Validation: list ends with 0, strictly decreasing, first entry 1000, length
≥ 2. `num_inference_steps` on the request, if provided, must equal
`len(timesteps)`; otherwise reject (matches LingBot at
`lingbot_world/pipeline.py:685`).

Per-step update (velocity parameterisation, matching the sign convention the
bidirectional path documents at `pipeline_sana_wm.py:830-845`, where the
model output `v` satisfies `x_prev = x - (σ - σ_next) v`):

```text
x0      = x_t - σ_s * v_θ(x_t, t_s)
x_{t_{s+1}} = (1 - σ_{s+1}) * x0 + σ_{s+1} * ε_fresh      for s < last
final chunk latent = x0 of the last step
```

`ε_fresh` is drawn from the request generator each step (upstream re-noises
between steps). Conditioning frame 0 of chunk 0 is held at `t=0` and restored
after every step with the same `condition_mask` mechanism as the bidirectional
loop (`pipeline_sana_wm.py:875-880`).

Open decision D1 (§14): whether upstream re-noises with fresh noise or with the
original noise; unit test the two against a reference dump once
`SelfForcingFlowEulerCamCtrl` has been read.

## 7. Pipeline: `sana_wm/pipeline_sana_wm_streaming.py`

```python
class SanaWmStreamingPipeline(
    SanaWmPipeline,               # inherits loading, encoders, VAE, camera helpers
    SupportsStepExecution,
):
    supports_step_execution: ClassVar[bool] = True
    dummy_run_num_frames: ClassVar[int] = 0     # keep warmup skipped (same reason as parent)
```

`forward()` (request mode) is inherited and still works: it runs the
bidirectional loop, which is wrong for a causal checkpoint. Override it to
raise `NotImplementedError("SanaWmStreamingPipeline requires step execution; start with --diffusion-streaming-output")`
so a misconfigured server fails at the first request instead of producing
garbage.

### 7.1 `prepare_encode(state)`

Runs once per request. Everything below lands in `state.extra`.

1. `payload = state.prompt["additional_information"]["sana_wm"]` (already
   normalised by the pre-process hook). `params = self._native_params(...)`.
2. Geometry: `latent_T, lh, lw = resolve_video_latent_shape(...)`. Reject
   unless `(latent_T - 1) % num_frame_per_block == 0`, with a message listing
   the nearest valid `num_frames` (`8*3*k + 1`, e.g. 145, 169, 193). Reject
   `cfg_scale > 1.0` and `cfg_parallel_size > 1` for v1.
3. Text: `prompt_embeds`, `prompt_attention_mask` via `_native_prompt_embeds`
   (no negative branch). Stored on `state.prompt_embeds` / `extra`.
4. First frame: `first_latent = _vae_encode_first_frame(...)` (normalised).
5. Camera: `camera_tensors = build_plucker_condition(condition)` once for the
   whole trajectory (`chunk_plucker: [C, latent_T, lh, lw]`,
   `raymap: [latent_T, 20]`, `spatial_raymap: [3, latent_T, lh, lw]`), kept in
   `extra["camera_full"]`; per-chunk slices are views taken in
   `_prepare_next_chunk`. No change to `camera_control.py`.
6. Cache: `extra["cache"] = SanaWmStreamingCache.new(...)`.
7. History: `extra["history_latents"] = first_latent` (`[B, 128, 1, lh, lw]`,
   fp32), `extra["frames_committed"] = 1`.
8. Chunking: `state.total_chunks = (latent_T - 1) // 3`; chunk 0 generates
   frames `[1, 4)` with the conditioning frame in the same forward, chunk k>0
   generates `[1+3k, 4+3k)`.
9. Schedule: `extra["schedule"] = SanaWmSelfForcingSchedule.from_config(...)`.
10. `self._prepare_next_chunk(state)`.

### 7.2 `_prepare_next_chunk(state)`

```text
k = state.chunk_index
gen_start = 1 + 3k ; gen_end = gen_start + 3
if k == 0: frame_index = [0, 1, 2, 3]; latents = cat([first_latent, noise(3)])
else:      frame_index = [gen_start .. gen_end); latents = noise(3)
extra["frame_index"] = frame_index
extra["chunk_camera"] = slices of camera_full at frame_index
state.latents = latents (model dtype)
state.timesteps = tensor(schedule.timesteps)   # 4 entries
state.chunk_num_steps = 4 ; state.step_in_chunk = 0 ; state.step_index = 0
```

### 7.3 `denoise_step(input_batch, states)`

Single request only (assert like Helios). Build the per-frame timestep
`(B, 1, F_chunk)` from `state.current_timestep`, zero at frame 0 when
`chunk_index == 0`. Call `self.transformer.forward_streaming(... save_cache=False)`.
Return `v`.

### 7.4 `step_scheduler(state, noise_pred)`

Apply §6 with `σ_s = state.timesteps[state.step_in_chunk] / 1000`. Restore
frame 0 for chunk 0. Advance `step_in_chunk` / `step_index`. On the last step
keep `x0` in `state.latents`.

### 7.5 `post_decode(state)`

Called by the runner when `chunk_denoise_completed` (`diffusion_model_runner.py:1197`).

1. **Cache write:** `forward_streaming(x0, t=0, save_cache=True)` on the clean
   chunk; then `cache.trim(...)`; `frames_committed += F_chunk`.
2. **History:** append the generated frames (chunk 0: frames 1..3) to
   `history_latents`.
3. **Decode (§8):** `decode_latents = history[-4:]` (previous last frame +
   3 new; for chunk 0 that is frames 0..3). `video = _decode_native_latents(...)`.
   For `k > 0` drop the first pixel frame. Chunk 0 yields 25 frames, every
   later chunk 24; for 169 frames total `25 + 6 * 24 = 169`.
4. **Envelope:** `build_sana_wm_output_envelope(output=video, output_type=..., metadata={"backend": "native_gdn_streaming", "chunk_index": k, "frame_index": [...], "cache_bytes": ...})`.
5. **Advance:** `completed = state.chunk_index`; `state.chunk_index += 1`;
   `finished = state.request_denoise_completed`; if not finished
   `_prepare_next_chunk(state)` else free `cache`, `camera_full`, `history_latents`.
6. `return DiffusionOutput(output=envelope, chunk_index=completed, total_chunks=state.total_chunks, finished=finished, stage_durations=...)`.

`output_type="latent"` is honoured (returns the 3 new latent frames) so the
parity harness (§12) can compare against the batch path without a VAE.

## 8. VAE decode without a streaming decoder

The LTX-2 VAE maps `n` latent frames to `1 + 8(n-1)` pixel frames, so decoding
a 3-frame chunk in isolation yields 17 frames instead of the 24 it contributes
in a contiguous decode (the "63 instead of 81" effect in #6844 for Wan).
Decoding with a one-frame overlap and dropping the first pixel frame restores
the frame count exactly and gives the decoder one latent frame of temporal
context. Quality at chunk seams is still not identical to a whole-clip decode;
this is the documented v1 limitation and the reason a streaming decoder
(#6533-style) remains a follow-up.

Before merge, add a measurement to the PR description: PSNR/SSIM of
overlap-decode vs whole-clip decode on the same latents (script under
`tests/diffusion/models/sana_wm/` gated `advanced_model`). This can be produced
with the **existing bidirectional** pipeline, so it does not block on the
causal model.

VAE tiling stays forced on (`pipeline_sana_wm.py:483`); with 4-latent-frame
inputs only spatial tiling is exercised and peak memory drops well below the
22.6 GB whole-clip figure in the recipe.

## 9. Registry, metadata, deploy, CLI

- `diffusion/registry.py`: add `"SanaWmStreamingPipeline": ("sana_wm", "pipeline_sana_wm_streaming", "SanaWmStreamingPipeline")` and the same pre-process func as `SanaWmPipeline`.
- `diffusion/model_metadata.py`: `"SanaWmStreamingPipeline": DiffusionModelMetadata(final_output_type="video", supports_multimodal_inputs=True, max_multimodal_image_inputs=1)`. **`final_output_type="video"` is required**: `is_video_generation_pipeline` (`entrypoints/openai/utils.py:23`) gates the WS handler and the current `SanaWmPipeline` entry lacks it (`model_metadata.py:92`); fix that entry too.
- `model_index.json` of the converted streaming repo names `SanaWmStreamingPipeline` so class resolution needs no flag.
- `vllm_omni/deploy/sana_wm_streaming.yaml`: single stage, `max_num_seqs: 1`, `model_class_name: SanaWmStreamingPipeline`, `engine_args.streaming_output: true`, `default_sampling_params: {num_inference_steps: 4, guidance_scale: 1.0, num_frames: 169, height: 704, width: 1280, fps: 16}`. Single-stage models resolved through `create_default_diffusion` do not read YAML defaults (comment at `pipeline_sana_wm.py:91`), so the pipeline also carries these as class constants and the YAML is the documented CLI entry point. `--diffusion-streaming-output` on the CLI is equivalent.
- Startup validation in `SanaWmStreamingPipeline.__init__`: `config.streaming` must be true, else raise naming the expected repo.

## 10. Serving contract (no new endpoint)

`session.start` example:

```json
{
  "type": "session.start",
  "model": "<streaming repo id>",
  "prompt": "A slow forward camera move through a quiet city street.",
  "image_reference": "data:image/png;base64,...",
  "width": 1280, "height": 704, "num_frames": 169, "fps": 16,
  "num_inference_steps": 4, "guidance_scale": 1.0, "seed": 42,
  "extra_params": {
    "sana_wm": {
      "action": "w-168",
      "translation_speed": 0.055,
      "rotation_speed_deg": 1.2,
      "intrinsics": {"fx": 640, "fy": 640, "cx": 640, "cy": 352}
    }
  }
}
```

Server behaviour, all provided by the existing handler:

- `video.start`, then per chunk `video.chunk_metadata` with
  `generation_chunk_index = k`, `num_frames = 25 | 24`, followed by the binary
  m4s frame; `session.done` with `chunks = total_chunks (+1 trailer)`.
- `session.stop` / disconnect → `_abort_request`; the runner drops the state
  and the cache with it.
- `session.interaction` → the pipeline does not mix in `PromptUpdateMixin`;
  `submit_interaction` reaches the runner and returns an error that the
  handler forwards as `{"type": "error"}`. Documented in the recipe.

The offline path is `examples/offline_inference/diffusion/` **not** extended
(repo rule: no new model-specific examples); the e2e test in §12 is the
reference invocation.

## 11. Checkpoint conversion

Extend the existing offline conversion (the one that produced
`BBBBruce/SANA-WM_bidirectional-stage1-diffusers`) to accept the streaming
release (`SANA-WM_streaming/sana_dit/model.pt` + `sana_wm_streaming_1600m_720p.yaml`)
and emit the same diffusers layout with `transformer/config.json` carrying the
§5.1 fields and `model_index.json` naming `SanaWmStreamingPipeline`. Weight
keys are expected to match the bidirectional checkpoint one-to-one (same
architecture; only attention/FFN behaviour differs); the conversion asserts
that and fails loudly on any unmapped key. VAE is copied unchanged.

## 12. Tests

CPU / L1 (`tests/diffusion/models/sana_wm/`, marker `cpu`):

- `test_streaming_cache.py`: `SanaWmStreamingCache.new/trim` shapes, sink
  retention, `num_cached_blocks=-1`, `frames_committed` bookkeeping.
- `test_delta_scan_state_carry.py`: on a tiny config, running `_delta_scan`
  over frames `[0, T)` in one call equals running `[0, a)` then `[a, T)` with
  the carried state (forward direction); the bidirectional chunked variant
  equals the upstream "backward isolated at boundary" definition built by
  hand.
- `test_temporal_conv_tail.py` / `test_ffn_causal_tail.py`: chunked causal
  conv with tail equals a single causal conv over the concatenation.
- `test_softmax_cache.py`: cached K/V + `frame_index` RoPE equals full
  attention over the concatenated frames with absolute positions.
- `test_self_forcing_schedule.py`: validation rules, sigma mapping, one-step
  update equals the bidirectional per-token Euler step at the same sigma
  (sign convention guard).
- `test_streaming_pipeline_stepwise.py`: drive `prepare_encode → denoise_step
  → step_scheduler → post_decode` with the transformer and VAE mocked; assert
  `total_chunks`, `frame_index` per chunk, `chunk_index/finished` in each
  `DiffusionOutput`, 25/24 frame counts, `num_frames` rejection message,
  `cfg_scale>1` rejection, cache freed on the last chunk.
- Extend `tests/diffusion/test_diffusion_streaming_output.py` fixtures only
  if the envelope shape needs a new case (it should not).

GPU / L3-L4:

- `tests/e2e/online_serving/test_sana_wm_streaming.py`: boot with the
  streaming repo and `--diffusion-streaming-output`, run the WS client from
  `tests/helpers/client.py`, assert `generation_chunk_index` is contiguous,
  total frames = `num_frames`, and MP4 decodes. Small profile: 704x1280,
  `num_frames=49` (2 chunks).
- Parity harness (`advanced_model`): the sglang PR verified its batch and
  realtime paths bit-exact at latent level. Our equivalent: run
  `forward_streaming` chunk by chunk with `save_cache=True` on **clean**
  latents and compare against one `forward()` call on the same latents with a
  hand-built chunk-causal reference; expected `atol` documented. This is the
  test that catches a wrong slot layout or RoPE offset.
- Keep `tests/e2e/online_serving/test_sana_wm.py` green on the bidirectional
  repo and add the alignment run to the PR description as evidence the
  request-mode path is unchanged.

## 13. Docs

- `recipes/NVIDIA/SANA-WM.md`: new "Streaming (distilled Stage-1)" section
  with the command, `session.start` example, chunk timing and memory numbers
  measured on the same GPU as the existing section, and a "Known limitations"
  list (independent chunk decode, no interaction, no refiner, TP unvalidated).
- `docs/serving/streaming_video_output_api.md`: add SANA-WM to the model list
  with the `extra_params.sana_wm` block and note that `session.interaction` is
  rejected for it.
- `docs/models/supported_models.md`: mark streaming support.
- This document: flip status to "implemented" and record D1-D4 outcomes.

## 14. Implementation prerequisites and open decisions

Read before coding §5/§6 (each is a bounded, one-file read in NVlabs/Sana):

- P1 `CachedChunkCausalGDN` and `CachedChunkCausalSoftmaxAttn` in
  `diffusion/model/nets/sana_gdn_blocks*.py`: slot indices and tensor shapes →
  fixes §5.2.
- P2 `SelfForcingFlowEulerCamCtrl` (`create_autoregressive_segments`,
  x0/re-noise formula, cache-write call) → fixes §6 and D1/D2.
- P3 how `frame_index` is assembled from `sink_token` + `num_cached_blocks`
  in `SanaWMPipeline` → fixes `SanaWmStreamingCache.trim` and §5.6.
- P4 `ChunkGLUMBConvTemp` → confirms §5.7 (left-pad + tail, same weights).

Decisions to record in the PR:

- D1: re-noise with fresh noise per step vs. the chunk's original noise.
- D2: cache write as a separate `t=0` forward (assumed) vs. folded into the
  last denoising step.
- D3: chunk 0 as `[0,4)` with the conditioning frame in the forward (assumed,
  matches `streaming_pipeline.py`) vs. `[0,3)` generated frames only.
- D4: default `num_frames` for the streaming deploy: 169 (nearest valid ≥ the
  current 161 default) unless upstream's default differs.

## 15. Acceptance criteria

1. `vllm serve <streaming repo> --omni --diffusion-streaming-output` accepts
   the §10 `session.start` and streams `total_chunks` media chunks whose frame
   counts sum to `num_frames`; first chunk latency and per-chunk cadence are
   reported in the recipe.
2. All §12 CPU tests pass in the `cpu` lane; the parity harness passes at the
   documented tolerance on one GPU.
3. Bidirectional `SanaWmPipeline` request-mode output is byte-identical to
   `main` (alignment test), and `tests/e2e/online_serving/test_sana_wm.py`
   still passes.
4. No changes under `vllm_omni/diffusion/worker/`, `vllm_omni/engine/`,
   `vllm_omni/entrypoints/`, or `vllm_omni/experimental/`.
5. PR description lists the non-goals from §1 as follow-ups and the D1-D4
   outcomes.
