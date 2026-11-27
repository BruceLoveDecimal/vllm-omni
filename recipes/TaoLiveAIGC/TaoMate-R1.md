# TaoMate-R1: reference-conditioned realtime streaming

[TaoLiveAIGC/TaoMate-R1](https://huggingface.co/TaoLiveAIGC/TaoMate-R1) streams joint
audio and video from a prompt and ordered image references on the
[MiniMax-H3 Ref2VA checkpoint](https://huggingface.co/MiniMaxAI/MiniMax-H3). It ships two
rank-128 LoRA banks: an opening ("cold") bank for the first chunk and a continuation
("warm") bank for every later chunk. vLLM-Omni serves it through the AR-Diffusion engine
and `WS /v1/realtime/video`.

> Status: experimental. The chunk plan, packed layouts, reference conditioning noise and
> Qwen presentation match the release on CPU. GPU numerical parity with the release
> runtime has not been measured yet.

## Execution and cache ownership

A session has a fixed length chosen at `session.start`. Its frame count is aligned to
H3's `17 j + 5` grid, and the whole Ref2VA packed sequence is built once.

| Operation | Model behavior |
| --- | --- |
| Reference encode | Each image is resized to a 2048-pixel short edge (32-pixel grid, Lanczos). It becomes `<Picture i>` in request order for Qwen3-VL, and a video-VAE anchor with the 0.999 condition noise. |
| Chunk plan | 39 frames (12 video latents) open the session; later chunks add 34 frames (10 latents). A 17-frame tail closes sessions whose aligned length requires it. |
| Denoise | Three Euler steps per chunk at sigma indices 0, 16, 33 and 49 of the 50-point shifted schedule (video shift 12, audio shift 3). Audio and video are generated together. |
| Banks | Chunk 0 runs the cold bank, including its clean commit; every later chunk runs the warm bank. |
| Clean commit | Every chunk except the last recomputes its clean audio/video K/V at sigma 0. The last chunk has no successor and skips the write. |
| History | The first chunk's video is the sink, plus the two most recent chunks. Text and reference rows are recomputed in every forward. |
| Decode | Incremental video VAE windows and a sliding-window audio decode publish each chunk before the session ends. |

The LoRA manager pins both banks. The pipeline selects the active bank at each chunk
boundary. With one GPU adapter slot, the switch from cold to warm before chunk 1 reloads
the adapter weights once per session.

## Requirements

- MiniMax-H3 `Ref2VA/` weights and the TaoMate-R1 directory that holds `config.json`,
  `cold/` and `warm/`:

  ```bash
  hf download MiniMaxAI/MiniMax-H3 --include "model_index.json" "Ref2VA/*" --local-dir MiniMax-H3
  hf download TaoLiveAIGC/TaoMate-R1 --local-dir TaoMate-R1
  ```

- Hopper or newer GPUs with FlashAttention-3. The supplied configuration uses four
  H200-class 141 GB GPUs: BF16 DiT replicated per rank, Ulysses 4, and text encoder TP4.
- One session per server (`max_num_seqs: 1`).

## Realtime serving

```bash
vllm serve /path/to/MiniMax-H3/Ref2VA --omni --trust-remote-code \
  --deploy-config vllm_omni/deploy/taomate_r1_usp4_realtime.yaml \
  --lora-path /path/to/TaoMate-R1 --port 8000 \
  --stage-init-timeout 1800 --init-timeout 2100
```

The engine runs a three-chunk throwaway session before serving, covering the cold bank,
the warm bank and the terminal chunk.

Stream a ten-second clip with two references, in order:

```bash
python examples/online_serving/streaming_video_generation/streaming_video_client.py \
  --port 8000 --model /path/to/MiniMax-H3/Ref2VA --size 768x1344 --fps 24 --num-frames 243 \
  --seed 9111 --no-helios-distilled-preset \
  --image-reference presenter.png --image-reference product.png \
  --prompt "The presenter from <Picture 1> holds the mug from <Picture 2> and describes it." \
  --output taomate_r1.mp4
```

`session.start` takes `prompt`, an ordered `image_reference` list (1 to 9 images),
`width`/`height` matching the deployment, `fps: 24`, `seed` and `num_frames`.
Each `video.chunk_metadata` message carries
`model: {"name": "taomate_r1", "chunk_index", "lora_bank", "clean_commit"}`.
The `session.start` message is limited to 4 MiB; use image URLs or compressed images
for large references.

## Ordinary full-request inference

`taomate_r1_usp4_full_request.yaml` returns one MP4 with stereo audio through
`/v1/videos`:

```bash
vllm serve /path/to/MiniMax-H3/Ref2VA --omni --trust-remote-code \
  --deploy-config vllm_omni/deploy/taomate_r1_usp4_full_request.yaml \
  --lora-path /path/to/TaoMate-R1 --port 8000

curl --fail-with-body --silent --show-error http://localhost:8000/v1/videos \
  -F 'prompt=The presenter from <Picture 1> holds the mug from <Picture 2> and describes it.' \
  -F 'image_reference=[{"image_url":"https://example.com/presenter.png"},{"image_url":"https://example.com/product.png"}]' \
  -F 'size=768x1344' -F 'num_frames=243' -F 'fps=24' -F 'seed=9111'
```

`image_reference` is a JSON list in reference order; each entry is an http(s) URL or a
`data:image/...;base64,` URL. Uploaded `input_references` files are not used for this
model. Poll the returned job and download `/v1/videos/{id}/content` as with other video
models.

## Single-GPU smoke

This capacity profile checks that the model loads and generates on one GPU. It is not
the release configuration and does not reproduce the release outputs.

- FP8 per-channel DiT linears.
- A 480x864 canvas.
- References resized to a 768-pixel short edge.
- The Qwen3-VL text encoder streamed from host memory.

On one GPU the clean K/V history and attention staging keep all 56 heads, about
1.4 MB per token across the 50 layers. Use an H200-class 141 GB card and host memory
for the offloaded encoder.

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /path/to/MiniMax-H3/Ref2VA --omni --trust-remote-code \
  --deploy-config vllm_omni/deploy/taomate_r1_1gpu_smoke.yaml \
  --lora-path /path/to/TaoMate-R1 --port 8000 \
  --diffusion-offload-config '{"mode":"layer","components":["text_encoder"]}' \
  --stage-init-timeout 1800 --init-timeout 2100
```

The default request is three chunks (107 frames), which covers both banks and the
terminal chunk. `taomate_r1_1gpu_realtime_smoke.yaml` serves the same profile over
`/v1/realtime/video`.

## Deployment controls

These keys live under the stage's `model_config`.

| Key | Default | Meaning |
| --- | --- | --- |
| `taomate_r1_width` / `taomate_r1_height` | 768 / 1344 | Fixed canvas. Both edges must be multiples of 32, with a 480-, 768- or 1088-pixel short edge. |
| `taomate_r1_seed` | 42 | Default seed. Chunk `k` draws its noise from `seed + 97 k`. The reference condition noise uses `seed`. |
| `taomate_r1_default_frames` | 240 | Frame budget when a request names none; aligns to 243 frames (seven chunks). |
| `taomate_r1_max_frames` | 1440 | Longest accepted session. Positions beyond the release's typical lengths are untested. |
| `taomate_r1_warmup_chunks` | 3 | Chunks in the startup rollout. |
| `taomate_r1_reference_short_edge` | 2048 | Reference resize short edge, a multiple of 32 in [256, 4096]. Values other than 2048 depart from the release and only serve capacity smoke tests. |
| `taomate_r1_dump_latents_dir` | unset | Parity debugging: the output rank saves each chunk's final video/audio rows to `<dir>/<session>/chunk_<i>.pt`. Adds a device-to-host copy per chunk. |

The TaoMate decoder and logging controls keep their TaoMate-H3 names because the
mechanisms are shared: `taomate_h3_vae_decoder_tile_size`,
`taomate_h3_vae_decoder_tile_overlap_min`, `taomate_h3_vae_stack_tiling`,
`taomate_h3_cudnn_benchmark`, `taomate_h3_fp8_quant_cuda_op`, `taomate_h3_adaln_cache`
and `taomate_h3_log_timings`. TaoMate-H3's audio teacher, prompt hold and merged-LoRA
options do not apply.

## Validation and limits

- Image references only. Video and audio references, and image/video interleaving, are
  rejected; the release's video-reference preprocessing has not been ported.
- The prompt is fixed for the session; `session.interaction` events are rejected.
- Text-to-video without a reference is rejected; use TaoMate-H3 for T2AV.
- The parallel layout differs from the release (Ulysses 4 with TP 1 here, TP2 x Ulysses
  in the release), so outputs match only to numerical tolerance.
- Per-chunk audio decoding has less right context at chunk edges than the release's
  offline decode of the whole session.

Before relying on a deployment, check the following:

- Two distinguishable references keep their order.
- A three-chunk session reports `cold, warm, warm` and `clean_commit` true, true, false.
- Audio/video chunks arrive before the session ends.
- The complete MP4 decodes with both tracks.
