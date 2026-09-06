# SANA-WM

> Camera-controllable first-frame image-to-video world model.

## Summary

- Vendor: Efficient-Large-Model / NVlabs SANA
- Model: `BBBBruce/SANA-WM_bidirectional-stage1-diffusers` (standard diffusers layout, converted offline from the NVlabs release; Stage-1 transformer + VAE only)
- Task: First-frame image-to-video generation with camera control
- Mode: Online serving with the OpenAI-compatible video API; the distilled
  chunk-causal release additionally streams over `WS /v1/realtime/video`
  (see "Streaming (distilled chunk-causal Stage-1)" below)
- Model weights: about 13 GB for the Stage-1 transformer (10 GB) and VAE (2.3 GB)
- Local disk: reserve about 40 GB for the Hugging Face cache and runtime artifacts
- Recommended GPU: 24 GB or larger CUDA GPU
- Maintainer: Community

## When to use this recipe

Use this recipe when you want to serve SANA-WM through `/v1/videos` or
`/v1/videos/sync`. The model takes a text prompt, a first-frame image, and
either an action DSL string or explicit camera poses. vLLM-Omni serves the
SANA-WM Stage-1 DiT, decoded through the SANA VAE. The optional LTX-2 refiner
stage is not supported by this integration; it is a planned follow-up.

## References

- Upstream model card: <https://huggingface.co/Efficient-Large-Model/SANA-WM_bidirectional>
- Video API: [`docs/serving/videos_api.md`](../../docs/serving/videos_api.md)

## Hardware Support

## GPU

### 1x NVIDIA RTX PRO 6000 Blackwell 96GB

#### Capacity

- Model storage: the Stage-1 transformer is about 10 GB and the VAE about
  2.3 GB. The Gemma text encoder is a separate 4.9 GB repo. The engine prefetches
  the whole model repo at startup (`allow_patterns=["*"]`), which is why the
  Stage-1 weights live in their own repo — the two-stage one carries an
  additional 84 GB `refiner/` that this path never loads.
- Text encoder: the pipeline tries `google/gemma-2-2b-it` first, then falls back
  to the ungated mirror `Efficient-Large-Model/gemma-2-2b-it`. The first repo is
  gated, so without an accepted licence and `hf auth login` you will see one
  failed load before the fallback succeeds — that is expected, not an error to
  chase. Set `VLLM_OMNI_SANA_WM_STAGE1_TEXT_ENCODER` to pin a specific repo or a
  local path and skip both.
- Disk sizing: provision about 40 GB of local disk or Hugging Face cache volume
  so the model, temporary downloads, and generated artifacts fit without cache
  eviction.
- GPU sizing: the default 1280x704, 161-frame, 60-step serving profile peaks at
  22.6 GB of device memory and takes about 133 s to generate on one RTX PRO
  6000 Blackwell. The peak lands in the VAE decode, which is why the pipeline
  forces VAE tiling on regardless of `vae_use_tiling` — without it the same
  request costs about 9 GB more, and a 321-frame one OOMs outright. On smaller
  GPUs, lower `width`, `height`, or `num_frames` before serving production
  requests.

#### Environment

- OS: Linux
- Python: 3.10+
- Driver / runtime: NVIDIA driver with CUDA runtime supported by your PyTorch
  build
- Recommended operator library: Triton, installed through the vLLM/vLLM-Omni
  Python environment
- vLLM version: Match the repository requirements for your checkout
- vLLM-Omni version or commit: Use the commit you are deploying from

#### Command

The repo ships the standard Diffusers layout (`model_index.json` +
`transformer/`, `vae/`), and its `model_index.json` names `SanaWmPipeline`, so
the pipeline class resolves on its own.

```bash
CUDA_VISIBLE_DEVICES=0 \
vllm serve BBBBruce/SANA-WM_bidirectional-stage1-diffusers \
  --omni \
  --host 0.0.0.0 \
  --port 8091
```

No deploy config: single-stage diffusion models are deliberately absent from
`OMNI_PIPELINES` (`vllm_omni/config/pipeline_registry.py`), so stage resolution
falls back to the default stage config and a YAML's stage settings — including
`default_sampling_params` — would not be applied. The production generation
settings therefore live in the model (`num_inference_steps=60`,
`guidance_scale=5.0`); a request that omits a field gets them. The examples
below still pass every field explicitly so the numbers are visible.

If you point this at the older two-stage repo
(`BBBBruce/SANA-WM_bidirectional-diffusers`), startup fails with `Model class
SanaWmTwoStagesPipeline not found in diffusion model registry`, because that
repo's `model_index.json` names a class this build does not register. Add
`--model-class-name SanaWmPipeline` to override it.

#### Verification

Use a short smoke request first:

```bash
curl -sS -X POST http://localhost:8091/v1/videos/sync \
  -H "Accept: video/mp4" \
  -F "prompt=A slow forward camera move through a quiet city street." \
  -F "negative_prompt=blurry, low quality, distorted, watermark" \
  -F "input_reference=@/path/to/first_frame.png;type=image/png" \
  -F "width=1280" \
  -F "height=704" \
  -F "num_frames=9" \
  -F "fps=16" \
  -F "num_inference_steps=2" \
  -F "guidance_scale=5.0" \
  -F "seed=42" \
  --form-string 'extra_params={"sana_wm":{"action":"w-8","translation_speed":0.055,"rotation_speed_deg":1.2,"intrinsics":{"fx":640,"fy":640,"cx":640,"cy":352}}}' \
  -o sana_wm_smoke.mp4
```

For a production-length request, note that the action durations must sum to
`num_frames - 1` — the rollout includes the identity start pose — and a
mismatch is rejected rather than padded or truncated:

```bash
curl -sS -X POST http://localhost:8091/v1/videos/sync \
  -H "Accept: video/mp4" \
  -F "prompt=A slow forward camera move through a quiet city street." \
  -F "negative_prompt=blurry, low quality, distorted, watermark" \
  -F "input_reference=@/path/to/first_frame.png;type=image/png" \
  -F "width=1280" \
  -F "height=704" \
  -F "num_frames=161" \
  -F "fps=16" \
  -F "num_inference_steps=60" \
  -F "guidance_scale=5.0" \
  -F "seed=42" \
  --form-string 'extra_params={"sana_wm":{"action":"w-160","translation_speed":0.055,"rotation_speed_deg":1.2,"intrinsics":{"fx":640,"fy":640,"cx":640,"cy":352}}}' \
  -o sana_wm_output.mp4
```

Use `POST /v1/videos` instead when you want job storage and polling rather than
inline MP4 bytes. It accepts the same form fields as `/v1/videos/sync`.

```bash
create_response=$(curl -sS -X POST http://localhost:8091/v1/videos \
  -H "Accept: application/json" \
  -F "prompt=A slow forward camera move through a quiet city street." \
  -F "negative_prompt=blurry, low quality, distorted, watermark" \
  -F "input_reference=@/path/to/first_frame.png;type=image/png" \
  -F "width=1280" \
  -F "height=704" \
  -F "num_frames=161" \
  -F "fps=16" \
  -F "num_inference_steps=60" \
  -F "guidance_scale=5.0" \
  -F "seed=42" \
  --form-string 'extra_params={"sana_wm":{"action":"w-160","translation_speed":0.055,"rotation_speed_deg":1.2,"intrinsics":{"fx":640,"fy":640,"cx":640,"cy":352}}}')

video_id=$(echo "$create_response" | jq -r '.id')
curl -sS "http://localhost:8091/v1/videos/${video_id}" | jq .
curl -L "http://localhost:8091/v1/videos/${video_id}/content" -o sana_wm_output.mp4
```

#### Notes

- Sequence parallelism is not supported. The bidirectional gated delta
  recurrence carries state across frames, so a rank cannot denoise a slice
  of the token sequence in isolation; supporting it needs a distributed scan
  or an all-gather before the GDN blocks.

- `input_reference` is required for the first frame. Use `image_reference` only
  when you need a JSON-safe image URL or data URL instead of a multipart upload.
- `sana_wm` must provide exactly one of `action` or `camera`.
- Action strings use comma-separated `<keys>-<duration>` segments. Supported
  keys are `w`, `a`, `s`, `d` for translation and `i`, `j`, `k`, `l` for
  pitch/yaw rotation. The durations must sum to `num_frames - 1`.
- Explicit camera control (alternative to `action`): pass
  `"camera": {"poses": [...]}` where `poses` is a list of `num_frames`
  camera-to-world 4x4 matrices (row-major, OpenCV `+X right, +Y down, +Z forward`
  convention), e.g.
  `extra_params={"sana_wm":{"camera":{"poses":[[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], ...]},"intrinsics":{...}}}`.
  Most callers should prefer `action`; explicit poses exist for callers that
  already have a per-frame trajectory.
- Explicit `intrinsics` are recommended and take the mapping form
  `{"fx":640,"fy":640,"cx":640,"cy":352}` (for 1280x704). This `{fx,fy,cx,cy}`
  mapping is the only accepted intrinsics form; omit `intrinsics` to derive them
  from the output resolution. All four values must be finite, and `fx`/`fy`
  must be positive — the ray map divides by them.
- The video API returns decoded MP4 bytes and has no `output_type` field, so
  raw Stage-1 latents are reachable only from the offline API
  (`OmniDiffusionSamplingParams(output_type="latent")`); see
  [`tests/e2e/offline_inference/test_sana_wm.py`](../../tests/e2e/offline_inference/test_sana_wm.py).
  Putting `output_type` in `extra_params` does not work: it lands in
  `sampling_params.extra_args`, while the pipeline reads the top-level field.

### Streaming (distilled chunk-causal Stage-1)

The distilled `SANA-WM_streaming` release is a chunk-causal Stage-1 student:
it generates three latent frames (24 pixel frames) per chunk conditioned on a
KV/recurrent-state cache of the previous chunks, so a clip can be streamed
while it is being generated. vLLM-Omni serves it through
`SanaWmStreamingPipeline` and the step-execution framework behind
[`WS /v1/realtime/video`](../../docs/serving/streaming_video_output_api.md):
one `session.start` yields one fragmented-MP4 chunk per generated block.
Design document: [`docs/design/feature/sana_wm_realtime_streaming.md`](../../docs/design/feature/sana_wm_realtime_streaming.md).

- Model: `BBBBruce/SANA-WM_streaming-stage1-diffusers` (Stage-1 transformer in
  bf16, about 5.3 GB, plus the Stage-1 VAE; converted from
  `Efficient-Large-Model/SANA-WM_streaming` with
  [`tools/convert_sana_wm_streaming_to_diffusers.py`](../../tools/convert_sana_wm_streaming_to_diffusers.py)).
  Its `model_index.json` names `SanaWmStreamingPipeline`; the pipeline refuses
  bidirectional checkpoints at startup.
- Sampler: the distilled self-forcing schedule `1000, 960, 889, 727 -> 0`
  (4 denoising forwards plus one clean cache-write forward per chunk),
  `guidance_scale=1.0`, `num_cached_blocks=2` with the chunk-0 sink anchor.
  `num_inference_steps` must be 4 or omitted; `guidance_scale > 1` is rejected.
- Geometry: `num_frames` must be `24k + 1` (25, 49, ..., 169, ...); the default
  streaming length is 169 frames (7 chunks).

#### Command

```bash
CUDA_VISIBLE_DEVICES=0 \
vllm serve BBBBruce/SANA-WM_streaming-stage1-diffusers \
  --omni \
  --diffusion-streaming-output \
  --host 0.0.0.0 \
  --port 8091
```

`--diffusion-streaming-output` is required: it switches the runner to step
execution and forwards every chunk as soon as it is decoded. Without it the
first request fails with `SanaWmStreamingPipeline requires step execution`.
Request-mode endpoints (`/v1/videos`) are not served by this pipeline.

#### Verification

```bash
pip install av websockets
python - <<'EOF'
import asyncio, base64, io, json, websockets
from PIL import Image

img = Image.new("RGB", (1280, 704), (96, 128, 160))
buf = io.BytesIO(); img.save(buf, format="PNG")
payload = {
    "type": "session.start",
    "model": "BBBBruce/SANA-WM_streaming-stage1-diffusers",
    "prompt": "A slow forward camera move through a quiet city street.",
    "image_reference": {"image_url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()},
    "width": 1280, "height": 704, "num_frames": 49, "fps": 16,
    "num_inference_steps": 4, "guidance_scale": 1.0, "seed": 42, "format": "m4s",
    "extra_params": {"sana_wm": {"action": "w-48", "translation_speed": 0.055, "rotation_speed_deg": 1.2,
                                 "intrinsics": {"fx": 640, "fy": 640, "cx": 640, "cy": 352}}},
}

async def main():
    async with websockets.connect("ws://localhost:8091/v1/realtime/video", max_size=None) as ws:
        await ws.send(json.dumps(payload))
        with open("sana_wm_stream.m4s", "wb") as out:
            while True:
                msg = await ws.recv()
                if isinstance(msg, bytes):
                    out.write(msg); continue
                event = json.loads(msg); print(event)
                if event["type"] in ("session.done", "error"):
                    break

asyncio.run(main())
EOF
ffmpeg -y -i sana_wm_stream.m4s -c copy sana_wm_stream.mp4
```

Each `video.chunk_metadata` carries `generation_chunk_index` and `num_frames`
(25 for the first chunk, 24 afterwards). The e2e in
[`tests/e2e/online_serving/test_sana_wm_streaming.py`](../../tests/e2e/online_serving/test_sana_wm_streaming.py)
is the reference invocation.

#### Measured (1x RTX PRO 6000 Blackwell 96GB, 1280x704, 49 frames, seed 42)

- Server startup 18 s; model load 12.2 GiB (transformer + VAE + Gemma-2-2B).
- First media chunk (25 frames) 1.65 s after `session.start`; both chunks
  (49 frames) in 2.72 s, i.e. faster than the 16 fps playback rate
  (steady state about 0.9 s per 24-frame chunk).
- Parity against the NVlabs streaming reference (`forward_long` +
  `SelfForcingFlowEulerCamCtrl`, identical noise / text embeddings / camera
  tensors, same VAE for both decodes): SSIM 0.876 (chunk 0: 0.921, chunk 1:
  0.829), PSNR 20.8 dB over 49 frames against the reference's default bf16
  tensor-core GDN kernels, and SSIM 0.906 (0.937 / 0.872), PSNR 22.3 dB
  against the reference with fp32 GDN dots (`FUSED_GDN_PRECISION=0`). The
  distilled 4-step solver compounds bf16-level kernel differences, as in the
  bidirectional 161-frame figures above.

#### Known limitations (v1)

- Chunks are decoded independently with a one-latent-frame overlap (the
  previous chunk's last latent frame is decoded again and its pixel frame
  dropped), not with a stateful streaming VAE; seams between chunks are not
  identical to a whole-clip decode. The streaming release's causal VAE decoder
  is a follow-up.
- The camera trajectory is fixed at `session.start`; `session.interaction`
  (prompt update) is rejected.
- The LTX-2 refiner stage is not served.
- One request per session; `cfg_parallel_size > 1` is rejected and tensor
  parallelism is not validated for the streaming pipeline.
