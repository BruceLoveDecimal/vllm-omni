# SANA-WM

> Camera-controllable first-frame image-to-video world model.

## Summary

- Vendor: Efficient-Large-Model / NVlabs SANA
- Model: `BBBBruce/SANA-WM_bidirectional-stage1-diffusers` (standard diffusers layout, converted offline from the NVlabs release; Stage-1 transformer + VAE only)
- Model (two-stage): `BBBBruce/SANA-WM_bidirectional-diffusers` — the same Stage-1 tree plus the LTX-2 refiner. See [Two-stage serving](#two-stage-serving-stage-1--ltx-2-refiner).
- Task: First-frame image-to-video generation with camera control
- Mode: Online serving with the OpenAI-compatible video API
- Model weights: about 13 GB for the Stage-1 transformer (10 GB) and VAE (2.3 GB)
- Local disk: reserve about 40 GB for the Hugging Face cache and runtime artifacts
- Recommended GPU: 24 GB or larger CUDA GPU
- Maintainer: Community

## When to use this recipe

Use this recipe when you want to serve SANA-WM through `/v1/videos` or
`/v1/videos/sync`. The model takes a text prompt, a first-frame image, and
either an action DSL string or explicit camera poses. vLLM-Omni serves the
SANA-WM Stage-1 DiT, decoded through the SANA VAE. The optional LTX-2 refiner
stage is served by a second pipeline class against a second repo — see
[Two-stage serving](#two-stage-serving-stage-1--ltx-2-refiner). Stage-1 alone is
the cheaper default and is what the rest of this section describes.

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
  additional 84 GB `refiner/` that this path never loads. Deploy the two-stage
  repo only when you actually want the refiner.
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

The two-stage repo (`BBBBruce/SANA-WM_bidirectional-diffusers`) resolves to
`SanaWmTwoStagesPipeline` from its own `model_index.json`. To serve only Stage 1
from that repo — skipping the refiner load but still paying its download — add
`--model-class-name SanaWmPipeline`.

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

### Two-stage serving (Stage 1 + LTX-2 refiner)

`SanaWmTwoStagesPipeline` runs the same Stage-1 sampler and then refines its
latents with the LTX-2 refiner bundled in the two-stage repo, so the decoded
video comes out of the refiner rather than straight out of the SANA VAE.

- Model: `BBBBruce/SANA-WM_bidirectional-diffusers` — Stage-1 tree plus
  `refiner/` (transformer, connectors, Gemma-3 text encoder).
- Model storage: about 102 GB total. Provision roughly 180 GB of local disk or
  Hugging Face cache volume.
- GPU: one RTX PRO 6000 Blackwell 96 GB, measured — see
  [Two-stage GPU sizing](#two-stage-gpu-sizing) for the numbers and the ceiling.

```bash
CUDA_VISIBLE_DEVICES=0 \
vllm serve BBBBruce/SANA-WM_bidirectional-diffusers \
  --omni \
  --host 0.0.0.0 \
  --port 8091
```

No deploy config here either, for the same reason as Stage 1: the refiner runs
inside the pipeline, so this is still a single-stage diffusion deployment and a
YAML's stage settings would not be applied.

The refiner runs by default. Pass `extra_params={"sana_wm_inprocess_refiner":
false}` to return the Stage-1 result instead — note that over `/v1/videos` that
yields a raw latent the endpoint cannot encode, so it is only useful from the
offline API.

Request it over `/v1/videos/sync` exactly like Stage 1 — the smoke request in
[Verification](#verification) works unchanged against this deployment; the MP4
it returns is the refiner's output rather than the SANA VAE's.

#### Two-stage GPU sizing

Measured on one RTX PRO 6000 Blackwell 96 GB (97887 MiB reported by
`nvidia-smi`, 95.6 GiB), default 1280x704 profile, everything on one device:

| Point in the deployment | Default | `--enable-layerwise-offload` |
| --- | --- | --- |
| Startup complete, no request in flight | 77083 MiB (75.3 GiB) | 36515 MiB (35.7 GiB) |
| Peak during a 161-frame, 60-step request | 86799 MiB (84.8 GiB) | 47227 MiB (46.1 GiB) |
| Wall clock for that request | 145 s | 149 s |

**By default, 161 frames at 1280x704 is the ceiling for this profile**, not a
comfortable operating point: the peak is 89% of the card, leaving about 10.8 GiB
of headroom, and a longer clip is expected to exceed it. Lower `num_frames`,
`width`, or `height` before serving longer requests on a 96 GB card, or use
layerwise offload — it streams both DiTs' blocks for about 2 s on a 150 s
request and leaves ~49 GiB of headroom.

By default everything is loaded onto the device at startup — the pipeline builds
the refiner in its constructor so a deployment that will not fit fails at deploy
time rather than on the first request. Under offload or HSDP the refiner is
built on CPU instead and the backend places it.

Do not size the deployment from the 102 GB on disk. The Gemma-3 refiner text
encoder is stored as fp32 (`refiner/text_encoder/config.json` sets
`"dtype": "float32"`) but is loaded in the pipeline's runtime dtype, which is
bf16 on CUDA, so it costs half its on-disk size:

| Component | On disk | Device (bf16) |
| --- | --- | --- |
| refiner text encoder (Gemma-3) | 45.40 GiB fp32 | ~22.7 GiB |
| refiner transformer (LTX-2) | 35.17 GiB bf16 | ~34.5 GiB |
| Stage-1 transformer | 9.92 GiB | ~9.9 GiB |
| refiner connectors | 2.67 GiB bf16 | ~2.7 GiB |
| SANA VAE | 2.28 GiB | ~2.3 GiB |
| **Total** | **95.4 GiB** | **~72.1 GiB** |

The remainder of the measured 75.3 GiB is the Stage-1 `gemma-2-2b` text encoder
plus the CUDA context.

#### Splitting the two-stage model across GPUs

Both denoisers are declared as DiTs (`_dit_modules = ["transformer",
"refiner_transformer"]`), so the shared parallelism and offload machinery
reaches the LTX-2 refiner — the single largest component at ~34.5 GiB — the same
way it reaches Stage 1: tensor parallelism, HSDP sharding through the LTX-2
`_hsdp_shard_conditions`, and layerwise offload streaming its 48
`transformer_blocks`. Under offload or HSDP the refiner is built on CPU and the
backend places it, so startup no longer needs the full 75.3 GiB on one device.

`--enable-layerwise-offload` is the measured case above; the log line to look
for is:

```text
Applying layer-wise offloading on ['transformer', 'refiner_transformer']
Applying hooks on refiner_transformer (LTX2VideoTransformer3DModel)
Layer-wise offloading enabled on 48 layers (blocks)
```

Sequence parallelism is the exception. The refiner's self-attention keeps the
sink frames from attending to the frames after them, and that prefix boundary is
not a rank-local index once the sequence is sharded, so `sequence_parallel_size
> 1` raises rather than returning a wrong result.

#### Two-stage accuracy

`tests/e2e/accuracy/sana_wm/test_sana_wm_reference_similarity.py` gates the
two-stage pipeline against the NVlabs reference. Measured on one RTX PRO 6000
Blackwell 96 GB at 1280x704, cfg 5.0, seed 42, refiner 3-step distilled Euler,
with `SANA_WM_REF_NUM_FRAMES` / `SANA_WM_REF_STEPS` selecting the profile:

| Comparison | 9 frames / 20 steps | 161 frames / 60 steps (production) |
| --- | --- | --- |
| `SanaWmTwoStagesPipeline` vs NVlabs Stage-1 + refiner | 0.9709 SSIM, 36.50 dB | 0.8234 SSIM, 21.08 dB |
| `SanaWmPipeline` vs NVlabs Stage-1 (historical) | 0.9785 SSIM, 37.97 dB | 0.9059 SSIM, 23.60 dB |

Only the two-stage row is gated by a test. The two-stage pipeline drives the
same Stage-1 sampler and decodes through the same SANA VAE, so a Stage-1
regression surfaces there; the Stage-1 row is kept as the measurement taken
while a separate Stage-1 parity test existed. Pass
`SANA_WM_E2E_MODEL_CLASS=SanaWmPipeline` to the offline e2e to exercise
Stage-1 alone.

The two are not bit-identical because the native side samples in the Omni
worker subprocess while the reference samples in-process, so the initial
latents differ; these numbers are the reported evidence. Similarity falls off
with clip length because 161 frames of 60-step sampling accumulate that initial
divergence.

The test gates on 0.80 SSIM / 20.0 dB (`SANA_WM_REF_MIN_SSIM` /
`SANA_WM_REF_MIN_PSNR` override them). That is set just under the two-stage
production row — the worst of the four — so it leaves 0.023 SSIM and 1.1 dB of
headroom there, while the 9-frame profile clears it by a wide margin. Only the
production profile is close enough to the gate to be worth watching.

Note the frame alignment when comparing by hand: the NVlabs `_refine` drops the
decoded sink anchor (`video[1:]`) while `SanaWmTwoStagesPipeline` keeps it, so
the native clip is one frame longer and native frame `i+1` corresponds to
reference frame `i`.

Moving the refiner onto `LTX2VideoTransformer3DModel.forward` changed its
numerics — that route uses the shared LTX-2 path (RoPE materialized in the
hidden-state dtype, the project's attention backend) where the previous
hand-rolled loop kept RoPE in fp32 and called `F.scaled_dot_product_attention`
directly. Both cells above are measured on the shared-forward implementation:
the production profile reproduces `mean_ssim=0.8234 mean_psnr=21.08dB` over
160 aligned frames, and the 9-frame profile lands at 0.9709 / 36.50 dB (the
hand-rolled loop scored 0.9765 / 38.59 dB there). The refiner reaches the
shared forward with a one-token dummy audio stream and both audio<->video
cross-attentions disabled — audio only enters video through a2v, so the video
output is byte-identical to a dedicated video-only path (verified: identical
MP4 checksums at the production profile).

#### Where the Stage-1 difference comes from

End-to-end SSIM says how far apart the two implementations end up, not which
module takes them there. The numbers below come from a module-level probe that
drove both DiTs from byte-identical inputs and reported per-block divergence as
a multiple of bf16 machine epsilon (`2**-8` = 3.9e-3), the scale below which a
difference is rounding rather than different math. The probe was a one-off
diagnostic and is not kept in-tree; it lives in this file's git history.

On the same box, one denoising step, 9 frames:

- Every shared embedder agrees. `x_embedder`, `t_embedder` and
  `plucker_embedder` are **bit-identical**; `y_embedder` and `attention_y_norm`
  land under 1x bf16 eps.
- Teacher-forced (each block handed the reference's own input, so the number is
  that block's own contribution), every block output is **below** bf16 eps —
  0.66x for GDN blocks, 0.44x for softmax ones. No block is wrong.
- Gated DeltaNet is not the outlier. On the attention path GDN blocks sit at
  1.35x bf16 eps against 0.89x for the softmax blocks — slightly noisier, same
  order. The largest per-module contribution is the `GLUMBConvTemp` FFN at
  2.86x, and `plucker_proj` is effectively exact at 0.54x.
- Free-running, the cumulative per-block drift stays flat at 1.6–2.0x bf16 eps
  across all 20 blocks — it does not compound — and the DiT output lands at
  4.67x (relL2 1.8e-2, cosine 0.99983).

So a single step is bf16-accurate everywhere and no module is individually
suspect; the end-to-end gap is that ~2% per-step difference amplified by the
sampling trajectory over 60 steps, which is also why the 161-frame numbers sit
below the 9-frame ones.
