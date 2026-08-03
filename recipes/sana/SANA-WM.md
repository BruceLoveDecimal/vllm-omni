# SANA-WM

> Camera-controllable first-frame image-to-video world model.

## Summary

- Vendor: Efficient-Large-Model / NVlabs SANA
- Model: `BBBBruce/SANA-WM_bidirectional-diffusers` (standard diffusers layout, converted offline from the NVlabs release)
- Task: First-frame image-to-video generation with camera control
- Mode: Online serving with the OpenAI-compatible video API
- Model weights: TODO GB for the Stage-1 transformer and VAE
- Local disk: reserve TODO GB for the Hugging Face cache and runtime artifacts
- Recommended GPU: TODO GB or larger CUDA GPU
- Maintainer: Community

## When to use this recipe

Use this recipe when you want to serve SANA-WM through `/v1/videos` or
`/v1/videos/sync`. The model takes a text prompt, a first-frame image, and
either an action DSL string or explicit camera poses. vLLM-Omni serves the
SANA-WM Stage-1 DiT, decoded through the SANA VAE. The optional LTX-2 refiner
stage is not supported by this PR; it is a planned follow-up.

## References

- Upstream model card: <https://huggingface.co/Efficient-Large-Model/SANA-WM_bidirectional>
- Online serving example: [`examples/online_serving/sana_wm/README.md`](../../examples/online_serving/sana_wm/README.md)
- Deploy config: [`vllm_omni/deploy/sana_wm.yaml`](../../vllm_omni/deploy/sana_wm.yaml)
- Video API: [`docs/serving/videos_api.md`](../../docs/serving/videos_api.md)

## Hardware Support

## GPU

### 1x NVIDIA RTX PRO 6000 Blackwell 96GB

#### Capacity

- Model storage: the Stage-1 transformer and VAE are about TODO GB.
- Disk sizing: provision about TODO GB of local disk or Hugging Face cache volume
  so the model, temporary downloads, and generated artifacts fit without cache
  eviction.
- GPU sizing: the default 1280x704, 161-frame, 60-step serving profile peaks at
  about TODO GB of device memory. On smaller GPUs, lower `width`, `height`, or
  `num_frames` before serving production requests.

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

The repo ships the standard Diffusers layout (`model_index.json` + `transformer/`,
`vae/`). SANA-WM still needs the deploy YAML to wire its omni serving
stages, so pass it through `--deploy-config`.

```bash
CUDA_VISIBLE_DEVICES=0 \
vllm serve BBBBruce/SANA-WM_bidirectional-diffusers \
  --omni \
  --deploy-config vllm_omni/deploy/sana_wm.yaml \
  --host 0.0.0.0 \
  --port 8091
```

The deploy YAML path is `vllm_omni/deploy/sana_wm.yaml`. Some wrappers document
this target as `vllm serve vllm_omni/deploy/sana_wm.yaml`; with the standard
vLLM-Omni CLI, pass it through `--deploy-config` as shown above.

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
  --form-string 'sana_wm={"action":"w-8","translation_speed":0.055,"rotation_speed_deg":1.2,"intrinsics":{"fx":640,"fy":640,"cx":640,"cy":352}}' \
  -o sana_wm_smoke.mp4
```

For a production-length request using the deploy defaults, use an action length
that matches `num_frames - 1`:

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
  --form-string 'sana_wm={"action":"w-160","translation_speed":0.055,"rotation_speed_deg":1.2,"intrinsics":{"fx":640,"fy":640,"cx":640,"cy":352}}' \
  -o sana_wm_output.mp4
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
  pitch/yaw rotation.
- Explicit camera control (alternative to `action`): pass
  `"camera": {"poses": [...]}` where `poses` is a list of `num_frames`
  camera-to-world 4x4 matrices (row-major, OpenCV `+X right, +Y down, +Z forward`
  convention), e.g.
  `sana_wm={"camera":{"poses":[[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], ...]},"intrinsics":{...}}`.
  Most callers should prefer `action`; explicit poses exist for callers that
  already have a per-frame trajectory.
- Explicit `intrinsics` are recommended and take the mapping form
  `{"fx":640,"fy":640,"cx":640,"cy":352}` (for 1280x704). This `{fx,fy,cx,cy}`
  mapping is the only accepted intrinsics form; omit `intrinsics` to derive them
  from the output resolution.
- The deploy config sets `sana_wm_output_type: "np"` so the video API returns
  decoded MP4 bytes; the pipeline default is `latent`, which serving cannot
  render.
