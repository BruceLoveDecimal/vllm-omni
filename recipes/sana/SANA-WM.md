# SANA-WM

> Camera-controllable first-frame image-to-video world model.

## Summary

- Vendor: Efficient-Large-Model / NVlabs SANA
- Model: `Efficient-Large-Model/SANA-WM_bidirectional`
- Task: First-frame image-to-video generation with camera control
- Mode: Online serving with the OpenAI-compatible video API
- Model weights: approximately 102 GB for the full SANA-WM pipeline with refiner
- Local disk: reserve approximately 180 GB for the Hugging Face cache, refiner
  weights, and runtime artifacts
- Recommended GPU: 1x NVIDIA RTX PRO 6000 Blackwell 96 GB, or a larger CUDA GPU
- Maintainer: Community

## When to use this recipe

Use this recipe when you want to serve SANA-WM through `/v1/videos` or
`/v1/videos/sync`. The model takes a text prompt, a first-frame image, and
either an action DSL string or explicit camera poses. vLLM-Omni serves the
two-stage pipeline: SANA-WM Stage-1 DiT plus the bundled LTX-2 refiner.

## References

- Upstream model card: <https://huggingface.co/Efficient-Large-Model/SANA-WM_bidirectional>
- Online serving example: [`examples/online_serving/sana_wm/README.md`](../../examples/online_serving/sana_wm/README.md)
- Deploy config: [`vllm_omni/deploy/sana_wm.yaml`](../../vllm_omni/deploy/sana_wm.yaml)
- Video API: [`docs/serving/videos_api.md`](../../docs/serving/videos_api.md)

## Hardware Support

## GPU

### 1x NVIDIA RTX PRO 6000 Blackwell 96GB

#### Capacity

- Model storage: the SANA-WM model weights plus the bundled refiner are about
  102 GB.
- Disk sizing: provision about 180 GB of local disk or Hugging Face cache volume
  so the model, refiner, temporary downloads, and generated artifacts fit
  without cache eviction.
- GPU sizing: use one RTX PRO 6000 Blackwell 96 GB class GPU or larger for the
  default 1280x704, 161-frame, 60-step serving profile below. On smaller GPUs,
  lower `width`, `height`, `num_frames`, or refiner settings before serving
  production requests.

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

SANA-WM needs the deploy YAML because the Hugging Face repo does not include a
standard Diffusers `model_index.json`.

```bash
CUDA_VISIBLE_DEVICES=0 \
vllm serve Efficient-Large-Model/SANA-WM_bidirectional \
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

- `input_reference` is required for the first frame. Use `image_reference` only
  when you need a JSON-safe image URL or data URL instead of a multipart upload.
- `sana_wm` must provide exactly one of `action` or `camera`.
- Action strings use comma-separated `<keys>-<duration>` segments. Supported
  keys are `w`, `a`, `s`, `d` for translation and `i`, `j`, `k`, `l` for
  pitch/yaw rotation.
- Explicit `intrinsics` are recommended. The mapping form is
  `{"fx":640,"fy":640,"cx":640,"cy":352}` for 1280x704 examples.
- The deploy config enables the in-process LTX-2 refiner so the video API
  returns decoded MP4 bytes rather than Stage-1 latents.
