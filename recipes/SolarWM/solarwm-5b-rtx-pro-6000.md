# SolarWM-5B Stage2 — RTX PRO 6000 Blackwell

## Summary

Native SolarWM Wan2.2-5B Stage2 image/camera-to-video generation through
`vllm_omni.entrypoints.omni.Omni` and the shared image-to-video example.
The provided profile uses one RTX PRO 6000 Blackwell Server Edition (96 GB),
resident components, eager execution, and Torch SDPA.

## Supported model contract

| Item | Contract |
| --- | --- |
| Model | `junchaoh-cs/SolarWM`: `SolarWM-5B-base` + `SolarWM-5B-sgf-stage2-81f`, EMA role |
| Input | One first-frame RGB image and a text prompt; optional local camera NPZ |
| Geometry | Default 480×864; dimensions must be positive multiples of 32 |
| Sampling | Exactly four denoising evaluations per three-latent chunk, guidance 1 |
| History | Five clean chunks (15 latent frames), no attention sink, window-relative RoPE |
| Output | RGB video at 16 fps; no audio |
| Duration | Arbitrary positive requested frame count; round the latent horizon up to a full chunk and trim decoded excess frames |
| First frame | Bilinear resize and center crop, then deterministic Wan VAE encoding |
| Runtime | Native pipeline; no runtime import from the SolarWM repository |

The Stage0.5, AnyFlow Stage1, 14B, LTX and MiniMax variants are outside this profile.
A longer rollout does not guarantee preservation of scene identity indefinitely.

## References

- [Pinned upstream source](https://github.com/Junchao-cs/SolarWM/tree/a3a3fac16466102a2b97df867f7703df7172cb2a)
- [Released weights](https://huggingface.co/junchaoh-cs/SolarWM/tree/fe1587a9392bb91df5c90bcbe849064a5e88c546)
- [Native architecture and validation](../../vllm_omni/diffusion/models/solarwm/INTEGRATION.md)
- [Shared image-to-video example](../../examples/offline_inference/image_to_video/README.md)
- [Model support](../../docs/models/supported_models.md) and [feature matrix](../../docs/user_guide/diffusion_features.md)

## Checkpoint preparation

Install vLLM-Omni and its dependencies, including `ftfy`. Obtain the two release
folders at revision `fe1587a9392bb91df5c90bcbe849064a5e88c546`, following the
upstream weight access requirements. Together they occupy about 71 GB.
The assembly command creates a small separate index without copying or changing weights:

```bash
python examples/offline_inference/solarwm/prepare_checkpoint.py \
  --base /path/to/SolarWM-5B-base \
  --stage /path/to/SolarWM-5B-sgf-stage2-81f \
  --output /path/to/SolarWM-Omni
```

The index stores absolute paths; reassemble it after relocating the release folders.

## Hardware and software

| Item | Qualification environment |
| --- | --- |
| Accelerator | 1× NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB reported memory |
| Parallelism | TP/SP/PP/CFG all 1; no inter-device communication |
| Driver / PyTorch | 590.44.01 / 2.13.0+cu130 |
| Python / vLLM / Diffusers | 3.12 / 0.28.0 / 0.40.0 |
| vLLM-Omni | Base `6fb7b0a05` plus the native SolarWM changes |
| Precision | BF16 DiT; FP32 UMT5 and VAE weights; BF16 autocast during VAE decode |

## Command

Run from the vLLM-Omni checkout. `PYTHONPATH` ensures worker processes use this
checkout when another copy is installed in the environment.

```bash
PYTHONPATH="$PWD" DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA \
VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT=7200 \
python examples/offline_inference/image_to_video/image_to_video.py \
  --model /path/to/SolarWM-Omni --image /path/to/first-frame.jpg \
  --prompt "The camera slowly pulls back as the scene moves naturally." \
  --height 480 --width 864 --num-frames 1920 --fps 16 \
  --num-inference-steps 4 --guidance-scale 1 --seed 42 --enforce-eager \
  --output solarwm_120s.mp4
```

This requests 120 seconds: 483 latent frames produce 1929 decoded frames, of
which the first 1920 are returned. It uses newly generated chunks and one
continuous temporal VAE cache; it does not repeat a short clip.
For an 81-frame smoke, replace `--num-frames 1920` with `--num-frames 81`.
HTTP serving is not qualified by this recipe.

### Camera control

Omitting camera input gives a static camera. To pass a trajectory, add:

```bash
--extra-body '{"camera_path":"/absolute/path/camera.npz"}'
```

The NPZ must contain either:

- `c2w`: absolute camera-to-world poses `[pixel_frames,4,4]`. Cover the complete
  rounded generation horizon. For 1920 requested frames, supply 1929 poses.
  The model selects source frames `0,1,5,9,...`, rebases to the first camera,
  and converts to W2C in FP32.
- `viewmats`: already relative W2C poses `[latent_frames,4,4]`; exactly 483 for
  this two-minute command.

Use the release's normalized focal lengths (`fx≈0.50505`, `fy≈0.89787`). Custom
intrinsics are not exposed. A CPU request generator is recreated on the model
device from its initial seed to preserve the release's CUDA noise distribution.

## Supported features

| Feature | Status |
| --- | --- |
| Single GPU, Torch SDPA, eager | Qualified |
| Rolling clean KV history | Native model semantics, qualified across eviction |
| [Parallel execution](../../docs/user_guide/diffusion/parallelism/overview.md) | Unsupported |
| [CPU/layerwise offload](../../docs/user_guide/diffusion/cpu_offload.md) | Unsupported in this profile |
| [Cache-DiT](../../docs/user_guide/diffusion/cache_acceleration/cache_dit.md) | Unsupported; unrelated to the model's exact causal KV cache |
| Quantization, LoRA, step execution, spatial VAE tiling | Not qualified |

## Validation

Prerequisites for CPU tests: repository dependencies, compatible vLLM, PyTorch,
pytest, and ftfy. No weights are needed:

```bash
python -m pytest -o addopts='' tests/diffusion/models/solarwm/test_inputs.py -q
python -m pytest -o addopts='' tests/diffusion/models/solarwm \
  -m 'core_model and cpu' --run-level core_model -q
```

Released-weight regression uses the shared Omni test client and covers window
eviction. It requires the assembled checkpoint, an image, and sufficient CUDA memory:

```bash
SOLARWM_MODEL=/path/to/SolarWM-Omni SOLARWM_IMAGE=/path/to/first-frame.jpg \
PYTHONPATH="$PWD" DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA \
python -m pytest -o addopts='' tests/e2e/offline_inference/test_solarwm_expansion.py \
  -m 'full_model and cuda and cards_1' --run-level full_model -q
```

The nightly X2V job is conditional on provisioning `SOLARWM_MODEL` and
`SOLARWM_IMAGE`. Its H100 marker is a CI resource selection, not an H100
performance qualification.

The pinned-reference CUDA comparisons use the same SDPA kernel on both sides:
16 full 5B forwards including eviction had maximum error 0; UMT5 and VAE decode
also had maximum error 0; VAE encode had maximum error `2.3841858e-7`.
The complete 24-latent rollout also matched the upstream sampler bitwise.
The reference comparison scripts are under `tests/diffusion/models/solarwm`.

## Two-minute rollout evidence and limitation

The shared CLI generated 1920 distinct decoded frames at 864×480 and 16 fps
(120.000 seconds) on the stated single-GPU environment. The moving-camera
cat run took 176.17 seconds inside Omni generation; the static-camera cat run
took 176.00 seconds. These measurements exclude model loading and MP4 export.
Peak driver-reported GPU memory was 52137 MiB (about 50.9 GiB); this is device
usage sampled once per second, not PyTorch allocated/reserved memory.

**Visual stability is not qualified.** The moving/static-camera cat runs and an independent static interior input develop severe artifacts
around 20–30 seconds and remain degraded later in the rollout. Frame uniqueness
and successful numerical checks establish execution, not usable long-video
quality. Do not treat this profile as a quality guarantee for two-minute clips.

A complete 483-latent comparison against the original upstream `CausalWanModel`
and Stage2 sampler was bitwise equal (`max_abs=0`) for the static-camera cat
input, with both sides using SDPA. This rules out native rollout differences
for that measured case; it does not establish quality on other inputs or
equivalence to unmeasured attention kernels.
