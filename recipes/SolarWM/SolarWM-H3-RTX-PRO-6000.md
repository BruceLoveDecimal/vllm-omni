# SolarWM-H3 — RTX PRO 6000 Blackwell

> Camera-controlled first-frame image-to-video with the 33B SolarWM-H3 causal student.

## Summary

- Vendor: SolarWM (Junchao Huang et al.)
- Model: [`junchaoh-cs/SolarWM-H3-33B`](https://huggingface.co/junchaoh-cs/SolarWM-H3-33B) — `SolarWM-h3-33B-base` (the MiniMax-H3 33B base in the Diffusers layout) plus the Stage-2 EMA LoRA package `SolarWM-h3-33B-sgf-stage2-158f`
- Task: first-frame image-to-video with a per-frame camera trajectory
- Mode: offline `Omni` API and OpenAI-compatible `/v1/videos` serving
- Hardware: 1x NVIDIA RTX PRO 6000 Blackwell 96 GB
- Maintainer: Community

## When to use this recipe

Use this recipe to generate camera-controlled videos from one image with the
released SolarWM-H3 Stage-2 student on a single 96 GB GPU. The student is the
MiniMax-H3 DiT with a rank-384 LoRA merged at load time, running a six-chunk
causal window with four denoiser evaluations per five-latent chunk.

## Supported model contract

| Item | Value |
| --- | --- |
| Inputs | one RGB first frame (stretched to 1344x768 with LANCZOS), a caption, `camera_c2w` = `num_frames` absolute 4x4 camera-to-world poses |
| Output | 1344x768 video, `num_frames` frames (default 158) at the requested `fps` (default 24), no audio |
| Frame count | at least 22; the rollout generates whole five-latent chunks and trims the decoded video to `num_frames` |
| Steps | fixed: 4 evaluations per chunk on the shift-12 rectified-flow grid; `num_inference_steps` is ignored |
| Guidance | none (CFG-distilled); `cfg_parallel_size` must stay 1 |
| Intrinsics | fixed to the checkpoint's normalized Wan focal lengths; user intrinsics are not read |
| Parallelism | tensor parallelism only; Ulysses/Ring sequence parallelism is not supported by the windowed attention |

Camera poses are made first-frame relative inside the pipeline, so any world
frame works. Latent `i` takes the pose of source frame
`17 * (i // 5) + (0, 1, 5, 9, 13)[i % 5]`; latents past the last frame reuse
its pose.

## References

- Paper and code: <https://github.com/Junchao-cs/SolarWM>
- Reference inference path: `solarwm.backends.minimax_h3.full_inference` (Stage-2 source-length rollout)
- Shared example: [`examples/offline_inference/image_to_video`](../../examples/offline_inference/image_to_video/README.md#solarwm-h3-camera-controlled-world-model)

## Hardware

- Accelerator: NVIDIA RTX PRO 6000 Blackwell, 96 GB
- Number of devices: 1
- Interconnect: PCIe
- Host memory: 128 GB or more. The Qwen3-VL encoder (50 retained layers, ~50 GB bf16) streams from pinned host memory, and the six-chunk raw KV window (~36 GB bf16 for 768x1344) also lives in pinned host memory by default.
- Disk: ~150 GB for the base checkpoint and the Stage-2 package
- Qualification scope: single-GPU bf16 offline generation at 768x1344 x 158 frames

## Software environment

- OS: Linux
- Python: 3.12
- Driver / runtime: NVIDIA driver 590 / CUDA 13
- vLLM: 0.29.0
- vLLM-Omni: this checkout (source install)
- diffusers 0.40.0 provides `AutoencoderKLMiniMaxH3` and `AutoencoderKLMiniMaxH3Audio`

## Checkpoint layout

Download both directories of the gated repository into one root; the pipeline
resolves the repository root on its own:

```bash
hf auth login
hf download junchaoh-cs/SolarWM-H3-33B \
  --include "SolarWM-h3-33B-base/**" "SolarWM-h3-33B-sgf-stage2-158f/**" \
  --local-dir /path/to/SolarWM-H3-33B
```

`SolarWM-h3-33B-base` is byte-identical to the root Diffusers layout of
`MiniMaxAI/MiniMax-H3` (transformer, vae, audio_vae, text_encoder, tokenizer,
processor), so an existing copy of that layout can be placed under the same
name. The audio branch is a fixed condition: the pipeline encodes the 158-frame
stereo silence with the audio VAE once at start-up, no support artifact is needed.

## Offline generation

```bash
python examples/offline_inference/image_to_video/image_to_video.py \
  --model /path/to/SolarWM-H3-33B \
  --image first_frame.png \
  --prompt "A slow dolly forward through a sunlit forest path." \
  --num-frames 158 --fps 24 --seed 42 \
  --diffusion-offload-config '{"mode": "layer", "components": ["text_encoder"]}' \
  --extra-body '{"camera_c2w": "/path/to/camera_c2w.npy"}' \
  --output solarwm_h3.mp4
```

`camera_c2w` accepts a nested `[num_frames][4][4]` list or a path to a
`.npy`, `.npz` (key `c2w`) or `.json` file. A static camera is
`np.repeat(np.eye(4)[None], num_frames, axis=0)`.

## Online serving

```bash
vllm serve /path/to/SolarWM-H3-33B \
  --omni --host 0.0.0.0 --port 8000 \
  --num-gpus 1 --tensor-parallel-size 1 \
  --diffusion-offload-config '{"mode": "layer", "components": ["text_encoder"]}' \
  --enforce-eager
```

```bash
curl -sS -X POST http://localhost:8000/v1/videos/sync \
  -H "Accept: video/mp4" \
  -F "prompt=A slow dolly forward through a sunlit forest path." \
  -F "input_reference=@first_frame.png;type=image/png" \
  -F "width=1344" -F "height=768" -F "num_frames=158" -F "fps=24" -F "seed=42" \
  --form-string "extra_params=$(python -c 'import json,numpy as np; print(json.dumps({"camera_c2w": np.load("camera_c2w.npy").tolist()}))')" \
  -o solarwm_h3.mp4
```

## Memory and runtime profile

Measured on this recipe's hardware with the offline command above
(768x1344, 158 frames, seed 42, bf16, text encoder streamed, KV window in
pinned host memory):

| Metric | Value |
| --- | --- |
| Peak device memory | see validation notes below |
| Generation time (10 chunks x 5 forwards) | see validation notes below |
| Host pinned memory | ~50 GB encoder + ~36 GB KV window |

Pass `"kv_cache_on_device": true` in `extra_params` / `--extra-body` to keep
the KV window on the GPU when the device has room (for example with tensor
parallelism over two GPUs); it removes the per-layer host copies.

## Validation

- L1 (CPU): `pytest tests/diffusion/models/solarwm_h3 -m "core_model and cpu"` covers the camera PRoPE algebra, the packed layout and rollout geometry, the Diffusers-to-native weight adaptation with LoRA merge, and the KV window.
- Reference parity: the pipeline draws its noise on the accelerator in the reference order, so `seed` reproduces the official SolarWM rollout's noise stream; compare generated latents (`output_type="latent"`) with the reference `*.latents.safetensors` from `solarwm infer` on the same image, caption and trajectory.

## Limitations

- Sequence parallelism, Cache-DiT and step execution are not supported.
- The resolution is fixed at 768x1344; other sizes are rejected.
- Audio is a fixed silence condition and is not generated.
