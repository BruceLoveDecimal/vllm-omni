# Wan2.2-Animate-2

> Character animation: retarget a driving video's motion onto a reference person

## Summary

- Vendor: Wan-AI
- Models: `Wan-AI/Wan2.2-Animate-2-14B-Diffusers` (base), `Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers` (distilled)
- Task: Character animation (motion / expression retargeting)
- Mode: Offline inference
- Maintainer: Community

## When to use this recipe

Use this recipe to animate the person in a reference image with the motion and
expression of a driving video. Unlike Wan2.2-Animate (v1), Animate-2 takes the
**raw driving video**: no pose-skeleton extraction and no face crop are needed.

Two configurations are provided:

1. **Distilled, single GPU**: 10 steps, no CFG. The fastest way to get output.
2. **Base, TP=2**: 40 steps with CFG 3.0. Higher quality at roughly 8x the compute.

## References

- Model cards: <https://huggingface.co/Wan-AI/Wan2.2-Animate-2-14B-Diffusers>,
  <https://huggingface.co/Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers>
  (mirrored on ModelScope under the same ids)
- Upstream inference code: <https://github.com/Wan-Video/Wan-Animate-2>
- Example reference assets: <https://github.com/Wan-Video/Wan-Animate-2/tree/main/examples>

## Checkpoint layout

Both releases ship in the Diffusers layout with a standard `model_index.json`
(`_class_name: WanAnimate2Pipeline`), so no `--model-class-name` is needed.
The pipeline tells the distilled release apart from the base one through its
`modular_model_index.json`.

```
Wan2.2-Animate-2-14B-Diffusers/
├── model_index.json
├── modular_model_index.json
├── transformer/            # 4 safetensors shards, 32.8 GB
├── text_encoder/ tokenizer/
├── image_encoder/ image_processor/
├── vae/
└── scheduler/
```

## Hardware Support

## CUDA

### 1x NVIDIA H100 (80 GB): distilled

#### Environment

- OS: Linux
- Python: 3.10+
- Driver: NVIDIA driver with CUDA 12.x
- vLLM version: Match the repository requirements for your checkout
- vLLM-Omni version or commit: Use the commit you are deploying from

#### Prerequisites

None

#### Command

```bash
python examples/offline_inference/image_to_video/image_to_video.py \
  --model Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers \
  --image reference.png \
  --video driving.mp4 \
  --prompt "static background." \
  --height 800 --width 640 --fps 24 --num-frames 241 \
  --num-inference-steps 10 --guidance-scale 1.0 \
  --seed 42 --output animate2_distilled.mp4
```

### 2x NVIDIA H100 (80 GB): base with CFG

#### Command

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python examples/offline_inference/image_to_video/image_to_video.py \
  --model Wan-AI/Wan2.2-Animate-2-14B-Diffusers \
  --image reference.png \
  --video driving.mp4 \
  --prompt "static background." \
  --height 1280 --width 720 --fps 24 --num-frames 241 \
  --num-inference-steps 40 --guidance-scale 3.0 \
  --tensor-parallel-size 2 \
  --seed 42 --output animate2_base.mp4
```

## Notes

- **Steps and guidance follow the release.** The shared example defaults to the
  base settings (40 steps, CFG 3.0). Pass `--num-inference-steps 10
  --guidance-scale 1.0` for the distilled release; it is trained without CFG,
  and a guidance scale above 1 only doubles the cost.
- **Output length follows the driving video, capped by `--num-frames`.** The
  video is generated in 81-frame segments with a 1-frame overlap; segment count
  grows linearly with the driving video's duration, and so does runtime. The
  driving video is resampled to `--fps` before segmentation.
- **Reference-image aspect ratio wins.** `--height` / `--width` set the *target
  area*; the frame is letterboxed to the reference image's aspect ratio, aligned
  to 16 px, and cropped back on output. 640x800 is the Diffusers default area
  and fits the reference K/V cache of one segment on a single 80 GB card;
  720x1280 needs tensor parallelism.
- **Request-level extras** go through `--extra-body`: `prompt_ref` (prompt of
  the reference-extraction pass, default "人物动作的参考视频"),
  `segment_frame_length` (4k+1, default 81) and `max_driving_frames`.
- Animate-2 is a single-transformer model; there is no `--boundary-ratio` MoE
  split as in Wan2.2 T2V/I2V.
