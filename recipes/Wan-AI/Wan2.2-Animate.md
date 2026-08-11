# Wan2.2 Animate

> Character animation from a reference image plus a driving performance (Wan2.2 14B)

## Summary

- Vendor: Wan-AI
- Model: `Wan-AI/Wan2.2-Animate-14B-Diffusers`
- Task: Character animation (reference image + pose video + face video → video)
- Mode: Offline inference
- Maintainer: Community

## When to use this recipe

Use this recipe to animate a character image with motion taken from a driving
video. vLLM-Omni supports **animation mode**; replacement mode (compositing the
character into a background video with a mask and the relighting LoRA) is not
supported yet.

The driving inputs must be pre-processed. Wan-Animate depends on external pose
estimation and face retargeting models that vLLM-Omni deliberately does not
bundle — generate `pose` and `face` videos with the upstream repo first, the
same convention used for other conditioning videos.

## References

- Upstream model card: <https://huggingface.co/Wan-AI/Wan2.2-Animate-14B>
- Diffusers checkpoint: <https://huggingface.co/Wan-AI/Wan2.2-Animate-14B-Diffusers>
- Upstream preprocessing: <https://github.com/Wan-Video/Wan2.2>

## Hardware Support

## CUDA

### 1× NVIDIA H100/H20 (80 GB) or RTX PRO 6000 (96 GB)

#### Environment

- OS: Linux
- Python: 3.10+
- Driver: NVIDIA driver with CUDA 12.x
- vLLM-Omni version or commit: Use the commit you are deploying from

#### Prerequisites

The diffusers checkpoint is FP32 and about 69 GB on disk; weights are cast to
bf16 at load time. In bf16 the DiT is roughly 35 GB and the text and image
encoders add about 14 GB, so a single 80 GB card runs without offload. Budget
~70 GB of free disk for the download.

Produce the driving videos with the upstream preprocessing script:

```bash
python ./wan/modules/animate/preprocess/preprocess_data.py \
  --ckpt_path ./Wan2.2-Animate-14B/process_checkpoint \
  --video_path ./driving_video.mp4 \
  --refer_path ./character.png \
  --save_path ./preprocessed \
  --resolution_area 1280 720
```

#### Command

```bash
python examples/offline_inference/animate/wan_animate.py \
  --model Wan-AI/Wan2.2-Animate-14B-Diffusers \
  --image ./character.png \
  --pose-video ./preprocessed/src_pose.mp4 \
  --face-video ./preprocessed/src_face.mp4 \
  --prompt "A person dancing in a bright studio" \
  --num-inference-steps 20 \
  --output animate_output.mp4
```

### 1× 48 GB card (L40S / A6000)

Weights alone exceed 48 GB, so offloading is required:

```bash
python examples/offline_inference/animate/wan_animate.py \
  --model Wan-AI/Wan2.2-Animate-14B-Diffusers \
  --image ./character.png \
  --pose-video ./preprocessed/src_pose.mp4 \
  --face-video ./preprocessed/src_face.mp4 \
  --prompt "A person dancing in a bright studio" \
  --enable-layerwise-offload \
  --vae-use-tiling \
  --output animate_output.mp4
```

### 2× GPUs

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python examples/offline_inference/animate/wan_animate.py \
  --model Wan-AI/Wan2.2-Animate-14B-Diffusers \
  --image ./character.png \
  --pose-video ./preprocessed/src_pose.mp4 \
  --face-video ./preprocessed/src_face.mp4 \
  --prompt "A person dancing in a bright studio" \
  --tensor-parallel-size 2 \
  --output animate_output.mp4
```

## Notes

- **CFG is off by default** (`--guidance-scale 1.0`), matching the reference
  implementation. Values above 1.0 add a negative branch that drops the text
  prompt and blanks the face signal, roughly doubling per-step compute.
- **Long videos** are generated in segments of `--segment-frame-length` frames
  (`4n + 1`, default 77). Each subsequent segment is conditioned on the last
  `--prev-segment-conditioning-frames` decoded frames (1 or 5, default 1).
- **Resolution** must be a multiple of 16 in both dimensions; it defaults to the
  reference image's aspect ratio at roughly 720p.
- **Unsupported acceleration**: sequence parallelism (`--ulysses-degree`,
  `--ring-degree`) and Cache-DiT. The face adapter attends within each latent
  frame, so sequence shards would cut across frame boundaries, and it is
  injected between backbone blocks where a generic block cache would skip it.
  Both are rejected explicitly rather than degrading output silently.
