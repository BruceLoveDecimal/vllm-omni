# Wan2.2-Animate (character animation)

Drives a reference character image with a driving performance, using
[Wan2.2-Animate-14B](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B-Diffusers).
vLLM-Omni currently supports **animation mode** (reference image + driving
motion). Replacement mode (compositing the character into a background video
with a mask and the relighting LoRA) is not supported yet.

## Inputs

The pipeline takes three inputs:

| Input | Description |
|-------|-------------|
| `image` | Reference character image. Scaled to fit the target resolution and centered on a black canvas. |
| `pose_video` | Pre-processed skeleton video, one frame per output frame. |
| `face_video` | Pre-processed face video, resized internally to 512x512. |

**Pose and face videos must be prepared beforehand.** Wan-Animate relies on
external pose estimation (DWPose) and face retargeting models, which live in the
upstream [Wan-Animate repo](https://github.com/Wan-Video/Wan2.2) as
`preprocess_data.py`. vLLM-Omni does not run them; it expects their output, the
same convention the repo uses for other conditioning videos.

```bash
# In the upstream Wan2.2 repo, produce pose.mp4 and face.mp4:
python ./wan/modules/animate/preprocess/preprocess_data.py \
    --ckpt_path ./Wan2.2-Animate-14B/process_checkpoint \
    --video_path ./driving_video.mp4 \
    --refer_path ./character.png \
    --save_path ./preprocessed \
    --resolution_area 1280 720
```

## Generate

```bash
python wan_animate.py \
    --model Wan-AI/Wan2.2-Animate-14B-Diffusers \
    --image character.png \
    --pose-video preprocessed/src_pose.mp4 \
    --face-video preprocessed/src_face.mp4 \
    --prompt "A person dancing in a bright studio" \
    --output animate_output.mp4
```

Resolution defaults to the reference image's aspect ratio at roughly 720p, and
both dimensions must be multiples of 16. Pass `--height`/`--width` to override.

## Long videos

Video is generated in segments of `--segment-frame-length` frames (must be
`4n + 1`, default 77). Each segment after the first is conditioned on the last
`--prev-segment-conditioning-frames` decoded frames of the previous one
(1 or 5, default 1), which is what keeps motion continuous across segment
boundaries. The driving videos determine the total length; pass `--num-frames`
to generate only a prefix.

## Guidance

Wan-Animate runs **without CFG** by default (`--guidance-scale 1.0`), matching
the reference implementation. Raising it above 1.0 enables a negative branch
that drops the text prompt and blanks the face signal, at roughly double the
compute per step.

## Multi-GPU and memory

| Flag | Notes |
|------|-------|
| `--tensor-parallel-size N` | Shards the backbone's linear layers. |
| `--cfg-parallel-size 2` | Splits the CFG branches; only useful with `--guidance-scale > 1`. |
| `--vae-patch-parallel-size N` | Shards VAE decode. |
| `--use-hsdp` | Shards transformer weights to cut per-GPU memory. |
| `--enable-cpu-offload` / `--enable-layerwise-offload` | Trades speed for memory on smaller cards. |

Sequence parallelism (`--ulysses-degree` / `--ring-degree`) and Cache-DiT are
**not supported**: the face adapter attends within each latent frame, so
sequence shards would cut across frame boundaries, and it is injected between
backbone blocks, where a generic block cache would skip it. Both are rejected
explicitly rather than silently producing wrong output.

The bf16 model needs roughly 48 GB for weights alone, so a single 80 GB card is
the comfortable configuration; 48 GB cards work with CPU offload enabled.
