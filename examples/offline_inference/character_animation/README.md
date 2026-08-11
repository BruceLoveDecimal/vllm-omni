# Character Animation (Wan2.2-Animate)

This example generates character animation and character replacement videos with the Wan2.2-Animate pipeline using vLLM-Omni's offline inference API.

Wan-Animate has two modes, selected by which inputs you pass — there is no mode flag:

| Mode | Inputs | Result |
|------|--------|--------|
| Animation | reference image + pose video + face video | The reference character performs the driving motion; the background is generated |
| Replacement | the above + background video + mask video | The person in the background video is swapped for the reference character, keeping the original background |

Long videos are produced segment by segment. Each segment is conditioned on the last `--prev-segment-conditioning-frames` decoded frames of the previous one, so the motion stays continuous across segment boundaries.

## Prerequisites

The pose and face videos must be **pre-extracted**. This pipeline consumes them; it does not run pose estimation or face retargeting. Use the Wan-Animate `preprocess_data.py` tooling from the upstream repository to produce them from a driving video.

```bash
pip install decord
```

## Animation mode

```bash
python character_animation.py \
  --model Wan-AI/Wan2.2-Animate-14B-Diffusers \
  --image character.png \
  --pose-video pose.mp4 \
  --face-video face.mp4 \
  --prompt "a person dancing in a studio" \
  --height 720 --width 1280 \
  --num-inference-steps 20 \
  --output animation.mp4
```

## Replacement mode

```bash
python character_animation.py \
  --model Wan-AI/Wan2.2-Animate-14B-Diffusers \
  --image character.png \
  --pose-video pose.mp4 \
  --face-video face.mp4 \
  --background-video source.mp4 \
  --mask-video mask.mp4 \
  --prompt "a person walking through a market" \
  --output replacement.mp4
```

The mask video marks — in white — the region the new character should occupy, and in black the region to preserve.

## Multi-GPU

Tensor parallelism shards the DiT across GPUs:

```bash
python character_animation.py ... --tensor-parallel-size 4
```

CFG parallelism only helps when CFG is actually on (`--guidance-scale > 1`). Wan-Animate is trained to run without CFG, so the default `--guidance-scale 1.0` runs a single branch and `--cfg-parallel-size 2` would be idle.

**Sequence parallelism is not supported for Wan-Animate.** The face adapter cross-attends per latent frame and needs frame-aligned token shards, which the generic token-axis sharding does not guarantee; passing `--ring-degree`/`--ulysses-degree` > 1 raises rather than producing wrong output.

**Cache-DiT is not supported for Wan-Animate** either: the face adapter injects conditioning between transformer blocks, which cached steps would skip.

## Memory

- `--motion-encode-batch-size` controls how many face frames the motion encoder processes at once. Lowering it trades speed for peak memory.
- `--enable-layerwise-offload` and `--enable-cpu-offload` move DiT blocks and encoders off the GPU when idle.
- `--vae-use-slicing` / `--vae-use-tiling` reduce VAE decode memory.
