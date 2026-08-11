# Wan2.2-Animate online serving

Character animation: drive a reference character image with a pre-processed
performance. vLLM-Omni supports **animation mode**; replacement mode
(compositing into a background video with a mask and the relighting LoRA) is
not supported.

## Prerequisites

The two driving videos must be produced beforehand by the upstream
[Wan-Animate `preprocess_data.py`](https://github.com/Wan-Video/Wan2.2). This
server deliberately does not bundle the pose-estimation and face-retargeting
models, matching how other conditioning videos are handled in this project.

The diffusers checkpoint is FP32 and about 69 GB on disk (cast to bf16 at load
time). A single 80 GB card runs it without offload.

## Start the server

```bash
bash examples/online_serving/animate/run_server.sh
```

Environment overrides: `MODEL`, `PORT` (default 8092), `TP`, `FLOW_SHIFT`.

## Send a request

```bash
IMAGE_URL=https://example.com/character.png \
POSE_VIDEO_URL=https://example.com/pose.mp4 \
FACE_VIDEO_URL=https://example.com/face.mp4 \
bash examples/online_serving/animate/run_curl_animate.sh
```

## Request shape

Wan-Animate needs three media inputs, and the video API has fields for only
two of them:

| Input | Field | Notes |
|---|---|---|
| Reference character image | `image_reference` | Standard media field |
| Pose (skeleton) video | `video_reference` | Standard media field; the API allows it together with `image_reference` |
| Face video | `extra_params.face_video` | No HTTP field exists for a second video, so this is a model-specific extra |

`extra_params` also accepts `segment_frame_length` (frames per segment, must be
4n+1, default 77) and `prev_segment_conditioning_frames` (1 or 5, default 1),
which control long-video chunking.

Frame count follows `num_frames`. Videos longer than one segment are generated
clip-by-clip, with each segment conditioned on the tail of the previous one.

## Notes

- CFG is off by default (`guidance_scale=1.0`), matching the reference
  implementation. Raising it enables a negative branch that drops the text
  prompt and blanks the face signal.
- Cache acceleration (Cache-DiT / TeaCache) is not available: the face adapter
  is injected between backbone blocks, which a generic block-level cache would
  skip.
- Sequence parallelism and pipeline parallelism are rejected at load time — the
  face adapter attends frame-by-frame, so a sequence shard would have to align
  with latent frame boundaries. Use tensor parallelism or CFG parallelism.
