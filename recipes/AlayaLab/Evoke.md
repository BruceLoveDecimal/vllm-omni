# Alaya-EVOKE real-time world model

## Summary

- Vendor: AlayaLab
- Model: `AlayaLab/Evoke` (`evoke/stage3_post_distillation`); `AlayaLab/Evoke-Turbo` shares the architecture and recipe
- Task: Camera-controlled image-to-video world exploration, streamed chunk by chunk (the text-to-video path exists but has not been validated on GPU)
- Mode: Online serving over `WS /v1/realtime/video` with step execution
- Maintainer: Community

## When to use this recipe

Use this recipe to serve the CFG-free real-time EVOKE checkpoint and drive it
with live camera and prompt updates. Every chunk is a three-stage pyramid with
one UniPC step per stage, and produces 1.5 s of 640x384 video at 24 fps.
Camera motion reaches the model only through a geometry warp: ViGeo predicts
depth for the frames generated so far, the frames are kept in a point-cloud
bank, and the bank is re-rendered from the requested camera path for the next
chunk.

The Stage-1 checkpoints (`evoke/stage1_*`, 50-step CFG) are not supported; they
carry camera-control tensors that the loader rejects.

## References

- Upstream repository: <https://github.com/AlayaLab/Evoke>
- Model weights: <https://huggingface.co/AlayaLab/Evoke>
- Depth model: <https://huggingface.co/pkqbajng/ViGeo> (CC-BY-NC-4.0)
- Protocol: [`docs/serving/streaming_video_output_api.md`](../../docs/serving/streaming_video_output_api.md)
- Client: [`examples/online_serving/streaming_video_generation`](../../examples/online_serving/streaming_video_generation/README.md)

## Setup

`AlayaLab/Evoke` holds several checkpoints and is not a loadable pipeline root.
Assemble a model directory (about 85 GB; components are symlinked into the
Hugging Face cache):

```bash
python examples/offline_inference/evoke/download_evoke.py --output /models/evoke-realtime
```

Install ViGeo from its source tree. Its wheel also installs top-level `utils`
and `videoldcm` packages into site-packages; the `utils` package fails to
import and shadows any other module named `utils`, so install it into a
dedicated environment:

```bash
pip install --no-deps /path/to/ViGeo
```

## Hardware Support

## GPU

### 1x NVIDIA RTX PRO 6000 Blackwell (96 GB)

#### Environment

- Python 3.12, torch 2.13 (CUDA 13.0), vLLM 0.31.0
- Weights: `AlayaLab/Evoke-Turbo` transformer with the `evoke-base` VAE, UMT5 and ViGeo 1.1

#### Command

```bash
vllm serve /models/evoke-realtime \
  --omni \
  --deploy-config vllm_omni/deploy/evoke_realtime.yaml \
  --port 8000
```

The deploy config defaults to the checkpoint recipe (3 steps, guidance 1.0,
24 fps, 640x384; sizes must be multiples of 16) and serves one session at a time, since the ViGeo cache, point-cloud
bank and causal VAE cache are per session.

Start an image-conditioned session that walks forward:

```bash
cd examples/online_serving/streaming_video_generation

python streaming_video_client.py \
  --model /models/evoke-realtime \
  --prompt "A sea turtle glides through a sunlit coral reef." \
  --image-reference /path/to/first_frame.jpg \
  --fps 24 --num-frames 1320 --seed 42 \
  --camera-updates '[{"at": 0, "camera": {"mode": "velocity", "data": {"translation": [0.0, 0.0, 0.0208], "rotation": [0.0, 0.0, 0.0, 1.0]}}}]' \
  --output evoke_walk.mp4
```

#### Verification

The client prints one line per streamed chunk and remuxes the stream into a
progressive MP4 after `session.done`. Measured with Evoke-Turbo weights,
image-conditioned, with camera and prompt updates during the session:

| Session | Chunks | Frames | Wall time | Per chunk | GPU memory |
| --- | --- | --- | --- | --- | --- |
| 1 min (`num_frames=1320`) | 40 | 1437 | 4.6 min | 6.5 s | 57 GB |
| 3 min (`num_frames=3960`) | 120 | 4317 | 14 min | 6.8 s | 57 GB |
| 5 min (`num_frames=6600`) | 200 | 7197 | 24 min | 7.2 s | 60 GB |

One GPU runs at about 0.22x real time; each chunk is 1.5 s of video. Upstream
reaches interactive rates with four-GPU sequence parallelism, which this
pipeline does not support yet.

## Request semantics

- Length: a session streams `ceil(num_frames / 33)` chunks of 36 frames (the
  last one 33), the same rounding as the upstream launchers. `num_frames=1320`
  gives 40 chunks, 1437 frames, about 60 s.
- Camera: `session.interaction` camera payloads use the shared Unity frame
  (+x right, +y up, +z forward). A `velocity` delta is applied once per output
  frame (1/24 s), not once per latent frame. The upstream interactive UI moves
  at 0.5 units/s and turns at 15 deg/s, so a forward walk is
  `translation [0, 0, 0.0208]` and a right turn is a rotation of 0.625 deg about
  +y per frame. The `actions` (WASD) helper in the example client is tuned for
  LingBot and moves about 2.4x faster and turns about 10x faster than that.
- Requests may pass `extra_params.camera_c2ws` (OpenCV camera-to-world, one
  4x4 per output frame) instead of interactive camera updates, and
  `camera_intrinsics` or `camera_hfov_degrees` (default 70 deg on a 1280x720
  source, centre-cropped to 640x384).
- Prompts can change at chunk boundaries through `session.interaction`.
- Sampling knobs are fixed by the distilled checkpoint: requests with
  `guidance_scale > 1`, a step count other than 3, an fps other than 24 or
  pyramid/history overrides are rejected.

## Known limitations

- Single GPU only; tensor and sequence parallelism are not wired up yet.
- `evoke_geometry_max_bank_frames: 0` keeps every ingested frame for the warp,
  matching upstream. Memory grows with session length; set a positive bound for
  very long sessions at the cost of what the warp can recall.
- Cache acceleration (TeaCache, cache-dit) is disabled for this pipeline.
