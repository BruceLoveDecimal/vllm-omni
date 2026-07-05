# SANA-WM Online Serving

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/sana_wm>.


This example shows how to serve SANA-WM for first-frame image-to-video
generation with camera control. SANA-WM is primarily exposed through the video
serving API because every request needs a first-frame image plus an action or
camera trajectory.

For the video endpoint contract, see
[Videos API](https://github.com/vllm-project/vllm-omni/tree/main/docs/serving/videos_api.md). For deployment guidance, see
the [SANA-WM recipe](https://github.com/vllm-project/vllm-omni/tree/main/recipes/Efficient-Large-Model/SANA-WM.md).

## Start Server

SANA-WM uses a deploy config because the Hugging Face repository does not ship a
standard Diffusers `model_index.json`. Use the model ID as the positional
argument and pass the deploy YAML explicitly:

```bash
vllm serve Efficient-Large-Model/SANA-WM_bidirectional \
  --omni \
  --deploy-config vllm_omni/deploy/sana_wm.yaml \
  --host 0.0.0.0 \
  --port 8091
```

The deploy target is `vllm_omni/deploy/sana_wm.yaml`. If your deployment wrapper
accepts a deploy YAML as the positional target, the shorthand is:

```bash
vllm serve vllm_omni/deploy/sana_wm.yaml
```

With the standard vLLM-Omni CLI, prefer the `--deploy-config` form above.

## Sync API

`POST /v1/videos/sync` blocks until generation finishes and returns the MP4
bytes directly. The example below uses a short 9-frame clip for a quick smoke
test. Production-length clips work with the deploy defaults: the native token
cap now covers the model's full envelope (up to 321 frames at 704x1280). Raise
`sana_wm_native_max_tokens` through `extra_params` only for requests beyond it.

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
  -o sana_wm_output.mp4
```

## Camera Control Payload

Pass SANA-WM camera controls in the multipart `sana_wm` field as JSON. The
payload must contain exactly one of:

- `action`: a comma-separated action DSL string, such as `w-8` or `w-40,l-20`.
- `camera`: explicit camera poses, usually `{"format":"c2w_4x4","poses":[...]}`.

The action DSL uses `<keys>-<duration>` segments. Supported keys are `w`, `a`,
`s`, `d` for translation and `i`, `j`, `k`, `l` for pitch/yaw rotation. For
example, `w-8` produces 8 forward-motion steps, which pairs with
`num_frames=9`.

Explicit `intrinsics` are recommended so serving does not depend on optional
camera-calibration packages. The mapping form is:

```json
{
  "fx": 640,
  "fy": 640,
  "cx": 640,
  "cy": 352
}
```

## Async API

Use `POST /v1/videos` when you want job storage and polling instead of inline
MP4 bytes. It accepts the same form fields as `/v1/videos/sync`.

```bash
create_response=$(curl -sS -X POST http://localhost:8091/v1/videos \
  -H "Accept: application/json" \
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
  --form-string 'sana_wm={"action":"w-8","translation_speed":0.055,"rotation_speed_deg":1.2,"intrinsics":{"fx":640,"fy":640,"cx":640,"cy":352}}')

video_id=$(echo "$create_response" | jq -r '.id')
curl -sS "http://localhost:8091/v1/videos/${video_id}" | jq .
curl -L "http://localhost:8091/v1/videos/${video_id}/content" -o sana_wm_output.mp4
```
