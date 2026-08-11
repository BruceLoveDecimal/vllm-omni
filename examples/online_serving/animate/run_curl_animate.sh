#!/bin/bash
# Wan2.2-Animate curl example using the sync video API.
#
# Three inputs travel three different ways:
#   - reference image -> image_reference   (standard media field)
#   - pose video      -> video_reference   (standard media field; the API
#                                           permits it alongside image_reference)
#   - face video      -> extra_params      (no HTTP field exists for a second
#                                           video, so it is passed as a
#                                           model-specific extra)
#
# Both driving videos must already be pre-processed by the upstream Wan-Animate
# preprocess_data.py; this server does not run pose estimation or face
# retargeting.

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8092}"
OUTPUT_PATH="${OUTPUT_PATH:-animate_480p_serve.mp4}"
IMAGE_URL="${IMAGE_URL:?set IMAGE_URL to the reference character image}"
POSE_VIDEO_URL="${POSE_VIDEO_URL:?set POSE_VIDEO_URL to the pre-processed pose video}"
FACE_VIDEO_URL="${FACE_VIDEO_URL:?set FACE_VIDEO_URL to the pre-processed face video}"
PROMPT="${PROMPT:-A person dancing energetically in a studio}"
WIDTH="${WIDTH:-832}"
HEIGHT="${HEIGHT:-480}"
NUM_FRAMES="${NUM_FRAMES:-77}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-20}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
FPS="${FPS:-30}"

echo "Sending Wan-Animate request..."
echo "  Image URL: $IMAGE_URL"
echo "  Pose video: $POSE_VIDEO_URL"
echo "  Face video: $FACE_VIDEO_URL"
echo "  Resolution: ${WIDTH}x${HEIGHT}, ${NUM_FRAMES} frames"
echo "  Steps: $NUM_INFERENCE_STEPS, guidance: $GUIDANCE_SCALE"

IMAGE_REF_JSON="{\"image_url\": \"${IMAGE_URL}\"}"
VIDEO_REF_JSON="{\"video_url\": \"${POSE_VIDEO_URL}\"}"
EXTRA_PARAMS_JSON="{\"face_video\": \"${FACE_VIDEO_URL}\"}"

no_proxy=127.0.0.1 \
curl -X POST "${BASE_URL}/v1/videos/sync" \
  -F "prompt=${PROMPT}" \
  -F "image_reference=${IMAGE_REF_JSON}" \
  -F "video_reference=${VIDEO_REF_JSON}" \
  -F "extra_params=${EXTRA_PARAMS_JSON}" \
  -F "width=${WIDTH}" -F "height=${HEIGHT}" \
  -F "num_frames=${NUM_FRAMES}" \
  -F "num_inference_steps=${NUM_INFERENCE_STEPS}" \
  -F "guidance_scale=${GUIDANCE_SCALE}" \
  -F "fps=${FPS}" \
  --output "${OUTPUT_PATH}"

if [ ! -s "$OUTPUT_PATH" ]; then
    echo "ERROR: Output file is empty or missing"
    exit 1
fi

# A failed request still writes a body -- the server's JSON error lands in the
# .mp4. Check that this is actually a video before declaring success.
if head -c 1 "$OUTPUT_PATH" | grep -q '{'; then
    echo "ERROR: server returned an error instead of a video:"
    cat "$OUTPUT_PATH"
    echo
    exit 1
fi

echo "Saved video to ${OUTPUT_PATH} ($(du -h "$OUTPUT_PATH" | cut -f1))"
