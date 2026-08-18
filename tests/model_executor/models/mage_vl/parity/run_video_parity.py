"""Video e2e: frame-sampled clip through the online chat endpoint vs the HF reference.

This mirrors the reference repo's own online path (``inference_base.py::run_online``),
which sends sampled frames as multiple images -- the checkpoint's supported online video
mode. Also covers multi-image prompts and concurrency in one pass.
"""

import base64
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor

import requests
from PIL import Image

BASE = os.environ.get("MAGE_BASE_URL", "http://127.0.0.1:8077/v1")
MODEL_NAME = os.environ.get("MAGE_SERVED_NAME", "Mage-VL")
MODEL = os.environ.get("MAGE_VL_PATH", "/root/autodl-tmp/models/Mage-VL")
VIDEO = f"{MODEL}/examples/soccer-broadcast.mp4"
QUESTION = "Describe what is happening in this video."
NUM_FRAMES = int(os.environ.get("NUM_FRAMES", "4"))
MAX_TOKENS = 48


def sample_video(path, num_frames):
    import cv2
    import numpy as np
    from PIL import Image

    cap = cv2.VideoCapture(path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, count - 1, min(num_frames, count), dtype=int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def encode_frames(frames):
    """Encode once; both sides then consume the exact same bytes.

    Sending JPEG to the server while handing raw decoded frames to the reference would
    compare different pixels, not different implementations.
    """
    payloads = []
    for img in frames:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        payloads.append(buf.getvalue())
    return payloads


def frame_url(jpeg_bytes: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode()


def chat(frames, question, max_tokens=MAX_TOKENS):
    content = [{"type": "image_url", "image_url": {"url": frame_url(f)}} for f in frames]
    content.append({"type": "text", "text": question})
    r = requests.post(
        f"{BASE}/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=600,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    raw_frames = sample_video(VIDEO, NUM_FRAMES)
    jpegs = encode_frames(raw_frames)
    # Decode back so the reference sees exactly the pixels the server received.
    frames = [Image.open(io.BytesIO(b)).convert("RGB") for b in jpegs]
    print(f"sampled {len(frames)} frames from {VIDEO}")

    online_text = chat(jpegs, QUESTION)
    print(f"\n[online video] {online_text!r}")

    # HF reference over the same frames, same greedy settings.
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .eval()
        .cuda()
    )
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"} for _ in frames] + [{"type": "text", "text": QUESTION}],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=frames, return_tensors="pt")
    inputs = {k: (v.cuda() if hasattr(v, "to") else v) for k, v in inputs.items()}
    inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False)
    hf_text = processor.tokenizer.decode(
        gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    print(f"[hf     video] {hf_text!r}")

    tok = processor.tokenizer
    match = tok(online_text, add_special_tokens=False)["input_ids"] == tok(
        hf_text, add_special_tokens=False
    )["input_ids"]
    print(f"\nVIDEO_TOKEN_MATCH: {match}")

    # Concurrency: 4 parallel requests must not cross-contaminate.
    with ThreadPoolExecutor(max_workers=4) as ex:
        outs = list(ex.map(lambda _: chat(jpegs, QUESTION, 24), range(4)))
    distinct = sorted(set(outs))
    consistent = len(distinct) == 1
    print(f"CONCURRENCY: {len(distinct)} distinct output(s) across 4 parallel requests")
    for i, o in enumerate(distinct):
        print(f"  variant {i}: {o[:120]!r}")

    print("VIDEO_RESULT:", "PASS" if match and consistent else "FAIL")


if __name__ == "__main__":
    main()
