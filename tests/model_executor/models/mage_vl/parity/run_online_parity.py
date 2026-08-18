"""Online e2e check: OpenAI-compatible /v1/chat/completions vs the HF reference.

Sends the same images/questions the HF baseline used, through the chat endpoint, and
compares generated text token-for-token (via the served tokenizer) against the reference.
"""

import base64
import json
import mimetypes
import os
from pathlib import Path

import requests

BASE = os.environ.get("MAGE_BASE_URL", "http://127.0.0.1:8077/v1")
MODEL_NAME = os.environ.get("MAGE_SERVED_NAME", "Mage-VL")
MODEL = os.environ.get("MAGE_VL_PATH", "/root/autodl-tmp/models/Mage-VL")
REF = Path(os.environ.get("REF_OUT", "/root/autodl-tmp/mage_ref"))


def data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(Path(path).read_bytes()).decode()}"


def main():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    summary = json.loads((REF / "summary.json").read_text())

    models = requests.get(f"{BASE}/models", timeout=30).json()
    print("served models:", [m["id"] for m in models.get("data", [])])

    passed = 0
    total = 0
    for name, rec in summary.items():
        total += 1
        body = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url(rec["image"])}},
                        {"type": "text", "text": rec["question"]},
                    ],
                }
            ],
            "max_tokens": len(rec["gen_tokens"]),
            "temperature": 0.0,
        }
        r = requests.post(f"{BASE}/chat/completions", json=body, timeout=300)
        r.raise_for_status()
        got_text = r.json()["choices"][0]["message"]["content"].strip()
        want_text = rec["gen_text"].strip()

        got_ids = tok(got_text, add_special_tokens=False)["input_ids"]
        want_ids = tok(want_text, add_special_tokens=False)["input_ids"]
        ok = got_ids == want_ids
        passed += 1 if ok else 0
        print(f"\n[{name}] token_match={ok} ({len(got_ids)} vs {len(want_ids)} tokens)")
        print(f"  online: {got_text[:150]!r}")
        print(f"  hf    : {want_text[:150]!r}")

    # streaming smoke on one case
    rec = next(iter(summary.values()))
    body = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url(rec["image"])}},
                    {"type": "text", "text": rec["question"]},
                ],
            }
        ],
        "max_tokens": 24,
        "temperature": 0.0,
        "stream": True,
    }
    chunks = 0
    text = ""
    with requests.post(f"{BASE}/chat/completions", json=body, stream=True, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            payload = line[len(b"data: "):]
            if payload.strip() == b"[DONE]":
                break
            delta = json.loads(payload)["choices"][0]["delta"].get("content")
            if delta:
                chunks += 1
                text += delta
    print(f"\n[streaming] chunks={chunks} text={text[:110]!r}")

    print(f"\nONLINE_PARITY: {passed}/{total}")
    print("ONLINE_RESULT:", "PASS" if passed == total and chunks > 1 else "FAIL")


if __name__ == "__main__":
    main()
