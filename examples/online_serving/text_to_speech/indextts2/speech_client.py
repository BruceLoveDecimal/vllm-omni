# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenAI-compatible speech client for IndexTTS2."""

from __future__ import annotations

import base64
from pathlib import Path

import httpx
from vllm.utils.argparse_utils import FlexibleArgumentParser


def _audio_data_url(path: str) -> str:
    data = Path(path).read_bytes()
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:audio/wav;base64,{encoded}"


def main(args) -> None:
    payload: dict = {
        "model": args.model,
        "input": args.text,
        "voice": "default",
        "ref_audio": _audio_data_url(args.ref_audio),
        "response_format": args.response_format,
        "extra_params": {
            "top_p": args.top_p,
            "top_k": args.top_k,
            "temperature": args.temperature,
            "num_beams": args.num_beams,
            "repetition_penalty": args.repetition_penalty,
            "max_mel_tokens": args.max_mel_tokens,
            "max_text_tokens_per_segment": args.max_text_tokens_per_segment,
        },
    }
    if args.emo_audio:
        payload["extra_params"]["emo_audio"] = _audio_data_url(args.emo_audio)
    if args.emo_text:
        payload["instructions"] = args.emo_text
    if args.stream:
        payload["stream"] = True
        payload["response_format"] = "pcm"

    with httpx.Client(timeout=args.timeout) as client:
        with client.stream("POST", f"{args.api_base}/audio/speech", json=payload) as response:
            response.raise_for_status()
            mode = "wb"
            with open(args.output, mode) as f:
                for chunk in response.iter_bytes():
                    if chunk:
                        f.write(chunk)

    print(f"Saved {args.output}")


def parse_args():
    parser = FlexibleArgumentParser(description="IndexTTS2 speech client")
    parser.add_argument("--api-base", default="http://localhost:8091/v1")
    parser.add_argument("--model", default="indextts2")
    parser.add_argument("--text", default="Hello, this is IndexTTS2 served by vLLM-Omni.")
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--emo-audio", default=None)
    parser.add_argument("--emo-text", default=None)
    parser.add_argument("--response-format", default="wav", choices=["wav", "pcm", "flac", "mp3", "aac", "opus"])
    parser.add_argument("--stream", action="store_true", help="Stream raw PCM chunks.")
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--repetition-penalty", type=float, default=10.0)
    parser.add_argument("--max-mel-tokens", type=int, default=1500)
    parser.add_argument("--max-text-tokens-per-segment", type=int, default=120)
    parser.add_argument("--output", default="indextts2_output.wav")
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
