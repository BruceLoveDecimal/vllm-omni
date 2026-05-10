# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline inference example for IndexTTS2 via vLLM-Omni.

This MVP uses a single-stage wrapper around upstream IndexTTS2. The upstream
runtime owns GPT mel-code generation, s2mel/CFM, and BigVGAN. Output is
22.05 kHz mono audio.
"""

from __future__ import annotations

import os
from pathlib import Path

import soundfile as sf
import torch
from vllm import SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from vllm_omni import Omni  # noqa: E402


def build_request(args) -> dict:
    additional: dict = {
        "text": [args.text],
        "spk_audio_prompt": [str(args.ref_audio)],
        "max_text_tokens_per_segment": [args.max_text_tokens_per_segment],
        "interval_silence": [args.interval_silence],
        "top_p": [args.top_p],
        "top_k": [args.top_k],
        "temperature": [args.temperature],
        "num_beams": [args.num_beams],
        "repetition_penalty": [args.repetition_penalty],
        "max_mel_tokens": [args.max_mel_tokens],
    }
    if args.emo_audio is not None:
        additional["emo_audio_prompt"] = [str(args.emo_audio)]
    if args.emo_text is not None:
        additional["use_emo_text"] = [True]
        additional["emo_text"] = [args.emo_text]
    if args.emo_vector is not None:
        additional["emo_vector"] = [[float(v) for v in args.emo_vector.split(",")]]

    return {
        "prompt": "<|im_start|>assistant\n",
        "additional_information": additional,
    }


def _extract_audio(mm: dict) -> tuple[torch.Tensor | None, int]:
    audio = mm.get("audio")
    if audio is None:
        audio = mm.get("model_outputs")
    if isinstance(audio, list):
        chunks = [chunk.reshape(-1) for chunk in audio if hasattr(chunk, "numel") and chunk.numel() > 0]
        audio = torch.cat(chunks, dim=-1) if chunks else None
    sr_raw = mm.get("sr")
    if isinstance(sr_raw, list):
        sr_raw = sr_raw[-1] if sr_raw else None
    sr = int(sr_raw.item()) if hasattr(sr_raw, "item") else int(sr_raw or 22050)
    return audio, sr


def main(args) -> None:
    omni = Omni(
        model=args.model,
        deploy_config=args.deploy_config,
        stage_init_timeout=args.stage_init_timeout,
    )

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        top_k=50,
        max_tokens=4096,
        seed=args.seed if args.seed is not None else 42,
        detokenize=False,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = build_request(args)
    for stage_outputs in omni.generate(inputs, sampling_params):
        for i, req_output in enumerate(stage_outputs.request_output):
            for j, out in enumerate(req_output.outputs):
                mm = out.multimodal_output
                if mm is None:
                    print(f"  [req {i}] No audio output.")
                    continue
                audio, sr = _extract_audio(mm)
                if audio is None:
                    print(f"  [req {i}] No waveform in multimodal_output.")
                    continue
                out_path = output_dir / f"indextts2_{i}_{j}.wav"
                sf.write(str(out_path), audio.detach().cpu().float().numpy(), sr)
                print(f"  Saved {out_path} ({audio.numel()} samples, {sr} Hz)")


def parse_args():
    parser = FlexibleArgumentParser(description="IndexTTS2 offline inference")
    parser.add_argument("--model", required=True, help="Path to an IndexTTS2 checkpoint directory.")
    parser.add_argument("--text", default="Hello, this is IndexTTS2 running in vLLM-Omni.")
    parser.add_argument("--ref-audio", required=True, help="Speaker reference WAV path.")
    parser.add_argument("--emo-audio", default=None, help="Optional emotion reference WAV path.")
    parser.add_argument("--emo-text", default=None, help="Optional text used by IndexTTS2 emotion classifier.")
    parser.add_argument("--emo-vector", default=None, help="Optional comma-separated 8-value emotion vector.")
    parser.add_argument("--max-text-tokens-per-segment", type=int, default=120)
    parser.add_argument("--interval-silence", type=int, default=200, help="Silence between generated segments in ms.")
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--repetition-penalty", type=float, default=10.0)
    parser.add_argument("--max-mel-tokens", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(
            os.environ.get("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache")),
            "indextts2_output",
        ),
    )
    parser.add_argument("--deploy-config", default=None)
    parser.add_argument("--stage-init-timeout", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
