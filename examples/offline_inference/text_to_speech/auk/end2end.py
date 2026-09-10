# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate or edit speech with AuK's single-stage diffusion pipeline."""

import argparse

import numpy as np
import soundfile as sf

from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tencent/AuK-Flash")
    parser.add_argument("--qwen-path", help="Local Qwen2.5-Omni-3B snapshot or Hugging Face ID")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--text", help="Text to synthesize")
    mode.add_argument("--instruction", help="Full AuK generation/editing instruction")
    parser.add_argument("--ref-audio")
    parser.add_argument("--gen-seconds", type=float)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cfg-strength", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="auk.wav")
    args = parser.parse_args()
    if args.gen_seconds is None and args.ref_audio is None:
        parser.error("--gen-seconds is required without --ref-audio")

    prompt = {"input": args.text} if args.text is not None else {"instruction": args.instruction}
    if args.ref_audio:
        waveform, sr = sf.read(args.ref_audio, dtype="float32", always_2d=True)
        prompt["ref_audio"] = (waveform.T, sr)
    if args.gen_seconds is not None:
        prompt["gen_seconds"] = args.gen_seconds
    params = OmniDiffusionSamplingParams(
        seed=args.seed,
        num_inference_steps=args.steps,
        extra_args={"cfg_strength": args.cfg_strength},
    )
    additional = {"qwen_path": args.qwen_path} if args.qwen_path else {}
    engine = Omni(model=args.model, dtype="bfloat16", enforce_eager=True, additional_config=additional)
    try:
        results = list(engine.generate([prompt], sampling_params_list=[params]))
        output = results[-1]
        mm = output.multimodal_output
        if not mm and output.outputs:
            mm = output.outputs[0].multimodal_output
        audio = mm["audio"]
        if hasattr(audio, "cpu"):
            audio = audio.cpu().numpy()
        sf.write(args.output, np.asarray(audio).squeeze(), mm.get("audio_sample_rate", 24000))
    finally:
        engine.close()


if __name__ == "__main__":
    main()
