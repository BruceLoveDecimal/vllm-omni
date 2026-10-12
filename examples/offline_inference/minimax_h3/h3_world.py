# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""H3-World action-conditioned I2V through the existing MiniMax-H3 pipeline."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="MiniMaxAI/MiniMax-H3")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--first-frame", required=True, type=Path)
    parser.add_argument("--scene-prompt", required=True)
    controls = parser.add_mutually_exclusive_group()
    controls.add_argument(
        "--action-preset",
        default="forward",
        choices=[
            "still",
            "forward",
            "back",
            "strafe-left",
            "strafe-right",
            "tilt-down",
            "tilt-up",
            "pan-left",
            "pan-right",
            "pan-left-fast",
            "pan-right-fast",
        ],
    )
    controls.add_argument("--actions-json", type=Path, help="JSON with keys or one key list per video latent")
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--out", type=Path, default=Path("h3_world.mp4"))
    args = parser.parse_args()
    if args.num_frames < 107 or (args.num_frames - 5) % 17:
        parser.error("num-frames must be 17k+5 and at least 107 (H3's minimum four seconds)")
    if args.width % 32 or args.height % 32 or min(args.width, args.height, args.steps, args.tp_size) <= 0:
        parser.error("canvas must be positive multiples of 32; steps and tp-size must be positive")
    if not args.checkpoint.is_file() or args.checkpoint.name != "step-10000.safetensors":
        parser.error("checkpoint must point to the released step-10000.safetensors")

    # Imports stay below argument validation so --help works without vLLM.
    from vllm_omni.diffusion.models.minimax_h3.world import H3_WORLD_PRESETS, action_script
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.lora.request import LoRARequest
    from vllm_omni.lora.utils import stable_lora_int_id

    actions = (
        json.loads(args.actions_json.read_text())
        if args.actions_json
        else {
            "keys": list(H3_WORLD_PRESETS[args.action_preset]),
        }
    )
    action_script(actions, (args.num_frames - 5) // 17 * 5 + 2)
    with Image.open(args.first_frame) as source:
        image = ImageOps.fit(source.convert("RGB"), (args.width, args.height), method=Image.Resampling.LANCZOS)
    checkpoint = str(args.checkpoint.resolve())
    params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        fps=24,
        num_inference_steps=args.steps,
        seed=args.seed,
        guidance_scale=1.0,
        quality="lossless",
        lora_request=LoRARequest("h3-world", stable_lora_int_id(checkpoint), checkpoint),
        lora_scale=1.0,
        extra_args={"task": "fl2va", "frame_indices": [0], "h3_world_actions": actions, "preencode_mp4": True},
    )
    omni = Omni(
        model=args.model,
        task_type="fl2va",
        num_gpus=args.tp_size,
        tensor_parallel_size=args.tp_size,
        text_encoder_tp_size=args.tp_size,
        vae_patch_parallel_size=args.tp_size,
        usp=1,
        ring=1,
        diffusion_attention_backend="FLASH_ATTN",
        enforce_eager=True,
        cache_backend="none",
        lora_path=checkpoint,
        vae_parallel_mode="tile",
        vae_use_tiling=True,
    )
    try:
        outputs = omni.generate({"prompt": args.scene_prompt, "multi_modal_data": {"image": image}}, params)
    finally:
        omni.close()
    output = outputs[0]
    videos = output.images or (output.multimodal_output or {}).get("video")
    if not isinstance(videos, list) or len(videos) != 1 or not isinstance(videos[0], bytes):
        raise RuntimeError("Expected one pre-encoded H3 video with audio")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(videos[0])
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
