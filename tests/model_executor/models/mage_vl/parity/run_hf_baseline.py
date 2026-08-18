"""Dump HF Mage-VL reference tensors for parity testing.

Saves, for each case:
  - processor outputs (input_ids, pixel_values, image_grid_thw, patch_positions)
  - vision tower output (merged visual embeddings)
  - logits of the prompt's last position
  - greedy generated token ids + text
"""

import argparse
import json
import os
from pathlib import Path

import torch

MODEL = os.environ.get("MAGE_VL_PATH", "/root/autodl-tmp/models/Mage-VL")
OUT = Path(os.environ.get("REF_OUT", "/root/autodl-tmp/mage_ref"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    args = ap.parse_args()

    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor

    OUT.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation=args.attn
    ).eval().cuda()
    print("model loaded:", type(model).__name__, flush=True)

    cases = [
        ("img_dog_describe", f"{MODEL}/examples/dog.jpg", "Describe this image in detail."),
        ("img_dog_short", f"{MODEL}/examples/dog.jpg", "What animal is in the picture?"),
        ("img_cover_ocr", f"{MODEL}/assets/mage-vl-cover.png", "What text appears in this image?"),
    ]

    summary = {}
    for name, img_path, question in cases:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[Image.open(img_path).convert("RGB")], return_tensors="pt")
        inputs = {k: (v.cuda() if hasattr(v, "to") else v) for k, v in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

        rec = {
            "question": question,
            "image": img_path,
            "prompt_text": text,
            "input_ids": inputs["input_ids"][0].tolist(),
            "image_grid_thw": inputs["image_grid_thw"].tolist(),
        }

        with torch.inference_mode():
            vis = model.model.get_image_features(
                inputs["pixel_values"],
                inputs["image_grid_thw"],
                patch_positions=inputs.get("patch_positions"),
            )
            vis_cat = torch.cat(vis, dim=0)

            out = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                pixel_values=inputs["pixel_values"],
                image_grid_thw=inputs["image_grid_thw"],
                patch_positions=inputs.get("patch_positions"),
            )
            logits = out["logits"] if isinstance(out, dict) else out.logits

            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            new_tokens = gen[0, inputs["input_ids"].shape[1]:]

        torch.save(
            {
                "pixel_values": inputs["pixel_values"].cpu(),
                "image_grid_thw": inputs["image_grid_thw"].cpu(),
                "patch_positions": (
                    inputs["patch_positions"].cpu() if inputs.get("patch_positions") is not None else None
                ),
                "input_ids": inputs["input_ids"].cpu(),
                "vision_embeds": vis_cat.float().cpu(),
                "last_logits": logits[0, -1].float().cpu(),
                "gen_tokens": new_tokens.cpu(),
            },
            OUT / f"{name}.pt",
        )

        rec["vision_embeds_shape"] = list(vis_cat.shape)
        rec["gen_tokens"] = new_tokens.tolist()
        rec["gen_text"] = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        summary[name] = rec
        print(f"[{name}] vis={tuple(vis_cat.shape)} gen={rec['gen_text'][:110]!r}", flush=True)

    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("saved to", OUT, flush=True)


if __name__ == "__main__":
    main()
