"""Offline vLLM run of Mage-VL, compared against the HF reference generations."""

import json
import os
from pathlib import Path

MODEL = os.environ.get("MAGE_VL_PATH", "/root/autodl-tmp/models/Mage-VL")
REF = Path(os.environ.get("REF_OUT", "/root/autodl-tmp/mage_ref"))


def main():
    from PIL import Image
    from vllm import LLM, SamplingParams

    summary = json.loads((REF / "summary.json").read_text())

    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=8192,
        limit_mm_per_prompt={"image": 4},
        gpu_memory_utilization=0.85,
        enforce_eager=os.environ.get("MAGE_EAGER", "0") == "1",
    )

    n_match = 0
    n_total = 0
    for name, rec in summary.items():
        prompt = rec["prompt_text"]
        image = Image.open(rec["image"]).convert("RGB")
        out = llm.generate(
            {"prompt": prompt, "multi_modal_data": {"image": image}},
            SamplingParams(temperature=0.0, max_tokens=len(rec["gen_tokens"])),
        )
        got_ids = list(out[0].outputs[0].token_ids)
        want_ids = rec["gen_tokens"]
        got_text = out[0].outputs[0].text.strip()
        n_total += 1

        exact = got_ids == want_ids
        # Length can differ if one side stopped at EOS earlier; compare the overlap too.
        overlap = min(len(got_ids), len(want_ids))
        prefix_match = sum(1 for a, b in zip(got_ids[:overlap], want_ids[:overlap]) if a == b)
        n_match += 1 if exact else 0
        print(f"\n[{name}] exact_token_match={exact} "
              f"prefix_agreement={prefix_match}/{overlap}")
        print(f"  vllm: {got_text[:160]!r}")
        print(f"  hf  : {rec['gen_text'][:160]!r}")

    print(f"\nTOKEN_PARITY: {n_match}/{n_total} exact")
    print("SMOKE_RESULT:", "PASS" if n_match == n_total else "PARTIAL")


if __name__ == "__main__":
    main()
