"""Greedy generation parity on codec inputs, end to end through the vLLM engine.

``run_codec_parity.py`` stops at the vision tower. This drives the whole thing: codec
canvases -> our tower -> vLLM's decoder and sampler, against the checkpoint's own
``generate`` on the same inputs.

vLLM's multimodal plumbing is bypassed on purpose. It decodes video into frames before a
model sees it, while the codec engine needs the encoded bitstream, so the codec path is
built processor-side and handed to the engine as ``prompt_embeds`` -- text embeddings with
the tower's canvas embeddings scattered onto the ``<|image_pad|>`` positions, exactly the
layout the rewritten prompt describes.

    python run_codec_generation_parity.py
"""

import gc
import json
import os

import torch

MODEL = os.environ.get("MAGE_VL_PATH", "/root/autodl-tmp/models/Mage-VL")
VIDEO = os.environ.get("MAGE_VIDEO", f"{MODEL}/examples/soccer-broadcast.mp4")
ASSETS = os.environ.get("MAGE_CODEC_ASSETS")
MAX_NEW_TOKENS = int(os.environ.get("MAGE_MAX_NEW_TOKENS", "32"))
# bf16 by default (what serving runs). MAGE_PARITY_DTYPE=float32 separates encoder
# accumulation from an actual porting difference; the vLLM leg is skipped there, since
# it is already established that the engine matches HF given identical embeddings.
DTYPE = torch.bfloat16
RUN_VLLM = os.environ.get("MAGE_PARITY_DTYPE", "bfloat16") == "bfloat16"
IMAGE_PAD_ID = 151655  # <|image_pad|>

PROMPT = (
    "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>\n"
    "What happens in this video?<|im_end|>\n<|im_start|>assistant\n"
)


def build_codec_inputs():
    from vllm_omni.transformers_utils.processors.mage_vl import MageVLProcessor

    processor = MageVLProcessor.from_pretrained(MODEL)
    codec_config = {"asset_dir": ASSETS} if ASSETS else None
    out = processor(text=PROMPT, videos=VIDEO, video_backend="codec", codec_config=codec_config)
    print(
        f"codec inputs: {out['image_grid_thw'].shape[0]} canvases, "
        f"{out['pixel_values'].shape[0]} patches, {out['input_ids'].shape[1]} prompt tokens, "
        f"{int((out['input_ids'][0] == IMAGE_PAD_ID).sum())} visual tokens",
        flush=True,
    )
    return out


def reference_pass(inputs):
    """HF greedy tokens, plus the text embedding table we reuse for the vLLM side."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL, trust_remote_code=True, dtype=DTYPE, attn_implementation="sdpa"
        )
        .eval()
        .cuda()
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    cuda_inputs = {
        "input_ids": inputs["input_ids"].cuda(),
        "pixel_values": inputs["pixel_values"].cuda().to(DTYPE),
        "image_grid_thw": inputs["image_grid_thw"].cuda(),
        "patch_positions": inputs["patch_positions"].cuda(),
    }
    with torch.inference_mode():
        generated = model.generate(**cuda_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        new_tokens = generated[0, cuda_inputs["input_ids"].shape[1] :]
        text_embeds = model.get_input_embeddings()(cuda_inputs["input_ids"]).cpu()

    tokens = new_tokens.cpu().tolist()
    print(f"[reference] {tokenizer.decode(tokens, skip_special_tokens=True)!r}", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return tokens, text_embeds, tokenizer


def reference_decoder_with_our_embeds(inputs, visual_embeds, text_embeds, tokenizer):
    """Same decoder, our tower: separates encoder drift from anything engine-side."""
    from transformers import AutoModelForCausalLM

    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL, trust_remote_code=True, dtype=DTYPE, attn_implementation="sdpa"
        )
        .eval()
        .cuda()
    )
    mask = inputs["input_ids"][0] == IMAGE_PAD_ID
    embeds = text_embeds[0].clone()
    embeds[mask] = visual_embeds.to(embeds.dtype)

    with torch.inference_mode():
        generated = model.generate(
            inputs_embeds=embeds.unsqueeze(0).cuda(),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    tokens = generated[0, -MAX_NEW_TOKENS:].cpu().tolist()
    print(f"[hf+ours]   {tokenizer.decode(tokens, skip_special_tokens=True)!r}", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return tokens


def reference_visual_embeds(inputs, dtype):
    """The checkpoint's own tower at ``dtype``; only the embeddings survive the call."""
    from transformers import AutoModelForCausalLM

    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL, trust_remote_code=True, dtype=dtype, attn_implementation="sdpa"
        )
        .eval()
        .cuda()
    )
    with torch.inference_mode():
        features = model.model.get_image_features(
            inputs["pixel_values"].cuda().to(dtype),
            inputs["image_grid_thw"].cuda(),
            patch_positions=inputs["patch_positions"].cuda(),
        )
        embeds = torch.cat(features, dim=0).cpu()

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return embeds


def decode_with_embeds(inputs, text_embeds, visual_embeds, tokenizer, label):
    """Greedy decode with the bf16 decoder over a given set of visual embeddings."""
    from transformers import AutoModelForCausalLM

    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .eval()
        .cuda()
    )
    mask = inputs["input_ids"][0] == IMAGE_PAD_ID
    embeds = text_embeds[0].clone()
    embeds[mask] = visual_embeds.to(embeds.dtype)

    with torch.inference_mode():
        generated = model.generate(
            inputs_embeds=embeds.unsqueeze(0).cuda(),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    tokens = generated[0, -MAX_NEW_TOKENS:].cpu().tolist()
    print(f"[{label}] {tokenizer.decode(tokens, skip_special_tokens=True)!r}", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return tokens


def fp32_attribution(inputs, text_embeds, tokenizer):
    """Is the bf16 divergence accumulation, or a real difference in the port?

    Both towers run in fp32 and both sequences decode through the same bf16 decoder, so
    the only thing that varies is which tower produced the visual embeddings.
    """
    reference_embeds = reference_visual_embeds(inputs, torch.float32)
    ours = our_visual_embeds(inputs, torch.float32)
    diff = (ours.float() - reference_embeds.float())
    print(
        f"fp32 visual embeds: rel_l2="
        f"{(diff.norm() / reference_embeds.float().norm()).item():.4e}",
        flush=True,
    )

    reference_tokens = decode_with_embeds(
        inputs, text_embeds, reference_embeds, tokenizer, "fp32 ref  "
    )
    our_tokens = decode_with_embeds(inputs, text_embeds, ours, tokenizer, "fp32 ours ")
    print(
        f"fp32 towers, same decoder: prefix={_prefix(our_tokens, reference_tokens)}/"
        f"{len(reference_tokens)} exact={our_tokens == reference_tokens}"
    )

    # Control: the reference against *itself* at two precisions, same decoder, same path.
    # If this also diverges, greedy token equality cannot discriminate a port -- the
    # continuation is simply unstable under perturbations of this size.
    reference_bf16 = reference_visual_embeds(inputs, torch.bfloat16)
    control_tokens = decode_with_embeds(
        inputs, text_embeds, reference_bf16, tokenizer, "bf16 ref  "
    )
    print(
        f"reference fp32 vs reference bf16 (control): "
        f"prefix={_prefix(control_tokens, reference_tokens)}/{len(reference_tokens)} "
        f"exact={control_tokens == reference_tokens}"
    )
    return our_tokens == reference_tokens


def our_visual_embeds(inputs, dtype=None):
    """Run the ported tower on the codec canvases."""
    dtype = DTYPE if dtype is None else dtype
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29611")

    from safetensors import safe_open
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    from vllm_omni.model_executor.models.mage_vl.vision import MageVLVisionTransformer
    from vllm_omni.transformers_utils.configs.mage_vl import MageVLConfig

    ctx = set_current_vllm_config(VllmConfig())
    ctx.__enter__()
    init_distributed_environment(
        world_size=1, rank=0, distributed_init_method="env://", local_rank=0, backend="nccl"
    )
    initialize_model_parallel(tensor_model_parallel_size=1)

    cfg = MageVLConfig.from_pretrained(MODEL)
    torch.set_default_dtype(dtype)
    with torch.device("cuda"):
        tower = MageVLVisionTransformer(cfg.vision_config, prefix="visual")
    tower.eval()
    for layer in tower.encoder.layers:
        layer.self_attn.attn.attn_backend = AttentionBackendEnum.TORCH_SDPA
        layer.self_attn.attn.is_flash_attn_backend = False

    with open(f"{MODEL}/model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]
    by_file: dict[str, list[str]] = {}
    for key in [k for k in weight_map if k.startswith("model.visual.")]:
        by_file.setdefault(weight_map[key], []).append(key)

    def weights():
        for filename, keys in by_file.items():
            with safe_open(f"{MODEL}/{filename}", framework="pt", device="cpu") as f:
                for key in keys:
                    yield key[len("model.visual.") :], f.get_tensor(key).cuda().to(dtype)

    tower.load_weights(weights())
    with torch.inference_mode():
        embeds = tower(
            inputs["pixel_values"].cuda().to(dtype),
            inputs["image_grid_thw"].cuda(),
            inputs["patch_positions"].cuda(),
        ).cpu()

    del tower
    gc.collect()
    torch.cuda.empty_cache()
    torch.set_default_dtype(torch.float32)
    return embeds


def vllm_pass(inputs, text_embeds, visual_embeds, tokenizer):
    from vllm import LLM, SamplingParams

    token_ids = inputs["input_ids"][0]
    mask = token_ids == IMAGE_PAD_ID
    if int(mask.sum()) != visual_embeds.shape[0]:
        raise ValueError(
            f"visual token count {int(mask.sum())} != tower output {visual_embeds.shape[0]}"
        )

    prompt_embeds = text_embeds[0].clone()
    prompt_embeds[mask] = visual_embeds.to(prompt_embeds.dtype)

    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.45,
        enable_prompt_embeds=True,
        limit_mm_per_prompt={"image": 0},
    )
    out = llm.generate(
        {"prompt_embeds": prompt_embeds},
        SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS),
    )
    tokens = list(out[0].outputs[0].token_ids)
    print(f"[vllm]      {tokenizer.decode(tokens, skip_special_tokens=True)!r}", flush=True)
    return tokens


def _prefix(a, b):
    count = 0
    for lhs, rhs in zip(a, b):
        if lhs != rhs:
            break
        count += 1
    return count


def main():
    inputs = build_codec_inputs()
    reference_tokens, text_embeds, tokenizer = reference_pass(inputs)

    if not RUN_VLLM:  # fp32 attribution mode
        ok = fp32_attribution(inputs, text_embeds, tokenizer)
        print(f"CODEC_GENERATION_PARITY[fp32-towers]: {'PASS' if ok else 'FAIL'}")
        return
    visual_embeds = our_visual_embeds(inputs)

    # Attribution: same decoder with our encoder isolates tower drift from the engine.
    hf_ours_tokens = reference_decoder_with_our_embeds(
        inputs, visual_embeds, text_embeds, tokenizer
    )
    tokens = vllm_pass(inputs, text_embeds, visual_embeds, tokenizer) if RUN_VLLM else None

    n = len(reference_tokens)
    print(
        f"hf_decoder+our_tower : prefix={_prefix(hf_ours_tokens, reference_tokens)}/{n} "
        f"exact={hf_ours_tokens == reference_tokens}"
    )
    if tokens is not None:
        print(
            f"vllm_decoder+our_tower: prefix={_prefix(tokens, reference_tokens)}/{n} "
            f"exact={tokens == reference_tokens}"
        )
        print(
            f"vllm vs hf(both our tower): prefix={_prefix(tokens, hf_ours_tokens)}/{n} "
            f"exact={tokens == hf_ours_tokens}"
        )
    verdict = hf_ours_tokens == reference_tokens and (tokens is None or tokens == hf_ours_tokens)
    print(f"CODEC_GENERATION_PARITY[{DTYPE}]: {'PASS' if verdict else 'FAIL'}")


if __name__ == "__main__":
    main()
