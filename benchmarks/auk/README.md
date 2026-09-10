# AuK native-output parity with Seed-TTS-Eval

This benchmark compares **AuK native output against vLLM-Omni pipeline output**,
using Seed-TTS-Eval text/reference inputs and its official WavLM-large SIM scorer.
It does not score generated speech against the input speaker reference.

Use one manifest and seed for both implementations. Each selected text is run
with and without reference audio. Optional editing cases are extracted, without
rewriting their instructions, from the official AuK Gradio examples. Reference
VAE sampling is also seeded before every native request because the native
`generate(seed=...)` argument only seeds target diffusion noise.

```bash
python benchmarks/auk/seed_tts_sim.py prepare \
  --dataset /path/to/seedtts_testset --languages en --per-language 100 \
  --seed 42 --auk-reference /path/to/AuK --manifest manifest.json

python benchmarks/auk/seed_tts_sim.py generate \
  --backend native --auk-reference /path/to/AuK \
  --model /path/to/ckpts/AuK --qwen-path /path/to/Qwen2.5-Omni-3B \
  --manifest manifest.json --output native

python benchmarks/auk/seed_tts_sim.py generate \
  --backend omni --model /path/to/ckpts/AuK \
  --qwen-path /path/to/Qwen2.5-Omni-3B \
  --manifest manifest.json --output omni

python benchmarks/auk/seed_tts_sim.py score \
  --manifest manifest.json --output omni --native-output native \
  --seed-tts-eval /path/to/seed-tts-eval \
  --wavlm-checkpoint /path/to/wavlm_large_finetune.pth
```

The generation harness currently targets AuK-Base, BF16 autocast, 32 Euler steps,
and the real Omni SDPA backend. It directly invokes the pipeline and does not
measure the HTTP server, scheduler, or FlashAttention backend. Run each backend
in a separate process to fit the models on one GPU. The native implementation
and its dependencies must be available in the generation environment.

Duration uses AuK's reference-duration/UTF-8-text-length estimate, rounded to
20 ms. Both modes and implementations receive the same duration. Samples exceeding
the pipeline's 30-second reference-plus-target limit are recorded as exclusions
before random sampling. The manifest records audio hashes and exact sample IDs.
Existing outputs can be resumed only with matching run parameters.

The official scorer can skip failed samples. This harness instead rejects missing,
duplicate, and non-finite scores before reporting aggregate results. SIM measures
speaker similarity; a high score alone does not prove numerical or content parity.
