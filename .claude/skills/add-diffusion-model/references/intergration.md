# Practical Integration Tips

Hard-won lessons from real PR reviews. Read this before opening a PR.

---

## 1. Weight Format: Upload a Clean Copy to HF

If the upstream model weights don't match vllm-omni's naming conventions, upload a renamed copy to HuggingFace rather than writing a complex remap table.

- `load_weights()` should be **≤ 30 lines** — closer to 10 is ideal.
- A 600-line weight-name remap is a code smell; a renamed HF upload eliminates it entirely.
- This mostly affects academic/research models. Industrial models (HunyuanVideo, Wan, Flux, etc.) typically match vllm-omni conventions already.

```python
# Good: minimal load_weights after uploading renamed weights to HF
def load_weights(self, weights):
    params = dict(self.named_parameters())
    loaded = set()
    for name, tensor in weights:
        if name in params:
            default_weight_loader(params[name], tensor)
            loaded.add(name)
    return loaded
```

---

## 2. Tests: Offline + Online E2E Only

Follow the CI level definitions in `docs/contributing/ci/CI_5levels.md`.

**Required:**
- `tests/e2e/offline_inference/test_{model_name}_expansion.py` — L2 offline test
- `tests/e2e/online_serving/test_{model_name}_expansion.py` — L2 online test

**Not required:** no module-level unit tests (L1), no component-level GPU tests.

**In your PR description**, paste the actual text output of both tests using markdown code blocks:

```
# ✅ Do this — paste raw terminal output in markdown
```python
$ pytest tests/e2e/offline_inference/test_yourmodel_expansion.py -v
...
PASSED tests/e2e/offline_inference/test_yourmodel_expansion.py::test_offline_inference
```

**Never use screenshots** — pasted text proves you ran the tests yourself.

---

## 3. Don't Touch Framework Code; Don't Add Dependencies

**No changes to:**
- `entrypoints/`
- `engine/`
- Any existing dependency in `requirements/commons.txt`

If you need a utility function from an upstream package, **copy the function** rather than adding the package as a dependency. Ask Claude:

> "Why can't this part be replaced with an existing package in pytorch, transformers, diffusers, torchaudio, vllm, or vllm-omni? Search the codebase."

Claude will find existing equivalents or explain why the function must be ported. If its reasoning seems weak, push back — it tends toward caution. Often a direct import from an already-available package works fine.

---

## 4. Model Architecture Diagram in PR Description

**Include a model architecture diagram** (ASCII, Mermaid, or image) in your PR description. This:
1. Proves a human reviewed the architecture (reviewers look for this signal).
2. Brings reviewers to your PR faster.
3. Forces you to understand what each component does before porting it.

DiT blocks are almost always reusable from existing vllm-omni code. VAE encoders often are too. TTS decoders are the exception — they tend to require direct porting because the TTS ecosystem hasn't yet modularized its components.

**For each non-trivial block**, ask:

> "Why can't this part be imported from pytorch / transformers / diffusers / torchaudio / vllm / vllm-omni?"

Reviewers ask this question about every unfamiliar block.

---

## 5. Code Style: Match Existing Models

Find the 2–3 structurally closest models already in vllm-omni and follow their patterns exactly. A good reference:

- **ming-flash** (`vllm_omni/model_executor/models/ming_flash_omni/`) — written by yuanheng-zhao, very clean structure, good to imitate.

Before submitting, ask Claude to remove:

- `try/except` blocks (unless wrapping a real external call)
- Duplicate code paths
- Dead code / unreachable branches
- Over-engineered abstractions (helper classes for one-time use)
- Redundant validation (don't validate vllm-omni-internal data)
- Defensive coding (don't guard against things that can't happen)

Also read `docs/contributing/` fully before writing a line of code.

---

## 6. Delete All Training Code

Remove every training artifact before opening a PR:

- `gradient_checkpointing` flags and branches
- `dropout` layers (set to 0 or remove)
- `nn.Dropout` usage
- `requires_grad` settings
- Loss computation code
- `model.train()` / `model.eval()` calls inside `__init__`
- Training-specific config keys

Claude sometimes misses these. Go line-by-line on the transformer and pipeline files and ask: "Is this line training code?" for anything that looks suspicious. Reviewers will ask if anything survives.

---

## 7. Finding a Model to Integrate

Search open GitHub issues for stalled model integration PRs:

```bash
gh issue list --repo vllm-project/vllm-omni --state open --label "new model" | head -20
```

Or search PR history for abandoned PRs (no update > 1 month):

```bash
gh pr list --repo vllm-project/vllm-omni --state open | grep -v "today\|yesterday"
```

Before picking one, ask Claude: "How complex is this model to integrate? How large is the checkpoint?" Prefer small checkpoints (< 7B) and models with a Diffusers pipeline already available.

---

## 8. DiT Attention: Always Use Tensor Parallel Versions

**Every attention layer in a DiT transformer must use parallel linear layers**, not plain `nn.Linear`. This is non-negotiable for TP support.

Replace:

| Original | Replacement |
|----------|-------------|
| `nn.Linear` for Q/K/V projections | `QKVParallelLinear` |
| `nn.Linear` for output projection | `RowParallelLinear` |
| `nn.Linear` for FFN gate/up | `ColumnParallelLinear` |
| `nn.Linear` for FFN down | `RowParallelLinear` |

```python
from vllm.model_executor.layers.linear import (
    QKVParallelLinear, RowParallelLinear, ColumnParallelLinear,
)
```

After replacing, **compare outputs against the upstream repo** at matching seeds, resolutions, and step counts. If a substitution degrades quality, revert that specific layer. Accuracy alignment is the acceptance criterion — upload both your output and the upstream output to the PR description.

Do not add quantization for new models.

---

## 9. Model Config: Follow ming-flash Pattern

For model configuration files:

- **YAML deploy configs** → `vllm_omni/deploy/{model_name}.yaml`  
  (see `vllm_omni/deploy/ming_flash_omni.yaml` as template)
- **Helper functions** (config parsing, transformer kwargs, etc.) → `vllm_omni/transformers_utils/`  
  (see `vllm_omni/transformers_utils/configs/ming_flash_omni.py`)
- Keep `pipeline_{name}.py` focused on orchestration; move helpers out.

Avoid putting config dicts inline in the pipeline class unless they are truly pipeline-specific constants with no config-file equivalent.

---

## Summary Checklist

Before opening a PR, verify:

- [ ] `load_weights()` ≤ 30 lines; no giant name-remap table
- [ ] Offline E2E test passes; output pasted as markdown in PR description
- [ ] Online E2E test passes; output pasted as markdown in PR description
- [ ] No changes to `entrypoints/`, `engine/`, or `requirements/commons.txt`
- [ ] No new pip/conda dependencies (copy needed functions instead)
- [ ] Model architecture diagram included in PR description
- [ ] All attention layers use `QKVParallelLinear` / `RowParallelLinear` / `ColumnParallelLinear`
- [ ] Accuracy compared against upstream repo; results in PR description
- [ ] All training code removed (gradient checkpointing, dropout, loss, etc.)
- [ ] `try/except`, dead code, and defensive coding removed
- [ ] YAML deploy config in `vllm_omni/deploy/`
- [ ] Config helpers in `vllm_omni/transformers_utils/`
- [ ] Code style matches closest existing model (ming-flash recommended as reference)
