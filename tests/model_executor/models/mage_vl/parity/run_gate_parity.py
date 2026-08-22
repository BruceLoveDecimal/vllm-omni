"""Cognition-gate parity: the ported perception encoder vs the checkpoint's own.

Two things are checked, and the second is the one the streaming path depends on:

1. **Against the reference.** The checkpoint's ``streammind_gate.py`` builds its Mamba
   block with ``mamba_ssm``. That package's CUDA extension has no wheel for the torch
   versions this project targets, so the reference runs here on ``selective_scan_ref`` --
   the pure-torch recurrence ``mamba_ssm`` ships as its own algorithmic definition. Import
   it by putting an empty ``selective_scan_cuda.py`` on the path (``MAGE_SHIM_DIR``);
   ``mamba_ssm/__init__.py`` imports the extension eagerly even when nothing uses it.

2. **Incremental equals full-sequence.** Streaming folds one segment into the state at a
   time. If that disagreed with scanning the whole stream at once, every gate decision
   after the first would be made on a state the offline numbers never validated.

    MAGE_SHIM_DIR=/path/with/selective_scan_cuda.py python run_gate_parity.py
"""

import os
import sys

import torch

MODEL = os.environ.get("MAGE_VL_PATH", "/root/autodl-tmp/models/Mage-VL")
SHIM_DIR = os.environ.get("MAGE_SHIM_DIR", "/root/autodl-tmp/shims")
SEGMENTS = int(os.environ.get("MAGE_GATE_SEGMENTS", "8"))
PATCHES = int(os.environ.get("MAGE_GATE_PATCHES", "144"))


def load_reference(device, dtype):
    sys.path.insert(0, SHIM_DIR)
    sys.path.insert(0, MODEL)

    import mamba_ssm.modules.mamba_simple as mamba_simple
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref

    mamba_simple.selective_scan_fn = lambda *args, **kwargs: selective_scan_ref(*args, **kwargs)
    mamba_simple.causal_conv1d_fn = None  # force the reference conv path too

    from safetensors.torch import load_file
    from streammind_gate import StreamMindGate

    gate = StreamMindGate(hidden_size=2560)
    state_dict = load_file(f"{MODEL}/streammind_gate.safetensors")
    missing, unexpected = gate.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise SystemExit(f"reference gate did not load cleanly: {missing=} {unexpected=}")
    return gate.to(device=device, dtype=dtype).eval(), state_dict


def load_ours(state_dict, device, dtype):
    from vllm_omni.model_executor.models.mage_vl.gate import MageVLPerceptionEncoder

    encoder = MageVLPerceptionEncoder()
    encoder.load_weights(state_dict.items())
    return encoder.to(device=device, dtype=dtype).eval()


def load_full_gate(state_dict, device, dtype):
    from vllm_omni.model_executor.models.mage_vl.gate import MageVLCognitionGate

    gate = MageVLCognitionGate()
    gate.load_weights(state_dict.items())
    return gate.to(device=device, dtype=dtype).eval()


def report(label, ours, reference):
    diff = (ours.float() - reference.float()).abs()
    rel_l2 = (diff.norm() / reference.float().norm()).item()
    cos = torch.nn.functional.cosine_similarity(
        ours.float().flatten(), reference.float().flatten(), dim=0
    ).item()
    print(f"{label:<28} max_abs={diff.max().item():.4e} rel_l2={rel_l2:.4e} cos={cos:.10f}")
    return rel_l2


def main():
    device, dtype = "cuda", torch.float32
    reference, state_dict = load_reference(device, dtype)
    ours = load_ours(state_dict, device, dtype)

    torch.manual_seed(0)
    vision = torch.randn(1, SEGMENTS, PATCHES, 2560, device=device, dtype=dtype)

    with torch.inference_mode():
        reference_tokens = reference.perception_tokens(vision)

        state = ours.new_state(device=device, dtype=dtype)
        full_tokens = ours(vision, state)

        # Same stream, one segment per call: the streaming contract.
        streaming_state = ours.new_state(device=device, dtype=dtype)
        streamed = torch.cat(
            [ours(vision[:, i : i + 1], streaming_state) for i in range(SEGMENTS)], dim=1
        )

    rel_full = report("ours vs reference", full_tokens, reference_tokens)
    rel_stream = report("streamed vs reference", streamed, reference_tokens)
    rel_self = report("streamed vs full-sequence", streamed, full_tokens)

    # The decision itself: p_speak per segment, which is what the streaming policy
    # thresholds. The spec's acceptance is a deviation of at most 1e-2.
    gate = load_full_gate(state_dict, device, dtype)
    with torch.inference_mode():
        reference_scores = torch.softmax(reference(vision).float(), dim=-1)[0, :, 1]
        our_scores = gate.score(vision, gate.new_state(device=device, dtype=dtype))[0]
        streaming_state = gate.new_state(device=device, dtype=dtype)
        streamed_scores = torch.cat(
            [gate.score(vision[:, i : i + 1], streaming_state)[0] for i in range(SEGMENTS)]
        )

    worst = (our_scores - reference_scores).abs().max().item()
    worst_streamed = (streamed_scores - reference_scores).abs().max().item()
    print(f"p_speak reference : {[round(v, 6) for v in reference_scores.tolist()]}")
    print(f"p_speak ours      : {[round(v, 6) for v in our_scores.tolist()]}")
    print(f"p_speak max |diff|: {worst:.3e} (one-shot)  {worst_streamed:.3e} (streamed)")

    ok = (
        rel_full < 1e-4
        and rel_stream < 1e-4
        and rel_self < 1e-5
        and worst < 1e-2
        and worst_streamed < 1e-2
    )
    print(f"GATE_PARITY: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
