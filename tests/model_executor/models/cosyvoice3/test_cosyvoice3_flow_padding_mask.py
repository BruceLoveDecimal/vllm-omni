# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Right-padded rows of a batched flow call must not reach each other's valid frames.

Batched flow pads every row to the call's longest. The TensorRT estimator
runs the exported DiT, whose attention mask is an engine input built on the
host, so that map has to carry the key padding for the padded frames to stay
out of the softmax. These tests fill the padding with large values and check
each row against the same row run alone at its own length, on the export view
of the DiT and on the exported ONNX graph itself.
"""

import pytest
import torch

from vllm_omni.diffusion.models.cosyvoice3_audio.cosyvoice3_dit import DiT
from vllm_omni.model_executor.models.cosyvoice3.flow_estimator_trt import _EstimatorWithMaskInput
from vllm_omni.model_executor.models.cosyvoice3.utils import build_dit_attention_mask

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

BLOCK = 50
# Shorter than, across and at a block boundary; the longest sets the width.
ROW_FRAMES = (37, 61, 100, 100)


def _tiny_dit() -> DiT:
    torch.manual_seed(0)
    dit = DiT(dim=64, depth=2, heads=2, dim_head=32, dropout=0.0, mel_dim=80, mu_dim=80, spk_dim=80)
    dit.static_chunk_size = BLOCK
    return dit.eval()


def _row(frames: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    return {
        "x": torch.randn(1, 80, frames, generator=g),
        "mask": torch.ones(1, 1, frames),
        "mu": torch.randn(1, 80, frames, generator=g),
        "t": torch.rand(1, generator=g),
        "spks": torch.randn(1, 80, generator=g),
        "cond": torch.randn(1, 80, frames, generator=g),
    }


def _pad_batch(rows, width: int):
    """Right-pad ``rows`` to ``width`` with large garbage, not zeros, so a leak shows."""
    g = torch.Generator().manual_seed(123)
    batch = {}
    for name in ("x", "mu", "cond"):
        padded = []
        for row in rows:
            frames = row[name].shape[-1]
            garbage = 50.0 * torch.randn(1, 80, width - frames, generator=g)
            padded.append(torch.cat([row[name], garbage], dim=-1))
        batch[name] = torch.cat(padded)
    batch["mask"] = torch.cat(
        [torch.nn.functional.pad(row["mask"], (0, width - row["mask"].shape[-1])) for row in rows]
    )
    batch["t"] = torch.cat([row["t"] for row in rows])
    batch["spks"] = torch.cat([row["spks"] for row in rows])
    return batch


def _attn_mask(pad_mask: torch.Tensor, streaming: bool) -> torch.Tensor:
    return build_dit_attention_mask(pad_mask.bool(), streaming=streaming, static_chunk_size=BLOCK)


def _run_view(view, inputs, attn_mask):
    names = ("x", "mask", "mu", "t", "spks", "cond")
    with torch.inference_mode():
        return view(*(inputs[name] for name in names), attn_mask)


def _rows_and_batch():
    rows = [_row(frames, seed) for seed, frames in enumerate(ROW_FRAMES)]
    return rows, _pad_batch(rows, max(ROW_FRAMES))


@pytest.mark.parametrize("streaming", [True, False])
def test_padded_batch_matches_each_row_alone(streaming):
    view = _EstimatorWithMaskInput(_tiny_dit()).eval()
    rows, batch = _rows_and_batch()

    batched = _run_view(view, batch, _attn_mask(batch["mask"], streaming))

    for index, (row, frames) in enumerate(zip(rows, ROW_FRAMES)):
        alone = _run_view(view, row, _attn_mask(row["mask"], streaming))
        torch.testing.assert_close(batched[index : index + 1, :, :frames], alone, rtol=0, atol=1e-4)
        # Padded frames come out zeroed, so nothing downstream picks them up.
        assert batched[index, :, frames:].eq(0).all()


@pytest.mark.parametrize("streaming", [True, False])
def test_padding_leaks_without_key_padding_in_the_map(streaming):
    """The test above is sensitive: a map without key padding lets the garbage in."""
    view = _EstimatorWithMaskInput(_tiny_dit()).eval()
    rows, batch = _rows_and_batch()
    no_key_padding = _attn_mask(torch.ones_like(batch["mask"]), streaming)

    batched = _run_view(view, batch, no_key_padding)

    shortest = ROW_FRAMES[0]
    alone = _run_view(view, rows[0], _attn_mask(rows[0]["mask"], streaming))
    assert (batched[:1, :, :shortest] - alone).abs().max() > 1e-2


@pytest.mark.parametrize("streaming", [True, False])
def test_exported_graph_masks_padding(tmp_path, streaming):
    """The ONNX TensorRT builds from masks padding the same way, at a batch
    and lengths other than the trace's."""
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

    from vllm_omni.model_executor.models.cosyvoice3.flow_estimator_trt import export_chunk_mask_estimator_onnx

    path = export_chunk_mask_estimator_onnx(_tiny_dit(), str(tmp_path / "est.onnx"), fp16=False)
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def run(inputs, attn_mask):
        feeds = {name: inputs[name].numpy() for name in ("x", "mask", "mu", "t", "spks", "cond")}
        feeds["attn_mask"] = attn_mask.numpy()
        (out,) = session.run(None, feeds)
        return torch.from_numpy(out)

    rows, batch = _rows_and_batch()
    batched = run(batch, _attn_mask(batch["mask"], streaming))

    for index, (row, frames) in enumerate(zip(rows, ROW_FRAMES)):
        alone = run(row, _attn_mask(row["mask"], streaming))
        torch.testing.assert_close(batched[index : index + 1, :, :frames], alone, rtol=0, atol=1e-4)
