# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

spec = importlib.util.spec_from_file_location("auk_sim", Path(__file__).parents[2] / "benchmarks/auk/seed_tts_sim.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_manifest_is_reproducible_and_balanced(tmp_path):
    for language in ("en", "zh"):
        directory = tmp_path / language
        directory.mkdir()
        sf.write(directory / "ref.wav", np.zeros(2400), 24000)
        sf.write(directory / "target.wav", np.zeros(4800), 24000)
        (directory / "meta.lst").write_text(
            "\n".join(f"{i}|reference text|ref.wav|target text {i}|target.wav" for i in range(10))
        )
    args = SimpleNamespace(dataset=tmp_path, languages="en,zh", seed=42, per_language=3, manifest=tmp_path / "one.json")
    module.prepare(args)
    first = args.manifest.read_bytes()
    module.prepare(args)
    assert first == args.manifest.read_bytes()
    data = module.read_json(args.manifest)
    assert len(data["samples"]) == 6
    assert [s["language"] for s in data["samples"]].count("en") == 3
    assert all(s["seconds"] > 0 for s in data["samples"])
    args.seed = 43
    module.prepare(args)
    assert first != args.manifest.read_bytes()


def test_score_summary_requires_complete_finite_pairs():
    rows = [
        dict(id=language, language=language, mode=mode, sim=0.8 if mode == "with_reference" else 0.2)
        for language in ("en", "zh")
        for mode in module.MODES
    ]
    assert module.summarize(rows, 4)["paired_delta"]["mean"] == pytest.approx(0.6)
    with pytest.raises(ValueError):
        module.summarize(rows[:-1], 4)
    with pytest.raises(ValueError):
        module.summarize(rows[:3] + [rows[0]], 4)
    rows[0]["sim"] = float("nan")
    with pytest.raises(ValueError):
        module.summarize(rows, 4)
