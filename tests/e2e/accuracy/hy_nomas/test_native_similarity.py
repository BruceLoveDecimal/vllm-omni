# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in E2E measurement; quality gates are explicit, never silently calibrated."""

import csv
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tests.e2e.accuracy.helpers import compute_image_ssim_psnr
from tests.helpers.mark import hardware_marks

pytestmark = [pytest.mark.diffusion, pytest.mark.full_model, *hardware_marks(res={"cuda": "H200"}, num_cards=1)]


@pytest.mark.parametrize("scenario,offload", [("video", False), ("session", False), ("session", True)])
def test_native_video_similarity(tmp_path, scenario, offload):
    model = os.environ.get("HY_NOMAS_MODEL")
    if not model or os.environ.get("HY_NOMAS_RUN_COMPARISON") != "1":
        pytest.skip("Set HY_NOMAS_MODEL and HY_NOMAS_RUN_COMPARISON=1 for real-weight measurements")
    component = os.environ["HY_NOMAS_COMPONENT"]
    height = int(os.environ.get("HY_NOMAS_HEIGHT", "704"))
    width = int(os.environ.get("HY_NOMAS_WIDTH", "1280"))
    frames = int(os.environ.get("HY_NOMAS_FRAMES", "121" if scenario == "video" else "49"))
    repeats = int(os.environ.get("HY_NOMAS_REPEATS", "5"))
    root = Path(os.environ.get("HY_NOMAS_RESULTS", str(tmp_path))) / tmp_path.name
    root.mkdir(parents=True, exist_ok=False)
    runner = Path(__file__).with_name("run_backend.py")
    revision = (
        os.environ.get("HY_NOMAS_SOURCE_REVISION")
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    )
    (root / "revision.txt").write_text(revision + "\n")
    # Sequential processes: neither model nor CUDA allocator is shared between backends.
    for backend in ("reference", "native"):
        command = [
            sys.executable,
            str(runner),
            "--model",
            model,
            "--component",
            component,
            "--backend",
            backend,
            "--scenario",
            scenario,
            "--output",
            str(root / backend),
            "--height",
            str(height),
            "--width",
            str(width),
            "--frames",
            str(frames),
            "--steps",
            os.environ.get("HY_NOMAS_STEPS", "8"),
            "--seed",
            os.environ.get("HY_NOMAS_SEED", "42"),
            "--warmups",
            os.environ.get("HY_NOMAS_WARMUPS", "2"),
            "--repeats",
            str(repeats),
        ]
        if offload:
            command.append("--offload")
        with (root / f"{backend}.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                status = process.wait(timeout=6 * 3600)
                assert status == 0, f"{backend} failed: see {root / f'{backend}.log'}"
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
    rows = []
    repeat = 0
    reference = np.load(root / "reference" / f"video_{repeat}.npy", mmap_mode="r")
    actual = np.load(root / "native" / f"video_{repeat}.npy", mmap_mode="r")
    assert reference.shape == actual.shape == (frames * (3 if scenario == "session" else 1), height, width, 3)
    for index, (ref, pred) in enumerate(zip(reference, actual)):
        ssim, psnr = compute_image_ssim_psnr(prediction=Image.fromarray(pred), reference=Image.fromarray(ref))
        rows.append(
            {"repeat": repeat, "frame": index, "ssim": ssim, "psnr": psnr, "exact": bool(np.array_equal(ref, pred))}
        )
    worst = min(rows, key=lambda row: row["ssim"])["frame"]
    Image.fromarray(np.concatenate([reference[worst], actual[worst]], axis=1)).save(root / "worst_frame.png")
    with (root / "frames.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "component": component,
        "scenario": scenario,
        "offload": offload,
        "ssim_mean": float(np.mean([row["ssim"] for row in rows])),
        "ssim_min": min(row["ssim"] for row in rows),
        "psnr_mean": float(np.mean([row["psnr"] for row in rows])),
        "psnr_min": min(row["psnr"] for row in rows),
        "exact_frames": sum(row["exact"] for row in rows),
        "total_frames": len(rows),
        "quality_gate": "unconfigured",
    }
    for backend in ("reference", "native"):
        runs = [json.loads(line) for line in (root / backend / "runs.jsonl").read_text().splitlines()]
        measured = [run for run in runs if run["measured"]]
        summary[backend] = {
            "latency_median_s": float(np.median([run["elapsed_s"] for run in measured])),
            "latency_range_s": [min(run["elapsed_s"] for run in measured), max(run["elapsed_s"] for run in measured)],
            "gpu_sampled_peak_bytes": max(run["gpu_sampled_peak_bytes"] for run in measured),
            "unique_output_hashes": len({run["rgb_sha256"] for run in measured}),
        }
        if scenario == "session":
            summary[backend]["first_output_median_s"] = float(
                np.median([next(turn["available_s"] for turn in run["turns"] if turn["frames"]) for run in measured])
            )
            boundaries = [turn["metadata"]["frame_start"] for turn in measured[0]["turns"][1:]]
            summary[backend]["emission_boundary_frames"] = [
                row for row in rows if any(abs(row["frame"] - boundary) <= 2 for boundary in boundaries)
            ]
    if scenario == "session":
        summary["generation_boundary_frames"] = [
            row for row in rows if any(abs(row["frame"] - boundary) <= 2 for boundary in (frames, 2 * frames))
        ]
    ssim_gate, psnr_gate = os.environ.get("HY_NOMAS_MIN_SSIM"), os.environ.get("HY_NOMAS_MIN_PSNR")
    if ssim_gate is not None and psnr_gate is not None:
        summary["quality_gate"] = {"min_frame_ssim": float(ssim_gate), "min_frame_psnr": float(psnr_gate)}
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if ssim_gate is not None and psnr_gate is not None:
        assert summary["ssim_min"] >= float(ssim_gate)
        assert summary["psnr_min"] >= float(psnr_gate)
