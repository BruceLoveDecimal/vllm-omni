#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Aggregate the AuK matrix results written by run_matrix.sh.

Reads ``<results>/<ckpt>/<arm>/<task>_<workload>_c<conc>_r<rep>.json`` plus
the per-arm ``memory.csv`` / ``runs.csv`` sidecar files and writes
``summary.csv`` and ``summary.md``.

The first repeat of every cell pays the per-shape CUDA graph captures (one
per distinct reference length), so the headline numbers are the warm
steady state: the best repeat (highest throughput, lowest latency). The
first repeat is kept as ``cold_*`` so that capture cost stays visible.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import re
from collections import defaultdict
from pathlib import Path

_RUN_RE = re.compile(r"^(?P<task>[a-z_]+)_(?P<workload>uniform|mixed|random)_c(?P<conc>\d+)_r(?P<rep>\d+)\.json$")

# (column, candidate JSON keys in preference order)
_METRICS = [
    ("req_per_s", ["request_throughput"]),
    ("audio_s_per_s", ["audio_throughput"]),
    ("rtf_mean", ["mean_audio_rtf"]),
    ("ttfa_p50_ms", ["median_audio_ttfp_ms", "median_ttft_ms"]),
    ("ttfa_p99_ms", ["p99_audio_ttfp_ms", "p99_ttft_ms"]),
    ("e2el_mean_ms", ["mean_e2el_ms"]),
    ("e2el_p50_ms", ["median_e2el_ms"]),
    ("e2el_p90_ms", ["p90_e2el_ms"]),
    ("e2el_p99_ms", ["p99_e2el_ms"]),
    ("completed", ["completed"]),
]
_ARM_ORDER = ["base", "E", "B", "V", "EB", "EV", "BV", "EBV"]
# Higher is better for these; everything else is a latency or a size.
_MAX_IS_BEST = {"req_per_s", "audio_s_per_s", "completed"}


def _pick(result: dict, keys: list[str]) -> float | None:
    for key in keys:
        value = result.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    # vllm nests percentiles as {"percentiles_e2el_ms": {"99": ...}} in some versions.
    for key in keys:
        m = re.match(r"p(\d+)_(.+)", key)
        if not m:
            continue
        nested = result.get(f"percentiles_{m.group(2)}")
        if isinstance(nested, dict):
            value = nested.get(m.group(1)) or nested.get(float(m.group(1))) or nested.get(f"{m.group(1)}.0")
            if isinstance(value, (int, float)):
                return float(value)
    return None


def _load_runs(arm_dir: Path) -> dict[str, tuple[float, float]]:
    """Map run filename to (start, end) wall-clock from runs.csv."""
    windows: dict[str, tuple[float, float]] = {}
    path = arm_dir / "runs.csv"
    if not path.is_file():
        return windows
    for row in csv.reader(path.open()):
        if len(row) < 3:
            continue
        try:
            windows[row[0]] = (float(row[1]), float(row[2]))
        except ValueError:
            continue
    return windows


def _load_memory(arm_dir: Path) -> list[tuple[float, float | None, float | None]]:
    path = arm_dir / "memory.csv"
    samples: list[tuple[float, float | None, float | None]] = []
    if not path.is_file():
        return samples
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = float(row["ts"])
            except (KeyError, ValueError):
                continue
            used = float(row["gpu_mem_used_mib"]) if row.get("gpu_mem_used_mib") else None
            peak = float(row["stage1_peak_mb"]) if row.get("stage1_peak_mb") else None
            samples.append((ts, used, peak))
    return samples


def _window_max(samples, start: float, end: float, index: int) -> float | None:
    values = [s[index] for s in samples if start <= s[0] <= end and s[index] is not None]
    return max(values) if values else None


_PEAK_RE = re.compile(r"(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d).*Peak GPU memory \(this request\): ([\d.]+) GB reserved, ([\d.]+) GB allocated")


def _load_peaks(arm_dir: Path, year: int) -> list[tuple[float, float, float]]:
    """(epoch, reserved_gb, allocated_gb) per stage-1 forward, from the server log."""
    import datetime

    path = arm_dir / "server.log"
    out: list[tuple[float, float, float]] = []
    if not path.is_file():
        return out
    for line in path.open(errors="ignore"):
        m = _PEAK_RE.search(line)
        if not m:
            continue
        mo, da, h, mi, s = (int(x) for x in m.groups()[:5])
        ts = datetime.datetime(year, mo, da, h, mi, s, tzinfo=datetime.timezone.utc).timestamp()
        out.append((ts, float(m.group(6)), float(m.group(7))))
    return out


def _static(arm_dir: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    path = arm_dir / "static.txt"
    if path.is_file():
        for line in path.read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                try:
                    out[key.strip()] = float(value)
                except ValueError:
                    pass
    return out


def collect(results: Path) -> list[dict]:
    rows: list[dict] = []
    for ckpt_dir in sorted(p for p in results.iterdir() if p.is_dir()):
        for arm_dir in sorted(p for p in ckpt_dir.iterdir() if p.is_dir()):
            windows = _load_runs(arm_dir)
            memory = _load_memory(arm_dir)
            year = datetime.datetime.now(datetime.timezone.utc).year
            peaks = _load_peaks(arm_dir, year)
            static = _static(arm_dir)
            groups: dict[tuple, list[dict]] = defaultdict(list)
            for path in sorted(arm_dir.glob("*.json")):
                m = _RUN_RE.match(path.name)
                if not m:
                    continue
                try:
                    result = json.loads(path.read_text())
                except json.JSONDecodeError:
                    continue
                entry = {col: _pick(result, keys) for col, keys in _METRICS}
                if path.name in windows:
                    start, end = windows[path.name]
                    entry["gpu_used_peak_mib"] = _window_max(memory, start, end, 1)
                    entry["stage1_peak_mb"] = _window_max(memory, start, end, 2)
                    # server-log timestamps are whole seconds; widen the window by one on each side
                    entry["stage1_alloc_peak_gb"] = _window_max(peaks, start - 1, end + 1, 2)
                    entry["stage1_reserved_peak_gb"] = _window_max(peaks, start - 1, end + 1, 1)
                else:
                    entry["gpu_used_peak_mib"] = None
                    entry["stage1_peak_mb"] = None
                    entry["stage1_alloc_peak_gb"] = None
                    entry["stage1_reserved_peak_gb"] = None
                entry["rep"] = int(m["rep"])
                groups[(m["task"], m["workload"], int(m["conc"]))].append(entry)
            for (task, workload, conc), entries in sorted(groups.items()):
                row = {
                    "ckpt": ckpt_dir.name,
                    "arm": arm_dir.name,
                    "task": task,
                    "workload": workload,
                    "conc": conc,
                    "repeats": len(entries),
                    "idle_mib": static.get("idle_after_probe_mib"),
                }
                for col in list(dict(_METRICS)) + ["gpu_used_peak_mib", "stage1_peak_mb", "stage1_alloc_peak_gb", "stage1_reserved_peak_gb"]:
                    values = [e[col] for e in entries if e.get(col) is not None]
                    if not values:
                        row[col] = None
                    elif col in _MAX_IS_BEST or col.endswith("_mb") or col.endswith("_mib"):
                        row[col] = max(values)
                    else:
                        row[col] = min(values)
                first = min(entries, key=lambda e: e["rep"])
                row["cold_e2el_p50_ms"] = first.get("e2el_p50_ms")
                row["cold_e2el_p99_ms"] = first.get("e2el_p99_ms")
                row["cold_req_per_s"] = first.get("req_per_s")
                rows.append(row)
    return rows


def _fmt(value, digits=1) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def write_markdown(rows: list[dict], path: Path) -> None:
    lines = ["# AuK perf matrix", ""]
    keys = sorted({(r["ckpt"], r["task"], r["workload"]) for r in rows})
    for ckpt, task, workload in keys:
        lines.append(f"## {ckpt} / {task} / {workload}")
        lines.append("")
        subset = [r for r in rows if (r["ckpt"], r["task"], r["workload"]) == (ckpt, task, workload)]
        concs = sorted({r["conc"] for r in subset})
        lines.append("| arm | metric | " + " | ".join(f"c={c}" for c in concs) + " |")
        lines.append("|---|---|" + "---|" * len(concs))
        for arm in sorted({r["arm"] for r in subset}, key=lambda a: _ARM_ORDER.index(a) if a in _ARM_ORDER else 99):
            by_conc = {r["conc"]: r for r in subset if r["arm"] == arm}
            for col, digits in (
                ("req_per_s", 2),
                ("audio_s_per_s", 1),
                ("e2el_p50_ms", 0),
                ("e2el_p99_ms", 0),
                ("ttfa_p50_ms", 0),
                ("cold_e2el_p50_ms", 0),
                ("stage1_alloc_peak_gb", 2),
                ("stage1_reserved_peak_gb", 2),
                ("gpu_used_peak_mib", 0),
            ):
                cells = [_fmt(by_conc[c][col], digits) if c in by_conc else "-" for c in concs]
                lines.append(f"| {arm} | {col} | " + " | ".join(cells) + " |")
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="Output directory (default: results dir)")
    args = parser.parse_args()
    out = args.out or args.results
    rows = collect(args.results)
    if not rows:
        raise SystemExit(f"no results under {args.results}")
    columns = list(rows[0].keys())
    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, out / "summary.md")
    print(f"{len(rows)} rows -> {out / 'summary.csv'}, {out / 'summary.md'}")


if __name__ == "__main__":
    main()
