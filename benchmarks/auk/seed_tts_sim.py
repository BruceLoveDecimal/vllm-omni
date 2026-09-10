# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired AuK evaluation using seed-tts-eval's official WavLM SIM scorer."""

import argparse
import ast
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
from pathlib import Path

MODES = ("with_reference", "without_reference")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def official_edits(reference_root):
    root = Path(reference_root).resolve()
    source = root / "src/auk/infer/infer_gradio.py"
    tree = ast.parse(source.read_text())
    groups = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "DEMO_EXAMPLE_GROUPS"
    )
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    choices = [
        ("Enhancement & Separation", "Speech Enhancement"),
        ("Paralinguistic Editing", "Emotion Editing"),
        ("Paralinguistic Editing", "Timbre Editing"),
        ("Paralinguistic Editing", "Whisper Conversion"),
    ]
    result = []
    for group, category in choices:
        for index, (audio, instruction, seconds) in enumerate(groups[group][category]):
            path = root / audio
            result.append(
                dict(
                    id="edit_" + category.lower().replace(" ", "_") + "_" + str(index),
                    language="en" if category == "Emotion Editing" else "zh",
                    reference=str(path),
                    reference_sha256=digest(path),
                    instruction=instruction,
                    seconds=seconds,
                    category=category,
                    source=f"https://github.com/Tencent-Hunyuan/AuK/blob/{commit}/src/auk/infer/infer_gradio.py",
                )
            )
    return result


def prepare(args):
    import soundfile as sf

    root = Path(args.dataset).resolve()
    rng = random.Random(args.seed)
    samples, excluded = [], []
    for language in args.languages.split(","):
        meta = root / language / "meta.lst"
        candidates = []
        for line in meta.read_text().splitlines():
            fields = line.strip().split("|")
            if len(fields) not in (4, 5):
                raise ValueError(f"Expected four or five metadata fields: {meta}: {line}")
            utt, prompt_text, prompt_wav, text = fields[:4]
            reference = (meta.parent / prompt_wav).resolve()
            ref_info = sf.info(reference)
            ref_seconds = ref_info.frames / ref_info.samplerate
            seconds = ref_seconds * len(text.encode("utf-8")) / max(1, len(prompt_text.encode("utf-8")))
            seconds = math.ceil(seconds * 50) / 50
            identifier = language + "_" + Path(utt).stem
            if not text.strip() or seconds <= 0 or ref_seconds < 0.02:
                raise ValueError(f"Invalid sample: {identifier}")
            # Select from the supported duration range, recording every exclusion.
            if math.ceil(seconds * 50) + math.floor(ref_seconds * 50) > 1500:
                excluded.append({"id": identifier, "reason": "reference + target exceeds 30 seconds"})
                continue
            candidates.append(
                dict(
                    id=identifier,
                    language=language,
                    text=text,
                    prompt_text=prompt_text,
                    reference=str(reference),
                    seconds=seconds,
                    reference_sha256=digest(reference),
                )
            )
        candidates.sort(key=lambda row: row["id"])
        if len(candidates) < args.per_language:
            raise ValueError(f"Not enough eligible {language} samples")
        samples.extend(rng.sample(candidates, args.per_language))
    if len({row["id"] for row in samples}) != len(samples):
        raise ValueError("Duplicate sample IDs")
    write_json(
        args.manifest,
        dict(
            seed=args.seed,
            samples=samples,
            excluded=excluded,
            edits=official_edits(args.auk_reference) if getattr(args, "auk_reference", None) else [],
            duration_policy=(
                "reference duration times UTF-8 text length ratio, rounded to 20 ms; shared by both backends/modes"
            ),
            selection="seeded sampling without replacement, equal counts per selected language",
        ),
    )
    print(f"Prepared {len(samples)} paired samples; excluded {len(excluded)} overlength samples")


def generate(args):
    import numpy as np
    import soundfile as sf
    import torch

    from vllm_omni.diffusion.models.auk.request import parse_request

    manifest = read_json(args.manifest)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = dict(
        manifest_sha256=digest(args.manifest),
        model=args.model,
        qwen_path=args.qwen_path,
        seed=manifest["seed"],
        steps=args.steps,
        cfg_strength=args.cfg_strength,
        backend=args.backend,
        harness_sha256=digest(__file__),
    )
    run_file = output / "run.json"
    if run_file.exists() and read_json(run_file) != config:
        raise ValueError("Output directory belongs to a different run; use a new directory")
    write_json(run_file, config)
    if args.backend == "native":
        sys.path.insert(0, str(Path(args.auk_reference).resolve() / "src"))
        from auk.infer.infer_auk import AukInfer

        model = AukInfer(
            str(Path(args.model) / "config.yaml"),
            str(Path(args.model) / "auk_base.safetensors"),
            qwen_path=args.qwen_path,
            device="cuda",
            dtype="bf16",
        )
    else:
        from vllm_omni.diffusion.config import set_current_diffusion_config
        from vllm_omni.diffusion.data import OmniDiffusionConfig
        from vllm_omni.diffusion.models.auk.pipeline_auk import AuKPipeline
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        config = OmniDiffusionConfig(
            model=args.model,
            dtype=torch.bfloat16,
            additional_config={"qwen_path": args.qwen_path},
            diffusion_attention_config={"default": {"backend": "TORCH_SDPA"}},
        )
        with set_current_diffusion_config(config):
            model = AuKPipeline(od_config=config)
        model.load_weights(())
    tasks = [(mode, row) for mode in MODES for row in manifest["samples"]]
    tasks += [("instruction", row) for row in manifest.get("edits", [])]
    for index, (mode, row) in enumerate(tasks):
        directory = output / mode
        directory.mkdir(exist_ok=True)
        destination = directory / (row["id"] + ".wav")
        if destination.exists():
            continue
        prompt = {"gen_seconds": row["seconds"]}
        if mode == "instruction":
            prompt["instruction"] = row["instruction"]
        else:
            prompt["input"] = row["text"]
        waveform = None
        if mode != "without_reference":
            if digest(row["reference"]) != row["reference_sha256"]:
                raise ValueError(f"Reference changed: {row['id']}")
            waveform, sr = sf.read(row["reference"], dtype="float32", always_2d=True)
            prompt["ref_audio"] = (waveform.T, sr)
        if args.backend == "native":
            instruction = parse_request(prompt, {}).instruction
            content = [{"type": "text", "text": instruction}]
            if waveform is not None:
                content.append({"type": "audio", "audio": row["reference"]})
            # Upstream's seed only controls target noise; also align VAE sampling.
            torch.manual_seed(manifest["seed"])
            audio, sr = model.generate(
                [{"role": "user", "content": content}],
                audio=None if waveform is None else (torch.from_numpy(waveform.T.copy()), sr),
                gen_seconds=row["seconds"],
                nfe=args.steps,
                cfg_strength=args.cfg_strength,
                seed=manifest["seed"],
            )
        else:
            params = OmniDiffusionSamplingParams(
                seed=manifest["seed"], num_inference_steps=args.steps, extra_args={"cfg_strength": args.cfg_strength}
            )
            batch = DiffusionRequestBatch(
                [OmniDiffusionRequest(prompt=prompt, sampling_params=params, request_id=row["id"])]
            )
            audio, sr = model(batch)[0].output, 24000
        audio = audio.cpu().numpy().squeeze()
        if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
            raise ValueError(f"Invalid generated audio: {row['id']}")
        temporary = destination.with_suffix(".tmp.wav")
        sf.write(temporary, audio, sr, subtype="FLOAT")
        temporary.replace(destination)
        print(f"{args.backend} {mode} {index + 1}/{len(tasks)} {row['id']}", flush=True)


def summarize(rows, expected):
    if len(rows) != expected or len({(r["id"], r["mode"]) for r in rows}) != expected:
        raise ValueError("Incomplete or duplicate score rows")
    groups = {}
    for mode in sorted({r["mode"] for r in rows}):
        for language in ("all", *sorted({r["language"] for r in rows})):
            values = [r["sim"] for r in rows if r["mode"] == mode and (language == "all" or r["language"] == language)]
            if not values:
                continue
            if not all(math.isfinite(v) for v in values):
                raise ValueError("Missing or non-finite scores")
            groups[f"{mode}/{language}"] = dict(
                n=len(values),
                mean=statistics.mean(values),
                median=statistics.median(values),
                stdev=statistics.stdev(values) if len(values) > 1 else 0,
            )
    pairs = {}
    for row in rows:
        pairs.setdefault(row["id"], {})[row["mode"]] = row["sim"]
    deltas = [p["with_reference"] - p["without_reference"] for p in pairs.values() if "with_reference" in p]
    groups["paired_delta"] = dict(n=len(deltas), mean=statistics.mean(deltas))
    return groups


def score(args):
    manifest = read_json(args.manifest)
    root = Path(args.output).resolve()
    upstream = Path(args.seed_tts_eval).resolve()
    verification_dir = upstream / "thirdparty/UniSpeech/downstreams/speaker_verification"
    script = verification_dir / "verification_pair_list_v2.py"
    rows = []
    for mode in (*MODES, *(("instruction",) if manifest.get("edits") else ())):
        directory = root / mode
        pairs, expected = [], {}
        for row in manifest["edits"] if mode == "instruction" else manifest["samples"]:
            audio = directory / (row["id"] + ".wav")
            if not audio.is_file():
                raise FileNotFoundError(audio)
            reference = Path(args.native_output).resolve() / mode / (row["id"] + ".wav")
            if not reference.is_file():
                raise FileNotFoundError(reference)
            pairs.append(f"{audio}|{reference}")
            expected[f"{audio}_0_-1|{reference}_0_-1"] = row
        pair_file = directory / "sim_pairs.lst"
        pair_file.write_text("\n".join(pairs) + "\n")
        scores = directory / "sim_scores.txt"
        subprocess.run(
            [
                sys.executable,
                str(script),
                str(pair_file),
                "--model_name",
                "wavlm_large",
                "--checkpoint",
                str(Path(args.wavlm_checkpoint).resolve()),
                "--scores",
                str(scores),
                "--wav1_start_sr",
                "0",
                "--wav2_start_sr",
                "0",
                "--wav1_end_sr",
                "-1",
                "--wav2_end_sr",
                "-1",
                "--device",
                args.device,
            ],
            cwd=verification_dir,
            check=True,
        )
        seen = set()
        for line in scores.read_text().splitlines():
            if line.startswith("avg score:"):
                continue
            key, value = line.rsplit("\t", 1)
            if key not in expected or key in seen:
                raise ValueError(f"Unexpected or duplicate score: {key}")
            seen.add(key)
            sample = expected[key]
            rows.append(dict(id=sample["id"], language=sample["language"], mode=mode, sim=float(value)))
        if seen != set(expected):
            raise ValueError(f"Official scorer skipped {len(expected) - len(seen)} samples in {mode}")
    report = dict(
        results=summarize(rows, 2 * len(manifest["samples"]) + len(manifest.get("edits", []))),
        scores=rows,
        manifest_sha256=digest(args.manifest),
        wavlm_sha256=digest(args.wavlm_checkpoint),
        evaluator_commit=subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip(),
        interpretation=(
            "SIM compares vLLM-Omni-generated audio against native AuK-generated audio, "
            "separately for each reference mode."
        ),
    )
    write_json(root / "sim_report.json", report)
    print(json.dumps(report["results"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--dataset", required=True)
    prep.add_argument("--auk-reference", help="Add unmodified official editing examples from this checkout")
    prep.add_argument("--manifest", required=True)
    prep.add_argument("--per-language", type=int, default=50)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--languages", default="en,zh")
    gen = commands.add_parser("generate")
    gen.add_argument("--manifest", required=True)
    gen.add_argument("--output", required=True)
    gen.add_argument("--model", required=True)
    gen.add_argument("--backend", choices=("native", "omni"), required=True)
    gen.add_argument("--auk-reference")
    gen.add_argument("--qwen-path")
    gen.add_argument("--steps", type=int, default=32)
    gen.add_argument("--cfg-strength", type=float, default=2.0)
    sim = commands.add_parser("score")
    sim.add_argument("--manifest", required=True)
    sim.add_argument("--output", required=True)
    sim.add_argument("--native-output", required=True)
    sim.add_argument("--seed-tts-eval", required=True)
    sim.add_argument("--wavlm-checkpoint", required=True)
    sim.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    {"prepare": prepare, "generate": generate, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
