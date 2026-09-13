# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Assemble an Omni index around existing, unmodified SolarWM release files."""

import argparse
import json
from pathlib import Path

import diffusers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base, stage, output = args.base.resolve(), args.stage.resolve(), args.output.resolve()
    for path in (
        base / "transformer/config.json",
        base / "vae/Wan2.2_VAE.pth",
        base / "text_encoder/models_t5_umt5-xxl-enc-bf16.pth",
        stage / "model.pt",
        stage / "release-manifest.json",
        stage / "checkpoint-manifest.json",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    index = {
        "_class_name": "SolarWMStage2Pipeline",
        "_diffusers_version": diffusers.__version__,
        "base_path": str(base),
        "stage_path": str(stage),
    }
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "model_index.json"
    if index_path.exists() and json.loads(index_path.read_text()) != index:
        raise FileExistsError(f"Refusing to replace a different model index: {index_path}")
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    config_dir = output / "transformer"
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "config.json"
    config = (base / "transformer/config.json").read_text()
    if config_path.exists() and config_path.read_text() != config:
        raise FileExistsError(config_path)
    config_path.write_text(config)
    print(index_path)


if __name__ == "__main__":
    main()
