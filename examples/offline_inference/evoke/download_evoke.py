# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Download and assemble the Alaya-EVOKE real-time model directory for vLLM-Omni.

AlayaLab/Evoke is a multi-checkpoint repository, not a loadable pipeline root.
This fetches only what the real-time model needs and assembles:

    <output>/
      model_index.json   (_class_name: EvokePipeline)
      transformer/       evoke/stage3_post_distillation/transformer
      vae/ text_encoder/ tokenizer/ scheduler/   from evoke-base/
      vigeo/vigeo.pt     pkqbajng/ViGeo (CC-BY-NC-4.0)

Components are symlinked into the download cache, so nothing is stored twice.
About 85 GB is downloaded (transformer 57 GB, UMT5 23 GB, ViGeo 5 GB).

    python examples/offline_inference/evoke/download_evoke.py --output /models/evoke-realtime
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

EVOKE_REPO = "AlayaLab/Evoke"
EVOKE_REVISION = "7fa34ecef85754fde6f08996b1ece9d195dcd2f4"
VIGEO_REPO = "pkqbajng/ViGeo"
VIGEO_REVISION = "320bbfa59000df0732afe54511cd7669bb342386"
TRANSFORMER = "evoke/stage3_post_distillation/transformer"
BASE_COMPONENTS = ("vae", "text_encoder", "tokenizer", "scheduler")


def _link(target: Path, link: Path) -> None:
    if link.is_symlink() or link.exists():
        if link.is_symlink() and link.resolve() == target.resolve():
            return
        raise FileExistsError(f"{link} already exists and points elsewhere; choose an empty --output")
    link.symlink_to(target, target_is_directory=target.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, help="Directory to assemble the model in.")
    parser.add_argument("--cache-dir", default=None, help="Hugging Face cache directory.")
    parser.add_argument("--skip-vigeo", action="store_true", help="Text-to-video only; no camera geometry.")
    args = parser.parse_args()

    evoke = Path(
        snapshot_download(
            EVOKE_REPO,
            revision=EVOKE_REVISION,
            cache_dir=args.cache_dir,
            allow_patterns=[f"{TRANSFORMER}/*", *(f"evoke-base/{c}/*" for c in BASE_COMPONENTS)],
        )
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    _link(evoke / TRANSFORMER, output / "transformer")
    for component in BASE_COMPONENTS:
        _link(evoke / "evoke-base" / component, output / component)

    model_index = {
        "_class_name": "EvokePipeline",
        "_evoke_source": {"repo": EVOKE_REPO, "revision": EVOKE_REVISION, "transformer": TRANSFORMER},
    }
    if not args.skip_vigeo:
        vigeo = Path(snapshot_download(VIGEO_REPO, revision=VIGEO_REVISION, cache_dir=args.cache_dir))
        (output / "vigeo").mkdir(exist_ok=True)
        _link(vigeo / "vigeo.pt", output / "vigeo" / "vigeo.pt")
        model_index["_vigeo_source"] = {"repo": VIGEO_REPO, "revision": VIGEO_REVISION}
    (output / "model_index.json").write_text(json.dumps(model_index, indent=2) + "\n")
    print(f"Assembled Evoke real-time model at {os.fspath(output)}")


if __name__ == "__main__":
    main()
