# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read AuK's native YAML checkpoint layout without importing model code."""

from pathlib import Path

import yaml


def read_auk_config(model: str, revision: str | None = None) -> dict | None:
    path = Path(model) / "config.yaml"
    if not Path(model).is_dir():
        from transformers.utils.hub import cached_file

        resolved = cached_file(
            model,
            "config.yaml",
            revision=revision,
            _raise_exceptions_for_missing_entries=False,
        )
        if resolved is None:
            return None
        path = Path(resolved)
    if not path.is_file():
        return None
    with path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        return None
    model_config = config.get("model", {})
    if not isinstance(model_config, dict) or model_config.get("name") not in ("AuK", "AuK-Base", "AuK-Flash"):
        return None
    for component in ("arch", "vae", "text_encoder"):
        if not isinstance(model_config.get(component), dict):
            raise ValueError(f"AuK config is missing model.{component}")
    from omegaconf import OmegaConf

    # Released VAE configs interpolate latent_dim from model.vae.latent_dim.
    return OmegaConf.to_container(OmegaConf.create(config), resolve=True)


def is_auk_model(model: str, revision: str | None = None) -> bool:
    try:
        return read_auk_config(model, revision) is not None
    except (OSError, ValueError):
        return False
