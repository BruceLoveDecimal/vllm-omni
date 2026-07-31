# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.diffusion.models.mage_flow.autoencoder_mage import MageVAE

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_mage_vae_maps_relative_checkpoint_names():
    weights = [
        ("pipeline.y_embedder.encoder.weight", object()),
        ("pipeline.y_embedder.bottleneck.weight", object()),
        ("student.dconv_encoder.proj.weight", object()),
        ("pipeline.final_layer.weight", object()),
    ]

    mapped_names = [name for name, _ in MageVAE.hf_to_vllm_mapper.apply(weights)]

    assert mapped_names == [
        "dconv_encoder.proj.weight",
        "decoder_model.final_layer.weight",
    ]
