# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opt-in full thinker generation parity; requires the complete Ming checkpoint."""

import json
import os

import numpy as np
import pytest
from PIL import Image
from vllm import SamplingParams

from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config

pytestmark = [
    pytest.mark.omni,
    pytest.mark.slow,
    pytest.mark.skipif(not os.environ.get("MING_CHECKPOINT"), reason="set MING_CHECKPOINT for full thinker validation"),
]


class EncoderGraphTestWorkerExtension:
    """Inspect real worker state through vLLM's existing worker-extension RPC."""

    def encoder_graph_test_stats(self):
        manager = self.model_runner.encoder_cudagraph_manager
        if manager is None:
            return {"captured": False, "hits": 0, "misses": 0}
        stats = manager.get_cumulative_stats()
        return {
            "captured": stats["num_budgets"] > 0,
            "hits": stats["graph_hits"],
            "misses": stats["graph_misses"],
        }


def test_thinker_encoder_graph_generation_parity(record_property):
    from vllm_omni.entrypoints.omni import Omni

    model = os.environ["MING_CHECKPOINT"]
    deploy_config = os.environ.get("MING_CG_DEPLOY_CONFIG", get_deploy_config_path("ming_flash_omni_thinker_only.yaml"))
    image = Image.new("RGB", (224, 224), color=(220, 40, 30))
    frames = np.stack([np.asarray(image)] * 4)
    audio = (np.zeros(16000, dtype=np.float32), 16000)
    cases = [
        ("text", "Say hello briefly.", {}),
        ("image", "<IMAGE>What is the main color?", {"image": image}),
        ("video", "<VIDEO>Describe the video briefly.", {"video": frames}),
        ("mixed", "<IMAGE><AUDIO>Describe the image and audio briefly.", {"image": image, "audio": audio}),
    ]
    results = {}
    for enabled in [False, True]:
        config = modify_stage_config(
            deploy_config,
            updates={
                "stages": {
                    0: {
                        "compilation_config.cudagraph_mm_encoder": enabled,
                        "worker_extension_cls": (
                            "tests.e2e.offline_inference.test_ming_vision_cudagraph.EncoderGraphTestWorkerExtension"
                        ),
                    }
                }
            },
        )
        omni = Omni(model=model, deploy_config=config, stage_init_timeout=1800, init_timeout=2400, log_stats=True)
        tokens_by_case = {}
        try:
            for name, question, multimodal_data in cases:
                prompt = (
                    "<role>SYSTEM</role>你是一个友好的AI助手。\n\ndetailed thinking off<|role_end|>"
                    f"<role>HUMAN</role>{question}<|role_end|><role>ASSISTANT</role>"
                )
                request = {"prompt": prompt, "modalities": ["text"]}
                if multimodal_data:
                    request["multi_modal_data"] = multimodal_data
                outputs = omni.generate([request], [SamplingParams(temperature=0, max_tokens=8, seed=17)])
                text_outputs = [output for output in outputs if output.final_output_type == "text"]
                assert len(text_outputs) == 1
                completion = text_outputs[0].outputs[0]
                assert completion.token_ids, f"no generated tokens for {name}"
                tokens_by_case[name] = list(completion.token_ids)
                record_property(f"encoder_graph_{enabled}_{name}", completion.text)
                record_property(f"encoder_graph_{enabled}_{name}_tokens", json.dumps(tokens_by_case[name]))
            stats = omni.engine.collective_rpc(method="encoder_graph_test_stats", stage_ids=[0])
            assert stats, "worker statistics RPC returned no stages"
            for stage_stats in stats:
                assert isinstance(stage_stats, list) and stage_stats, f"worker statistics RPC failed: {stage_stats}"
                for rank_stats in stage_stats:
                    assert rank_stats["captured"] == enabled
                    if enabled:
                        assert rank_stats["hits"] >= 2, "requests did not exercise encoder graph replay"
            record_property(f"encoder_graph_{enabled}_stats", json.dumps(stats))
        finally:
            omni.close()
        results[enabled] = tokens_by_case
    assert results[True] == results[False], "greedy thinker tokens differ between encoder eager and graph paths"
