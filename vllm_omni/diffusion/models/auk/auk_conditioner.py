# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frozen Qwen2.5-Omni conditioning and checkpoint-compatible layer fusion."""

import torch
import torch.nn.functional as F
from torch import nn


class AuKConditioner(nn.Module):
    def __init__(self, thinker: nn.Module, processor):
        super().__init__()
        self.text_encoder = thinker
        self.processor = processor
        self.layer_weights = nn.Parameter(torch.zeros(thinker.config.text_config.num_hidden_layers))
        self.layer_scale = nn.Parameter(torch.ones(1))

    def fuse(self, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        # Match the release's stack/sum order, including the final normalized
        # decoder output. The embedding layer is excluded.
        hidden = torch.stack([F.layer_norm(h, [h.shape[-1]]) for h in hidden_states[1:]])
        weights = self.layer_weights.softmax(dim=0)
        return (hidden * weights[:, None, None, None]).sum(dim=0) * self.layer_scale

    @torch.inference_mode()
    def forward(self, instruction: str, audio: torch.Tensor | None, sample_rate: int, device):
        if audio is None and not instruction.endswith("|<no_prompt_audio>|"):
            instruction += "|<no_prompt_audio>|"
        content = [{"type": "text", "text": instruction}]
        if audio is not None:
            content.append({"type": "audio", "audio": "reference.wav"})
        messages = [{"role": "user", "content": content}]
        formatted = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        kwargs = {}
        if audio is not None:
            from vllm_omni.diffusion.models.auk.audio import resample_semantic_audio

            encoder_sr = self.processor.feature_extractor.sampling_rate
            waveform = audio.float().cpu().squeeze(0).numpy()
            # qwen-omni-utils reads reference files through librosa.load;
            # use the same soxr resampler for in-memory audio.
            waveform = resample_semantic_audio(waveform, sample_rate, encoder_sr)
            kwargs["audio"] = [waveform]
            kwargs["use_audio_in_video"] = True
        inputs = self.processor(text=formatted, padding=True, return_tensors="pt", **kwargs).to(device)
        outputs = self.text_encoder(**inputs, output_hidden_states=True, use_cache=False)
        return self.fuse(outputs.hidden_states), inputs["attention_mask"].bool()
