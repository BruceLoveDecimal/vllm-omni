# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from vllm_omni.diffusion.models.solarwm.pipeline import get_solarwm_post_process_func, prepare_camera, prepare_image

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_camera_alignment_rebases_and_inverts(tmp_path):
    # The latent camera positions are source frames 0, 1, 5, not 0, 4, 8.
    poses = np.broadcast_to(np.eye(4, dtype=np.float32), (6, 4, 4)).copy()
    poses[:, 0, 3] = np.arange(6) + 10
    path = tmp_path / "camera.npz"
    np.savez(path, c2w=poses)
    views, _ = prepare_camera(path, 3, "cpu")
    torch.testing.assert_close(views[0, :, 0, 3], torch.tensor([0.0, -1.0, -5.0]))
    with pytest.raises(ValueError, match="cover"):
        prepare_camera(path, 6, "cpu")


def test_latent_camera_validation(tmp_path):
    path = tmp_path / "camera.npz"
    poses = np.broadcast_to(np.eye(4, dtype=np.float32), (3, 4, 4)).copy()
    np.savez(path, viewmats=poses)
    views, ks = prepare_camera(path, 3, "cpu")
    torch.testing.assert_close(views[0], torch.from_numpy(poses))
    assert ks.shape == (1, 3, 3, 3)
    poses[0, 0, 0] = np.nan
    np.savez(path, viewmats=poses)
    with pytest.raises(ValueError, match="finite"):
        prepare_camera(path, 3, "cpu")


def test_first_frame_center_crop_and_video_output():
    pixels = np.zeros((32, 96, 3), dtype=np.uint8)
    pixels[:, 32:64, 0] = 255
    output = prepare_image(Image.fromarray(pixels), 32, 32, "cpu")
    assert output.shape == (1, 3, 1, 32, 32)
    assert (output[:, 0] == 1).all() and (output[:, 1:] == -1).all()
    video = torch.tensor([[[[[255, 0, 0]]], [[[0, 255, 0]]]]], dtype=torch.uint8)
    result = get_solarwm_post_process_func(None)(video, output_type="pil")
    frames = result["payload"]["video"][0]
    assert len(frames) == 2 and frames[1].getpixel((0, 0)) == (0, 255, 0)


@pytest.mark.parametrize("values", [{"num_frames": 0}, {"height": 0}, {"width": -32}])
def test_invalid_geometry_rejected_before_loading_inputs(values):
    from vllm_omni.diffusion.models.solarwm.pipeline import SolarWMStage2Pipeline
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    pipe = SolarWMStage2Pipeline.__new__(SolarWMStage2Pipeline)
    sampling = OmniDiffusionSamplingParams(num_inference_steps=4, guidance_scale=1.0, **values)
    req = SimpleNamespace(prompts=[{"prompt": "test"}], sampling_params=sampling)
    with pytest.raises(ValueError, match="Positive frame count"):
        pipe.forward(req)
