# SolarWM checkpoint assembly

`prepare_checkpoint.py` creates a vLLM-Omni index pointing to the existing
SolarWM-5B base and Stage2 release folders. It does not duplicate the weights.

Generation uses the shared
[`image_to_video.py`](../image_to_video/image_to_video.py) entrypoint.
See the [SolarWM recipe](../../../recipes/SolarWM/solarwm-5b-rtx-pro-6000.md)
for pinned checkpoints, camera inputs, two-minute generation, and validation.
