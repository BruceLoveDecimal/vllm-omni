# SolarWM native integration

The registered `SolarWMStage2Pipeline` implements the released Wan2.2-5B
Stage2 EMA student through the shared Omni request/output contract.
The runtime never imports SolarWM; reference imports are confined to tests.

## Pinned contract

- Source: [Junchao-cs/SolarWM](https://github.com/Junchao-cs/SolarWM/tree/a3a3fac16466102a2b97df867f7703df7172cb2a),
  revision `a3a3fac16466102a2b97df867f7703df7172cb2a`.
- Weights: `junchaoh-cs/SolarWM`, revision
  `fe1587a9392bb91df5c90bcbe849064a5e88c546`, directories
  `SolarWM-5B-base` and `SolarWM-5B-sgf-stage2-81f`.
- Resolve the role from `release-manifest.json`; the release chooses
  `generator_ema`. Validate the Stage2 flow-matching, linear
  translation, 18-frame window, no-sink contract before strict loading.
- Default: one CUDA GPU, eager, Torch SDPA, resident weights. Other stages,
  variants, parallel layouts and diffusion block caching are not supported.
- Source notices, including the PRoPE MIT notice, are retained in `NOTICE`.

## Architecture and ownership

| Component | Implementation and reuse |
| --- | --- |
| Text | Shared Wan `.pth`→UMT5 converter, Transformers UMT5 encoder and tokenizer; FP32 output, masked padding, 512×4096 context |
| VAE | Shared `OmniAutoencoderKLWan`, original 48-channel statistics and strict Wan2.2 residual-block key conversion |
| Attention | Shared `Attention`/Torch SDPA kernels; model-local PRoPE camera transforms |
| DiT | Native 30-layer, width-3072, 24-head Wan student; original checkpoint names and norm/precision semantics |
| Cache | Per-request raw normalized K, raw V, text KV and FP32 camera metadata; no model-global request cache |
| Sampler | Native Stage2 four-evaluation self-forcing loop, plus clean commit after each chunk |
| Entrypoint | Shared `Omni`, image-to-video CLI, `model_extras` defaults and registry postprocessing |
| Checkpoints | A small separate assembled `model_index.json`; original release files remain unchanged |

### Geometry and cameras

The VAE has spatial compression 16 and temporal compression 4. Patch size
`(1,2,2)` gives 405 tokens per latent frame at 480×864. Three latent frames
form one chunk. Temporal decoding produces `4*latent_frames-3` pixel frames.
The pipeline rounds the latent horizon to full chunks and trims excess output.
1920 requested frames therefore require 483 latents and 161 chunks.

Camera NPZ accepts absolute per-pixel C2W or latent-aligned relative W2C.
C2W source indices are `0,1,5,9,...`: rebase to the first camera in NumPy FP32,
then invert with Torch FP32. Preserve the release's normalized focal lengths;
principal point and skew are discarded by PRoPE.

Within each block: Q/K normalization → window-relative Wan RoPE → camera
projection. With `P=lift(K)@W2C`, transform Q by `P^T`, K/V by `P^-1`, and the
attention output by P. A chunk is internally bidirectional and attends only
its clean history. Token-causal attention would implement the wrong mask.

### Rolling history and numerical closure

Retain five clean three-latent chunks. Before each forward, append the current
three-frame chunk and recompute RoPE relative to the visible 18-frame window.
Cached K is unrotated; retaining old RoPE coordinates after eviction is wrong.
Denoising reads history without committing noisy K/V. A final zero-timestep
forward commits the clean chunk; each request owns its cache and cursor.

Map raw labels `[1000,750,500,250]` through the shifted 1000-point CPU schedule
(`shift=5`). Retain its original sigma values; deriving sigma again from a raw
timestep divided by 1000 on CUDA changes BF16 rounding ties in re-noising. Predict `x0=x_t-t*velocity/1000`, then re-noise with fresh BF16
CUDA noise before the next non-final evaluation. Restore the initial image
latent during every evaluation of the first chunk. The initial noise tensor
is sampled in full BTCHW order to match upstream RNG consumption.
This distilled conditional student uses guidance 1 and no CFG branches.

The reference's autocast boundaries are part of correctness. The DiT uses
BF16 autocast, including FP32-promoted `pow` in QK normalization; LayerNorm
casts its output back to the activation dtype. UMT5 and VAE encode run in
FP32. VAE denormalization runs **inside** decode autocast, where division is
promoted to FP32. Moving these operations outside autocast changes numerical results,
even when a five-second smoke looks plausible. This is explicitly covered
by the CUDA comparison.

VAE decoding uses the existing continuous temporal cache and emits bounded
chunks to CPU uint8 tensors before assembling the output. No repeated clips
or endpoint frame padding are used to reach the requested duration.

## Validation

- Fifteen CPU regressions cover independent camera algebra, window visibility,
  C2W alignment, image crop, output layout, request isolation, clean commit,
  cursor validation and cache eviction.
- `compare_reference.py`: 12 PRoPE/window cases (`max_abs=9.53674316e-7`) and
  24 tiny-DiT forwards including eviction (`max_abs=5.96046448e-8`).
- `qualify_components.py`: actual weights; VAE encode maximum error
  `2.3841858e-7`, VAE decode and UMT5 maximum error 0.
  The 63-latent case crosses the reference 60-latent streaming boundary.
- `qualify_transformer.py`: actual 5B EMA, 480×864 token geometry, real BF16
  autocast, 16 full forwards across eight chunks including eviction. Both
  implementations use SDPA; maximum error 0.
- `qualify_sampler.py`: executes the pinned upstream sampler with an adapter
  to the qualified DiT, isolating RNG, schedule and clean-commit semantics.
  The full 24-latent rollout is bitwise equal to the reference sampler.
- Additional full-horizon qualification: the original upstream `CausalWanModel`
  and original Stage2 sampler matched all 483 native latents bitwise for the
  static-camera cat input (both using SDPA). The two-minute sample still shows
  severe visual degradation; numerical parity is not a quality guarantee.
- `test_solarwm_expansion.py`: real-weight 93-frame shared-client regression;
  nightly X2V selection requires explicitly provisioned model/image assets.

For deployment commands, software/hardware qualification, feature boundaries,
and local/CI-like test commands, see the
[SolarWM recipe](../../../../recipes/SolarWM/solarwm-5b-rtx-pro-6000.md).

Reference comparison commands (a separate clean pinned checkout is required):

```bash
python -m tests.diffusion.models.solarwm.compare_reference /path/to/SolarWM --transformer
python -m tests.diffusion.models.solarwm.qualify_components \
  /path/to/SolarWM /path/to/SolarWM-5B-base --latent-frames 63 --output components.json
python -m tests.diffusion.models.solarwm.qualify_transformer \
  /path/to/SolarWM /path/to/SolarWM-5B-base /path/to/SolarWM-5B-sgf-stage2-81f \
  --output transformer.json
python -m tests.diffusion.models.solarwm.qualify_sampler \
  /path/to/SolarWM /path/to/SolarWM-Omni /path/to/first-frame.jpg --output sampler.json
```
