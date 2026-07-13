# Python API

Everything below is importable from the top-level package. All images
are 2D NumPy arrays in count units unless stated otherwise.

## Simulation

```python
import numpy as np
from stemdenoise import make_field, preset, simulate_pair, simulate_n2n_pair

rng = np.random.default_rng(0)

# One clean field with exact ground truth.
fld = make_field(rng, size=256, dose=10.0, spec=preset("hexagonal"))
fld.clean       # expected-counts image
fld.positions   # (N, 2) ground-truth column centres, (row, col) pixels
fld.weights     # (N,) relative column weights

# Noisy view plus ground truth in one call.
noisy, fld = simulate_pair(rng, size=256, dose=10.0)

# Two independent noisy views of the same field (Noise2Noise training).
view_a, view_b, fld = simulate_n2n_pair(rng, size=256, dose=10.0)
```

`dose` is the expected count in the brightest pixel of an isolated
full-weight column. The detector model is Poisson shot noise plus
Gaussian readout noise (sigma 0.5 counts by default).

Presets: `"hexagonal"` (single-species honeycomb) and `"binary_square"`
(rocksalt-like projection, faint sublattice at 0.3 relative weight).
Custom geometry goes through `LatticeSpec`:

```python
from stemdenoise import LatticeSpec
spec = LatticeSpec(kind="hexagonal", spacing_px=9.0, vacancy_frac=0.1)
```

## Classical denoisers

All three run inside a generalized Anscombe variance-stabilizing
transform (except the Gaussian filter, which commutes with it) and
return images in count units:

```python
from stemdenoise import gaussian_denoise, nlm_denoise, wavelet_denoise

restored = nlm_denoise(noisy, h=0.8)
restored = wavelet_denoise(noisy, sigma_scale=1.0)
restored = gaussian_denoise(noisy, sigma=1.5)
```

The transform itself is public for building your own pipeline:

```python
from stemdenoise.classical import anscombe, inverse_anscombe
z = anscombe(noisy)              # approximately unit-variance Gaussian
restored = inverse_anscombe(z)   # bias-corrected closed-form inverse
```

## CNN denoiser

```python
from stemdenoise import load_checkpoint, denoise_counts

model, meta = load_checkpoint("models/unet_supervised.pt")
restored = denoise_counts(model, noisy, dose=10.0)
```

`dose` normalizes the input; for real data of unknown scale use
`stemdenoise.io.estimate_dose(noisy)`. Training from scratch:

```python
from stemdenoise import TrainConfig, train_denoiser

cfg = TrainConfig(mode="noise2noise", dose_min=2.0, dose_max=500.0, steps=2500)
model, history = train_denoiser(cfg, out_path="my_model.pt")
```

## Detection and metrics

```python
from stemdenoise import find_peaks, match_positions, psnr, ssim
from stemdenoise.detect import detector_params_for

det_kw = detector_params_for(spacing_px=12.0, probe_sigma_px=2.2)
found = find_peaks(restored, **det_kw)
score = match_positions(fld.positions, found, tolerance_px=3.0)
score.f1, score.rmse_px, score.precision, score.recall

psnr(fld.clean, restored)   # dB, clean image's range as peak
ssim(fld.clean, restored)
```

## Benchmark harness

```python
from stemdenoise.benchmark import run_benchmark
out = run_benchmark("configs/dose_sweep.yaml")   # writes results/dose_sweep.json
```

Configs are plain YAML; see `configs/` for the two committed ones. The
output JSON stores every aggregated metric plus the tuned parameter
chosen for each classical method in each condition.

## Bring your own data

```python
from stemdenoise.io import load_image, estimate_dose
img = load_image("frame.tif")
scale = estimate_dose(img)
restored = denoise_counts(model, img, dose=scale)
```
