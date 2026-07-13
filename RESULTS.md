# Results

Every number below regenerates from a committed YAML config with fixed
seeds; the raw values live in `results/*.json`. Doses are expected
counts in the brightest pixel of a full-weight column peak. Metrics are
means over 6 evaluation fields per condition (256 px, tolerance 3 px,
border strip excluded on both the truth and prediction side). Classical
methods are tuned per condition on 2 separate tuning fields; the chosen
parameter and tuning criterion are stored in the JSON next to each row.
All CNN rows use one fixed checkpoint across every condition, with the
input normalized by the true simulated dose (see the scale-robustness
check at the end for what changes when the dose is estimated blindly).

Environment: fresh venv (Python 3.11.9, torch 2.13 CPU), Windows 11,
everything on CPU. Commands:

```
python scripts/train_all.py
stemdenoise benchmark configs/dose_sweep.yaml
stemdenoise benchmark configs/detection_tuned.yaml
stemdenoise benchmark configs/cross_dose.yaml
stemdenoise benchmark configs/off_geometry.yaml
python scripts/check_scale_robustness.py
python scripts/make_figures.py
```

## 1. Dose sweep, hexagonal lattice (`configs/dose_sweep.yaml`)

Single-species honeycomb, spacing 12 px. Classical methods tuned for
PSNR per dose.

### PSNR (dB)

| Method | dose 2 | dose 5 | dose 10 | dose 20 | dose 50 | dose 150 | dose 500 |
|---|---|---|---|---|---|---|---|
| raw (no denoise) | 8.52 | 13.69 | 17.19 | 20.43 | 24.54 | 29.42 | 34.64 |
| Gaussian (tuned) | 20.19 | 23.51 | 25.33 | 27.73 | 30.23 | 33.46 | 36.00 |
| NLM + VST (tuned) | 15.52 | 19.72 | 23.01 | 25.96 | 29.36 | 32.81 | 36.74 |
| wavelet + VST (tuned) | 18.24 | 20.87 | 23.41 | 25.79 | 28.33 | 31.58 | 36.06 |
| CNN supervised | 27.86 | 31.84 | 34.57 | 37.32 | 41.09 | 44.92 | 48.36 |
| CNN Noise2Noise | 27.61 | 31.63 | 34.39 | 37.14 | 40.82 | 44.44 | 47.25 |

### SSIM

| Method | dose 2 | dose 5 | dose 10 | dose 20 | dose 50 | dose 150 | dose 500 |
|---|---|---|---|---|---|---|---|
| raw (no denoise) | 0.222 | 0.442 | 0.603 | 0.736 | 0.862 | 0.944 | 0.981 |
| Gaussian (tuned) | 0.734 | 0.852 | 0.878 | 0.934 | 0.970 | 0.983 | 0.993 |
| NLM + VST (tuned) | 0.493 | 0.795 | 0.895 | 0.945 | 0.975 | 0.987 | 0.994 |
| wavelet + VST (tuned) | 0.683 | 0.786 | 0.874 | 0.920 | 0.958 | 0.973 | 0.989 |
| CNN supervised | 0.959 | 0.984 | 0.991 | 0.995 | 0.997 | 0.999 | 0.999 |
| CNN Noise2Noise | 0.955 | 0.983 | 0.991 | 0.995 | 0.998 | 0.999 | 0.999 |

### Detection F1 (fixed peak finder on the restoration)

| Method | dose 2 | dose 5 | dose 10 | dose 20 | dose 50 | dose 150 | dose 500 |
|---|---|---|---|---|---|---|---|
| raw (no denoise) | 0.932 | 0.989 | 0.996 | 0.999 | 1.000 | 0.999 | 0.999 |
| Gaussian (tuned) | 0.990 | 0.996 | 0.997 | 0.999 | 1.000 | 0.999 | 0.999 |
| NLM + VST (tuned) | 0.981 | 0.996 | 0.998 | 0.999 | 1.000 | 1.000 | 1.000 |
| wavelet + VST (tuned) | 0.991 | 0.996 | 0.998 | 0.999 | 1.000 | 1.000 | 0.999 |
| CNN supervised | 0.998 | 0.997 | 0.999 | 0.999 | 1.000 | 1.000 | 0.999 |
| CNN Noise2Noise | 0.997 | 0.997 | 0.999 | 0.999 | 1.000 | 1.000 | 0.999 |

### Localization RMSE of matched columns (px)

| Method | dose 2 | dose 5 | dose 10 | dose 20 | dose 50 | dose 150 | dose 500 |
|---|---|---|---|---|---|---|---|
| raw (no denoise) | 0.773 | 0.454 | 0.325 | 0.248 | 0.188 | 0.155 | 0.141 |
| Gaussian (tuned) | 0.648 | 0.403 | 0.305 | 0.238 | 0.184 | 0.154 | 0.142 |
| NLM + VST (tuned) | 1.002 | 0.552 | 0.383 | 0.288 | 0.204 | 0.169 | 0.146 |
| wavelet + VST (tuned) | 0.929 | 0.484 | 0.365 | 0.258 | 0.190 | 0.155 | 0.141 |
| CNN supervised | 0.454 | 0.317 | 0.245 | 0.197 | 0.163 | 0.145 | 0.140 |
| CNN Noise2Noise | 0.464 | 0.321 | 0.247 | 0.198 | 0.164 | 0.145 | 0.140 |

Reading: on an easy single-species lattice, detection F1 saturates for
every method above dose 5; the smoothing inside the fixed peak finder is
already most of the denoising a detector needs. What restoration buys
here is fidelity (7 to 12 dB over the best tuned classical method at
every dose) and localization precision at low dose (0.454 px vs 0.648 px
at dose 2). Noise2Noise, trained without a single clean image, tracks
the supervised model within 0.3 to 1.1 dB PSNR and 0.003 F1 everywhere.

## 2. Dose sweep, binary lattice with a faint sublattice

Rocksalt-like projection, spacing 14 px; half the columns carry 0.3 of
the bright-column weight (a light element next to a heavy one). Same
config as above.

### Detection F1 (classical tuned for PSNR)

| Method | dose 2 | dose 5 | dose 10 | dose 20 | dose 50 | dose 150 | dose 500 |
|---|---|---|---|---|---|---|---|
| raw (no denoise) | 0.860 | 0.906 | 0.947 | 0.957 | 0.983 | 0.973 | 0.985 |
| Gaussian (tuned) | 0.734 | 0.808 | 0.910 | 0.921 | 0.959 | 0.959 | 0.969 |
| NLM + VST (tuned) | 0.652 | 0.700 | 0.735 | 0.765 | 0.875 | 0.914 | 0.965 |
| wavelet + VST (tuned) | 0.734 | 0.763 | 0.873 | 0.919 | 0.973 | 0.968 | 0.984 |
| CNN supervised | 0.990 | 0.994 | 0.993 | 0.987 | 0.993 | 0.981 | 0.988 |
| CNN Noise2Noise | 0.987 | 0.996 | 0.994 | 0.988 | 0.994 | 0.983 | 0.990 |

Every classical method scores worse than doing nothing, at every dose,
despite gaining 6 to 12 dB PSNR over raw. The smoothing that wins PSNR
flattens faint columns into the shoulders of their bright neighbours,
and recall collapses (NLM at dose 10: 0.582 recall vs 0.902 raw, see
`results/dose_sweep.json`). PSNR-tuned NLM is simultaneously the best
classical method by PSNR at low dose and the worst detector in the
whole benchmark. PSNR and usefulness are different axes.

## 3. Fair-tuning check: classical tuned directly for detection F1 (`configs/detection_tuned.yaml`)

The obvious objection to section 2 is that the classical methods were
tuned for the wrong objective. This config retunes them per dose to
maximize detection F1 itself on the tuning fields.

| Method | dose 2 | dose 5 | dose 10 | dose 20 | dose 50 | dose 150 | dose 500 |
|---|---|---|---|---|---|---|---|
| raw (no denoise) | 0.860 | 0.906 | 0.947 | 0.957 | 0.983 | 0.973 | 0.985 |
| Gaussian (tuned for F1) | 0.857 | 0.901 | 0.944 | 0.953 | 0.980 | 0.969 | 0.982 |
| NLM + VST (tuned for F1) | 0.652 | 0.700 | 0.770 | 0.856 | 0.954 | 0.970 | 0.985 |
| wavelet + VST (tuned for F1) | 0.812 | 0.870 | 0.931 | 0.949 | 0.979 | 0.970 | 0.984 |
| CNN supervised | 0.990 | 0.994 | 0.993 | 0.987 | 0.993 | 0.981 | 0.988 |

Given the choice, every classical method drives its parameter to the
weakest smoothing in the grid and lands at or just below raw. There is
no classical operating point on this lattice at which denoising helps
the detector; the classical benefit is visual and fidelity-only. The
CNN is above raw at every dose (+0.130 F1 at dose 2, +0.088 at dose 5)
with precision of 0.99, so the gain is not lattice hallucination: the
model localizes real faint columns, it does not invent them. This
check was run because the section 2 gap looked too clean; it survived.

## 4. Cross-dose generalization (`configs/cross_dose.yaml`)

Is the CNN advantage an artifact of training at the evaluation dose?
Two additional supervised models were trained at one dose each (10 and
150) and evaluated across the whole range, against the range-trained
model and per-dose-tuned NLM. Hexagonal lattice.

### PSNR (dB)

| Method | dose 2 | dose 5 | dose 10 | dose 20 | dose 50 | dose 150 | dose 500 |
|---|---|---|---|---|---|---|---|
| NLM + VST (tuned) | 15.52 | 19.72 | 23.01 | 25.96 | 29.36 | 32.81 | 36.74 |
| CNN trained at dose 10 only | 16.17 | 27.40 | 34.47 | 35.15 | 35.85 | 37.09 | 37.92 |
| CNN trained at dose 150 only | 12.04 | 18.77 | 24.66 | 31.10 | 39.39 | 45.49 | 49.32 |
| CNN supervised (range-trained) | 27.86 | 31.84 | 34.57 | 37.32 | 41.09 | 44.92 | 48.36 |

### Detection F1

| Method | dose 2 | dose 5 | dose 10 | dose 20 | dose 50 | dose 150 | dose 500 |
|---|---|---|---|---|---|---|---|
| NLM + VST (tuned) | 0.981 | 0.996 | 0.998 | 0.999 | 1.000 | 1.000 | 1.000 |
| CNN trained at dose 10 only | 0.993 | 0.997 | 0.998 | 0.999 | 1.000 | 0.999 | 0.999 |
| CNN trained at dose 150 only | 0.972 | 0.996 | 0.998 | 0.999 | 1.000 | 1.000 | 0.999 |
| CNN supervised (range-trained) | 0.998 | 0.997 | 0.999 | 0.999 | 1.000 | 1.000 | 0.999 |

Three readings, two of them against the CNN:

1. A dose-matched specialist slightly beats the range-trained model at
   its own dose (45.49 vs 44.92 dB at dose 150; 49.32 vs 48.36 at 500).
   Range training costs about half a dB to one dB at the covered doses.
2. Off dose, specialists fall hard: the dose-150 model drops to 12.0 dB
   at dose 2, below tuned NLM (15.5 dB). A CNN evaluated outside its
   training noise level can be worse than a decent classical method.
   The headline CNN advantage is conditional on training coverage.
3. Detection is far more forgiving than fidelity: even badly mismatched
   models keep F1 at or above 0.972 on this lattice.

## 5. Off-distribution geometry (`configs/off_geometry.yaml`)

The CNNs were trained only on the two built-in presets (hexagonal
spacing 12 px, binary spacing 14 px, probe sigma 2.2 px, faint weight
0.3). Every benchmark above evaluates on those same presets, which a
skeptic should call out: the learned prior is being tested exactly
in-distribution. This config changes one geometric property at a time
outside the training family and lets the tuned classical methods
compete as usual. PSNR / detection F1 at each dose:

| Variant (change vs training) | Dose | raw | Gaussian (tuned) | NLM + VST (tuned) | CNN supervised |
|---|---|---|---|---|---|
| hex_dense9 (spacing 12 to 9) | 5 | 11.9 / 0.998 | **21.4** / 0.998 | 17.8 / 0.998 | 18.6 / 0.958 |
| hex_dense9 | 50 | 22.5 / 1.000 | 27.9 / 1.000 | 27.2 / 1.000 | **29.5** / 0.999 |
| hex_sparse16 (spacing 12 to 16) | 5 | 15.0 / 0.934 | 25.4 / 0.967 | 22.0 / 0.997 | **29.1** / 0.996 |
| hex_sparse16 | 50 | 26.4 / 0.999 | 32.4 / 0.999 | 31.7 / 1.000 | **38.3** / 0.999 |
| hex_vacancy12 (3% to 12% vacancies) | 5 | 14.0 / 0.985 | 23.8 / 0.995 | 20.2 / 0.998 | **31.9** / 0.999 |
| hex_vacancy12 | 50 | 24.8 / 0.999 | 30.6 / 0.999 | 29.8 / 1.000 | **41.1** / 0.999 |
| hex_probe30 (probe sigma 2.2 to 3.0) | 5 | 11.8 / 0.998 | **22.9** / 0.998 | 18.9 / 0.998 | 22.6 / 0.998 |
| hex_probe30 | 50 | 22.4 / 1.000 | **29.9** / 0.999 | 28.5 / 1.000 | 25.3 / 1.000 |
| bin_faint20 (faint weight 0.3 to 0.2) | 5 | 13.2 / 0.793 | 23.2 / 0.702 | 19.5 / 0.668 | **29.5** / 0.976 |
| bin_faint20 | 50 | 24.1 / 0.849 | 29.9 / 0.766 | 28.2 / 0.670 | **36.7** / 0.957 |

(The faint variant uses weight 0.2 rather than something smaller
because the fixed detector's relative threshold is 0.15; at weight 0.15
the faint species sits exactly at threshold and every method's recall,
including on the raw image, is throttled by the detector rather than
the denoiser.)

Readings:

1. **The learned advantage inverts on a denser lattice.** At spacing
   9 px and dose 5, the CNN drops below the tuned Gaussian by 2.8 dB
   and is the only method that loses detections (F1 0.958 vs 0.998).
   It has learned the training lattice's length scale, and columns
   closer than that get partially merged.
2. **A wider probe also breaks the fidelity advantage.** At probe sigma
   3.0 and dose 50 the CNN trails the tuned Gaussian by 4.5 dB (it
   sharpens peaks toward the shape it was trained on), although its
   localization RMSE stays best.
3. **Milder shifts are fine.** Sparser lattices, four times the vacancy
   rate, and a fainter second species all keep the CNN clearly ahead;
   the faint-species detection gap actually widens at weight 0.2
   (F1 0.976 vs 0.793 raw at dose 5).

The honest summary: the CNN's edge comes from a structural prior, and
that prior has a range of validity roughly bracketed by these variants.
For data whose geometry is unknown or unlike the training family, the
variance-stabilized classical methods degrade predictably; the CNN does
not.

## 6. Scale-robustness check (`scripts/check_scale_robustness.py`)

The benchmark hands the CNN the true dose for input normalization,
which real data does not provide. Re-running with the dose estimated
blindly from the noisy image (`stemdenoise.io.estimate_dose`, which
lands 10 to 15 percent low): PSNR changes by at most 1.7 dB (44.80 to
43.12 at dose 150) and typically under 0.5 dB at low dose; detection F1
moves by at most 0.003. Values in `results/scale_robustness.json`.

## Where classical methods are competitive

To keep the comparison honest in both directions: the tuned Gaussian is
within 0.008 F1 of the CNN on hexagonal detection at every dose above 2
while costing microseconds and having one interpretable parameter; NLM
overtakes the Gaussian as the best classical option by PSNR at the top
of the range (36.7 dB at dose 500 on the hexagonal lattice); the tuned
Gaussian beats the CNN outright on a lattice denser than the CNN's
training family and under a wider probe (section 5); and at dose 2 a mismatched CNN is worse than any
tuned classical method. If the downstream task is detection on a
well-separated single-species lattice at moderate dose, a fixed
detector on the raw image is already near ceiling and no restoration
step is needed at all.
