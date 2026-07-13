# Model card: stemdenoise ResUNet denoisers

Four checkpoints, all the same architecture, differing only in training
signal and dose range. All were trained in this repository by
`scripts/train_all.py` with the seeds recorded inside each checkpoint;
no external data, weights or pretrained components were used anywhere.

## Architecture

Residual U-Net (`stemdenoise.net.ResUNet`, base width 24): two encoder
levels, a middle block, two decoder levels with skip connections,
GroupNorm and SiLU throughout, and a zero-initialized 1x1 output head so
the untrained network is exactly the identity. 263,257 parameters,
about 1.1 MB per checkpoint. Input and output are single-channel images
normalized so a full-weight column peak sits near 1.0.

## Checkpoints

| File | Training signal | Dose range | Steps | Seed |
|---|---|---|---|---|
| unet_supervised.pt | noisy to clean (MSE) | 2 to 500, log-uniform | 2500 | 0 |
| unet_n2n.pt | noisy to independent noisy (Noise2Noise) | 2 to 500, log-uniform | 2500 | 1 |
| unet_fixed_low.pt | noisy to clean | 10 only | 1500 | 2 |
| unet_fixed_high.pt | noisy to clean | 150 only | 1500 | 3 |

Training data is simulated on the fly (no fixed dataset): 48 clean
fields per run, mixing the hexagonal and binary_square lattice presets,
96 px patches, batch 16, Adam at 2e-3 with cosine decay. Every noise
realization is fresh, so no noisy image repeats. Full details in
`stemdenoise.train.TrainConfig`; the exact config is stored inside each
checkpoint's metadata.

## Measured performance

All numbers regenerate from `configs/dose_sweep.yaml` and are averaged
over 6 held-out evaluation fields per condition (see
`results/dose_sweep.json` and RESULTS.md for full tables). On the
hexagonal lattice, unet_supervised reaches 27.9 dB PSNR at dose 2
(raw input: 8.5 dB, best tuned classical: 20.2 dB) and 48.4 dB at dose
500. Detection F1 through the fixed peak finder is 0.998 at dose 2. The
Noise2Noise checkpoint trails the supervised one by 0.2 to 1.1 dB PSNR
across the range with detection scores equal to within 0.003 F1.

## Intended use and limitations

These models restore atomic-resolution HAADF-STEM-like images whose
noise is dominated by counting statistics. They were trained entirely
on the two synthetic lattice presets in `stemdenoise.sim` and carry that
structural prior: on very different structures (amorphous regions,
grain boundaries, large defects, strong scan distortion) the prior is
wrong and output should not be trusted quantitatively. The
input must be roughly count-scaled; `stemdenoise.io.estimate_dose`
provides the normalization for data of unknown scale, and
`results/scale_robustness.json` shows the cost of using that estimate
instead of the true dose is at most 1.7 dB PSNR and 0.003 F1 on
simulated fields. The fixed-dose checkpoints exist only for the
cross-dose generalization experiment and are not recommended for use.

No real micrograph was used in training or evaluation. Treat any
application to experimental data as qualitative until validated against
ground truth you trust.
