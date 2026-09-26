# Changelog

## 0.1.1 (2026-09-26)

- Package metadata completed: keywords, classifiers, and project URLs in `pyproject.toml`.
- Citation file (`CITATION.cff`) and this changelog added.
- Author and license block aligned with the sibling repositories.
- CI badge added to the README.
- Related repositories section in the README linking the six sibling electron-microscopy repositories.
- Test guarding the package `__version__` against the installed distribution metadata.

## 0.1.0 (2026-07-13)

- HAADF image simulator with exact ground truth: two lattice presets, rotation and offset, jitter, vacancies, contamination background, Poisson shot noise plus Gaussian readout.
- Classical baselines (Gaussian, non-local means, BayesShrink wavelet) inside a generalized Anscombe transform with a bias-corrected inverse, tuned per condition for PSNR or for detection F1 on separate tuning fields.
- A 263k-parameter residual U-Net trained on-the-fly on simulated patches, supervised and as self-supervised Noise2Noise, with the training history committed.
- Dual scoring: PSNR and SSIM plus Hungarian-matched detection F1 and localisation RMSE through one fixed peak finder; dose sweep, cross-dose generalisation, off-distribution geometry, and scale-robustness benchmarks.
- The `stemdenoise` CLI, a bring-your-own-data path, committed weights, results JSON, figures including the dose-ladder triptych, model card, API docs, executed tutorial, and a CI workflow.
- Maintenance after publication: unsafe checkpoint loading flag fixed, ruff and black pinned to exact versions, README expanded and restructured, figures re-exported at higher resolution.
