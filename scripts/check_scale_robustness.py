"""Fairness check: does the CNN need the true dose for normalization?

In the benchmark the CNN input is normalized by the true simulated dose,
which a real microscope does not hand you. This script re-evaluates the
range-trained supervised model with the dose replaced by the blind
estimate from :func:`stemdenoise.io.estimate_dose` and writes both sets
of numbers to results/scale_robustness.json.

Run from the repo root:
    python scripts/check_scale_robustness.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stemdenoise import (
    denoise_counts,
    find_peaks,
    load_checkpoint,
    make_field,
    match_positions,
    preset,
    psnr,
)
from stemdenoise.detect import detector_params_for
from stemdenoise.io import estimate_dose
from stemdenoise.metrics import filter_margin
from stemdenoise.sim import add_noise

N_FIELDS = 4
DOSES = (2.0, 10.0, 150.0)


def main() -> None:
    model, _ = load_checkpoint("models/unet_supervised.pt")
    rows = []
    for pname in ("hexagonal", "binary_square"):
        spec = preset(pname)
        det_kw = detector_params_for(spec.spacing_px, spec.probe_sigma_px)
        margin = spec.spacing_px / 2
        for dose in DOSES:
            rng = np.random.default_rng(int(dose) * 7 + (0 if pname == "hexagonal" else 1))
            per = {"true": [], "est": [], "est_dose": []}
            for _ in range(N_FIELDS):
                fld = make_field(rng, size=256, dose=dose, spec=spec)
                noisy = add_noise(fld.clean, rng)
                est_d = estimate_dose(noisy)
                per["est_dose"].append(est_d)
                gt = filter_margin(fld.positions, fld.clean.shape, margin)
                for tag, d in (("true", dose), ("est", est_d)):
                    den = denoise_counts(model, noisy, dose=d)
                    found = filter_margin(find_peaks(den, **det_kw), fld.clean.shape, margin)
                    score = match_positions(gt, found, tolerance_px=3.0)
                    per[tag].append((psnr(fld.clean, den), score.f1))
            row = {
                "preset": pname,
                "dose": dose,
                "mean_estimated_dose": float(np.mean(per["est_dose"])),
                "psnr_true_dose": float(np.mean([p for p, _ in per["true"]])),
                "psnr_estimated_dose": float(np.mean([p for p, _ in per["est"]])),
                "f1_true_dose": float(np.mean([f for _, f in per["true"]])),
                "f1_estimated_dose": float(np.mean([f for _, f in per["est"]])),
            }
            rows.append(row)
            print(
                f"{pname:>14} dose {dose:>5g} (est {row['mean_estimated_dose']:6.1f}): "
                f"psnr {row['psnr_true_dose']:.2f} -> {row['psnr_estimated_dose']:.2f}, "
                f"f1 {row['f1_true_dose']:.3f} -> {row['f1_estimated_dose']:.3f}"
            )
    out = Path("results/scale_robustness.json")
    out.write_text(json.dumps({"n_fields": N_FIELDS, "results": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
