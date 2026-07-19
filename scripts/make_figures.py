"""Regenerate every committed figure from results JSON and fixed seeds.

Run from the repo root after scripts/train_all.py and the two
benchmarks:
    python scripts/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stemdenoise.benchmark import _condition_seed
from stemdenoise.classical import nlm_denoise
from stemdenoise.net import load_checkpoint
from stemdenoise.plots import metric_vs_dose, restoration_ladder, restoration_quad, training_curves
from stemdenoise.sim import add_noise, make_field, preset
from stemdenoise.train import denoise_counts

FIG = Path("figures")
LADDER_DOSES = [2, 5, 20, 150]
COL_TITLES = ["noisy input", "NLM + VST (tuned)", "CNN supervised", "ground truth"]


def _tuned_nlm_params() -> dict:
    return {
        r["dose"]: r.get("tuned_param")
        for r in json.loads(Path("results/dose_sweep.json").read_text())["results"]
        if r["method"] == "nlm" and r["preset"] == "hexagonal"
    }


def _dose_panel(model, tuned: dict, spec, dose: int) -> dict:
    """Build the four restoration images for one dose, identical to the hero row."""
    # Same seed stream as the benchmark eval fields for this condition.
    rng = np.random.default_rng(_condition_seed(42, "hexagonal", dose, "eval"))
    fld = make_field(rng, size=192, dose=dose, spec=spec)
    noisy = add_noise(fld.clean, rng)
    nlm = nlm_denoise(noisy, h=tuned.get(dose, 0.8))
    cnn = denoise_counts(model, noisy, dose=dose)
    return {"dose": dose, "images": [noisy, nlm, cnn, fld.clean]}


def hero_ladder() -> None:
    """Dose-ladder restoration triptych: noisy, tuned NLM, CNN, ground truth."""
    model, _ = load_checkpoint("models/unet_supervised.pt")
    tuned = _tuned_nlm_params()
    spec = preset("hexagonal")
    panels = [_dose_panel(model, tuned, spec, dose) for dose in LADDER_DOSES]
    restoration_ladder(
        panels,
        FIG / "dose_ladder_triptych.png",
        col_titles=COL_TITLES,
    )
    print("wrote figures/dose_ladder_triptych.png")


def dose2_grid() -> None:
    """The dose-2 case alone as a square 2x2 grid for a single-image post.

    Reuses the exact dose-2 row of the hero triptych, so the four panels match
    that row image for image; only the layout differs.
    """
    model, _ = load_checkpoint("models/unet_supervised.pt")
    panel = _dose_panel(model, _tuned_nlm_params(), preset("hexagonal"), 2)
    restoration_quad(
        panel["images"],
        COL_TITLES,
        FIG / "dose2_grid.png",
        suptitle="Low-dose HAADF-STEM restoration, dose 2",
    )
    print("wrote figures/dose2_grid.png")


def sweep_curves() -> None:
    metric_vs_dose(
        "results/dose_sweep.json",
        FIG / "fidelity_vs_dose.png",
        keys=("psnr", "ssim"),
        titles=("PSNR (dB)", "SSIM"),
        preset="hexagonal",
        suptitle="Image fidelity vs dose (hexagonal lattice)",
    )
    metric_vs_dose(
        "results/dose_sweep.json",
        FIG / "detection_vs_dose.png",
        keys=("f1", "rmse_px"),
        titles=("detection F1", "localization RMSE (px)"),
        preset="hexagonal",
        suptitle="Downstream atom detection vs dose (hexagonal lattice)",
    )
    metric_vs_dose(
        "results/dose_sweep.json",
        FIG / "detection_vs_dose_binary.png",
        keys=("f1", "recall"),
        titles=("detection F1", "detection recall"),
        preset="binary_square",
        suptitle="Detection vs dose (binary lattice with faint sublattice)",
    )
    print("wrote sweep curve figures")


def operating_point_figure() -> None:
    metric_vs_dose(
        "results/detection_tuned.json",
        FIG / "operating_point.png",
        keys=("f1", "recall"),
        titles=("detection F1", "detection recall"),
        preset="binary_square",
        suptitle="Classical methods tuned directly for detection F1 (binary lattice)",
        methods=["raw", "gaussian", "nlm", "wavelet", "cnn_supervised"],
    )
    print("wrote figures/operating_point.png")


def cross_dose_figure() -> None:
    metric_vs_dose(
        "results/cross_dose.json",
        FIG / "cross_dose.png",
        keys=("psnr", "f1"),
        titles=("PSNR (dB)", "detection F1"),
        preset="hexagonal",
        suptitle="Cross-dose generalization: fixed-dose vs range-trained CNNs",
        methods=["nlm", "cnn_fixed_low", "cnn_fixed_high", "cnn_supervised"],
    )
    print("wrote figures/cross_dose.png")


def training_figure() -> None:
    hist = json.loads(Path("results/training_history.json").read_text())
    training_curves(
        {k: v for k, v in hist.items() if k in ("cnn_supervised", "cnn_n2n")},
        FIG / "training_curves.png",
    )
    print("wrote figures/training_curves.png")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    hero_ladder()
    sweep_curves()
    operating_point_figure()
    cross_dose_figure()
    training_figure()
