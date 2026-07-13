"""Smoke tests for the CLI."""

import numpy as np

from stemdenoise.cli import main


def test_simulate_then_denoise(tmp_path):
    field = str(tmp_path / "f.npz")
    restored = str(tmp_path / "r.npz")
    assert main(["simulate", "--dose", "10", "--size", "96", "--out", field]) == 0
    assert main(["denoise", field, "--method", "gaussian", "--out", restored]) == 0
    with np.load(restored) as z:
        assert z["restored"].shape == (96, 96)


def test_denoise_real_image_path(tmp_path):
    # A bare .npy with no dose triggers the estimate_dose path.
    img = np.random.default_rng(0).poisson(20.0, size=(64, 64)).astype(float)
    path = tmp_path / "img.npy"
    np.save(path, img)
    out = str(tmp_path / "r.npz")
    assert main(["denoise", str(path), "--method", "wavelet", "--out", out]) == 0


def test_cnn_requires_checkpoint(tmp_path):
    field = str(tmp_path / "f.npz")
    main(["simulate", "--size", "64", "--out", field])
    assert main(["denoise", field, "--method", "cnn"]) == 2
