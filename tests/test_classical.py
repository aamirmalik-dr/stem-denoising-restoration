"""Tests for the VST and classical denoisers."""

import numpy as np

from stemdenoise.classical import (
    CLASSICAL,
    anscombe,
    gaussian_denoise,
    inverse_anscombe,
    nlm_denoise,
    wavelet_denoise,
)
from stemdenoise.sim import add_noise, make_field


def test_anscombe_roundtrip():
    lam = np.linspace(2.0, 500.0, 50)
    back = inverse_anscombe(anscombe(lam))
    # The bias-corrected inverse is accurate to a fraction of a count.
    assert np.max(np.abs(back - lam)) < 0.3


def test_anscombe_stabilizes_variance():
    rng = np.random.default_rng(0)
    for lam in (5.0, 50.0, 500.0):
        x = rng.poisson(lam, size=200_000) + rng.normal(scale=0.5, size=200_000)
        z = anscombe(x)
        assert abs(z.var() - 1.0) < 0.1, f"variance not stabilized at lambda={lam}"


def test_inverse_anscombe_bounded_at_zero():
    # Over-shrunk stabilized values must map to ~0 counts, never explode
    # (the raw series inverse has 1/y**3 terms that blow up near zero).
    tiny = np.array([-5.0, 0.0, 0.5, 1.0])
    out = inverse_anscombe(tiny)
    assert np.all(np.isfinite(out))
    assert np.all(out < 1.0)


def _mse(a, b):
    return float(np.mean((a - b) ** 2))


def test_denoisers_reduce_mse():
    rng = np.random.default_rng(1)
    fld = make_field(rng, size=128, dose=20.0)
    noisy = add_noise(fld.clean, rng)
    base = _mse(fld.clean, noisy)
    for fn in (gaussian_denoise, nlm_denoise, wavelet_denoise):
        est = fn(noisy)
        assert est.shape == noisy.shape
        assert _mse(fld.clean, est) < base, fn.__name__


def test_registry_defaults_in_grid_range():
    for name, entry in CLASSICAL.items():
        grid = entry["grid"]
        assert min(grid) <= entry["default"] <= max(grid), name
