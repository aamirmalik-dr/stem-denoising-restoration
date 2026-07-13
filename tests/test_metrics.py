"""Tests for fidelity and detection metrics."""

import numpy as np

from stemdenoise.metrics import filter_margin, match_positions, psnr, ssim


def test_filter_margin():
    pts = np.array([[1.0, 50.0], [50.0, 50.0], [99.0, 50.0], [50.0, 2.0]])
    kept = filter_margin(pts, (100, 100), margin=6.0)
    np.testing.assert_array_equal(kept, np.array([[50.0, 50.0]]))
    assert len(filter_margin(np.empty((0, 2)), (100, 100), 6.0)) == 0


def test_psnr_identical_is_inf():
    img = np.random.default_rng(0).uniform(0, 10, (32, 32))
    assert psnr(img, img) == float("inf")


def test_psnr_known_value():
    clean = np.zeros((10, 10))
    clean[0, 0] = 10.0  # data_range 10
    est = clean + 1.0  # mse 1
    assert abs(psnr(clean, est) - 20.0) < 1e-9


def test_ssim_bounds():
    rng = np.random.default_rng(1)
    img = rng.uniform(0, 1, (64, 64))
    assert ssim(img, img) > 0.999
    assert ssim(img, rng.uniform(0, 1, (64, 64))) < 0.5


def test_match_perfect():
    pts = np.array([[10.0, 10.0], [30.0, 40.0], [50.0, 20.0]])
    s = match_positions(pts, pts + 0.3, tolerance_px=3.0)
    assert s.f1 == 1.0
    assert abs(s.rmse_px - 0.3 * np.sqrt(2)) < 1e-6


def test_match_partial():
    true = np.array([[10.0, 10.0], [30.0, 40.0]])
    pred = np.array([[10.5, 10.0], [90.0, 90.0], [80.0, 10.0]])
    s = match_positions(true, pred, tolerance_px=3.0)
    assert s.n_matched == 1
    assert s.precision == 1 / 3
    assert s.recall == 0.5


def test_match_one_to_one():
    # Two predictions near one truth: only one may match.
    true = np.array([[10.0, 10.0]])
    pred = np.array([[10.2, 10.0], [10.0, 10.4]])
    s = match_positions(true, pred, tolerance_px=3.0)
    assert s.n_matched == 1


def test_match_empty():
    s = match_positions(np.empty((0, 2)), np.array([[1.0, 1.0]]))
    assert s.f1 == 0.0
    assert np.isnan(s.rmse_px)
