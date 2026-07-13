"""Tests for the fixed peak-finding detector."""

import numpy as np

from stemdenoise.detect import detector_params_for, find_peaks
from stemdenoise.metrics import match_positions
from stemdenoise.sim import make_field, preset


def test_detects_clean_hexagonal():
    spec = preset("hexagonal")
    fld = make_field(np.random.default_rng(0), size=256, dose=100.0, spec=spec)
    det = find_peaks(fld.clean, **detector_params_for(spec.spacing_px, spec.probe_sigma_px))
    score = match_positions(fld.positions, det, tolerance_px=3.0)
    assert score.f1 > 0.97, score


def test_detects_clean_binary_square():
    spec = preset("binary_square")
    fld = make_field(np.random.default_rng(1), size=256, dose=100.0, spec=spec)
    det = find_peaks(fld.clean, **detector_params_for(spec.spacing_px, spec.probe_sigma_px))
    score = match_positions(fld.positions, det, tolerance_px=3.0)
    assert score.f1 > 0.9, score


def test_flat_image_no_peaks():
    assert len(find_peaks(np.zeros((64, 64)))) == 0


def test_params_scale_with_geometry():
    p = detector_params_for(spacing_px=14.0, probe_sigma_px=2.2)
    assert p["min_distance"] == 3
    assert 1.0 < p["smooth_sigma"] < 2.2
