"""Tests for the simulator: determinism, calibration, noise statistics."""

import numpy as np

from stemdenoise.sim import (
    LatticeSpec,
    add_noise,
    make_field,
    preset,
    simulate_n2n_pair,
    simulate_pair,
)


def test_same_seed_same_field():
    a = make_field(np.random.default_rng(7), size=128, dose=50.0)
    b = make_field(np.random.default_rng(7), size=128, dose=50.0)
    np.testing.assert_array_equal(a.clean, b.clean)
    np.testing.assert_array_equal(a.positions, b.positions)


def test_positions_inside_field():
    fld = make_field(np.random.default_rng(0), size=128, dose=10.0)
    assert len(fld.positions) > 20
    assert fld.positions.min() >= 0.0
    assert fld.positions.max() < 128.0


def test_dose_calibration():
    # An isolated full-weight column peaks near the dose; the smooth
    # background can add at most spec.background on top.
    spec = LatticeSpec(vacancy_frac=0.0, jitter_px=0.0, background=0.0)
    fld = make_field(np.random.default_rng(1), size=128, dose=200.0, spec=spec)
    peak = fld.clean.max()
    # Neighbour overlap can add a little; peak must be within 25% of dose.
    assert 200.0 * 0.9 <= peak <= 200.0 * 1.25


def test_binary_square_has_two_weights():
    fld = make_field(np.random.default_rng(2), size=128, dose=10.0, spec=preset("binary_square"))
    assert set(np.round(np.unique(fld.weights), 3)) == {0.3, 1.0}


def test_poisson_statistics():
    clean = np.full((200, 200), 40.0)
    noisy = add_noise(clean, np.random.default_rng(3), readout_sigma=0.5)
    assert abs(noisy.mean() - 40.0) < 0.5
    # Variance = lambda + readout^2 = 40.25
    assert abs(noisy.var() - 40.25) < 2.0


def test_n2n_pair_independent():
    a, b, fld = simulate_n2n_pair(np.random.default_rng(4), size=96, dose=20.0)
    assert a.shape == b.shape == fld.clean.shape
    assert not np.array_equal(a, b)


def test_simulate_pair_shapes():
    noisy, fld = simulate_pair(np.random.default_rng(5), size=96, dose=5.0)
    assert noisy.shape == (96, 96)
    assert fld.dose == 5.0
