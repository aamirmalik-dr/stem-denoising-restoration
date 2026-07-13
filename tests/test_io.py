"""Tests for image loading and dose estimation."""

import numpy as np
from PIL import Image

from stemdenoise.io import estimate_dose, load_image, save_npz
from stemdenoise.sim import add_noise, make_field


def test_load_npy(tmp_path):
    arr = np.arange(12.0).reshape(3, 4)
    np.save(tmp_path / "a.npy", arr)
    np.testing.assert_array_equal(load_image(tmp_path / "a.npy"), arr)


def test_load_png_rgb_averaged(tmp_path):
    rgb = np.zeros((5, 5, 3), dtype=np.uint8)
    rgb[..., 0] = 30
    rgb[..., 1] = 60
    rgb[..., 2] = 90
    Image.fromarray(rgb).save(tmp_path / "a.png")
    img = load_image(tmp_path / "a.png")
    assert img.shape == (5, 5)
    assert abs(img[0, 0] - 60.0) < 1e-9


def test_save_npz_roundtrip(tmp_path):
    save_npz(tmp_path / "x.npz", a=np.ones((2, 2)))
    with np.load(tmp_path / "x.npz") as z:
        np.testing.assert_array_equal(z["a"], np.ones((2, 2)))


def test_estimate_dose_close_to_truth():
    rng = np.random.default_rng(0)
    for dose in (20.0, 200.0):
        fld = make_field(rng, size=256, dose=dose)
        est = estimate_dose(add_noise(fld.clean, rng))
        assert 0.6 * dose <= est <= 1.6 * dose, (dose, est)
