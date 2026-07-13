"""Loading and scaling real or saved images (bring your own data).

Real HAADF frames arrive in arbitrary units: raw counts if you are
lucky, detector ADU or rescaled 8/16-bit exports if you are not. The
CNN and the variance-stabilized classical methods both assume roughly
count-scaled data, so :func:`estimate_dose` provides a robust peak-scale
estimate that maps an unknown image onto the normalization the models
were trained with. That mapping is approximate for heavily processed
exports, and results on such data should be read qualitatively.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def load_image(path: str | Path) -> np.ndarray:
    """Load a 2D image from .npy, .npz, .png, .tif or .tiff.

    Multi-channel images are averaged to one channel. For .npz the first
    array in the archive is used.

    Args:
        path: File path.

    Returns:
        2D float64 array.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as z:
            arr = z[z.files[0]]
    elif suffix in {".png", ".tif", ".tiff"}:
        arr = np.array(Image.open(path))
    else:
        raise ValueError(f"unsupported image format: {suffix!r}")
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D image, got shape {arr.shape}")
    return arr


def estimate_dose(counts: np.ndarray, smooth_sigma: float = 1.0) -> float:
    """Estimate the peak scale of an image of unknown provenance.

    Smooths lightly to suppress single-pixel noise, then takes the 99.7th
    percentile as the typical bright-column peak level. The sigma is kept
    small because smoothing shrinks peak amplitude; at 1.0 px the
    estimate lands within roughly 15 percent of the true dose on
    simulated fields across doses 5 to 200.

    Args:
        counts: 2D image, ideally count-scaled.
        smooth_sigma: Pre-smoothing sigma in pixels.

    Returns:
        Estimated counts at a bright column peak (floored at a small
        positive value so downstream division is safe).
    """
    sm = gaussian_filter(counts, sigma=smooth_sigma)
    return float(max(np.percentile(sm, 99.7), 1e-3))


def save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    """Save named arrays to a compressed .npz archive."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
