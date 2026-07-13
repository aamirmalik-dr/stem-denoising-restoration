"""Downstream atom detection: the task the restored image is for.

A deliberately simple, fixed peak finder (Gaussian smooth, local maxima
above a relative threshold, minimum separation) is applied identically
to every restored image. Keeping the detector fixed and dumb is the
point: differences in detection F1 and localization RMSE then measure
what the denoiser did to the image, not what a clever detector can
rescue. The detector's three parameters are set once from the lattice
geometry and never retuned per method.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter


def find_peaks(
    image: np.ndarray,
    smooth_sigma: float = 1.5,
    min_distance: int = 5,
    rel_threshold: float = 0.15,
) -> np.ndarray:
    """Detect bright peaks in an image.

    Args:
        image: Input image (any non-negative scale).
        smooth_sigma: Gaussian pre-smoothing sigma in pixels.
        min_distance: Minimum separation between reported peaks, pixels.
        rel_threshold: Peaks must exceed background plus this fraction of
            the smoothed dynamic range.

    Returns:
        (N, 2) array of peak positions as (row, col), sub-pixel refined
        by a local centre of mass in a small window.
    """
    sm = gaussian_filter(image.astype(np.float64), sigma=smooth_sigma)
    lo, hi = np.percentile(sm, 1.0), np.percentile(sm, 99.9)
    if hi <= lo:
        return np.empty((0, 2))
    thresh = lo + rel_threshold * (hi - lo)
    footprint = maximum_filter(sm, size=2 * min_distance + 1)
    peaks = np.argwhere((sm == footprint) & (sm > thresh))
    return _com_refine(sm, peaks, radius=max(2, min_distance // 2))


def _com_refine(image: np.ndarray, peaks: np.ndarray, radius: int) -> np.ndarray:
    """Refine integer peak locations by local background-subtracted centre of mass."""
    out = []
    h, w = image.shape
    for r, c in peaks:
        r0, r1 = max(r - radius, 0), min(r + radius + 1, h)
        c0, c1 = max(c - radius, 0), min(c + radius + 1, w)
        win = image[r0:r1, c0:c1] - image[r0:r1, c0:c1].min()
        total = win.sum()
        if total <= 0:
            out.append([float(r), float(c)])
            continue
        rr, cc = np.mgrid[r0:r1, c0:c1]
        out.append([float((rr * win).sum() / total), float((cc * win).sum() / total)])
    return np.array(out) if out else np.empty((0, 2))


def detector_params_for(spacing_px: float, probe_sigma_px: float) -> dict:
    """Derive fixed detector parameters from lattice geometry.

    Args:
        spacing_px: Nearest-neighbour column spacing, pixels.
        probe_sigma_px: Column peak sigma, pixels.

    Returns:
        Keyword arguments for :func:`find_peaks`.
    """
    # The local-max window must stay clear of the rising slope of a
    # bright neighbour, or faint sublattice sites are never window
    # maxima: on the binary lattice the faint site sits spacing/sqrt(2)
    # from a bright one, so the half-window is capped at spacing/4.
    return {
        "smooth_sigma": probe_sigma_px * 0.7,
        "min_distance": max(2, int(spacing_px * 0.25)),
        "rel_threshold": 0.15,
    }
