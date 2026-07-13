"""Classical denoisers with proper Poisson handling.

Shot noise dominates low-dose STEM, and most classical denoisers assume
additive Gaussian noise, so every method here runs inside a generalized
Anscombe variance-stabilizing transform: stabilize, denoise in the
Gaussian domain, then invert with a bias-corrected closed form. Skipping
the VST and pretending counts are Gaussian is the single most common way
to make classical baselines look worse than they are, and this module
exists so that the benchmark's classical numbers are not strawmen.

All functions take and return images in raw count units.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_nl_means, denoise_wavelet

from .sim import DEFAULT_READOUT_SIGMA


def anscombe(counts: np.ndarray, readout_sigma: float = DEFAULT_READOUT_SIGMA) -> np.ndarray:
    """Generalized Anscombe transform for Poisson plus Gaussian readout noise.

    Maps counts with variance ``lambda + sigma_r**2`` to approximately
    unit-variance Gaussian data.

    Args:
        counts: Raw count image.
        readout_sigma: Std dev of the Gaussian readout component, counts.

    Returns:
        Variance-stabilized image.
    """
    arg = counts + 3.0 / 8.0 + readout_sigma**2
    return 2.0 * np.sqrt(np.clip(arg, 0.0, None))


def inverse_anscombe(
    stabilized: np.ndarray, readout_sigma: float = DEFAULT_READOUT_SIGMA
) -> np.ndarray:
    """Bias-corrected closed-form inverse of the generalized Anscombe transform.

    Uses the asymptotically unbiased series inverse rather than the naive
    algebraic inverse, which is biased low at the small counts this
    benchmark cares about.

    Args:
        stabilized: Image in the stabilized domain.
        readout_sigma: Same readout sigma passed to :func:`anscombe`.

    Returns:
        Estimated expected counts.
    """
    # The series inverse is only valid above the VST of zero counts;
    # below that the 1/y**3 terms explode, so clip to the physical floor
    # (an over-shrunk stabilized value can otherwise reach y ~ 0 and map
    # to astronomically large counts).
    floor = 2.0 * np.sqrt(3.0 / 8.0 + readout_sigma**2)
    y = np.clip(stabilized, floor, None)
    inv = (
        0.25 * y**2
        + 0.25 * np.sqrt(1.5) / y
        - 11.0 / 8.0 / y**2
        + 0.625 * np.sqrt(1.5) / y**3
        - 1.0 / 8.0
        - readout_sigma**2
    )
    return np.clip(inv, 0.0, None)


def gaussian_denoise(counts: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Gaussian low-pass baseline (no VST needed; linear filters commute).

    Args:
        counts: Raw count image.
        sigma: Filter sigma in pixels.

    Returns:
        Denoised image in count units.
    """
    return gaussian_filter(counts, sigma=sigma)


def nlm_denoise(
    counts: np.ndarray,
    h: float = 0.8,
    patch_size: int = 5,
    patch_distance: int = 6,
    readout_sigma: float = DEFAULT_READOUT_SIGMA,
) -> np.ndarray:
    """Non-local means inside the generalized Anscombe transform.

    Args:
        counts: Raw count image.
        h: NLM filtering strength in units of the stabilized noise sigma
            (which is 1 after the VST).
        patch_size: Side of the comparison patch.
        patch_distance: Search radius in patches.
        readout_sigma: Readout noise passed through the VST.

    Returns:
        Denoised image in count units.
    """
    z = anscombe(counts, readout_sigma)
    den = denoise_nl_means(
        z,
        h=h,
        sigma=1.0,
        patch_size=patch_size,
        patch_distance=patch_distance,
        fast_mode=True,
        preserve_range=True,
    )
    return inverse_anscombe(den, readout_sigma)


def wavelet_denoise(
    counts: np.ndarray,
    sigma_scale: float = 1.0,
    wavelet: str = "db2",
    levels: int | None = None,
    readout_sigma: float = DEFAULT_READOUT_SIGMA,
) -> np.ndarray:
    """Wavelet shrinkage (BayesShrink) inside the generalized Anscombe transform.

    Args:
        counts: Raw count image.
        sigma_scale: Multiplier on the stabilized noise sigma of 1.0;
            values above 1 shrink harder.
        wavelet: PyWavelets wavelet name.
        levels: Decomposition levels; None lets skimage choose.
        readout_sigma: Readout noise passed through the VST.

    Returns:
        Denoised image in count units.
    """
    z = anscombe(counts, readout_sigma)
    den = denoise_wavelet(
        z,
        sigma=sigma_scale,
        wavelet=wavelet,
        mode="soft",
        method="BayesShrink",
        wavelet_levels=levels,
        rescale_sigma=False,
    )
    return inverse_anscombe(den, readout_sigma)


# Registry used by the benchmark: name -> (callable, tunable parameter name,
# tuning grid). The grid is searched per condition on held-out validation
# fields so classical methods compete at their best, not at a default.
CLASSICAL: dict[str, dict] = {
    "gaussian": {
        "fn": gaussian_denoise,
        "param": "sigma",
        "grid": [0.4, 0.6, 0.8, 1.2, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0],
        "default": 1.5,
    },
    "nlm": {
        "fn": nlm_denoise,
        "param": "h",
        "grid": [0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0, 4.0],
        "default": 0.8,
    },
    "wavelet": {
        "fn": wavelet_denoise,
        "param": "sigma_scale",
        "grid": [0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0, 4.0],
        "default": 1.0,
    },
}
