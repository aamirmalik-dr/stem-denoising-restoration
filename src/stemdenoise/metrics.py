"""Image-fidelity and detection metrics.

PSNR and SSIM measure how much the restored image looks like the clean
one. Detection precision, recall, F1 and matched-pair localization RMSE
measure whether the restoration preserved the information a microscopist
actually extracts. The benchmark reports both because they disagree in
instructive ways: a blurry image can score a decent PSNR while smearing
neighbouring columns into undetectability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.metrics import structural_similarity


def psnr(clean: np.ndarray, estimate: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB, with the clean image's range as peak.

    Args:
        clean: Reference expected-counts image.
        estimate: Restored image in the same units.

    Returns:
        PSNR in dB (inf if the images are identical).
    """
    mse = float(np.mean((clean - estimate) ** 2))
    if mse == 0:
        return float("inf")
    data_range = float(clean.max() - clean.min())
    return 20.0 * np.log10(data_range) - 10.0 * np.log10(mse)


def ssim(clean: np.ndarray, estimate: np.ndarray) -> float:
    """Structural similarity, computed on the clean image's dynamic range."""
    data_range = float(clean.max() - clean.min())
    return float(structural_similarity(clean, estimate, data_range=data_range))


@dataclass
class DetectionScore:
    """Detection quality of predicted positions against ground truth.

    Attributes:
        precision: Matched predictions / all predictions.
        recall: Matched ground-truth columns / all ground-truth columns.
        f1: Harmonic mean of precision and recall.
        rmse_px: Root-mean-square distance of matched pairs, pixels.
        n_true: Ground-truth column count.
        n_pred: Predicted column count.
        n_matched: One-to-one matches within tolerance.
    """

    precision: float
    recall: float
    f1: float
    rmse_px: float
    n_true: int
    n_pred: int
    n_matched: int

    def as_dict(self) -> dict:
        """Plain-dict form for JSON serialization."""
        return asdict(self)


def filter_margin(positions: np.ndarray, shape: tuple[int, int], margin: float) -> np.ndarray:
    """Drop positions within ``margin`` pixels of the image border.

    Columns straddling the field edge render partially and are neither
    fairly detectable nor fairly missable, so the benchmark scores only
    the interior on both the ground-truth and the predicted side.

    Args:
        positions: (N, 2) positions as (row, col).
        shape: Image shape (height, width).
        margin: Border width to exclude, pixels.

    Returns:
        The interior subset of ``positions``.
    """
    if len(positions) == 0:
        return positions
    h, w = shape
    keep = (
        (positions[:, 0] >= margin)
        & (positions[:, 0] < h - margin)
        & (positions[:, 1] >= margin)
        & (positions[:, 1] < w - margin)
    )
    return positions[keep]


def match_positions(
    true_pos: np.ndarray, pred_pos: np.ndarray, tolerance_px: float = 3.0
) -> DetectionScore:
    """Score predicted positions by optimal one-to-one matching.

    Uses the Hungarian algorithm on the pairwise distance matrix and
    accepts only matches closer than ``tolerance_px``, so a single
    prediction can never claim two ground-truth columns.

    Args:
        true_pos: (N, 2) ground-truth positions.
        pred_pos: (M, 2) predicted positions.
        tolerance_px: Maximum accepted match distance, pixels.

    Returns:
        A DetectionScore. RMSE is NaN when nothing matched.
    """
    n_true, n_pred = len(true_pos), len(pred_pos)
    if n_true == 0 or n_pred == 0:
        return DetectionScore(0.0, 0.0, 0.0, float("nan"), n_true, n_pred, 0)
    d = np.linalg.norm(true_pos[:, None, :] - pred_pos[None, :, :], axis=2)
    cost = np.where(d <= tolerance_px, d, 1e6)
    rows, cols = linear_sum_assignment(cost)
    ok = d[rows, cols] <= tolerance_px
    n_matched = int(ok.sum())
    precision = n_matched / n_pred
    recall = n_matched / n_true
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    rmse = float(np.sqrt(np.mean(d[rows[ok], cols[ok]] ** 2))) if n_matched else float("nan")
    return DetectionScore(precision, recall, f1, rmse, n_true, n_pred, n_matched)
