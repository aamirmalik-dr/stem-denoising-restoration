"""Synthetic HAADF-STEM image formation with exact ground truth.

Images are built in three stages: place atomic columns on a lattice,
render each column as a probe-convolved Gaussian peak with a Z-contrast
weight, then apply the detector model (Poisson shot noise at a chosen
dose plus Gaussian readout noise). The dose knob is calibrated so that
``dose`` equals the expected count in the brightest pixel of an isolated
full-weight column, which makes "counts per peak pixel" the unit quoted
throughout the benchmark.

Ground-truth column positions and per-column weights are returned with
every field, so downstream detection metrics never rely on estimated
truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Readout noise (std dev, counts) applied after Poisson sampling.
DEFAULT_READOUT_SIGMA = 0.5


@dataclass
class LatticeSpec:
    """Geometry and contrast of one simulated crystal field.

    Attributes:
        kind: "hexagonal" for a single-species honeycomb sheet or
            "binary_square" for a rocksalt-like projection with a bright
            and a faint sublattice.
        spacing_px: Nearest-neighbour column spacing in pixels.
        probe_sigma_px: Gaussian sigma of the rendered column peak,
            standing in for the probe-convolved column profile.
        faint_weight: Relative HAADF weight of the faint sublattice
            (binary_square only). HAADF intensity scales roughly with
            Z^1.7, so 0.3 mimics a light element next to a heavy one.
        vacancy_frac: Fraction of columns removed at random.
        jitter_px: RMS random static displacement of each column, which
            keeps the fields from being perfectly periodic.
        background: Peak-relative amplitude of a smooth contamination
            background added to the clean image.
    """

    kind: str = "hexagonal"
    spacing_px: float = 12.0
    probe_sigma_px: float = 2.2
    faint_weight: float = 0.3
    vacancy_frac: float = 0.03
    jitter_px: float = 0.35
    background: float = 0.08


@dataclass
class Field:
    """One simulated field: clean expected-counts image plus ground truth.

    Attributes:
        clean: Expected-counts image at the requested dose (float64).
        positions: (N, 2) array of column centres as (row, col) pixels.
        weights: (N,) relative column weights in (0, 1].
        dose: Expected counts at the brightest pixel of a full-weight
            isolated column.
        spec: The lattice specification used to build the field.
    """

    clean: np.ndarray
    positions: np.ndarray
    weights: np.ndarray
    dose: float
    spec: LatticeSpec = field(repr=False, default_factory=LatticeSpec)


def _hexagonal_sites(size: int, spacing: float, rng: np.random.Generator) -> np.ndarray:
    """Honeycomb sites at random global rotation and offset covering a square field."""
    a1 = np.array([spacing * 1.5, spacing * np.sqrt(3.0) / 2.0])
    a2 = np.array([spacing * 1.5, -spacing * np.sqrt(3.0) / 2.0])
    basis = [np.zeros(2), np.array([spacing, 0.0])]
    return _tile_lattice(size, a1, a2, basis, rng)


def _binary_square_sites(
    size: int, spacing: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Square lattice with alternating sublattice labels (0 bright, 1 faint)."""
    a1 = np.array([spacing, 0.0])
    a2 = np.array([0.0, spacing])
    basis = [np.zeros(2), np.array([spacing / 2.0, spacing / 2.0])]
    sites = _tile_lattice(size, a1, a2, basis, rng, return_basis_index=True)
    return sites[:, :2], sites[:, 2].astype(int)


def _tile_lattice(
    size: int,
    a1: np.ndarray,
    a2: np.ndarray,
    basis: list[np.ndarray],
    rng: np.random.Generator,
    return_basis_index: bool = False,
) -> np.ndarray:
    theta = rng.uniform(0.0, 2.0 * np.pi)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    offset = rng.uniform(0.0, 1.0, size=2) * (a1 + a2) + size / 2.0
    reach = int(np.ceil(1.5 * size / min(np.linalg.norm(a1), np.linalg.norm(a2)))) + 2
    rows = []
    for i in range(-reach, reach + 1):
        for j in range(-reach, reach + 1):
            cell = i * a1 + j * a2
            for b_idx, b in enumerate(basis):
                p = rot @ (cell + b) + offset
                if -2.0 <= p[0] < size + 2.0 and -2.0 <= p[1] < size + 2.0:
                    rows.append([p[0], p[1], float(b_idx)])
    arr = np.array(rows) if rows else np.empty((0, 3))
    return arr if return_basis_index else arr[:, :2]


def _render_columns(
    size: int, positions: np.ndarray, weights: np.ndarray, sigma: float
) -> np.ndarray:
    """Sum of unit-height Gaussians, scaled per column by its weight."""
    img = np.zeros((size, size), dtype=np.float64)
    half = int(np.ceil(4.0 * sigma))
    for (r, c), w in zip(positions, weights):
        r0, c0 = int(np.floor(r)), int(np.floor(c))
        rr = np.arange(max(r0 - half, 0), min(r0 + half + 1, size))
        cc = np.arange(max(c0 - half, 0), min(c0 + half + 1, size))
        if rr.size == 0 or cc.size == 0:
            continue
        gr = np.exp(-0.5 * ((rr - r) / sigma) ** 2)
        gc = np.exp(-0.5 * ((cc - c) / sigma) ** 2)
        img[np.ix_(rr, cc)] += w * np.outer(gr, gc)
    return img


def _smooth_background(size: int, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    """Low-frequency positive background from a few random 2D cosine modes."""
    yy, xx = np.mgrid[0:size, 0:size] / size
    bg = np.zeros((size, size))
    for _ in range(3):
        kx, ky = rng.uniform(0.5, 2.0, size=2)
        phix, phiy = rng.uniform(0.0, 2.0 * np.pi, size=2)
        bg += np.cos(2.0 * np.pi * kx * xx + phix) * np.cos(2.0 * np.pi * ky * yy + phiy)
    bg -= bg.min()
    if bg.max() > 0:
        bg = bg / bg.max()
    return amplitude * bg


def make_field(
    rng: np.random.Generator,
    size: int = 256,
    dose: float = 100.0,
    spec: LatticeSpec | None = None,
) -> Field:
    """Build one clean field with exact ground truth.

    Args:
        rng: NumPy random generator; drives geometry, vacancies and jitter.
        size: Field edge length in pixels.
        dose: Expected counts at the brightest pixel of a full-weight column.
        spec: Lattice specification. Defaults to a hexagonal sheet.

    Returns:
        A Field whose ``clean`` image is in expected-counts units.
    """
    spec = spec or LatticeSpec()
    if spec.kind == "hexagonal":
        pos = _hexagonal_sites(size, spec.spacing_px, rng)
        wts = np.ones(len(pos))
    elif spec.kind == "binary_square":
        pos, sub = _binary_square_sites(size, spec.spacing_px, rng)
        wts = np.where(sub == 0, 1.0, spec.faint_weight)
    else:
        raise ValueError(f"unknown lattice kind: {spec.kind!r}")

    if len(pos) == 0:
        raise RuntimeError("lattice tiling produced no sites; check spacing vs size")

    keep = rng.uniform(size=len(pos)) >= spec.vacancy_frac
    pos, wts = pos[keep], wts[keep]
    pos = pos + rng.normal(scale=spec.jitter_px, size=pos.shape)

    inside = (pos[:, 0] >= 0) & (pos[:, 0] < size) & (pos[:, 1] >= 0) & (pos[:, 1] < size)
    unit = _render_columns(size, pos, wts, spec.probe_sigma_px)
    unit += _smooth_background(size, spec.background, rng)
    clean = dose * unit
    return Field(clean=clean, positions=pos[inside], weights=wts[inside], dose=dose, spec=spec)


def add_noise(
    clean: np.ndarray,
    rng: np.random.Generator,
    readout_sigma: float = DEFAULT_READOUT_SIGMA,
) -> np.ndarray:
    """Apply the detector model: Poisson counts plus Gaussian readout noise.

    Args:
        clean: Expected-counts image (non-negative).
        rng: NumPy random generator for the noise draw.
        readout_sigma: Std dev of additive Gaussian readout noise, in counts.

    Returns:
        Noisy count image (float64, can be slightly negative from readout).
    """
    shot = rng.poisson(np.clip(clean, 0.0, None)).astype(np.float64)
    return shot + rng.normal(scale=readout_sigma, size=clean.shape)


def simulate_pair(
    rng: np.random.Generator,
    size: int = 256,
    dose: float = 100.0,
    spec: LatticeSpec | None = None,
) -> tuple[np.ndarray, Field]:
    """Simulate one (noisy, ground-truth field) pair."""
    fld = make_field(rng, size=size, dose=dose, spec=spec)
    return add_noise(fld.clean, rng), fld


def simulate_n2n_pair(
    rng: np.random.Generator,
    size: int = 256,
    dose: float = 100.0,
    spec: LatticeSpec | None = None,
) -> tuple[np.ndarray, np.ndarray, Field]:
    """Simulate two independent noisy views of the same field (Noise2Noise)."""
    fld = make_field(rng, size=size, dose=dose, spec=spec)
    return add_noise(fld.clean, rng), add_noise(fld.clean, rng), fld


PRESETS: dict[str, LatticeSpec] = {
    "hexagonal": LatticeSpec(kind="hexagonal"),
    "binary_square": LatticeSpec(kind="binary_square", spacing_px=14.0),
}


def preset(name: str) -> LatticeSpec:
    """Look up a named lattice preset ("hexagonal" or "binary_square")."""
    try:
        return PRESETS[name]
    except KeyError as err:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}") from err
