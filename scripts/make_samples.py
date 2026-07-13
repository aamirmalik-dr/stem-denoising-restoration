"""Generate the small committed sample fields in data/sample/.

Run from the repo root:
    python scripts/make_samples.py
"""

from __future__ import annotations

import numpy as np

from stemdenoise.io import save_npz
from stemdenoise.sim import add_noise, make_field, preset

SAMPLES = [
    ("hexagonal", 5.0, 11),
    ("hexagonal", 50.0, 12),
    ("binary_square", 20.0, 13),
]


def main() -> None:
    for preset_name, dose, seed in SAMPLES:
        rng = np.random.default_rng(seed)
        fld = make_field(rng, size=192, dose=dose, spec=preset(preset_name))
        noisy = add_noise(fld.clean, rng)
        path = f"data/sample/{preset_name}_d{dose:g}.npz"
        save_npz(
            path,
            noisy=noisy.astype(np.float32),
            clean=fld.clean.astype(np.float32),
            positions=fld.positions.astype(np.float32),
            weights=fld.weights.astype(np.float32),
            dose=np.array(dose, dtype=np.float32),
        )
        print(f"wrote {path} ({len(fld.positions)} columns)")


if __name__ == "__main__":
    main()
