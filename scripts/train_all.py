"""Train every checkpoint the benchmarks need, with fixed seeds.

Four models: the two range-trained denoisers (supervised and
Noise2Noise, doses 2 to 500) and two single-dose supervised models used
only by the cross-dose generalization check. Histories are saved so the
training-curve figure regenerates from real logs.

Run from the repo root:
    python scripts/train_all.py [--steps 2500] [--fixed-steps 1500]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stemdenoise.train import TrainConfig, train_denoiser


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--fixed-steps", type=int, default=1500)
    args = ap.parse_args()

    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    runs = {
        "cnn_supervised": TrainConfig(mode="supervised", steps=args.steps, seed=0),
        "cnn_n2n": TrainConfig(mode="noise2noise", steps=args.steps, seed=1),
        "cnn_fixed_low": TrainConfig(
            mode="supervised", dose_min=10.0, dose_max=10.0, steps=args.fixed_steps, seed=2
        ),
        "cnn_fixed_high": TrainConfig(
            mode="supervised", dose_min=150.0, dose_max=150.0, steps=args.fixed_steps, seed=3
        ),
    }
    paths = {
        "cnn_supervised": "models/unet_supervised.pt",
        "cnn_n2n": "models/unet_n2n.pt",
        "cnn_fixed_low": "models/unet_fixed_low.pt",
        "cnn_fixed_high": "models/unet_fixed_high.pt",
    }

    hist_path = Path("results/training_history.json")
    histories: dict[str, list] = {}
    for name, cfg in runs.items():
        print(f"=== training {name} ===", flush=True)
        _, hist = train_denoiser(cfg, out_path=paths[name])
        histories[name] = hist
        # Save after every model so an interrupted run keeps its logs.
        hist_path.write_text(json.dumps(histories, indent=2))

    print("wrote results/training_history.json")


if __name__ == "__main__":
    main()
