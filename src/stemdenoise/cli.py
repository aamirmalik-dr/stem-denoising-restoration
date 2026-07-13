"""Command-line interface: simulate, denoise, train and benchmark.

Examples:
    stemdenoise simulate --preset hexagonal --dose 10 --out field.npz
    stemdenoise denoise field.npz --method nlm --out restored.npz
    stemdenoise denoise real_frame.tif --method cnn --checkpoint models/unet_supervised.pt
    stemdenoise train --mode noise2noise --steps 2500 --out models/unet_n2n.pt
    stemdenoise benchmark configs/dose_sweep.yaml
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from . import __version__
from .classical import CLASSICAL
from .io import estimate_dose, load_image, save_npz
from .net import load_checkpoint
from .sim import preset, simulate_pair
from .train import TrainConfig, denoise_counts, train_denoiser


def _cmd_simulate(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(args.seed)
    noisy, fld = simulate_pair(rng, size=args.size, dose=args.dose, spec=preset(args.preset))
    save_npz(
        args.out,
        noisy=noisy,
        clean=fld.clean,
        positions=fld.positions,
        weights=fld.weights,
        dose=np.array(fld.dose),
    )
    print(
        f"wrote {args.out}: {args.size}x{args.size} {args.preset} field, "
        f"dose {args.dose:g}, {len(fld.positions)} columns"
    )
    return 0


def _cmd_denoise(args: argparse.Namespace) -> int:
    if str(args.input).endswith(".npz"):
        with np.load(args.input) as z:
            counts = z["noisy"] if "noisy" in z.files else z[z.files[0]]
            dose = float(z["dose"]) if "dose" in z.files else None
    else:
        counts, dose = load_image(args.input), None
    if dose is None:
        dose = estimate_dose(counts)
        print(f"estimated peak scale: {dose:.2f} counts")
    if args.method == "cnn":
        if not args.checkpoint:
            print("error: --checkpoint is required for method cnn", file=sys.stderr)
            return 2
        model, meta = load_checkpoint(args.checkpoint)
        restored = denoise_counts(model, counts, dose=dose)
        print(f"applied CNN ({meta.get('mode', 'unknown')} training)")
    elif args.method in CLASSICAL:
        entry = CLASSICAL[args.method]
        val = args.param if args.param is not None else entry["default"]
        restored = entry["fn"](counts, **{entry["param"]: val})
        print(f"applied {args.method} ({entry['param']}={val:g})")
    else:
        print(f"error: unknown method {args.method!r}", file=sys.stderr)
        return 2
    save_npz(args.out, restored=restored, dose=np.array(dose))
    print(f"wrote {args.out}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    cfg = TrainConfig(
        mode=args.mode,
        dose_min=args.dose_min,
        dose_max=args.dose_max,
        steps=args.steps,
        seed=args.seed,
        base=args.base,
    )
    train_denoiser(cfg, out_path=args.out)
    print(f"wrote {args.out}")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from .benchmark import run_benchmark

    run_benchmark(args.config, out_path=args.out)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the stemdenoise console command."""
    ap = argparse.ArgumentParser(prog="stemdenoise", description=__doc__)
    ap.add_argument("--version", action="version", version=f"stemdenoise {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("simulate", help="simulate one noisy field with ground truth")
    p.add_argument("--preset", default="hexagonal", choices=["hexagonal", "binary_square"])
    p.add_argument("--dose", type=float, default=10.0)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="field.npz")
    p.set_defaults(fn=_cmd_simulate)

    p = sub.add_parser("denoise", help="denoise a saved field or a real image")
    p.add_argument("input", help=".npz from simulate, or .npy/.png/.tif of your own")
    p.add_argument("--method", default="nlm", choices=[*CLASSICAL, "cnn"])
    p.add_argument("--param", type=float, default=None, help="override the method's parameter")
    p.add_argument("--checkpoint", default=None, help="model checkpoint (method cnn)")
    p.add_argument("--out", default="restored.npz")
    p.set_defaults(fn=_cmd_denoise)

    p = sub.add_parser("train", help="train a CNN denoiser")
    p.add_argument("--mode", default="supervised", choices=["supervised", "noise2noise"])
    p.add_argument("--dose-min", type=float, default=2.0)
    p.add_argument("--dose-max", type=float, default=500.0)
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--base", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="model.pt")
    p.set_defaults(fn=_cmd_train)

    p = sub.add_parser("benchmark", help="run a YAML benchmark config")
    p.add_argument("config")
    p.add_argument("--out", default=None)
    p.set_defaults(fn=_cmd_benchmark)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
