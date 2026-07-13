"""Config-driven, fixed-seed benchmark harness.

Every number in RESULTS.md regenerates from a YAML config in configs/.
The harness holds three things constant across methods so comparisons
stay fair: the evaluation fields (seeded per condition), the downstream
detector (fixed parameters derived from lattice geometry), and the
matching tolerance. Classical methods are additionally tuned per
condition on separate held-out tuning fields, and the chosen parameter
values are written into the output JSON so the tuning is auditable.
"""

from __future__ import annotations

import json
import time
import zlib
from pathlib import Path

import numpy as np
import yaml

from .classical import CLASSICAL
from .detect import detector_params_for, find_peaks
from .metrics import filter_margin, match_positions, psnr, ssim
from .net import load_checkpoint
from .sim import LatticeSpec, add_noise, make_field, preset
from .train import denoise_counts


def _condition_seed(base_seed: int, preset_name: str, dose: float, stream: str) -> int:
    """Stable per-condition seed derived from the config seed.

    Uses CRC32 of a canonical string because Python's built-in hash of
    strings is salted per process and would break reproducibility.
    """
    key = f"{base_seed}|{preset_name}|{float(dose):.6f}|{stream}"
    return zlib.crc32(key.encode()) % (2**31 - 1)


def _resolve_preset(entry: str | dict) -> tuple[str, LatticeSpec]:
    """Resolve a config presets entry to (condition name, lattice spec).

    An entry is either the name of a built-in preset ("hexagonal",
    "binary_square") or a mapping with a ``name`` key plus LatticeSpec
    fields, which defines an inline lattice variant, e.g.::

        - {name: hex_spacing9, kind: hexagonal, spacing_px: 9.0}

    The name seeds the per-condition random stream, so built-in preset
    names keep their historical seeding.
    """
    if isinstance(entry, str):
        return entry, preset(entry)
    entry = dict(entry)
    name = entry.pop("name")
    return name, LatticeSpec(**entry)


def _make_eval_set(
    base_seed: int, cond_name: str, spec: LatticeSpec, dose: float, n: int, size: int, stream: str
) -> list[tuple[np.ndarray, object]]:
    """Generate n (noisy, field) pairs for one condition, deterministically."""
    rng = np.random.default_rng(_condition_seed(base_seed, cond_name, dose, stream))
    out = []
    for _ in range(n):
        fld = make_field(rng, size=size, dose=dose, spec=spec)
        out.append((add_noise(fld.clean, rng), fld))
    return out


def _tune_classical(
    method: str,
    tune_set: list[tuple[np.ndarray, object]],
    metric: str,
    det_kw: dict,
    margin: float,
    tolerance_px: float,
) -> float:
    """Pick the grid value maximizing the chosen metric on the tuning fields.

    ``metric`` is "psnr" (image fidelity) or "f1" (downstream detection).
    Offering both matters: a parameter tuned for PSNR can over-smooth and
    hurt detection, so the benchmark lets classical methods compete at
    their best operating point for the metric actually being reported.
    """
    entry = CLASSICAL[method]

    def score_one(noisy: np.ndarray, fld, val: float) -> float:
        est = entry["fn"](noisy, **{entry["param"]: val})
        if metric == "psnr":
            return psnr(fld.clean, est)
        if metric == "f1":
            shape = fld.clean.shape
            return match_positions(
                filter_margin(fld.positions, shape, margin),
                filter_margin(find_peaks(est, **det_kw), shape, margin),
                tolerance_px=tolerance_px,
            ).f1
        raise ValueError(f"unknown tune metric {metric!r}")

    best_val, best_score = entry["default"], -np.inf
    for val in entry["grid"]:
        score = float(np.mean([score_one(noisy, f, val) for noisy, f in tune_set]))
        if score > best_score:
            best_val, best_score = val, score
    return best_val


def _apply_method(
    method_cfg: dict,
    noisy: np.ndarray,
    dose: float,
    tuned_param: float | None,
    models: dict,
) -> np.ndarray:
    name = method_cfg["name"]
    if name == "raw":
        return noisy
    if name in CLASSICAL:
        entry = CLASSICAL[name]
        val = tuned_param if tuned_param is not None else entry["default"]
        return entry["fn"](noisy, **{entry["param"]: val})
    if name.startswith("cnn"):
        model, _ = models[name]
        return denoise_counts(model, noisy, dose=dose)
    raise ValueError(f"unknown method {name!r}")


def run_benchmark(config_path: str | Path, out_path: str | Path | None = None) -> dict:
    """Run one benchmark config end to end.

    Args:
        config_path: Path to a YAML config (see configs/ for examples).
        out_path: Where to write the results JSON. Defaults to
            results/<config name>.json next to the current directory.

    Returns:
        The results dict that was written.
    """
    config_path = Path(config_path)
    cfg = yaml.safe_load(config_path.read_text())
    t0 = time.time()

    models: dict = {}
    for m in cfg["methods"]:
        if m["name"].startswith("cnn"):
            models[m["name"]] = load_checkpoint(m["checkpoint"])

    rows = []
    for preset_entry in cfg["presets"]:
        cond_name, spec = _resolve_preset(preset_entry)
        det_kw = detector_params_for(spec.spacing_px, spec.probe_sigma_px)
        margin = spec.spacing_px / 2.0
        for dose in cfg["doses"]:
            eval_set = _make_eval_set(
                cfg["seed"], cond_name, spec, dose, cfg["n_eval_fields"], cfg["field_size"], "eval"
            )
            tune_set = _make_eval_set(
                cfg["seed"], cond_name, spec, dose, cfg["n_tune_fields"], cfg["field_size"], "tune"
            )
            for m in cfg["methods"]:
                tuned = None
                if m.get("tune") and m["name"] in CLASSICAL:
                    tuned = _tune_classical(
                        m["name"],
                        tune_set,
                        metric=m.get("tune_metric", "psnr"),
                        det_kw=det_kw,
                        margin=margin,
                        tolerance_px=cfg["tolerance_px"],
                    )
                per_field = []
                for noisy, fld in eval_set:
                    est = _apply_method(m, noisy, dose, tuned, models)
                    shape = fld.clean.shape
                    det = match_positions(
                        filter_margin(fld.positions, shape, margin),
                        filter_margin(find_peaks(est, **det_kw), shape, margin),
                        tolerance_px=cfg["tolerance_px"],
                    )
                    per_field.append(
                        {
                            "psnr": psnr(fld.clean, est),
                            "ssim": ssim(fld.clean, est),
                            **det.as_dict(),
                        }
                    )
                agg = {
                    k: float(np.nanmean([p[k] for p in per_field]))
                    for k in ("psnr", "ssim", "precision", "recall", "f1", "rmse_px")
                }
                row = {"preset": cond_name, "dose": dose, "method": m["name"], **agg}
                if tuned is not None:
                    row["tuned_param"] = tuned
                    row["tune_metric"] = m.get("tune_metric", "psnr")
                rows.append(row)
                print(
                    f"{cond_name:>14} dose {dose:>6g} {m['name']:>14}: "
                    f"psnr {agg['psnr']:6.2f} ssim {agg['ssim']:.3f} "
                    f"f1 {agg['f1']:.3f} rmse {agg['rmse_px']:.3f}px",
                    flush=True,
                )

    out = {
        "config": cfg,
        "config_file": config_path.name,
        "elapsed_s": round(time.time() - t0, 1),
        "results": rows,
    }
    if out_path is None:
        out_path = Path("results") / f"{cfg.get('name', config_path.stem)}.json"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path} ({out['elapsed_s']}s)")
    return out
