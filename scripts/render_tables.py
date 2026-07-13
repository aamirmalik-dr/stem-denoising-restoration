"""Print the markdown tables used in README.md and RESULTS.md.

Reads only the committed results JSON, so every number in the docs
regenerates from the same source the benchmark wrote.

Run from the repo root:
    python scripts/render_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path

ORDER = ["raw", "gaussian", "nlm", "wavelet", "cnn_supervised", "cnn_n2n"]
LABEL = {
    "raw": "raw (no denoise)",
    "gaussian": "Gaussian (tuned)",
    "nlm": "NLM + VST (tuned)",
    "wavelet": "wavelet + VST (tuned)",
    "cnn_supervised": "CNN supervised",
    "cnn_n2n": "CNN Noise2Noise",
    "cnn_fixed_low": "CNN trained at dose 10 only",
    "cnn_fixed_high": "CNN trained at dose 150 only",
}


def rows_of(path: str, preset: str) -> list[dict]:
    return [r for r in json.loads(Path(path).read_text())["results"] if r["preset"] == preset]


def table(rows: list[dict], methods: list[str], doses: list[float], key: str, fmt: str) -> str:
    lines = ["| Method | " + " | ".join(f"dose {d:g}" for d in doses) + " |"]
    lines.append("|---" * (len(doses) + 1) + "|")
    by = {(r["method"], r["dose"]): r for r in rows}
    for m in methods:
        cells = []
        for d in doses:
            r = by.get((m, d))
            cells.append(format(r[key], fmt) if r else "")
        lines.append(f"| {LABEL.get(m, m)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    hexa = rows_of("results/dose_sweep.json", "hexagonal")
    bino = rows_of("results/dose_sweep.json", "binary_square")

    print("### Detection F1, hexagonal, all doses\n")
    print(table(hexa, ORDER, [2, 5, 10, 20, 50, 150, 500], "f1", ".3f"))
    print("\n### PSNR (dB), hexagonal, all doses\n")
    print(table(hexa, ORDER, [2, 5, 10, 20, 50, 150, 500], "psnr", ".2f"))
    print("\n### SSIM, hexagonal, all doses\n")
    print(table(hexa, ORDER, [2, 5, 10, 20, 50, 150, 500], "ssim", ".3f"))
    print("\n### Localization RMSE (px), hexagonal\n")
    print(table(hexa, ORDER, [2, 5, 10, 20, 50, 150, 500], "rmse_px", ".3f"))
    print("\n### Detection F1, binary lattice (faint sublattice)\n")
    print(table(bino, ORDER, [2, 5, 10, 20, 50, 150, 500], "f1", ".3f"))
    print("\n### Detection recall, binary lattice\n")
    print(table(bino, ORDER, [2, 5, 10, 20, 50, 150, 500], "recall", ".3f"))

    print("\n### Tuned classical parameters (hexagonal)\n")
    doses = [2, 5, 10, 20, 50, 150, 500]
    lines = ["| Method | " + " | ".join(f"dose {d:g}" for d in doses) + " |"]
    lines.append("|---" * (len(doses) + 1) + "|")
    by = {(r["method"], r["dose"]): r for r in hexa}
    for m in ("gaussian", "nlm", "wavelet"):
        cells = [format(by[(m, d)].get("tuned_param", float("nan")), "g") for d in doses]
        lines.append(f"| {LABEL[m]} | " + " | ".join(cells) + " |")
    print("\n".join(lines))

    cross = rows_of("results/cross_dose.json", "hexagonal")
    cross_methods = ["nlm", "cnn_fixed_low", "cnn_fixed_high", "cnn_supervised"]
    print("\n### Cross-dose: detection F1, hexagonal\n")
    print(table(cross, cross_methods, doses, "f1", ".3f"))
    print("\n### Cross-dose: PSNR (dB), hexagonal\n")
    print(table(cross, cross_methods, doses, "psnr", ".2f"))


if __name__ == "__main__":
    main()
