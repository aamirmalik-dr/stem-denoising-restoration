"""Figure generation for the benchmark.

Method colors are fixed once here and reused in every figure, so a
method keeps its identity across the whole README. The palette is a
colorblind-validated categorical set; single-axis panels only.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHOD_COLORS: dict[str, str] = {
    "raw": "#8a8a85",
    "gaussian": "#eda100",
    "nlm": "#1baf7a",
    "wavelet": "#4a3aa7",
    "cnn_supervised": "#2a78d6",
    "cnn_n2n": "#e34948",
    "cnn_fixed_low": "#e87ba4",
    "cnn_fixed_high": "#eb6834",
}

METHOD_LABELS: dict[str, str] = {
    "raw": "raw (no denoise)",
    "gaussian": "Gaussian (tuned)",
    "nlm": "NLM + VST (tuned)",
    "wavelet": "wavelet + VST (tuned)",
    "cnn_supervised": "CNN supervised",
    "cnn_n2n": "CNN Noise2Noise",
    "cnn_fixed_low": "CNN trained at dose 10",
    "cnn_fixed_high": "CNN trained at dose 150",
}

_GRID_KW = {"color": "#dddddd", "linewidth": 0.6, "zorder": 0}


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, **_GRID_KW)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _rows(results_path: str | Path, preset: str | None = None) -> list[dict]:
    data = json.loads(Path(results_path).read_text())["results"]
    if preset:
        data = [r for r in data if r["preset"] == preset]
    return data


def _series(rows: list[dict], method: str, key: str) -> tuple[list[float], list[float]]:
    pts = sorted(((r["dose"], r[key]) for r in rows if r["method"] == method), key=lambda t: t[0])
    return [p[0] for p in pts], [p[1] for p in pts]


def metric_vs_dose(
    results_path: str | Path,
    out_path: str | Path,
    keys: tuple[str, ...] = ("psnr", "ssim"),
    titles: tuple[str, ...] = ("PSNR (dB)", "SSIM"),
    preset: str | None = None,
    suptitle: str | None = None,
    methods: list[str] | None = None,
) -> None:
    """Line panels of one or more metrics against dose (log-x), one axis each."""
    rows = _rows(results_path, preset)
    if methods is None:
        methods = [m for m in METHOD_COLORS if any(r["method"] == m for r in rows)]
    fig, axes = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 4.0))
    axes = np.atleast_1d(axes)
    for ax, key, title in zip(axes, keys, titles):
        for m in methods:
            x, y = _series(rows, m, key)
            if not x:
                continue
            ax.plot(
                x,
                y,
                marker="o",
                markersize=4,
                linewidth=2,
                color=METHOD_COLORS[m],
                label=METHOD_LABELS.get(m, m),
                zorder=3,
            )
        ax.set_xscale("log")
        ax.set_xlabel("dose (counts per peak pixel)")
        ax.set_ylabel(title)
        _style_axis(ax)
    axes[0].legend(fontsize=8, frameon=False)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    _save(fig, out_path)


def restoration_ladder(
    panels: list[dict],
    out_path: str | Path,
    col_titles: list[str],
) -> None:
    """The hero: rows are doses, columns are (noisy, restorations..., clean).

    Args:
        panels: One dict per dose row with keys "dose" and "images"
            (a list of 2D arrays, one per column, shared clean scale).
        out_path: Output PNG path.
        col_titles: Column headings, same length as each images list.
    """
    n_rows, n_cols = len(panels), len(col_titles)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.15 * n_cols, 2.3 * n_rows), squeeze=False)
    for i, row in enumerate(panels):
        vmax = float(np.percentile(row["images"][-1], 99.9))
        for j, img in enumerate(row["images"]):
            ax = axes[i][j]
            ax.imshow(img, cmap="inferno", vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(col_titles[j], fontsize=10)
        axes[i][0].set_ylabel(f"dose {row['dose']:g}", fontsize=10, rotation=90, labelpad=6)
    fig.tight_layout(pad=0.6)
    _save(fig, out_path)


def training_curves(histories: dict[str, list[tuple[int, float]]], out_path: str | Path) -> None:
    """Training loss against step for one or more runs, log-y."""
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for name, hist in histories.items():
        steps = [h[0] for h in hist]
        losses = [h[1] for h in hist]
        color = METHOD_COLORS.get(name, "#2a78d6")
        ax.plot(steps, losses, linewidth=2, color=color, label=METHOD_LABELS.get(name, name))
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("training MSE (running mean)")
    ax.set_title(
        "Noise2Noise loss sits on the irreducible noise floor of its noisy targets;\n"
        "the model underneath improves like the supervised one (see benchmark PSNR)",
        fontsize=9,
    )
    _style_axis(ax)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    _save(fig, out_path)


def _save(fig: plt.Figure, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, facecolor="white", bbox_inches="tight")
    plt.close(fig)
