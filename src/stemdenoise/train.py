"""Training loops for the CNN denoiser: supervised and Noise2Noise.

Training data is simulated on the fly, so the model never sees the same
noise realization twice. A pool of clean fields is built once per run;
each step crops random patches, draws a dose (log-uniform over the
configured range unless a fixed dose is requested), and Poisson-samples
fresh noise. Inputs and targets are normalized by the dose so that a
full-weight column peak sits near 1.0 regardless of dose, which is what
lets one network serve the whole dose range.

Noise2Noise uses a second independent noise draw of the same field as
the target instead of the clean image. Its optimum is the same
conditional expectation as the supervised loss, and it needs no clean
data, which matters because paired clean acquisitions barely exist for
real beam-sensitive samples.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .net import ResUNet, save_checkpoint
from .sim import DEFAULT_READOUT_SIGMA, LatticeSpec, add_noise, make_field, preset


@dataclass
class TrainConfig:
    """Configuration for one training run.

    Attributes:
        mode: "supervised" (noisy to clean) or "noise2noise" (noisy to
            independently noisy).
        dose_min: Lower end of the log-uniform training dose range.
        dose_max: Upper end. Set both equal to train at a fixed dose.
        steps: Optimizer steps.
        batch: Patches per step.
        patch: Patch edge length (multiple of 4).
        base: U-Net base channel width.
        lr: Adam learning rate.
        seed: Seed for field pool, patch sampling and noise.
        pool_fields: Clean fields in the training pool.
        field_size: Edge length of each pooled field.
        presets: Lattice preset names mixed into the pool.
    """

    mode: str = "supervised"
    dose_min: float = 2.0
    dose_max: float = 500.0
    steps: int = 2500
    batch: int = 16
    patch: int = 96
    base: int = 24
    lr: float = 2e-3
    seed: int = 0
    pool_fields: int = 48
    field_size: int = 192
    presets: list[str] = field(default_factory=lambda: ["hexagonal", "binary_square"])


def _build_pool(cfg: TrainConfig, rng: np.random.Generator) -> list[np.ndarray]:
    """Pre-render clean unit-intensity fields (dose applied later)."""
    pool = []
    for i in range(cfg.pool_fields):
        spec: LatticeSpec = preset(cfg.presets[i % len(cfg.presets)])
        fld = make_field(rng, size=cfg.field_size, dose=1.0, spec=spec)
        pool.append(fld.clean)
    return pool


def _sample_batch(
    cfg: TrainConfig, pool: list[np.ndarray], rng: np.random.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw one training batch of (input, target) normalized patches."""
    xs, ys = [], []
    for _ in range(cfg.batch):
        unit = pool[rng.integers(len(pool))]
        r = rng.integers(0, unit.shape[0] - cfg.patch + 1)
        c = rng.integers(0, unit.shape[1] - cfg.patch + 1)
        clean_unit = unit[r : r + cfg.patch, c : c + cfg.patch]
        if cfg.dose_min == cfg.dose_max:
            dose = cfg.dose_min
        else:
            dose = float(np.exp(rng.uniform(np.log(cfg.dose_min), np.log(cfg.dose_max))))
        clean = dose * clean_unit
        noisy = add_noise(clean, rng)
        xs.append(noisy / dose)
        if cfg.mode == "supervised":
            ys.append(clean / dose)
        elif cfg.mode == "noise2noise":
            ys.append(add_noise(clean, rng) / dose)
        else:
            raise ValueError(f"unknown mode {cfg.mode!r}")
    x = torch.from_numpy(np.stack(xs)[:, None]).float()
    y = torch.from_numpy(np.stack(ys)[:, None]).float()
    return x, y


def train_denoiser(
    cfg: TrainConfig, out_path: str | None = None, log_every: int = 100
) -> tuple[ResUNet, list[tuple[int, float]]]:
    """Train a ResUNet denoiser from scratch.

    Args:
        cfg: Training configuration.
        out_path: If given, save a checkpoint (weights plus config) here.
        log_every: Print running loss every this many steps.

    Returns:
        The trained model (eval mode) and a list of (step, loss) pairs.
    """
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    pool = _build_pool(cfg, rng)
    model = ResUNet(base=cfg.base)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps)
    loss_fn = nn.MSELoss()
    history: list[tuple[int, float]] = []
    running, t0 = 0.0, time.time()
    model.train()
    for step in range(1, cfg.steps + 1):
        x, y = _sample_batch(cfg, pool, rng)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        sched.step()
        running += loss.item()
        if step % log_every == 0:
            avg = running / log_every
            history.append((step, avg))
            print(
                f"[{cfg.mode} d{cfg.dose_min:g}-{cfg.dose_max:g}] "
                f"step {step}/{cfg.steps} loss {avg:.5f} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
            running = 0.0
    model.eval()
    if out_path:
        save_checkpoint(model, out_path, meta=vars(cfg).copy())
    return model, history


def denoise_counts(
    model: ResUNet,
    counts: np.ndarray,
    dose: float,
    readout_sigma: float = DEFAULT_READOUT_SIGMA,
) -> np.ndarray:
    """Apply a trained model to a count image of arbitrary size.

    The image is normalized by ``dose``, padded reflectively to a
    multiple of 4, denoised in one pass, cropped and rescaled back to
    counts.

    Args:
        model: Trained ResUNet in eval mode.
        counts: Raw count image.
        dose: Normalization scale; for simulated data the true dose, for
            real data an estimate (see :func:`stemdenoise.io.estimate_dose`).
        readout_sigma: Unused; accepted for signature parity with the
            classical denoisers.

    Returns:
        Denoised image in count units, clipped at zero.
    """
    del readout_sigma
    h, w = counts.shape
    pad_h, pad_w = (-h) % 4, (-w) % 4
    x = np.pad(counts / dose, ((0, pad_h), (0, pad_w)), mode="reflect")
    with torch.no_grad():
        out = model(torch.from_numpy(x[None, None]).float())[0, 0].numpy()
    return np.clip(out[:h, :w] * dose, 0.0, None)
