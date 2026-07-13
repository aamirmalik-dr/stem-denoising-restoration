"""Tests for the training loop and full-image inference."""

import numpy as np
import torch

from stemdenoise.net import ResUNet
from stemdenoise.train import (
    TrainConfig,
    _build_pool,
    _sample_batch,
    denoise_counts,
    train_denoiser,
)


def _tiny_cfg(mode: str) -> TrainConfig:
    return TrainConfig(
        mode=mode,
        dose_min=5.0,
        dose_max=50.0,
        steps=4,
        batch=2,
        patch=32,
        base=8,
        seed=0,
        pool_fields=2,
        field_size=64,
    )


def test_tiny_supervised_run(tmp_path):
    path = str(tmp_path / "m.pt")
    model, history = train_denoiser(_tiny_cfg("supervised"), out_path=path, log_every=2)
    assert len(history) == 2
    assert all(np.isfinite(loss) for _, loss in history)
    assert (tmp_path / "m.pt").exists()


def test_tiny_n2n_run():
    model, history = train_denoiser(_tiny_cfg("noise2noise"), log_every=2)
    assert len(history) == 2


def test_n2n_target_is_noisy():
    cfg = _tiny_cfg("noise2noise")
    rng = np.random.default_rng(0)
    pool = _build_pool(cfg, rng)
    x, y = _sample_batch(cfg, pool, rng)
    assert x.shape == y.shape == (2, 1, 32, 32)
    # Independent noise draws: input and target must differ.
    assert not torch.equal(x, y)


def test_fixed_dose_sampling():
    cfg = _tiny_cfg("supervised")
    cfg.dose_min = cfg.dose_max = 10.0
    rng = np.random.default_rng(0)
    pool = _build_pool(cfg, rng)
    x, _ = _sample_batch(cfg, pool, rng)
    assert torch.isfinite(x).all()


def test_denoise_counts_odd_size():
    model = ResUNet(base=8)
    model.eval()
    counts = np.random.default_rng(1).poisson(10.0, size=(70, 91)).astype(float)
    out = denoise_counts(model, counts, dose=10.0)
    assert out.shape == (70, 91)
    assert out.min() >= 0.0
