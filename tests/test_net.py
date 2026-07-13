"""Tests for the ResUNet architecture and checkpointing."""

import numpy as np
import torch

from stemdenoise.net import ResUNet, count_parameters, load_checkpoint, save_checkpoint


def test_forward_shape():
    model = ResUNet(base=8)
    x = torch.randn(2, 1, 64, 64)
    y = model(x)
    assert y.shape == x.shape


def test_identity_at_init():
    # The zero-initialized head makes the untrained network exactly the identity.
    model = ResUNet(base=8)
    model.eval()
    x = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        y = model(x)
    torch.testing.assert_close(y, x)


def test_parameter_budget():
    n = count_parameters(ResUNet(base=24))
    assert 100_000 < n < 500_000, f"unexpected size: {n}"


def test_checkpoint_roundtrip(tmp_path):
    model = ResUNet(base=8)
    path = str(tmp_path / "m.pt")
    save_checkpoint(model, path, meta={"mode": "supervised", "seed": 0})
    loaded, meta = load_checkpoint(path)
    assert meta["mode"] == "supervised"
    x = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        np.testing.assert_allclose(model(x).numpy(), loaded(x).numpy(), atol=1e-6)
