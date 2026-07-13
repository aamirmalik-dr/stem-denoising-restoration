"""A compact residual U-Net denoiser sized for CPU training.

The network predicts a residual correction to its input, so at
initialization it is close to the identity and training starts from "do
nothing" rather than from noise. Two downsampling levels and a small
channel budget (about 0.19M parameters at base=24) keep a full training
run under an hour on a laptop CPU while still giving the model enough
receptive field to average over several lattice periods.
"""

from __future__ import annotations

import torch
from torch import nn


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1),
        nn.GroupNorm(min(8, c_out), c_out),
        nn.SiLU(),
        nn.Conv2d(c_out, c_out, 3, padding=1),
        nn.GroupNorm(min(8, c_out), c_out),
        nn.SiLU(),
    )


class ResUNet(nn.Module):
    """Residual U-Net: input plus a learned correction.

    Args:
        base: Channel width of the first encoder level; deeper levels
            double it.
    """

    def __init__(self, base: int = 24):
        super().__init__()
        self.base = base
        self.enc1 = _block(1, base)
        self.enc2 = _block(base, base * 2)
        self.mid = _block(base * 2, base * 4)
        self.down = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _block(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)
        # Zero-init the head so the network is exactly the identity at
        # initialization; training then learns a correction from zero.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Denoise a batch of normalized images, shape (B, 1, H, W).

        H and W must be multiples of 4 (two pooling levels).
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.down(e1))
        m = self.mid(self.down(e2))
        d2 = self.dec2(torch.cat([self.up2(m), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return x + self.out(d1)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model: ResUNet, path: str, meta: dict) -> None:
    """Save weights plus training metadata (mode, doses, steps, seed)."""
    torch.save({"state_dict": model.state_dict(), "base": model.base, "meta": meta}, path)


def load_checkpoint(path: str) -> tuple[ResUNet, dict]:
    """Load a checkpoint saved by :func:`save_checkpoint`.

    Returns:
        The model in eval mode and the stored metadata dict.
    """
    # Checkpoints hold only tensors and plain Python types, so the safe
    # weights_only loader is sufficient.
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = ResUNet(base=ckpt["base"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt.get("meta", {})
