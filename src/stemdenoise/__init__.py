"""stemdenoise: restoration of low-dose HAADF-STEM images, scored by fidelity and by atoms found."""

__version__ = "0.1.0"

from .classical import gaussian_denoise, nlm_denoise, wavelet_denoise
from .detect import find_peaks
from .metrics import match_positions, psnr, ssim
from .net import ResUNet, load_checkpoint
from .sim import LatticeSpec, make_field, preset, simulate_n2n_pair, simulate_pair
from .train import TrainConfig, denoise_counts, train_denoiser

__all__ = [
    "LatticeSpec",
    "ResUNet",
    "TrainConfig",
    "denoise_counts",
    "find_peaks",
    "gaussian_denoise",
    "load_checkpoint",
    "make_field",
    "match_positions",
    "nlm_denoise",
    "preset",
    "psnr",
    "simulate_n2n_pair",
    "simulate_pair",
    "ssim",
    "train_denoiser",
    "wavelet_denoise",
]
