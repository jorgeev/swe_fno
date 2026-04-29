"""Loss functions and channel-wise Gaussian normalization."""
from __future__ import annotations
import torch


class LpLoss:
    """Relative L^p loss (default p=2), averaged over the batch."""

    def __init__(self, p: int = 2, eps: float = 1.0e-8):
        self.p = p
        self.eps = eps

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = (pred - target).flatten(1).norm(p=self.p, dim=1)
        denom = target.flatten(1).norm(p=self.p, dim=1) + self.eps
        return (diff / denom).mean()


class ChannelNormalizer:
    """Per-channel mean/std normalization for tensors shaped [N, C, ...]."""

    def __init__(self, x: torch.Tensor, eps: float = 1.0e-8):
        # reduce over batch and all spatial axes
        dims = (0,) + tuple(range(2, x.dim()))
        self.mean = x.mean(dim=dims, keepdim=True)        # [1, C, 1, ...]
        self.std = x.std(dim=dims, keepdim=True) + eps    # [1, C, 1, ...]

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


def all_tensors_from_dataset(dataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack the entire dataset into (X, Y) tensors. Used to fit the normalizer."""
    xs, ys = [], []
    for i in range(len(dataset)):
        xi, yi = dataset[i]
        xs.append(xi)
        ys.append(yi)
    return torch.stack(xs), torch.stack(ys)
