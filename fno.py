"""Fourier Neural Operator layers and models (1D and 2D)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 1D
# --------------------------------------------------------------------------- #
class SpectralConv1d(nn.Module):
    """Spectral convolution: learn complex weights on the lowest `modes` modes."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, C, N]
        B, _, N = x.shape
        x_ft = torch.fft.rfft(x, n=N)                           # [B, C, N//2+1]
        out_ft = torch.zeros(
            B, self.out_channels, N // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )
        m = min(self.modes, x_ft.shape[-1])
        out_ft[:, :, :m] = torch.einsum(
            "bcn,con->bon", x_ft[:, :, :m], self.weight[:, :, :m]
        )
        return torch.fft.irfft(out_ft, n=N)


class FNO1d(nn.Module):
    """Standard 1D FNO: lift -> (SpectralConv + 1x1 bypass + GELU) x L -> project."""

    def __init__(self, in_channels: int, out_channels: int,
                 modes: int, width: int, n_layers: int = 4):
        super().__init__()
        self.lift = nn.Conv1d(in_channels + 1, width, 1)        # +1 for x-grid channel
        self.spectral = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(n_layers)]
        )
        self.bypass = nn.ModuleList(
            [nn.Conv1d(width, width, 1) for _ in range(n_layers)]
        )
        self.proj = nn.Sequential(
            nn.Conv1d(width, 4 * width, 1),
            nn.GELU(),
            nn.Conv1d(4 * width, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, C, N]
        B, _, N = x.shape
        grid = torch.linspace(0.0, 1.0, N, device=x.device).view(1, 1, N).expand(B, 1, N)
        h = self.lift(torch.cat([x, grid], dim=1))
        for sc, bp in zip(self.spectral, self.bypass):
            h = F.gelu(sc(h) + bp(h))
        return self.proj(h)


# --------------------------------------------------------------------------- #
# 2D
# --------------------------------------------------------------------------- #
class SpectralConv2d(nn.Module):
    """Spectral conv 2D. Uses two weight tensors for the two retained corners
    of the rfft2 output (positive ky, negative ky)."""

    def __init__(self, in_channels: int, out_channels: int,
                 modes_x: int, modes_y: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        scale = 1.0 / (in_channels * out_channels)
        self.w1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat)
        )
        self.w2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, C, H, W] (H=y, W=x)
        B, _, H, W = x.shape
        x_ft = torch.fft.rfft2(x)                           # [B, C, H, W//2+1]
        out_ft = torch.zeros(
            B, self.out_channels, H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )
        mx = min(self.modes_x, H // 2)
        my = min(self.modes_y, W // 2 + 1)
        out_ft[:, :, :mx, :my] = torch.einsum(
            "bcxy,coxy->boxy", x_ft[:, :, :mx, :my], self.w1[:, :, :mx, :my]
        )
        out_ft[:, :, -mx:, :my] = torch.einsum(
            "bcxy,coxy->boxy", x_ft[:, :, -mx:, :my], self.w2[:, :, :mx, :my]
        )
        return torch.fft.irfft2(out_ft, s=(H, W))


class FNO2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 modes_x: int, modes_y: int, width: int, n_layers: int = 4):
        super().__init__()
        self.lift = nn.Conv2d(in_channels + 2, width, 1)        # +2 for (x,y) grid channels
        self.spectral = nn.ModuleList(
            [SpectralConv2d(width, width, modes_x, modes_y) for _ in range(n_layers)]
        )
        self.bypass = nn.ModuleList(
            [nn.Conv2d(width, width, 1) for _ in range(n_layers)]
        )
        self.proj = nn.Sequential(
            nn.Conv2d(width, 4 * width, 1),
            nn.GELU(),
            nn.Conv2d(4 * width, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, C, H, W]
        B, _, H, W = x.shape
        gy = torch.linspace(0.0, 1.0, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
        gx = torch.linspace(0.0, 1.0, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
        h = self.lift(torch.cat([x, gx, gy], dim=1))
        for sc, bp in zip(self.spectral, self.bypass):
            h = F.gelu(sc(h) + bp(h))
        return self.proj(h)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
