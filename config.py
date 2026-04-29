"""Configuration dataclasses for FNO shallow-water experiments."""
from dataclasses import dataclass


@dataclass
class Config1D:
    # Physics / domain
    Lx: float = 1.0
    nx: int = 128
    g: float = 1.0
    H0: float = 1.0           # reference water-surface height
    bottom_amp: float = 0.1   # max |b| (must stay < H0)
    eta_amp: float = 0.05     # initial surface anomaly amplitude
    u_amp: float = 0.10       # initial velocity amplitude
    n_modes_ic: int = 4       # Fourier modes used to build random IC / bottom

    # Time integration
    T: float = 0.30           # final time (state at T is the FNO target)
    cfl: float = 0.40
    visc: float = 5.0e-4      # hyperviscosity coefficient (k^4 damping)

    # Dataset sizes
    n_train: int = 800
    n_val: int = 100
    n_test: int = 100
    seed: int = 0

    # FNO1d
    modes: int = 16
    width: int = 64
    n_layers: int = 4
    in_channels: int = 3      # h0, u0, b
    out_channels: int = 2     # h(T), u(T)

    # Training
    batch_size: int = 32
    epochs: int = 100
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    sched_step: int = 25
    sched_gamma: float = 0.5

    device: str = "cuda"
    out_dir: str = "runs/swe1d"


@dataclass
class Config2D:
    # Physics / domain
    Lx: float = 1.0
    Ly: float = 1.0
    nx: int = 64
    ny: int = 64
    g: float = 1.0
    H0: float = 1.0
    bottom_amp: float = 0.1
    eta_amp: float = 0.05
    u_amp: float = 0.10
    n_modes_ic: int = 4
    f_min: float = 0.0        # Coriolis sampled in [f_min, f_max] per trajectory
    f_max: float = 2.0

    # Time integration
    T: float = 0.30
    cfl: float = 0.30
    visc: float = 5.0e-4

    # Dataset sizes
    n_train: int = 400
    n_val: int = 50
    n_test: int = 50
    seed: int = 0

    # FNO2d
    modes_x: int = 12
    modes_y: int = 12
    width: int = 32
    n_layers: int = 4
    in_channels: int = 5      # h0, u0, v0, b, f
    out_channels: int = 3     # h(T), u(T), v(T)

    # Training
    batch_size: int = 16
    epochs: int = 100
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    sched_step: int = 25
    sched_gamma: float = 0.5

    device: str = "cuda"
    out_dir: str = "runs/swe2d"
