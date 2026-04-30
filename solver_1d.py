"""1D shallow-water solver (periodic, variable bottom) — GPU-accelerated via CuPy.

Equations
---------
    h_t + (h u)_x = 0
    u_t + u u_x + g (h + b)_x = 0

Pseudo-spectral spatial derivatives (periodic), RK4 in time,
2/3-rule dealiasing on the nonlinear products plus a small k^4
hyperviscosity for stability.

Initial condition generation runs on CPU (NumPy); the RK4 integration
loop runs on GPU (CuPy).  CuPy is a hard requirement.
"""
from __future__ import annotations
import numpy as np
import cupy as cp


# --------------------------------------------------------------------------- #
# Random fields  (CPU — called once per trajectory)
# --------------------------------------------------------------------------- #
def _smooth_random_1d(nx: int, Lx: float, n_modes: int, rng: np.random.Generator) -> np.ndarray:
    x = np.linspace(0.0, Lx, nx, endpoint=False)
    f = np.zeros(nx)
    for k in range(1, n_modes + 1):
        f += rng.normal() * np.cos(2 * np.pi * k * x / Lx + rng.uniform(0, 2 * np.pi)) / k
    f -= f.mean()
    m = np.abs(f).max()
    return f / m if m > 0 else f


def make_bottom_1d(cfg, rng: np.random.Generator) -> np.ndarray:
    return cfg.bottom_amp * _smooth_random_1d(cfg.nx, cfg.Lx, cfg.n_modes_ic, rng)


def make_initial_1d(cfg, b: np.ndarray, rng: np.random.Generator):
    eta = cfg.eta_amp * _smooth_random_1d(cfg.nx, cfg.Lx, cfg.n_modes_ic, rng)
    u   = cfg.u_amp   * _smooth_random_1d(cfg.nx, cfg.Lx, cfg.n_modes_ic, rng)
    h   = (cfg.H0 - b) + eta
    return h, u


# --------------------------------------------------------------------------- #
# Spectral helpers  (CPU — one-time setup)
# --------------------------------------------------------------------------- #
def _make_spectral_ops(nx: int, Lx: float):
    kx   = 2.0 * np.pi * np.fft.fftfreq(nx, d=Lx / nx)
    kmax = np.abs(kx).max()
    dealias = (np.abs(kx) <= (2.0 / 3.0) * kmax).astype(np.float64)
    hyper   = (kx ** 2 / max(kmax, 1e-12) ** 2) ** 2
    return kx, dealias, hyper


# --------------------------------------------------------------------------- #
# RHS / time stepping  (GPU — cupy arrays throughout)
# --------------------------------------------------------------------------- #
def _rhs(h, u, b, kx, dealias, hyper, g, visc):
    Hk = cp.fft.fft(h)
    Uk = cp.fft.fft(u)
    Bk = cp.fft.fft(b)

    dh_dx   = cp.real(cp.fft.ifft(1j * kx * Hk))
    du_dx   = cp.real(cp.fft.ifft(1j * kx * Uk))
    deta_dx = cp.real(cp.fft.ifft(1j * kx * (Hk + Bk)))

    hu     = cp.real(cp.fft.ifft(cp.fft.fft(h * u) * dealias))
    dhu_dx = cp.real(cp.fft.ifft(1j * kx * cp.fft.fft(hu)))

    udu = cp.real(cp.fft.ifft(cp.fft.fft(u * du_dx) * dealias))

    dh_dt = -dhu_dx             - visc * cp.real(cp.fft.ifft(hyper * Hk))
    du_dt = -udu - g * deta_dx  - visc * cp.real(cp.fft.ifft(hyper * Uk))
    return dh_dt, du_dt


def _rk4(h, u, b, kx, dealias, hyper, dt, g, visc):
    k1h, k1u = _rhs(h,              u,              b, kx, dealias, hyper, g, visc)
    k2h, k2u = _rhs(h + 0.5*dt*k1h, u + 0.5*dt*k1u, b, kx, dealias, hyper, g, visc)
    k3h, k3u = _rhs(h + 0.5*dt*k2h, u + 0.5*dt*k2u, b, kx, dealias, hyper, g, visc)
    k4h, k4u = _rhs(h +     dt*k3h, u +     dt*k3u,  b, kx, dealias, hyper, g, visc)
    h_new = h + (dt / 6.0) * (k1h + 2*k2h + 2*k3h + k4h)
    u_new = u + (dt / 6.0) * (k1u + 2*k2u + 2*k3u + k4u)
    return h_new, u_new


# --------------------------------------------------------------------------- #
# Trajectory
# --------------------------------------------------------------------------- #
def simulate_1d(cfg, rng: np.random.Generator):
    """Run one trajectory on GPU. Returns dict {h0,u0,b,h,u} (numpy) or None on failure."""
    kx_np, dealias_np, hyper_np = _make_spectral_ops(cfg.nx, cfg.Lx)
    dx = cfg.Lx / cfg.nx

    b_np      = make_bottom_1d(cfg, rng)
    h_np, u_np = make_initial_1d(cfg, b_np, rng)
    h0, u0    = h_np.copy(), u_np.copy()

    c_max = float(np.sqrt(cfg.g * (cfg.H0 + cfg.bottom_amp)) + np.abs(u_np).max())
    dt = cfg.cfl * dx / max(c_max, 1e-6)
    nt = max(int(np.ceil(cfg.T / dt)), 1)
    dt = cfg.T / nt

    # Transfer IC and operators to GPU
    h       = cp.asarray(h_np)
    u       = cp.asarray(u_np)
    b       = cp.asarray(b_np)
    kx      = cp.asarray(kx_np)
    dealias = cp.asarray(dealias_np)
    hyper   = cp.asarray(hyper_np)

    for _ in range(nt):
        h, u = _rk4(h, u, b, kx, dealias, hyper, dt, cfg.g, cfg.visc)
        if not bool(cp.all(cp.isfinite(h))) or float(h.min()) <= 1e-3:
            return None

    return dict(h0=h0, u0=u0, b=b_np, h=cp.asnumpy(h), u=cp.asnumpy(u))


def generate_dataset_1d(cfg, n_samples: int, base_seed: int = 0):
    """Generate ``n_samples`` trajectories. Returns dict of stacked numpy arrays [N, nx]."""
    rng    = np.random.default_rng(base_seed)
    fields = {k: [] for k in ["h0", "u0", "b", "h", "u"]}
    n_done = 0
    n_try  = 0
    while n_done < n_samples:
        n_try += 1
        out = simulate_1d(cfg, rng)
        if out is None:
            continue
        for k in fields:
            fields[k].append(out[k])
        n_done += 1
        if n_done % max(1, n_samples // 10) == 0:
            print(f"  [1D] {n_done}/{n_samples} (rejected {n_try - n_done})")
    return {k: np.stack(v).astype(np.float32) for k, v in fields.items()}
