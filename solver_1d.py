"""1D shallow-water solver (periodic, variable bottom).

Equations
---------
    h_t + (h u)_x = 0
    u_t + u u_x + g (h + b)_x = 0

Pseudo-spectral spatial derivatives (periodic), RK4 in time,
2/3-rule dealiasing on the nonlinear products plus a small k^4
hyperviscosity for stability.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# Random fields
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
    # Equilibrium depth = H0 - b, perturbed by eta.
    h = (cfg.H0 - b) + eta
    return h, u


# --------------------------------------------------------------------------- #
# Spectral helpers
# --------------------------------------------------------------------------- #
def _make_spectral_ops(nx: int, Lx: float):
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=Lx / nx)
    kmax = np.abs(kx).max()
    # 2/3 dealiasing mask
    dealias = (np.abs(kx) <= (2.0 / 3.0) * kmax).astype(np.float64)
    # Hyperviscosity factor (k/kmax)^4, normalized so max value is 1
    hyper = (kx ** 2 / max(kmax, 1e-12) ** 2) ** 2
    return kx, dealias, hyper


# --------------------------------------------------------------------------- #
# RHS / time stepping
# --------------------------------------------------------------------------- #
def _rhs(h, u, b, kx, dealias, hyper, g, visc):
    Hk = np.fft.fft(h)
    Uk = np.fft.fft(u)
    Bk = np.fft.fft(b)

    dh_dx  = np.real(np.fft.ifft(1j * kx * Hk))
    du_dx  = np.real(np.fft.ifft(1j * kx * Uk))
    deta_dx = np.real(np.fft.ifft(1j * kx * (Hk + Bk)))

    # nonlinear products dealiased
    hu = np.real(np.fft.ifft(np.fft.fft(h * u) * dealias))
    dhu_dx = np.real(np.fft.ifft(1j * kx * np.fft.fft(hu)))

    udu = np.real(np.fft.ifft(np.fft.fft(u * du_dx) * dealias))

    dh_dt = -dhu_dx        - visc * np.real(np.fft.ifft(hyper * Hk))
    du_dt = -udu - g * deta_dx - visc * np.real(np.fft.ifft(hyper * Uk))
    return dh_dt, du_dt


def _rk4(h, u, b, kx, dealias, hyper, dt, g, visc):
    k1h, k1u = _rhs(h,            u,            b, kx, dealias, hyper, g, visc)
    k2h, k2u = _rhs(h + 0.5*dt*k1h, u + 0.5*dt*k1u, b, kx, dealias, hyper, g, visc)
    k3h, k3u = _rhs(h + 0.5*dt*k2h, u + 0.5*dt*k2u, b, kx, dealias, hyper, g, visc)
    k4h, k4u = _rhs(h +     dt*k3h, u +     dt*k3u, b, kx, dealias, hyper, g, visc)
    h_new = h + (dt / 6.0) * (k1h + 2*k2h + 2*k3h + k4h)
    u_new = u + (dt / 6.0) * (k1u + 2*k2u + 2*k3u + k4u)
    return h_new, u_new


# --------------------------------------------------------------------------- #
# Trajectory
# --------------------------------------------------------------------------- #
def simulate_1d(cfg, rng: np.random.Generator):
    """Run one trajectory. Returns dict with h0, u0, b, h, u (final time) or None on failure."""
    kx, dealias, hyper = _make_spectral_ops(cfg.nx, cfg.Lx)
    dx = cfg.Lx / cfg.nx

    b = make_bottom_1d(cfg, rng)
    h, u = make_initial_1d(cfg, b, rng)
    h0, u0 = h.copy(), u.copy()

    c_max = float(np.sqrt(cfg.g * (cfg.H0 + cfg.bottom_amp)) + np.abs(u).max())
    dt = cfg.cfl * dx / max(c_max, 1e-6)
    nt = max(int(np.ceil(cfg.T / dt)), 1)
    dt = cfg.T / nt

    for _ in range(nt):
        h, u = _rk4(h, u, b, kx, dealias, hyper, dt, cfg.g, cfg.visc)
        if (not np.all(np.isfinite(h))) or (h.min() <= 1e-3):
            return None  # reject blow-up / dry bed

    return dict(h0=h0, u0=u0, b=b, h=h, u=u)


def generate_dataset_1d(cfg, n_samples: int, base_seed: int = 0):
    """Generate `n_samples` trajectories. Returns dict of stacked arrays [N, nx]."""
    rng = np.random.default_rng(base_seed)
    fields = {k: [] for k in ["h0", "u0", "b", "h", "u"]}
    n_done = 0
    n_try = 0
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
