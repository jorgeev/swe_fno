"""2D shallow-water solver (periodic, variable bottom, f-plane Coriolis) — GPU-accelerated via CuPy.

Equations
---------
    h_t + (h u)_x + (h v)_y = 0
    u_t + u u_x + v u_y - f v + g (h + b)_x = 0
    v_t + u v_x + v v_y + f u + g (h + b)_y = 0

Arrays are shaped (ny, nx); axis 0 is y, axis 1 is x.
Pseudo-spectral derivatives, RK4 in time, 2/3-rule dealiasing,
small (k^2)^2 hyperviscosity.

Initial condition generation runs on CPU (NumPy); the RK4 integration
loop runs on GPU (CuPy).  CuPy is a hard requirement.
"""
from __future__ import annotations
import numpy as np
import cupy as cp


# --------------------------------------------------------------------------- #
# Random fields  (CPU — called once per trajectory)
# --------------------------------------------------------------------------- #
def _smooth_random_2d(ny, nx, Ly, Lx, n_modes, rng) -> np.ndarray:
    x = np.linspace(0.0, Lx, nx, endpoint=False)
    y = np.linspace(0.0, Ly, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="xy")
    f = np.zeros((ny, nx))
    for kx in range(0, n_modes + 1):
        for ky in range(0, n_modes + 1):
            if kx == 0 and ky == 0:
                continue
            scale = 1.0 / (kx + ky + 1)
            phx = rng.uniform(0, 2 * np.pi)
            phy = rng.uniform(0, 2 * np.pi)
            f += rng.normal() * scale * np.cos(2 * np.pi * kx * X / Lx + phx) \
                                       * np.cos(2 * np.pi * ky * Y / Ly + phy)
    f -= f.mean()
    m = np.abs(f).max()
    return f / m if m > 0 else f


def make_bottom_2d(cfg, rng):
    return cfg.bottom_amp * _smooth_random_2d(cfg.ny, cfg.nx, cfg.Ly, cfg.Lx,
                                              cfg.n_modes_ic, rng)


def make_initial_2d(cfg, b, rng):
    eta = cfg.eta_amp * _smooth_random_2d(cfg.ny, cfg.nx, cfg.Ly, cfg.Lx,
                                          cfg.n_modes_ic, rng)
    u = cfg.u_amp * _smooth_random_2d(cfg.ny, cfg.nx, cfg.Ly, cfg.Lx,
                                      cfg.n_modes_ic, rng)
    v = cfg.u_amp * _smooth_random_2d(cfg.ny, cfg.nx, cfg.Ly, cfg.Lx,
                                      cfg.n_modes_ic, rng)
    h = (cfg.H0 - b) + eta
    return h, u, v


# --------------------------------------------------------------------------- #
# Spectral helpers  (CPU — one-time setup)
# --------------------------------------------------------------------------- #
def _make_spectral_ops(ny, nx, Ly, Lx):
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=Lx / nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=Ly / ny)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")           # both (ny, nx)

    kmax_x = np.abs(kx).max()
    kmax_y = np.abs(ky).max()
    dealias = ((np.abs(KX) <= (2.0 / 3.0) * kmax_x) &
               (np.abs(KY) <= (2.0 / 3.0) * kmax_y)).astype(np.float64)

    K2    = KX ** 2 + KY ** 2
    K2max = max(K2.max(), 1e-12)
    hyper = (K2 / K2max) ** 2
    return KX, KY, dealias, hyper


# --------------------------------------------------------------------------- #
# RHS / time stepping  (GPU — cupy arrays throughout)
# --------------------------------------------------------------------------- #
def _rhs(h, u, v, b, KX, KY, dealias, hyper, f, g, visc):
    # axes=(-2,-1) makes this work for unbatched (ny,nx) and batched (B,ny,nx)
    _fft2  = lambda x: cp.fft.fft2(x, axes=(-2, -1))
    _re_ifft = lambda F: cp.real(cp.fft.ifft2(F, axes=(-2, -1)))

    Hk = _fft2(h)
    Uk = _fft2(u)
    Vk = _fft2(v)
    Bk = _fft2(b)

    dudx    = _re_ifft(1j * KX * Uk)
    dudy    = _re_ifft(1j * KY * Uk)
    dvdx    = _re_ifft(1j * KX * Vk)
    dvdy    = _re_ifft(1j * KY * Vk)
    deta_dx = _re_ifft(1j * KX * (Hk + Bk))
    deta_dy = _re_ifft(1j * KY * (Hk + Bk))

    hu_d   = _re_ifft(_fft2(h * u) * dealias)
    hv_d   = _re_ifft(_fft2(h * v) * dealias)
    dhu_dx = _re_ifft(1j * KX * _fft2(hu_d))
    dhv_dy = _re_ifft(1j * KY * _fft2(hv_d))

    udu_vdu = _re_ifft(_fft2(u * dudx + v * dudy) * dealias)
    udv_vdv = _re_ifft(_fft2(u * dvdx + v * dvdy) * dealias)

    dh_dt = -(dhu_dx + dhv_dy)              - visc * _re_ifft(hyper * Hk)
    du_dt = -udu_vdu + f * v - g * deta_dx  - visc * _re_ifft(hyper * Uk)
    dv_dt = -udv_vdv - f * u - g * deta_dy  - visc * _re_ifft(hyper * Vk)
    return dh_dt, du_dt, dv_dt


def _rk4(h, u, v, b, KX, KY, dealias, hyper, dt, f, g, visc):
    k1 = _rhs(h, u, v, b, KX, KY, dealias, hyper, f, g, visc)
    k2 = _rhs(h + 0.5*dt*k1[0], u + 0.5*dt*k1[1], v + 0.5*dt*k1[2],
              b, KX, KY, dealias, hyper, f, g, visc)
    k3 = _rhs(h + 0.5*dt*k2[0], u + 0.5*dt*k2[1], v + 0.5*dt*k2[2],
              b, KX, KY, dealias, hyper, f, g, visc)
    k4 = _rhs(h +     dt*k3[0], u +     dt*k3[1], v +     dt*k3[2],
              b, KX, KY, dealias, hyper, f, g, visc)
    h_new = h + (dt / 6.0) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
    u_new = u + (dt / 6.0) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    v_new = v + (dt / 6.0) * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
    return h_new, u_new, v_new


# --------------------------------------------------------------------------- #
# Trajectory
# --------------------------------------------------------------------------- #
def simulate_2d(cfg, rng):
    """Run one trajectory on GPU. Returns dict {h0,u0,v0,b,f,h,u,v} (numpy) or None on failure."""
    KX_np, KY_np, dealias_np, hyper_np = _make_spectral_ops(cfg.ny, cfg.nx, cfg.Ly, cfg.Lx)
    dx = min(cfg.Lx / cfg.nx, cfg.Ly / cfg.ny)

    b_np         = make_bottom_2d(cfg, rng)
    h_np, u_np, v_np = make_initial_2d(cfg, b_np, rng)
    h0, u0, v0   = h_np.copy(), u_np.copy(), v_np.copy()

    f = float(rng.uniform(cfg.f_min, cfg.f_max))

    c_max = float(np.sqrt(cfg.g * (cfg.H0 + cfg.bottom_amp))
                  + np.sqrt(np.maximum(u_np, 0).max() ** 2 + np.maximum(v_np, 0).max() ** 2)
                  + np.abs(u_np).max() + np.abs(v_np).max())
    dt = cfg.cfl * dx / max(c_max, 1e-6)
    nt = max(int(np.ceil(cfg.T / dt)), 1)
    dt = cfg.T / nt

    # Transfer IC and operators to GPU
    h       = cp.asarray(h_np)
    u       = cp.asarray(u_np)
    v       = cp.asarray(v_np)
    b       = cp.asarray(b_np)
    KX      = cp.asarray(KX_np)
    KY      = cp.asarray(KY_np)
    dealias = cp.asarray(dealias_np)
    hyper   = cp.asarray(hyper_np)

    for _ in range(nt):
        h, u, v = _rk4(h, u, v, b, KX, KY, dealias, hyper, dt, f, cfg.g, cfg.visc)
        if not bool(cp.all(cp.isfinite(h))) or float(h.min()) <= 1e-3:
            return None

    return dict(h0=h0, u0=u0, v0=v0, b=b_np, f=f,
                h=cp.asnumpy(h), u=cp.asnumpy(u), v=cp.asnumpy(v))


def simulate_batch_2d(cfg, rng, B: int) -> list:
    """Simulate B trajectories in parallel on GPU.

    Returns a list of length B; each element is a result dict
    {h0,u0,v0,b,f,h,u,v} (numpy arrays of shape (ny,nx), f is a float)
    or None if that trajectory was invalid.
    """
    KX_np, KY_np, dealias_np, hyper_np = _make_spectral_ops(cfg.ny, cfg.nx, cfg.Ly, cfg.Lx)
    dx = min(cfg.Lx / cfg.nx, cfg.Ly / cfg.ny)

    b_list   = [make_bottom_2d(cfg, rng) for _ in range(B)]
    huv_list = [make_initial_2d(cfg, b, rng) for b in b_list]
    f_list   = [float(rng.uniform(cfg.f_min, cfg.f_max)) for _ in range(B)]

    h0_arr = np.stack([huv[0] for huv in huv_list])  # (B, ny, nx)
    u0_arr = np.stack([huv[1] for huv in huv_list])
    v0_arr = np.stack([huv[2] for huv in huv_list])
    b_arr  = np.stack(b_list)

    # Conservative dt across batch
    c_max = max(
        float(np.sqrt(cfg.g * (cfg.H0 + cfg.bottom_amp))
              + np.sqrt(np.maximum(huv[1], 0).max()**2 + np.maximum(huv[2], 0).max()**2)
              + np.abs(huv[1]).max() + np.abs(huv[2]).max())
        for huv in huv_list
    )
    dt = cfg.cfl * dx / max(c_max, 1e-6)
    nt = max(int(np.ceil(cfg.T / dt)), 1)
    dt = cfg.T / nt

    h       = cp.asarray(h0_arr)                          # (B, ny, nx)
    u       = cp.asarray(u0_arr)
    v       = cp.asarray(v0_arr)
    b       = cp.asarray(b_arr)
    KX      = cp.asarray(KX_np)                           # (ny, nx) — broadcasts
    KY      = cp.asarray(KY_np)
    dealias = cp.asarray(dealias_np)
    hyper   = cp.asarray(hyper_np)
    f_gpu   = cp.asarray(np.array(f_list)).reshape(B, 1, 1)  # (B, 1, 1) for broadcast

    # Run all B trajectories without per-step GPU→CPU sync
    for _ in range(nt):
        h, u, v = _rk4(h, u, v, b, KX, KY, dealias, hyper, dt, f_gpu, cfg.g, cfg.visc)

    h_np = cp.asnumpy(h)   # single transfer
    u_np = cp.asnumpy(u)
    v_np = cp.asnumpy(v)

    results = []
    for i in range(B):
        hi = h_np[i]
        if not np.all(np.isfinite(hi)) or hi.min() <= 1e-3:
            results.append(None)
        else:
            results.append(dict(h0=h0_arr[i], u0=u0_arr[i], v0=v0_arr[i],
                                b=b_arr[i], f=f_list[i],
                                h=hi.copy(), u=u_np[i].copy(), v=v_np[i].copy()))
    return results


def simulate_step_2d(cfg, h_np: np.ndarray, u_np: np.ndarray, v_np: np.ndarray,
                     b_np: np.ndarray, f: float) -> tuple:
    """Advance the 2D SWE by cfg.T from a given state.

    Accepts float32 or float64 numpy arrays; returns numpy arrays of the same dtype.
    No validity check — intended for video generation from a known-good state.
    """
    KX_np, KY_np, dealias_np, hyper_np = _make_spectral_ops(cfg.ny, cfg.nx, cfg.Ly, cfg.Lx)
    dx = min(cfg.Lx / cfg.nx, cfg.Ly / cfg.ny)

    c_max = float(np.sqrt(cfg.g * (cfg.H0 + cfg.bottom_amp))
                  + np.sqrt(np.maximum(u_np, 0).max()**2 + np.maximum(v_np, 0).max()**2)
                  + np.abs(u_np).max() + np.abs(v_np).max())
    dt = cfg.cfl * dx / max(c_max, 1e-6)
    nt = max(int(np.ceil(cfg.T / dt)), 1)
    dt = cfg.T / nt

    h       = cp.asarray(h_np)
    u       = cp.asarray(u_np)
    v       = cp.asarray(v_np)
    b       = cp.asarray(b_np)
    KX      = cp.asarray(KX_np)
    KY      = cp.asarray(KY_np)
    dealias = cp.asarray(dealias_np)
    hyper   = cp.asarray(hyper_np)

    for _ in range(nt):
        h, u, v = _rk4(h, u, v, b, KX, KY, dealias, hyper, dt, f, cfg.g, cfg.visc)

    return cp.asnumpy(h), cp.asnumpy(u), cp.asnumpy(v)


def generate_dataset_2d(cfg, n_samples: int, base_seed: int = 0):
    """Generate ``n_samples`` trajectories. Returns dict of stacked numpy arrays."""
    rng    = np.random.default_rng(base_seed)
    keys   = ["h0", "u0", "v0", "b", "h", "u", "v"]
    fields = {k: [] for k in keys}
    f_list = []
    n_done = 0
    n_try  = 0
    B      = cfg.gen_batch_size
    next_print = max(1, n_samples // 10)

    while n_done < n_samples:
        for out in simulate_batch_2d(cfg, rng, B):
            n_try += 1
            if out is None:
                continue
            for k in keys:
                fields[k].append(out[k])
            f_list.append(out["f"])
            n_done += 1
            if n_done >= next_print or n_done >= n_samples:
                print(f"  [2D] {n_done}/{n_samples} (rejected {n_try - n_done})")
                next_print += max(1, n_samples // 10)
            if n_done >= n_samples:
                break

    data      = {k: np.stack(v[:n_samples]).astype(np.float32) for k, v in fields.items()}
    data["f"] = np.array(f_list[:n_samples], dtype=np.float32)
    return data
