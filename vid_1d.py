"""Animated 1D SWE: autoregressive FNO rollout vs true solver.

Usage:
    python vid_1d.py [--run-dir runs/swe1d] [--n-frames 20] [--fps 10] [--sample-idx 0]

Loads the trained checkpoint, rolls the FNO forward n_frames steps autoregressively
(using each prediction as the next input), runs the true solver for the same number
of steps, and saves a side-by-side MP4 to {run_dir}/plots/evolution_1d.mp4.
"""
from __future__ import annotations
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch

from config import Config1D
from fno import FNO1d
from solver_1d import simulate_step_1d
from utils import ChannelNormalizer


def _make_normalizer(mean: torch.Tensor, std: torch.Tensor) -> ChannelNormalizer:
    norm = object.__new__(ChannelNormalizer)
    norm.mean = mean
    norm.std  = std
    return norm


def load_run(run_dir: str):
    cfg  = Config1D()
    ckpt = torch.load(os.path.join(run_dir, "fno1d_best.pt"), map_location="cpu",
                      weights_only=False)
    model = FNO1d(cfg.in_channels, cfg.out_channels, cfg.modes, cfg.width, cfg.n_layers)
    model.load_state_dict(ckpt["model"])
    model.eval()
    x_norm = _make_normalizer(ckpt["x_norm_mean"], ckpt["x_norm_std"])
    y_norm = _make_normalizer(ckpt["y_norm_mean"], ckpt["y_norm_std"])
    npz    = np.load(os.path.join(run_dir, "data.npz"))
    test   = {k: npz[f"test_{k}"] for k in ["h0", "u0", "b", "h", "u"]}
    return cfg, model, x_norm, y_norm, test


def build_solver_trajectory(cfg, h0, u0, b, n_frames):
    """Run solver for n_frames steps of length T each, saving snapshots."""
    h_frames = [h0.copy()]
    u_frames = [u0.copy()]
    h_cur, u_cur = h0.astype(np.float64), u0.astype(np.float64)
    b64 = b.astype(np.float64)
    for _ in range(n_frames):
        h_cur, u_cur = simulate_step_1d(cfg, h_cur, u_cur, b64)
        h_frames.append(h_cur.copy())
        u_frames.append(u_cur.copy())
    return h_frames, u_frames


@torch.no_grad()
def build_fno_trajectory(model, x_norm, y_norm, h0, u0, b, n_frames):
    """Roll FNO forward n_frames steps autoregressively."""
    h_frames = [h0.copy()]
    u_frames = [u0.copy()]
    h_cur = torch.tensor(h0, dtype=torch.float32)
    u_cur = torch.tensor(u0, dtype=torch.float32)
    b_t   = torch.tensor(b,  dtype=torch.float32)
    for _ in range(n_frames):
        x    = torch.stack([h_cur, u_cur, b_t], dim=0).unsqueeze(0)  # [1, 3, nx]
        pred = y_norm.decode(model(x_norm.encode(x)))[0]              # [2, nx]
        h_cur = pred[0]
        u_cur = pred[1]
        h_frames.append(h_cur.numpy().copy())
        u_frames.append(u_cur.numpy().copy())
    return h_frames, u_frames


def make_animation(cfg, truth_h, truth_u, fno_h, fno_u, fps):
    x = np.linspace(0, cfg.Lx, cfg.nx, endpoint=False)
    n_frames = len(truth_h) - 1

    fig, (ax_h, ax_u) = plt.subplots(2, 1, figsize=(10, 6))

    line_th, = ax_h.plot(x, truth_h[0], color="C0",  lw=1.5, label="truth")
    line_fh, = ax_h.plot(x, fno_h[0],   color="C1",  lw=1.2, ls="--", label="FNO")
    line_tu, = ax_u.plot(x, truth_u[0], color="C0",  lw=1.5, label="truth")
    line_fu, = ax_u.plot(x, fno_u[0],   color="C1",  lw=1.2, ls="--", label="FNO")

    ax_h.set_ylabel("h");    ax_h.legend(loc="upper right", fontsize=8)
    ax_u.set_ylabel("u");    ax_u.set_xlabel("x")
    title = fig.suptitle("t = 0.000")
    fig.tight_layout()

    def update(frame):
        line_th.set_ydata(truth_h[frame])
        line_fh.set_ydata(fno_h[frame])
        line_tu.set_ydata(truth_u[frame])
        line_fu.set_ydata(fno_u[frame])
        for ax, dt, df in [(ax_h, truth_h[frame], fno_h[frame]),
                           (ax_u, truth_u[frame], fno_u[frame])]:
            lo = min(dt.min(), df.min())
            hi = max(dt.max(), df.max())
            pad = max((hi - lo) * 0.05, 1e-6)
            ax.set_ylim(lo - pad, hi + pad)
        title.set_text(f"t = {frame * cfg.T:.3f}")
        return line_th, line_fh, line_tu, line_fu

    anim = animation.FuncAnimation(fig, update, frames=n_frames + 1,
                                   interval=1000 // fps, blit=False)
    return fig, anim


def main():
    parser = argparse.ArgumentParser(description="Animated 1D SWE FNO vs truth")
    parser.add_argument("--run-dir",    default="runs/swe1d")
    parser.add_argument("--n-frames",   type=int, default=20)
    parser.add_argument("--fps",        type=int, default=10)
    parser.add_argument("--sample-idx", type=int, default=0)
    args = parser.parse_args()

    run_dir  = args.run_dir
    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    print(f"[vid1d] Loading run from {run_dir} ...")
    cfg, model, x_norm, y_norm, test = load_run(run_dir)

    idx = args.sample_idx
    h0  = test["h0"][idx]
    u0  = test["u0"][idx]
    b   = test["b"][idx]

    print(f"[vid1d] Building solver trajectory ({args.n_frames} steps) ...")
    truth_h, truth_u = build_solver_trajectory(cfg, h0, u0, b, args.n_frames)

    print(f"[vid1d] Building FNO trajectory ({args.n_frames} steps) ...")
    fno_h, fno_u = build_fno_trajectory(model, x_norm, y_norm, h0, u0, b, args.n_frames)

    print("[vid1d] Rendering animation ...")
    fig, anim = make_animation(cfg, truth_h, truth_u, fno_h, fno_u, args.fps)

    out_path = os.path.join(plot_dir, "evolution_1d.mp4")
    writer = animation.FFMpegWriter(fps=args.fps,
                                    metadata={"title": "SWE 1D FNO evolution"})
    anim.save(out_path, writer=writer)
    plt.close(fig)
    print(f"[vid1d] Written to {out_path}")


if __name__ == "__main__":
    main()
