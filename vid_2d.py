"""Animated 2D SWE: autoregressive FNO rollout vs true solver.

Usage:
    python vid_2d.py [--run-dir runs/swe2d] [--n-frames 20] [--fps 10] [--sample-idx 0]

Loads the trained checkpoint, rolls the FNO forward n_frames steps autoregressively,
runs the true solver for the same steps, and saves a side-by-side MP4 showing the
water depth h (heatmap) with velocity quiver arrows to {run_dir}/plots/evolution_2d.mp4.
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

from config import Config2D
from fno import FNO2d
from solver_2d import simulate_step_2d
from utils import ChannelNormalizer


def _make_normalizer(mean: torch.Tensor, std: torch.Tensor) -> ChannelNormalizer:
    norm = object.__new__(ChannelNormalizer)
    norm.mean = mean
    norm.std  = std
    return norm


def load_run(run_dir: str):
    cfg  = Config2D()
    ckpt = torch.load(os.path.join(run_dir, "fno2d_best.pt"), map_location="cpu",
                      weights_only=False)
    model = FNO2d(cfg.in_channels, cfg.out_channels,
                  cfg.modes_x, cfg.modes_y, cfg.width, cfg.n_layers)
    model.load_state_dict(ckpt["model"])
    model.eval()
    x_norm = _make_normalizer(ckpt["x_norm_mean"], ckpt["x_norm_std"])
    y_norm = _make_normalizer(ckpt["y_norm_mean"], ckpt["y_norm_std"])
    npz    = np.load(os.path.join(run_dir, "data.npz"))
    keys   = ["h0", "u0", "v0", "b", "h", "u", "v", "f"]
    test   = {k: npz[f"test_{k}"] for k in keys}
    return cfg, model, x_norm, y_norm, test


def build_solver_trajectory(cfg, h0, u0, v0, b, f, n_frames):
    """Run solver for n_frames steps of length T each, saving snapshots."""
    h_frames = [h0.copy()]
    u_frames = [u0.copy()]
    v_frames = [v0.copy()]
    h_cur = h0.astype(np.float64)
    u_cur = u0.astype(np.float64)
    v_cur = v0.astype(np.float64)
    b64   = b.astype(np.float64)
    for _ in range(n_frames):
        h_cur, u_cur, v_cur = simulate_step_2d(cfg, h_cur, u_cur, v_cur, b64, f)
        h_frames.append(h_cur.copy())
        u_frames.append(u_cur.copy())
        v_frames.append(v_cur.copy())
    return h_frames, u_frames, v_frames


@torch.no_grad()
def build_fno_trajectory(model, x_norm, y_norm, h0, u0, v0, b, f, n_frames):
    """Roll FNO forward n_frames steps autoregressively."""
    ny, nx = h0.shape
    f_field = np.full((ny, nx), f, dtype=np.float32)

    h_frames = [h0.copy()]
    u_frames = [u0.copy()]
    v_frames = [v0.copy()]
    h_cur = torch.tensor(h0, dtype=torch.float32)
    u_cur = torch.tensor(u0, dtype=torch.float32)
    v_cur = torch.tensor(v0, dtype=torch.float32)
    b_t   = torch.tensor(b,  dtype=torch.float32)
    f_t   = torch.tensor(f_field, dtype=torch.float32)

    for _ in range(n_frames):
        x    = torch.stack([h_cur, u_cur, v_cur, b_t, f_t], dim=0).unsqueeze(0)  # [1,5,ny,nx]
        pred = y_norm.decode(model(x_norm.encode(x)))[0]                          # [3,ny,nx]
        h_cur = pred[0]
        u_cur = pred[1]
        v_cur = pred[2]
        h_frames.append(h_cur.numpy().copy())
        u_frames.append(u_cur.numpy().copy())
        v_frames.append(v_cur.numpy().copy())
    return h_frames, u_frames, v_frames


def make_animation(cfg, truth_h, truth_u, truth_v, fno_h, fno_u, fno_v, fps):
    x = np.linspace(0, cfg.Lx, cfg.nx, endpoint=False)
    y = np.linspace(0, cfg.Ly, cfg.ny, endpoint=False)
    n_frames = len(truth_h) - 1
    stride   = max(1, cfg.nx // 16)
    X_q, Y_q = np.meshgrid(x[::stride], y[::stride])

    # Fixed colorscale from frame-0 truth amplitude
    vabs = max(float(np.abs(truth_h[0]).max()), 0.01)

    fig, (ax_t, ax_f) = plt.subplots(1, 2, figsize=(12, 5))
    im_t = ax_t.pcolormesh(x, y, truth_h[0], cmap="RdBu_r",
                            vmin=-vabs, vmax=vabs, shading="auto")
    im_f = ax_f.pcolormesh(x, y, fno_h[0],   cmap="RdBu_r",
                            vmin=-vabs, vmax=vabs, shading="auto")
    fig.colorbar(im_t, ax=ax_t, label="h")
    fig.colorbar(im_f, ax=ax_f, label="h")
    ax_t.set_title("Truth");    ax_t.set_xlabel("x"); ax_t.set_ylabel("y")
    ax_f.set_title("FNO");      ax_f.set_xlabel("x")

    q_t = ax_t.quiver(X_q, Y_q,
                       truth_u[0][::stride, ::stride],
                       truth_v[0][::stride, ::stride],
                       color="k", alpha=0.5, scale_units="xy")
    q_f = ax_f.quiver(X_q, Y_q,
                       fno_u[0][::stride, ::stride],
                       fno_v[0][::stride, ::stride],
                       color="k", alpha=0.5, scale_units="xy")

    title = fig.suptitle("t = 0.000")
    fig.tight_layout()

    def update(frame):
        im_t.set_array(truth_h[frame].ravel())
        im_f.set_array(fno_h[frame].ravel())
        q_t.set_UVC(truth_u[frame][::stride, ::stride],
                    truth_v[frame][::stride, ::stride])
        q_f.set_UVC(fno_u[frame][::stride, ::stride],
                    fno_v[frame][::stride, ::stride])
        title.set_text(f"t = {frame * cfg.T:.3f}")
        return im_t, im_f, q_t, q_f

    anim = animation.FuncAnimation(fig, update, frames=n_frames + 1,
                                   interval=1000 // fps, blit=False)
    return fig, anim


def main():
    parser = argparse.ArgumentParser(description="Animated 2D SWE FNO vs truth")
    parser.add_argument("--run-dir",    default="runs/swe2d")
    parser.add_argument("--n-frames",   type=int, default=20)
    parser.add_argument("--fps",        type=int, default=10)
    parser.add_argument("--sample-idx", type=int, default=0)
    args = parser.parse_args()

    run_dir  = args.run_dir
    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    print(f"[vid2d] Loading run from {run_dir} ...")
    cfg, model, x_norm, y_norm, test = load_run(run_dir)

    idx = args.sample_idx
    h0  = test["h0"][idx]
    u0  = test["u0"][idx]
    v0  = test["v0"][idx]
    b   = test["b"][idx]
    f   = float(test["f"][idx])

    print(f"[vid2d] Building solver trajectory ({args.n_frames} steps) ...")
    truth_h, truth_u, truth_v = build_solver_trajectory(
        cfg, h0, u0, v0, b, f, args.n_frames)

    print(f"[vid2d] Building FNO trajectory ({args.n_frames} steps) ...")
    fno_h, fno_u, fno_v = build_fno_trajectory(
        model, x_norm, y_norm, h0, u0, v0, b, f, args.n_frames)

    print("[vid2d] Rendering animation ...")
    fig, anim = make_animation(cfg, truth_h, truth_u, truth_v,
                                fno_h, fno_u, fno_v, args.fps)

    out_path = os.path.join(plot_dir, "evolution_2d.mp4")
    writer = animation.FFMpegWriter(fps=args.fps,
                                    metadata={"title": "SWE 2D FNO evolution"})
    anim.save(out_path, writer=writer)
    plt.close(fig)
    print(f"[vid2d] Written to {out_path}")


if __name__ == "__main__":
    main()
