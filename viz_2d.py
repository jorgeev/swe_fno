"""Standalone 2D visualization script.

Usage:
    python viz_2d.py [--run-dir runs/swe2d]

Loads fno2d_best.pt, data.npz, and (optionally) history.json from the run
directory, then writes figures to {run_dir}/plots/.
"""
from __future__ import annotations
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config2D
from dataset import SWEDataset2D
from fno import FNO2d
from utils import ChannelNormalizer


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #
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

    npz  = np.load(os.path.join(run_dir, "data.npz"))
    keys = ["h0", "u0", "v0", "b", "h", "u", "v", "f"]
    test = {k: npz[f"test_{k}"] for k in keys}

    history = None
    hist_path = os.path.join(run_dir, "history.json")
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            history = json.load(f)

    return cfg, model, x_norm, y_norm, test, history, ckpt


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def get_predictions(model, x_norm, y_norm, test, n: int = 3):
    """Return (truth [n,3,ny,nx], pred [n,3,ny,nx]) numpy arrays."""
    ds = SWEDataset2D(test)
    n  = min(n, len(ds))
    xs, ys = [], []
    for i in range(n):
        x, y = ds[i]
        xs.append(x); ys.append(y)
    x_batch = torch.stack(xs)   # [n, 5, ny, nx]
    y_batch = torch.stack(ys)   # [n, 3, ny, nx]
    pred    = y_norm.decode(model(x_norm.encode(x_batch)))
    return y_batch.numpy(), pred.numpy()


@torch.no_grad()
def compute_per_sample_errors(model, x_norm, y_norm, test):
    """Relative L2 per channel per sample. Returns (eh[N], eu[N], ev[N])."""
    ds     = SWEDataset2D(test)
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    errs   = [[], [], []]
    for x, y in loader:
        pred = y_norm.decode(model(x_norm.encode(x)))
        for i in range(x.shape[0]):
            for c in range(3):
                errs[c].append(
                    (pred[i,c] - y[i,c]).norm() / (y[i,c].norm() + 1e-8)
                )
    return [np.array([e.item() for e in col]) for col in errs]


# --------------------------------------------------------------------------- #
# Plot functions
# --------------------------------------------------------------------------- #
def plot_training_curves(history: dict, plot_dir: str, best_epoch: int | None):
    epochs = np.arange(1, len(history["train"]) + 1)
    fig, ax1 = plt.subplots(figsize=(9, 4))

    ax1.semilogy(epochs, history["train"], color="C0", lw=1.5, label="train loss")
    ax1.semilogy(epochs, history["val"],   color="C1", lw=1.5, label="val loss")
    if best_epoch is not None:
        ax1.axvline(best_epoch, color="gray", ls="--", alpha=0.7,
                    label=f"best (ep {best_epoch})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Relative L² loss")

    ax2 = ax1.twinx()
    ax2.plot(epochs, history["lr"], color="C2", ls=":", lw=1.2, label="lr")
    ax2.set_ylabel("Learning rate")
    ax2.set_yscale("log")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    ax1.set_title("Training curves — 2D SWE FNO")
    fig.tight_layout()
    path = os.path.join(plot_dir, "training_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def plot_field_comparison(truth, pred, cfg, plot_dir, n: int = 3):
    """For each of n samples: 3-row (h,u,v) × 3-col (truth|pred|error) grid."""
    n          = min(n, truth.shape[0])
    ch_names   = ["h", "u", "v"]
    col_labels = ["Truth", "FNO pred", "|Error|"]

    for si in range(n):
        fig, axes = plt.subplots(3, 3, figsize=(13, 9))
        for ci, name in enumerate(ch_names):
            tr  = truth[si, ci]
            pr  = pred[si, ci]
            err = np.abs(pr - tr)

            vabs = max(np.abs(tr).max(), 1e-12)
            im0 = axes[ci, 0].imshow(tr,  origin="lower", cmap="RdBu_r",
                                     vmin=-vabs, vmax=vabs)
            im1 = axes[ci, 1].imshow(pr,  origin="lower", cmap="RdBu_r",
                                     vmin=-vabs, vmax=vabs)
            im2 = axes[ci, 2].imshow(err, origin="lower", cmap="viridis",
                                     vmin=0)

            for ax, im, cl in zip(axes[ci], [im0, im1, im2], col_labels):
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_xlabel("x"); ax.set_ylabel("y")
                if ci == 0:
                    ax.set_title(cl)
            axes[ci, 0].set_ylabel(f"{name}\ny")

        fig.suptitle(f"2D SWE — Test sample {si} (T={cfg.T:.2f})", y=1.01)
        fig.tight_layout()
        path = os.path.join(plot_dir, f"field_comparison_sample{si}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {path}")


def plot_quiver_comparison(truth, pred, cfg, plot_dir, n: int = 2, stride: int = 4):
    """Velocity quiver overlaid on h depth colormap: truth (left) vs FNO (right)."""
    n  = min(n, truth.shape[0])
    nx, ny = truth.shape[3], truth.shape[2]
    x  = np.linspace(0, cfg.Lx, nx, endpoint=False)
    y  = np.linspace(0, cfg.Ly, ny, endpoint=False)
    X, Y = np.meshgrid(x[::stride], y[::stride])

    for si in range(n):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, src, label in zip(axes, [truth[si], pred[si]], ["Truth", "FNO pred"]):
            h = src[0]
            u = src[1][::stride, ::stride]
            v = src[2][::stride, ::stride]
            speed = np.hypot(u, v).max()
            scale = speed * nx / stride * 2 if speed > 0 else 1.0

            im = ax.pcolormesh(x, y, h, cmap="Blues", shading="auto")
            ax.quiver(X, Y, u, v, scale=scale, color="k", alpha=0.75, width=0.003)
            fig.colorbar(im, ax=ax, label="h (depth)")
            ax.set_title(f"{label} — sample {si}")
            ax.set_xlabel("x"); ax.set_ylabel("y")
            ax.set_aspect("equal")

        fig.suptitle("2D SWE — Velocity field over depth")
        fig.tight_layout()
        path = os.path.join(plot_dir, f"quiver_comparison_sample{si}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {path}")


def plot_error_histogram(errors: list[np.ndarray], plot_dir: str):
    names = ["h", "u", "v"]
    lo = min(e.min() for e in errors) * 0.5
    hi = max(e.max() for e in errors) * 2.0
    lo = max(lo, 1e-6)
    bins = np.logspace(np.log10(lo), np.log10(hi), 40)

    fig, ax = plt.subplots(figsize=(8, 4))
    for e, name in zip(errors, names):
        ax.hist(e, bins=bins, alpha=0.5,
                label=f"{name}  (med={np.median(e):.2e}, std={e.std():.2e})")
    ax.set_xscale("log")
    ax.set_xlabel("Per-sample relative L² error")
    ax.set_ylabel("Count")
    ax.set_title("2D SWE FNO — Test error distribution")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(plot_dir, "error_histogram.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Visualize 2D SWE FNO results")
    parser.add_argument("--run-dir", default="runs/swe2d",
                        help="Directory containing fno2d_best.pt and data.npz")
    args = parser.parse_args()

    run_dir  = args.run_dir
    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    print(f"[viz2d] Loading run from {run_dir} ...")
    cfg, model, x_norm, y_norm, test, history, ckpt = load_run(run_dir)

    if history is not None:
        plot_training_curves(history, plot_dir, best_epoch=ckpt.get("epoch"))
    else:
        print("  history.json not found — skipping training curves")

    print("[viz2d] Running inference on test samples ...")
    truth, pred = get_predictions(model, x_norm, y_norm, test, n=3)
    plot_field_comparison(truth, pred, cfg, plot_dir)
    plot_quiver_comparison(truth, pred, cfg, plot_dir)

    print("[viz2d] Computing per-sample errors across full test set ...")
    errors = compute_per_sample_errors(model, x_norm, y_norm, test)
    plot_error_histogram(errors, plot_dir)

    print(f"[viz2d] Done. Plots written to {plot_dir}/")


if __name__ == "__main__":
    main()
