"""Standalone 1D visualization script.

Usage:
    python viz_1d.py [--run-dir runs/swe1d]

Loads fno1d_best.pt, data.npz, and (optionally) history.json from the run
directory, then writes four PNG figures to {run_dir}/plots/.
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

from config import Config1D
from dataset import SWEDataset1D
from fno import FNO1d
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
    cfg  = Config1D()
    ckpt = torch.load(os.path.join(run_dir, "fno1d_best.pt"), map_location="cpu",
                      weights_only=False)

    model = FNO1d(cfg.in_channels, cfg.out_channels, cfg.modes, cfg.width, cfg.n_layers)
    model.load_state_dict(ckpt["model"])
    model.eval()

    x_norm = _make_normalizer(ckpt["x_norm_mean"], ckpt["x_norm_std"])
    y_norm = _make_normalizer(ckpt["y_norm_mean"], ckpt["y_norm_std"])

    npz  = np.load(os.path.join(run_dir, "data.npz"))
    test = {k: npz[f"test_{k}"] for k in ["h0", "u0", "b", "h", "u"]}

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
def get_predictions(model, x_norm, y_norm, test, n: int = 6):
    """Return (truth [n,2,nx], pred [n,2,nx]) numpy arrays."""
    n    = min(n, len(test["h0"]))
    h0   = torch.tensor(test["h0"][:n])
    u0   = torch.tensor(test["u0"][:n])
    b    = torch.tensor(test["b"][:n])
    h    = torch.tensor(test["h"][:n])
    u    = torch.tensor(test["u"][:n])

    x    = torch.stack([h0, u0, b], dim=1)   # [n, 3, nx]
    y    = torch.stack([h, u],      dim=1)    # [n, 2, nx]
    pred = y_norm.decode(model(x_norm.encode(x)))
    return y.numpy(), pred.numpy()


@torch.no_grad()
def compute_per_sample_errors(model, x_norm, y_norm, test):
    """Relative L2 per channel per sample. Returns (errors_h [N], errors_u [N])."""
    ds     = SWEDataset1D(test)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    eh, eu = [], []
    for x, y in loader:
        pred = y_norm.decode(model(x_norm.encode(x)))
        for i in range(x.shape[0]):
            eh.append((pred[i,0] - y[i,0]).norm() / (y[i,0].norm() + 1e-8))
            eu.append((pred[i,1] - y[i,1]).norm() / (y[i,1].norm() + 1e-8))
    return np.array([e.item() for e in eh]), np.array([e.item() for e in eu])


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
    ax1.set_title("Training curves — 1D SWE FNO")
    fig.tight_layout()
    path = os.path.join(plot_dir, "training_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def plot_prediction_comparison(truth, pred, cfg, plot_dir, n: int = 6):
    """Each row = one test sample; left col = h, right col = u (truth vs pred overlaid)."""
    n  = min(n, truth.shape[0])
    nx = truth.shape[-1]
    x  = np.linspace(0, cfg.Lx, nx, endpoint=False)

    fig, axes = plt.subplots(n, 2, figsize=(12, 2.0 * n), squeeze=False)
    ch_labels = ["h", "u"]
    for i in range(n):
        for c, name in enumerate(ch_labels):
            ax = axes[i, c]
            ax.plot(x, truth[i, c], lw=1.2, label="truth")
            ax.plot(x, pred[i, c],  lw=1.0, ls="--", alpha=0.85, label="FNO")
            ax.set_ylabel(name)
            ax.set_xlabel("x")
            if i == 0:
                ax.set_title(f"{name} — truth vs FNO prediction")
                ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("1D SWE — Ground truth vs FNO prediction (test samples)", y=1.01)
    fig.tight_layout()
    path = os.path.join(plot_dir, "prediction_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def plot_error_map(truth, pred, plot_dir, n: int = 6):
    """Absolute pointwise error |pred − truth| for h and u."""
    n   = min(n, truth.shape[0])
    err = np.abs(pred - truth)   # [n, 2, nx]

    fig, axes = plt.subplots(n, 2, figsize=(12, 1.8 * n), squeeze=False)
    for i in range(n):
        for c, name in enumerate(["h", "u"]):
            ax = axes[i, c]
            ax.plot(err[i, c], color="C3", lw=0.9)
            ax.set_ylabel("|err|")
            ax.set_xlabel("grid point")
            if i == 0:
                ax.set_title(f"|error| — {name}")

    fig.suptitle("1D SWE — Absolute pointwise error (test samples)", y=1.01)
    fig.tight_layout()
    path = os.path.join(plot_dir, "error_map.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def plot_error_histogram(errors_h, errors_u, plot_dir):
    lo = min(errors_h.min(), errors_u.min()) * 0.5
    hi = max(errors_h.max(), errors_u.max()) * 2.0
    lo = max(lo, 1e-6)
    bins = np.logspace(np.log10(lo), np.log10(hi), 40)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(errors_h, bins=bins, alpha=0.6,
            label=f"h  (med={np.median(errors_h):.2e}, std={errors_h.std():.2e})")
    ax.hist(errors_u, bins=bins, alpha=0.6,
            label=f"u  (med={np.median(errors_u):.2e}, std={errors_u.std():.2e})")
    ax.set_xscale("log")
    ax.set_xlabel("Per-sample relative L² error")
    ax.set_ylabel("Count")
    ax.set_title("1D SWE FNO — Test error distribution")
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
    parser = argparse.ArgumentParser(description="Visualize 1D SWE FNO results")
    parser.add_argument("--run-dir", default="runs/swe1d",
                        help="Directory containing fno1d_best.pt and data.npz")
    args = parser.parse_args()

    run_dir  = args.run_dir
    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    print(f"[viz1d] Loading run from {run_dir} ...")
    cfg, model, x_norm, y_norm, test, history, ckpt = load_run(run_dir)

    if history is not None:
        plot_training_curves(history, plot_dir, best_epoch=ckpt.get("epoch"))
    else:
        print("  history.json not found — skipping training curves")

    print("[viz1d] Running inference on test samples ...")
    truth, pred = get_predictions(model, x_norm, y_norm, test, n=6)
    plot_prediction_comparison(truth, pred, cfg, plot_dir)
    plot_error_map(truth, pred, plot_dir)

    print("[viz1d] Computing per-sample errors across full test set ...")
    errors_h, errors_u = compute_per_sample_errors(model, x_norm, y_norm, test)
    plot_error_histogram(errors_h, errors_u, plot_dir)

    print(f"[viz1d] Done. Plots written to {plot_dir}/")


if __name__ == "__main__":
    main()
