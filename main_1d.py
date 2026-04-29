"""Entry point for the 1D shallow-water FNO experiment.

Run:
    python main_1d.py
"""
from __future__ import annotations
import os
import numpy as np
import torch

from config import Config1D
from solver_1d import generate_dataset_1d
from dataset import SWEDataset1D
from fno import FNO1d, count_params
from train import train_model, evaluate


def main():
    cfg = Config1D()
    os.makedirs(cfg.out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print("[1D] Generating training set ...")
    train_data = generate_dataset_1d(cfg, cfg.n_train, base_seed=cfg.seed + 1)
    print("[1D] Generating validation set ...")
    val_data   = generate_dataset_1d(cfg, cfg.n_val,   base_seed=cfg.seed + 2)
    print("[1D] Generating test set ...")
    test_data  = generate_dataset_1d(cfg, cfg.n_test,  base_seed=cfg.seed + 3)

    np.savez_compressed(os.path.join(cfg.out_dir, "data.npz"),
                        **{f"train_{k}": v for k, v in train_data.items()},
                        **{f"val_{k}": v   for k, v in val_data.items()},
                        **{f"test_{k}": v  for k, v in test_data.items()})

    train_ds = SWEDataset1D(train_data)
    val_ds   = SWEDataset1D(val_data)
    test_ds  = SWEDataset1D(test_data)

    model = FNO1d(in_channels=cfg.in_channels,
                  out_channels=cfg.out_channels,
                  modes=cfg.modes,
                  width=cfg.width,
                  n_layers=cfg.n_layers)
    print(f"[1D] FNO1d parameters: {count_params(model):,}")

    _, x_norm, y_norm = train_model(model, train_ds, val_ds, cfg,
                                    save_name="fno1d_best.pt")

    test_loss = evaluate(model, test_ds, x_norm, y_norm, cfg)
    print(f"[1D] test relative L2 = {test_loss:.4e}")


if __name__ == "__main__":
    main()
