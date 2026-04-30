"""Entry point for the 1D shallow-water FNO experiment.

Run:
    python main_1d.py
"""
from __future__ import annotations
import json
import os

import numpy as np
import torch

from config import Config1D
from cache import load_or_generate_1d
from dataset import SWEDataset1D
from fno import FNO1d, count_params
from train import train_model, evaluate


def main():
    cfg = Config1D()
    os.makedirs(cfg.out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_data, val_data, test_data = load_or_generate_1d(cfg)

    train_ds = SWEDataset1D(train_data)
    val_ds   = SWEDataset1D(val_data)
    test_ds  = SWEDataset1D(test_data)

    model = FNO1d(in_channels=cfg.in_channels,
                  out_channels=cfg.out_channels,
                  modes=cfg.modes,
                  width=cfg.width,
                  n_layers=cfg.n_layers)
    print(f"[1D] FNO1d parameters: {count_params(model):,}")

    history, x_norm, y_norm = train_model(model, train_ds, val_ds, cfg,
                                          save_name="fno1d_best.pt")

    with open(os.path.join(cfg.out_dir, "history.json"), "w") as f:
        json.dump(history, f)

    test_loss = evaluate(model, test_ds, x_norm, y_norm, cfg)
    print(f"[1D] test relative L2 = {test_loss:.4e}")


if __name__ == "__main__":
    main()
