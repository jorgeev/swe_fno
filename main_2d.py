"""Entry point for the 2D shallow-water FNO experiment (with f-plane Coriolis).

Run:
    python main_2d.py
"""
from __future__ import annotations
import json
import os

import numpy as np
import torch

from config import Config2D
from cache import load_or_generate_2d
from dataset import SWEDataset2D
from fno import FNO2d, count_params
from train import train_model, evaluate


def main():
    cfg = Config2D()
    os.makedirs(cfg.out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_data, val_data, test_data = load_or_generate_2d(cfg)

    train_ds = SWEDataset2D(train_data)
    val_ds   = SWEDataset2D(val_data)
    test_ds  = SWEDataset2D(test_data)

    model = FNO2d(in_channels=cfg.in_channels,
                  out_channels=cfg.out_channels,
                  modes_x=cfg.modes_x,
                  modes_y=cfg.modes_y,
                  width=cfg.width,
                  n_layers=cfg.n_layers)
    print(f"[2D] FNO2d parameters: {count_params(model):,}")

    history, x_norm, y_norm = train_model(model, train_ds, val_ds, cfg,
                                          save_name="fno2d_best.pt")

    with open(os.path.join(cfg.out_dir, "history.json"), "w") as f:
        json.dump(history, f)

    test_loss = evaluate(model, test_ds, x_norm, y_norm, cfg)
    print(f"[2D] test relative L2 = {test_loss:.4e}")


if __name__ == "__main__":
    main()
