"""Training / evaluation loop. Dimension-agnostic (1D or 2D)."""
from __future__ import annotations
import os
import time
from typing import Callable

import torch
from torch.utils.data import DataLoader

from utils import LpLoss, ChannelNormalizer, all_tensors_from_dataset


def fit_normalizers(train_ds):
    Xtr, Ytr = all_tensors_from_dataset(train_ds)
    return ChannelNormalizer(Xtr), ChannelNormalizer(Ytr)


def _run_epoch(model, loader, x_norm, y_norm, loss_fn, device,
               optimizer=None) -> float:
    train = optimizer is not None
    model.train(train)
    total = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_in = x_norm.encode(x)
        y_tg = y_norm.encode(y)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            pred = model(x_in)
            loss = loss_fn(pred, y_tg)
            if train:
                loss.backward()
                optimizer.step()

        total += loss.item() * x.shape[0]
        n += x.shape[0]
    return total / max(n, 1)


def train_model(
    model: torch.nn.Module,
    train_ds,
    val_ds,
    cfg,
    save_name: str = "model.pt",
    log_fn: Callable[[str], None] = print,
):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    log_fn(f"[train] device = {device}")
    model = model.to(device)

    x_norm, y_norm = fit_normalizers(train_ds)
    x_norm.to(device); y_norm.to(device)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size,
                            shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.sched_step, gamma=cfg.sched_gamma
    )
    loss_fn = LpLoss(p=2)

    os.makedirs(cfg.out_dir, exist_ok=True)
    best_val = float("inf")
    history = {"train": [], "val": [], "lr": []}

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr = _run_epoch(model, train_loader, x_norm, y_norm,
                        loss_fn, device, optimizer)
        va = _run_epoch(model, val_loader,   x_norm, y_norm,
                        loss_fn, device, None)
        scheduler.step()

        history["train"].append(tr)
        history["val"].append(va)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        dt = time.time() - t0
        log_fn(f"epoch {epoch:3d}/{cfg.epochs} | "
               f"train {tr:.4e} | val {va:.4e} | "
               f"lr {optimizer.param_groups[0]['lr']:.2e} | {dt:.1f}s")

        if va < best_val:
            best_val = va
            torch.save({
                "model": model.state_dict(),
                "x_norm_mean": x_norm.mean.cpu(),
                "x_norm_std":  x_norm.std.cpu(),
                "y_norm_mean": y_norm.mean.cpu(),
                "y_norm_std":  y_norm.std.cpu(),
                "epoch": epoch,
                "val_loss": va,
            }, os.path.join(cfg.out_dir, save_name))

    log_fn(f"[train] best val loss = {best_val:.4e}")
    return history, x_norm, y_norm


@torch.no_grad()
def evaluate(model, dataset, x_norm, y_norm, cfg) -> float:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)
    loss_fn = LpLoss(p=2)
    return _run_epoch(model, loader, x_norm, y_norm, loss_fn, device, None)
