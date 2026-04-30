"""Hash-based dataset caching for 1D and 2D SWE experiments.

On first run, data is generated and saved alongside a hash of the config
fields that affect generation.  Subsequent runs with the same config load
from disk instead of re-running the solver.  Change any data-relevant
config field and the hash mismatches, triggering regeneration automatically.
"""
from __future__ import annotations
import hashlib
import json
import os

import numpy as np

from solver_1d import generate_dataset_1d
from solver_2d import generate_dataset_2d

_1D_HASH_KEYS = [
    "Lx", "nx", "g", "H0", "bottom_amp", "eta_amp", "u_amp",
    "n_modes_ic", "T", "cfl", "visc", "n_train", "n_val", "n_test", "seed",
]
_2D_HASH_KEYS = _1D_HASH_KEYS + ["Ly", "ny", "f_min", "f_max"]


def _cfg_hash(cfg, keys: list[str]) -> str:
    d = {k: getattr(cfg, k) for k in keys}
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def _split_npz(npz, prefix: str, keys: list[str]) -> dict:
    return {k: npz[f"{prefix}_{k}"] for k in keys}


def _cache_valid(npz_path: str, hash_path: str, current_hash: str) -> bool:
    if not (os.path.exists(npz_path) and os.path.exists(hash_path)):
        return False
    with open(hash_path) as f:
        stored = json.load(f)
    return stored.get("hash") == current_hash


def load_or_generate_1d(cfg):
    """Return ``(train_data, val_data, test_data)`` dicts, using on-disk cache when config matches."""
    os.makedirs(cfg.out_dir, exist_ok=True)
    npz_path  = os.path.join(cfg.out_dir, "data.npz")
    hash_path = os.path.join(cfg.out_dir, "data_hash.json")
    current_hash = _cfg_hash(cfg, _1D_HASH_KEYS)

    if _cache_valid(npz_path, hash_path, current_hash):
        print("[1D] Cache hit — loading data from data.npz")
        npz  = np.load(npz_path)
        keys = ["h0", "u0", "b", "h", "u"]
        return (
            _split_npz(npz, "train", keys),
            _split_npz(npz, "val",   keys),
            _split_npz(npz, "test",  keys),
        )

    print("[1D] Cache miss — regenerating dataset")
    print("[1D] Generating training set ...")
    train_data = generate_dataset_1d(cfg, cfg.n_train, base_seed=cfg.seed + 1)
    print("[1D] Generating validation set ...")
    val_data   = generate_dataset_1d(cfg, cfg.n_val,   base_seed=cfg.seed + 2)
    print("[1D] Generating test set ...")
    test_data  = generate_dataset_1d(cfg, cfg.n_test,  base_seed=cfg.seed + 3)

    np.savez_compressed(
        npz_path,
        **{f"train_{k}": v for k, v in train_data.items()},
        **{f"val_{k}":   v for k, v in val_data.items()},
        **{f"test_{k}":  v for k, v in test_data.items()},
    )
    with open(hash_path, "w") as f:
        json.dump({"hash": current_hash}, f)
    print(f"[1D] Dataset saved and hash written ({current_hash})")

    return train_data, val_data, test_data


def load_or_generate_2d(cfg):
    """Return ``(train_data, val_data, test_data)`` dicts, using on-disk cache when config matches."""
    os.makedirs(cfg.out_dir, exist_ok=True)
    npz_path  = os.path.join(cfg.out_dir, "data.npz")
    hash_path = os.path.join(cfg.out_dir, "data_hash.json")
    current_hash = _cfg_hash(cfg, _2D_HASH_KEYS)

    if _cache_valid(npz_path, hash_path, current_hash):
        print("[2D] Cache hit — loading data from data.npz")
        npz          = np.load(npz_path)
        spatial_keys = ["h0", "u0", "v0", "b", "h", "u", "v"]
        return (
            {**_split_npz(npz, "train", spatial_keys), "f": npz["train_f"]},
            {**_split_npz(npz, "val",   spatial_keys), "f": npz["val_f"]},
            {**_split_npz(npz, "test",  spatial_keys), "f": npz["test_f"]},
        )

    print("[2D] Cache miss — regenerating dataset")
    print("[2D] Generating training set ...")
    train_data = generate_dataset_2d(cfg, cfg.n_train, base_seed=cfg.seed + 1)
    print("[2D] Generating validation set ...")
    val_data   = generate_dataset_2d(cfg, cfg.n_val,   base_seed=cfg.seed + 2)
    print("[2D] Generating test set ...")
    test_data  = generate_dataset_2d(cfg, cfg.n_test,  base_seed=cfg.seed + 3)

    np.savez_compressed(
        npz_path,
        **{f"train_{k}": v for k, v in train_data.items()},
        **{f"val_{k}":   v for k, v in val_data.items()},
        **{f"test_{k}":  v for k, v in test_data.items()},
    )
    with open(hash_path, "w") as f:
        json.dump({"hash": current_hash}, f)
    print(f"[2D] Dataset saved and hash written ({current_hash})")

    return train_data, val_data, test_data
