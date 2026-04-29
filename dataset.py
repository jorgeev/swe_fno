"""PyTorch Dataset wrappers around the simulator output."""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset


class SWEDataset1D(Dataset):
    """Inputs:  channels (h0, u0, b)            shape [3, nx]
       Outputs: channels (h(T), u(T))           shape [2, nx]
    """

    def __init__(self, data: dict):
        self.h0 = torch.from_numpy(np.asarray(data["h0"], dtype=np.float32))
        self.u0 = torch.from_numpy(np.asarray(data["u0"], dtype=np.float32))
        self.b  = torch.from_numpy(np.asarray(data["b"],  dtype=np.float32))
        self.h  = torch.from_numpy(np.asarray(data["h"],  dtype=np.float32))
        self.u  = torch.from_numpy(np.asarray(data["u"],  dtype=np.float32))
        assert self.h0.shape == self.u0.shape == self.b.shape == self.h.shape == self.u.shape

    def __len__(self) -> int:
        return self.h0.shape[0]

    def __getitem__(self, idx: int):
        x = torch.stack([self.h0[idx], self.u0[idx], self.b[idx]], dim=0)   # [3, nx]
        y = torch.stack([self.h[idx],  self.u[idx]],               dim=0)   # [2, nx]
        return x, y


class SWEDataset2D(Dataset):
    """Inputs:  channels (h0, u0, v0, b, f)     shape [5, ny, nx]
       Outputs: channels (h(T), u(T), v(T))    shape [3, ny, nx]

    The Coriolis parameter f is constant per sample; we broadcast it to a full field.
    """

    def __init__(self, data: dict):
        self.h0 = torch.from_numpy(np.asarray(data["h0"], dtype=np.float32))
        self.u0 = torch.from_numpy(np.asarray(data["u0"], dtype=np.float32))
        self.v0 = torch.from_numpy(np.asarray(data["v0"], dtype=np.float32))
        self.b  = torch.from_numpy(np.asarray(data["b"],  dtype=np.float32))
        self.f  = torch.from_numpy(np.asarray(data["f"],  dtype=np.float32))   # [N]
        self.h  = torch.from_numpy(np.asarray(data["h"],  dtype=np.float32))
        self.u  = torch.from_numpy(np.asarray(data["u"],  dtype=np.float32))
        self.v  = torch.from_numpy(np.asarray(data["v"],  dtype=np.float32))

    def __len__(self) -> int:
        return self.h0.shape[0]

    def __getitem__(self, idx: int):
        ny, nx = self.h0.shape[1], self.h0.shape[2]
        f_field = torch.full((ny, nx), float(self.f[idx]))
        x = torch.stack([self.h0[idx], self.u0[idx], self.v0[idx],
                         self.b[idx], f_field], dim=0)            # [5, ny, nx]
        y = torch.stack([self.h[idx], self.u[idx], self.v[idx]], dim=0)  # [3, ny, nx]
        return x, y
