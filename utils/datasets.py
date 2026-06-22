import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

def mean_pool2d_numpy(x, scale=2):
    N, C, H, W = x.shape
    if H % scale == 0 and W % scale == 0:
        return x.reshape(N, C, H // scale, scale, W // scale, scale).mean(axis=(3, 5))
    out = np.empty((N, C, H // scale, W // scale), dtype=np.float32)
    for n in range(N):
        for c in range(C):
            out[n, c] = cv2.resize(
                x[n, c],
                (W // scale, H // scale),
                interpolation=cv2.INTER_AREA
            )
    return out

def standardize(arr, train=True, mean=None, std=None, eps=1e-6):
    if train:
        m = arr.mean(axis=(0, 2, 3), keepdims=False)
        s = arr.std(axis=(0, 2, 3), keepdims=False)
        s = np.maximum(s, eps)
        norm = (arr - m.reshape(1, -1, 1, 1)) / s.reshape(1, -1, 1, 1)
        return norm, m.astype(np.float32), s.astype(np.float32)
    else:
        s = np.maximum(std, eps)
        return (arr - mean.reshape(1, -1, 1, 1)) / s.reshape(1, -1, 1, 1)

class FixedWindTerrainDataset(Dataset):
    """
    wind_path: .npy，内部是 dict，包含键 'u100','v100'，每个形状 (N,H,W)
    ele_path : .npy，普通数组，形状 (N,1,H,W) / (1,H,W) / (H,W)
    """

    def __init__(self, wind_path, ele_path,
                 train=True, scale=2, save_stats_path=None):
        super().__init__()

        wind_dict = np.load(wind_path, allow_pickle=True).item()
        u = wind_dict['u100']
        v = wind_dict['v100']

        N, H, W = u.shape
        wind_hr = np.stack([u, v], axis=1).astype(np.float32)  # (N,2,H,W)

        ele_hr = np.load(ele_path, allow_pickle=True).astype(np.float32)

        if ele_hr.ndim == 2:
            ele_hr = ele_hr[None, None, :, :]
        elif ele_hr.ndim == 3:
            ele_hr = ele_hr[:, None, :, :]
        elif ele_hr.ndim == 4:
            pass
        else:
            raise ValueError(f"Unsupported ele_hr shape: {ele_hr.shape}")

        if ele_hr.shape[0] == 1 and N > 1:
            ele_hr = np.repeat(ele_hr, N, axis=0)

        if ele_hr.shape[0] != N:
            raise ValueError(
                f"Sample number mismatch: wind_hr has {N}, but ele_hr has {ele_hr.shape[0]}"
            )

        self.N = N
        self.scale = scale
        self.train = train

        wind_lr = mean_pool2d_numpy(wind_hr, scale=scale)
        ele_lr = mean_pool2d_numpy(ele_hr, scale=scale)

        if train:
            wind_lr_norm, wind_mean, wind_std = standardize(wind_lr, train=True)
            wind_hr_norm = standardize(wind_hr, train=False, mean=wind_mean, std=wind_std)

            ele_lr_norm = ele_lr
            ele_hr_norm = ele_hr

            self.stats = {
                "wind_mean": wind_mean.astype(np.float32),
                "wind_std": wind_std.astype(np.float32)
            }

            if save_stats_path is not None:
                np.save(save_stats_path, self.stats, allow_pickle=True)

        else:
            if save_stats_path is None or (not os.path.exists(save_stats_path)):
                raise ValueError("val/test 模式下必须提供有效的 save_stats_path")

            stats = np.load(save_stats_path, allow_pickle=True).item()
            wm = np.array(stats["wind_mean"], dtype=np.float32)
            ws = np.array(stats["wind_std"], dtype=np.float32)

            wind_lr_norm = standardize(wind_lr, train=False, mean=wm, std=ws)
            wind_hr_norm = standardize(wind_hr, train=False, mean=wm, std=ws)

            ele_lr_norm = ele_lr
            ele_hr_norm = ele_hr

            self.stats = stats

        self.wind_lr = wind_lr_norm.astype(np.float32)
        self.wind_hr = wind_hr_norm.astype(np.float32)
        self.ele_lr = ele_lr_norm.astype(np.float32)
        self.ele_hr = ele_hr_norm.astype(np.float32)

    def GetDataShape(self):
        return {
            'wind_LR': self.wind_lr.shape,
            'wind_HR': self.wind_hr.shape,
            'ele_LR': self.ele_lr.shape,
            'ele_HR': self.ele_hr.shape
        }

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        wl = torch.from_numpy(self.wind_lr[idx])
        wh = torch.from_numpy(self.wind_hr[idx])
        gl = torch.from_numpy(self.ele_lr[idx])
        gh = torch.from_numpy(self.ele_hr[idx])
        return wl, wh, gl, gh
