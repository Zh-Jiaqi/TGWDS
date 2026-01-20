import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

def mean_pool2d_numpy(x, scale=2):
    N, C, H, W = x.shape
    if H % scale == 0 and W % scale == 0:
        return x.reshape(N, C, H//scale, scale, W//scale, scale).mean(axis=(3,5))
    # fallback
    out = np.empty((N, C, H//scale, W//scale), dtype=np.float32)
    for n in range(N):
        for c in range(C):
            out[n, c] = cv2.resize(x[n, c], (W//scale, H//scale), interpolation=cv2.INTER_AREA)
    return out

def standardize(arr, train=True, mean=None, std=None, eps=1e-6):
    if train:
        m = arr.mean(axis=(0,2,3), keepdims=False)
        s = arr.std(axis=(0,2,3), keepdims=False)
        s = np.maximum(s, eps)
        norm = (arr - m.reshape(1,-1,1,1)) / s.reshape(1,-1,1,1)
        return norm, m.astype(np.float32), s.astype(np.float32)
    else:
        s = np.maximum(std, eps)
        return (arr - mean.reshape(1,-1,1,1)) / s.reshape(1,-1,1,1)

class FixedWindTerrainDataset(Dataset):
    """
    wind_path: .npy，内部是 dict，包含键 'u100','v100'，每个形状 (N,64,64)
    ele_path : .npy，普通数组，形状 (N,1,64,64) 或 (1,64,64)
    """
    def __init__(self, wind_path, ele_path,
                 train=True, scale=2, stats=None, save_stats_path=None):
        super().__init__()
        # 加载风速 dict
        
#         scale=1
        
        wind_dict = np.load(wind_path, allow_pickle=True).item()
        u = wind_dict['u100']  # (N,H,W)
        v = wind_dict['v100']
        N, H, W = u.shape
#         if N>10000: 
#             u = u[:8760]
#             v = v[:8760]
        if N>10000: 
            u = u[-8760:]
            v = v[-8760:]
            
        wind_hr = np.stack([u, v], axis=1).astype(np.float32)  # (N,2,H,W)
        
         ####验证跨倍率泛化 
        #wind_hr = mean_pool2d_numpy(wind_hr, scale=2)

      
        ele_hr = np.load(ele_path, allow_pickle=True).astype(np.float32)

        if ele_hr.ndim == 3:  # (N,H,W)
            ele_hr = ele_hr[:,None,:,:]
        elif ele_hr.ndim == 2:  # (H,W)
            ele_hr = ele_hr[None,None,:,:]
            

        if ele_hr.shape[0] == 1 and wind_hr.shape[0] > 1:
            ele_hr = np.repeat(ele_hr, wind_hr.shape[0], axis=0)
        
        ####验证跨倍率泛化    
        #ele_hr = mean_pool2d_numpy(ele_hr, scale=2)

        self.N = wind_hr.shape[0]
        self.scale = scale
        self.train = train

        # 下采样
        wind_lr = mean_pool2d_numpy(wind_hr, scale=scale)  # (N,2,32,32)
        ele_lr  = mean_pool2d_numpy(ele_hr,  scale=scale)  # (N,1,32,32)

        if train:
            # 风：用 LR 的统计量
            wind_lr_norm, wind_mean, wind_std = standardize(wind_lr, train=True)
            wind_hr_norm = standardize(wind_hr, train=False, mean=wind_mean, std=wind_std)


            ele_lr_norm = ele_lr
            ele_hr_norm = ele_hr

            self.stats = {
                "wind_mean":   wind_mean,
                "wind_std":    wind_std
            }
            
            if save_stats_path is not None:
                np.save(save_stats_path, self.stats, allow_pickle=True)
        else:
            assert stats is not None, "test 模式需要提供训练保存的 stats"
            wm, ws = np.array(stats["wind_mean"]), np.array(stats["wind_std"])

            wind_lr_norm = standardize(wind_lr, train=False, mean=wm, std=ws)
            wind_hr_norm = standardize(wind_hr, train=False, mean=wm, std=ws)
            ele_lr_norm  = ele_lr
            ele_hr_norm  = ele_hr

            self.stats = stats

        self.wind_lr = wind_lr_norm.astype(np.float32)
        self.wind_hr = wind_hr_norm.astype(np.float32)
        self.ele_lr  = ele_lr_norm.astype(np.float32)
        self.ele_hr  = ele_hr_norm.astype(np.float32)

    def GetDataShape(self):
        return {
            'wind_LR': self.wind_lr.shape,  # (N,2,32,32)
            'wind_HR': self.wind_hr.shape,  # (N,2,64,64)
            'ele_LR':  self.ele_lr.shape,   # (N,1,32,32)
            'ele_HR':  self.ele_hr.shape    # (N,1,64,64)
        }

    def __len__(self): return self.N

    def __getitem__(self, idx):
        wl = torch.from_numpy(self.wind_lr[idx])  # (2,32,32)
        wh = torch.from_numpy(self.wind_hr[idx])  # (2,64,64)
        gl = torch.from_numpy(self.ele_lr[idx])   # (1,32,32)
        gh = torch.from_numpy(self.ele_hr[idx])   # (1,64,64)
        return wl, wh, gl, gh


import torch
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # ====== 构造示例数据 ======
    # 假设你已经有 wind_dict.npy (dict: {'u100': (N,64,64), 'v100': (N,64,64)})
    # 和 ele.npy (N,1,64,64)，这里我用随机数生成再保存一份
    N = 20
    wind_dict = {
        'u100': np.random.randn(N, 32, 32).astype(np.float32)*5,
        'v100': np.random.randn(N, 32, 32).astype(np.float32)*5,
    }
    np.save("wind_dict.npy", wind_dict)  # 注意: 保存的是 dict
    ele = np.random.rand(N, 1, 32, 32).astype(np.float32) * 1000  # 海拔0~1000m
    np.save("ele.npy", ele)

    # ====== 构造训练集 ======
    train_set = FixedWindTerrainDataset("wind_dict.npy", "ele.npy",
                                        train=True, save_stats_path="stats.npy")
    print("Train dataset size:", len(train_set))

    # 用 DataLoader 包装，batch_size=4
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0)

    # 迭代一个 batch
    for batch in train_loader:
        wl, wh, gl, gh = batch  # torch.Tensor
        print("Batch shapes:")
        print("wind LR:", wl.shape)  # (B,2,32,32)
        print("wind HR:", wh.shape)  # (B,2,64,64)
        print("ele  LR:", gl.shape)  # (B,1,32,32)
        print("ele  HR:", gh.shape)  # (B,1,64,64)
        break

    # ====== 构造测试集 ======
    stats = np.load("stats.npy", allow_pickle=True).item()
    test_set = FixedWindTerrainDataset("wind_dict.npy", "ele.npy",
                                       train=False, stats=stats)
    test_loader = DataLoader(test_set, batch_size=2, shuffle=False)

    for batch in test_loader:
        wl, wh, gl, gh = batch
        print("\nTest batch shapes:")
        print("wind LR:", wl.shape)
        print("wind HR:", wh.shape)
        print("ele  LR:", gl.shape)
        print("ele  HR:", gh.shape)
        break
