# -*- coding: utf-8 -*-
import os
import math
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from config import configs
from utils.log import printwrite
from data.datasets import FixedWindTerrainDataset
from models.WSR_model import WindSR_Terrain
from pytorch_wavelets import DWTForward




class WarmupCosine:
    def __init__(self, optimizer, warmup_steps, total_steps, base_lr, min_lr=1e-6):
        self.opt = optimizer
        self.warmup = max(1, warmup_steps)
        self.total = max(self.warmup + 1, total_steps)
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.step_id = 0

    def step(self):
        self.step_id += 1
        if self.step_id <= self.warmup:
            lr = self.base_lr * self.step_id / self.warmup
        else:
            t = (self.step_id - self.warmup) / (self.total - self.warmup)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * t))
        for g in self.opt.param_groups:
            g['lr'] = lr
        return lr


# ============================= Trainer =============================
class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device

        # 模型
        self.network = WindSR_Terrain(
            upscale=cfg.scale, dim=64,
            group_depth=6, dem_every=2, sample_ids=(1, 3, 5),
            dem_ch=16, psa_kernels=(1, 3, 5), psa_groups=(1, 1, 1),
            enc_res_scale=0.1, decoder_feats=None,
            sft_hidden=64, cross_blocks_per_stage=4,
            stage_res_scale=0.2, dropout_max=0.1
        ).to(self.device)

        self.opt = torch.optim.AdamW(self.network.parameters(),
                                     lr=cfg.lr, weight_decay=1e-4, betas=(0.9, 0.999))
        
        steps_per_epoch = getattr(cfg, "steps_per_epoch_est", 270)
        self.total_steps = cfg.epochs * steps_per_epoch
        self.sched = WarmupCosine(self.opt,
                                  warmup_steps=int(0.05 * self.total_steps),
                                  total_steps=self.total_steps,
                                  base_lr=cfg.lr, min_lr=1e-6)
        
        self.early_stop = cfg.patience

        self.scaler = GradScaler()
        

        k1d = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32)
        k2d = (k1d[:, None] @ k1d[None, :]) / 256.0  # 归一化
        self.blur5 = k2d.view(1, 1, 5, 5).to(self.device)  # (1,1,5,5)


    def _lp_smooth(self, x: torch.Tensor) -> torch.Tensor:
        # depthwise conv
        w = self.blur5.repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, w, padding=2, groups=x.size(1))

    def split_low_high(self, x: torch.Tensor):
        low = self._lp_smooth(x)
        high = x - low
        return low, high

    def terrain_mask_inverse(self, dem, eps=1e-3):
        # dem: (B,1,H,W)
        dx = self.gradient_x(dem)
        dy = self.gradient_y(dem)
        grad = torch.sqrt(dx**2 + dy**2)  # 坡度大小
        grad_norm = grad / (grad.max() + eps)
        mask = 1.0 - grad_norm  # 平坦区=1，陡峭区=0
        return mask
    
    def gradient_x(self, f):
        kernel = torch.tensor([[-0.5, 0, 0.5]], dtype=f.dtype, device=f.device).view(1,1,1,3)
        return F.conv2d(f, kernel, padding=(0,1))

    def gradient_y(self, f):
        kernel = torch.tensor([[-0.5],[0],[0.5]], dtype=f.dtype, device=f.device).view(1,1,3,1)
        return F.conv2d(f, kernel, padding=(1,0))

    def divergence(self, u, v):
        return self.gradient_x(u) + self.gradient_y(v)

    def vorticity(self, u, v):
        return self.gradient_x(v) - self.gradient_y(u)

    # -------------------------------------------
    def combine_loss(self, y_pred, y_true, gh, epoch: int,
                 ramp_T: int = 20,
                 phys_start: int = 10, phys_ramp: int = 10,
                 w_phys_max: float = 0.1,
                 base_w_low: float = 1.0, base_w_high: float = 1.0,
                 ema_m: float = 0.9, gamma: float = 1.0,
                 w_high_min: float = 0.3, w_high_max: float = 1.2):

        # 1) L1
        L1_full = F.l1_loss(y_pred, y_true)

        # 2) 频率分解
        pred_low, pred_high = self.split_low_high(y_pred)
        true_low, true_high = self.split_low_high(y_true)
        Llow  = F.l1_loss(pred_low, true_low)

        scale = true_high.abs().mean(dim=(1,2,3), keepdim=True).clamp_min(1e-3)
        Lhigh = F.l1_loss(pred_high/scale, true_high/scale)

        # 3) ramp（主 loss 调度）
        t   = min(1.0, float(epoch) / max(1, ramp_T))
        lam = min(0.5, 0.5 * t)
        w_low_sched  = 1.0
        w_high_sched = 1.0 - 0.9 * t
        
        loss_hf = w_low_sched * Llow + w_high_sched * Lhigh
        

        # 5) 物理项（散度 + 涡度，带地形权重 mask）
        u_pred, v_pred = y_pred[:,0:1], y_pred[:,1:2]
        u_true, v_true = y_true[:,0:1], y_true[:,1:2]

        div_pred = self.divergence(u_pred, v_pred)
        div_true = self.divergence(u_true, v_true)
        vor_pred = self.vorticity(u_pred, v_pred)
        vor_true = self.vorticity(u_true, v_true)

        mask = self.terrain_mask_inverse(gh)  # 平坦区=1，陡峭区=0
        Ldiv = ((div_pred - div_true).abs() * mask).mean()
        Lvor = ((vor_pred - vor_true).abs() * mask).mean()
        Phys = Ldiv + Lvor

        # 物理权重随 epoch 调整
        if epoch < phys_start:
            w_phys = 0.0
        elif epoch < phys_start + phys_ramp:
            w_phys = w_phys_max * (epoch - phys_start) / phys_ramp
        else:
            w_phys = w_phys_max

        # 6) 总损失
        loss = lam * L1_full + (1 - lam) * (loss_hf + w_phys * Phys)

        info = {
            "L1": float(L1_full.detach().cpu()),
            "Llow": w_low_sched * Llow, "Lhigh": w_high_sched * Lhigh,
            "lam": lam, "w_low": w_low_sched * Llow, "w_high": w_high_sched * Lhigh,
            "Phys": float(Phys.detach().cpu()),
            "w_phys": w_phys
        }
        return loss, info
    
    # -------------------- 验证损失 L1 --------------------
    @staticmethod
    def test_loss(y_pred, y_true):
        return F.l1_loss(y_pred, y_true)

    # -------------------- 单步训练 --------------------
    def train_once(self, lr, hr, gl, gh, epoch):
        lr = lr.float().to(self.device)  # (B,2,H/scale,W/scale)
        hr = hr.float().to(self.device)  # (B,2,H,W)
        gl = gl.float().to(self.device)  # (B,1,H/scale,W/scale)
        gh = gh.float().to(self.device)  # (B,1,H,W)

        pred = self.network(lr, gl, gh)

        loss, info = self.combine_loss(pred, hr, gh, epoch=epoch, ramp_T=10)
        return loss, info

    # -------------------- 验证 --------------------
    def test(self, dataset_eval, dataloader_eval):
        self.network.eval()
        total_loss, n = 0.0, 0
        with torch.no_grad():
            for lr, hr, gl, gh in dataloader_eval:
                lr = lr.float().to(self.device)
                hr = hr.float().to(self.device)
                gl = gl.float().to(self.device)
                gh = gh.float().to(self.device)

                pred = self.network(lr, gl, gh)
                loss = self.test_loss(pred, hr)
                total_loss += float(loss.detach().cpu())
                n += 1
        return total_loss / max(1, n)

    # -------------------- 训练主循环 --------------------
    def train(self, dataset_train, dataset_eval, chk_dir):

        log_file = os.path.join(chk_dir, "log.txt")
        best = math.inf
        count = 0

        printwrite(log_file, 'loading train dataloader')
        dl_train = DataLoader(dataset_train, batch_size=self.cfg.batch_size, shuffle=True, drop_last=False)
        printwrite(log_file, 'loading eval dataloader')
        dl_eval = DataLoader(dataset_eval, batch_size=self.cfg.batch_size_test, shuffle=False, drop_last=False)

        self.network.train()
        step_global = 0

        for epoch in range(1, self.cfg.epochs + 1):
            printwrite(log_file, f'\nepoch: {epoch}')
            self.opt.zero_grad()

            for it, (lr, hr, gl, gh) in enumerate(dl_train, start=1):
                self.network.train()
                loss, info = self.train_once(lr, hr, gl, gh, epoch=epoch)

                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
                lr_now = self.sched.step()

                step_global += 1
                if (it % self.cfg.display_interval) == 0:
                    printwrite(log_file,
                               'batch training loss: {:.4f}, Llow={:.4f}, Lhigh={:.4f}, '
                               'lam={:.2f}, w_low={:.2f}, w_high={:.2f}, Phys={:.3f}, w_phys={:.3f}, lr={:.6f}'.format(
                                   float(loss.detach().cpu()), info["Llow"], info["Lhigh"],
                                   info["lam"], info["w_low"], info["w_high"], info["Phys"], info["w_phys"], lr_now
                               ))

            # epoch 末评估
            val = self.test(dataset_eval, dl_eval)
            printwrite(log_file, f'epoch eval loss: {val:.4f}')
            if val < best:
                count = 0
                printwrite(log_file, f'eval loss is reduced from {best:.5f} to {val:.5f}, saving model')
                self.save_model(os.path.join(chk_dir, f'{name}_best.chk'))
                best = val
            else:
                count += 1
                printwrite(log_file, f'eval loss is not reduced for {count} epoch')
                printwrite(log_file, f'best is {best} until now')

            # 保存最近一次
            self.save_model(os.path.join(chk_dir, f'{name}_last.chk'))
            
             #----------- Early Stopping 判断 -----------
            if count >= self.early_stop:
                printwrite(log_file, f'\nEarly stopping triggered after {count} epochs without improvement.')
                printwrite(log_file, f'Best validation loss: {best:.5f}')
                break  # 结束训练主循环
                

    def save_configs(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.cfg, f)

    def save_model(self, path):
        torch.save({'net': self.network.state_dict()}, path)


# ============================= main =============================
if __name__ == '__main__':
    name = getattr(configs, "name", "WSR")
    

    exp_dir = f"exp/{name}"
    os.makedirs(exp_dir, exist_ok=True)

    log_file = os.path.join(exp_dir, "log.txt")
    printwrite(log_file, 'Configs:\n' + str(configs.__dict__))
    

    stats_file = os.path.join(exp_dir, f"stats_{name}.npy")

    # 数据
    train_path = configs.train_path
    val_path = configs.val_path
    scale = configs.scale

    printwrite(log_file, 'processing training set')
    dataset_train = FixedWindTerrainDataset(train_path, configs.geo_path, train=True,
                                           scale=scale, save_stats_path=stats_file)
    print(dataset_train.GetDataShape())

    printwrite(log_file, 'processing eval set')
    stats = np.load(stats_file, allow_pickle=True).item()
    dataset_eval = FixedWindTerrainDataset(val_path, configs.geo_path, train=False,
                                          scale=scale, stats=stats)
    print(dataset_eval.GetDataShape())

    printwrite(log_file, 'Dataset_train Shape:\n' + str(dataset_train.GetDataShape()))
    printwrite(log_file, 'Dataset_test Shape:\n' + str(dataset_eval.GetDataShape()))

    # 训练
    trainer = Trainer(configs)
    trainer.save_configs(os.path.join(exp_dir, "configs.pkl"))
    trainer.train(dataset_train, dataset_eval, chk_dir=exp_dir)
