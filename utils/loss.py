import torch
import torch.nn as nn
import torch.nn.functional as F


class WindLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

        k1d = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32)
        k2d = (k1d[:, None] @ k1d[None, :]) / 256.0
        self.register_buffer("blur5", k2d.view(1, 1, 5, 5))

    # -------------------- 频域拆分（低频=模糊，高频=原图-低频） --------------------
    def _lp_smooth(self, x: torch.Tensor) -> torch.Tensor:
        w = self.blur5.repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, w, padding=2, groups=x.size(1))

    def split_low_high(self, x: torch.Tensor):
        low = self._lp_smooth(x)
        high = x - low
        return low, high

    # -------------------- 地形与物理约束 --------------------
    def gradient_x(self, f):
        kernel = torch.tensor([[-0.5, 0, 0.5]], dtype=f.dtype, device=f.device).view(1, 1, 1, 3)
        return F.conv2d(f, kernel, padding=(0, 1))

    def gradient_y(self, f):
        kernel = torch.tensor([[-0.5], [0], [0.5]], dtype=f.dtype, device=f.device).view(1, 1, 3, 1)
        return F.conv2d(f, kernel, padding=(1, 0))

    def divergence(self, u, v):
        return self.gradient_x(u) + self.gradient_y(v)

    def vorticity(self, u, v):
        return self.gradient_x(v) - self.gradient_y(u)

    def terrain_mask_inverse(self, dem, eps=1e-3):
        dx = self.gradient_x(dem)
        dy = self.gradient_y(dem)
        grad = torch.sqrt(dx**2 + dy**2)
        grad_norm = grad / (grad.max() + eps)
        mask = 1.0 - grad_norm
        return mask

    # -------------------- 训练损失 --------------------
    def forward(
        self,
        y_pred,
        y_true,
        gh,
        epoch: int,
        ramp_T: int = 20,
        phys_start: int = 10,
        phys_ramp: int = 10,
    ):
        # 1) 整图 L1
        L1_full = F.l1_loss(y_pred, y_true)

        # 2) 频率分解
        pred_low, pred_high = self.split_low_high(y_pred)
        true_low, true_high = self.split_low_high(y_true)

        scale = true_high.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-3)
        Lhigh = F.l1_loss(pred_high / scale, true_high / scale)

        # 3) ramp
        t = min(1.0, float(epoch) / max(1, ramp_T))
        
        w_high = 0.05 + 0.15 * t
   

        # 4) 物理项
        u_pred, v_pred = y_pred[:, 0:1], y_pred[:, 1:2]
        u_true, v_true = y_true[:, 0:1], y_true[:, 1:2]

        div_pred = self.divergence(u_pred, v_pred)
        div_true = self.divergence(u_true, v_true)
        vor_pred = self.vorticity(u_pred, v_pred)
        vor_true = self.vorticity(u_true, v_true)

        mask = self.terrain_mask_inverse(gh)
        Ldiv = ((div_pred - div_true).abs() * mask).mean()
        Lvor = ((vor_pred - vor_true).abs() * mask).mean()
        Phys = Ldiv + Lvor

        if epoch < phys_start:
            w_phys = 0.0
        elif epoch < phys_start + phys_ramp:
            w_phys = 1 * (epoch - phys_start) / phys_ramp
        else:
            w_phys = 1

        # 5) 总损失
        loss = L1_full + w_high * Lhigh + w_phys * Phys

        info = {
            "L1": float(L1_full.detach().cpu()),
            "Lhigh": float((Lhigh).detach().cpu()),
            "w_high": w_high,
            "Phys": float(Phys.detach().cpu()),
            "w_phys": w_phys,
        }
        return loss, info

    # -------------------- 验证损失 --------------------
    @staticmethod
    def val_loss(y_pred, y_true):
        return F.l1_loss(y_pred, y_true)
