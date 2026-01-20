# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import dropout
from utils.utils import TransformerBlock, DetailFeatureExtraction
from utils.PSA import PSAModule

# ========== 顶层网络 ==========
class WindSR_Terrain(nn.Module):
    def __init__(self,
                 upscale: int = 8,
                 dim: int = 64,
                 group_depth: int = 6,
                 dem_every: int = 2,
                 sample_ids=(1, 3, 5),
                 dem_ch: int = 16,
                 psa_kernels=(1, 3, 5),
                 psa_groups=(1, 1, 1),
                 enc_res_scale: float = 0.1,
                 decoder_feats=None,
                 sft_hidden: int = 64,
                 cross_blocks_per_stage: int = 4,
                 stage_res_scale: float = 0.2,
                 dropout_max: float = 0.1):
        super().__init__()

        self.upscale = upscale
        self.encoder = EncoderTF(dim=dim,
                                 depth=group_depth,
                                 sample_blocks=sample_ids,
                                 dem_every = dem_every,
                                 dem_ch=dem_ch,
                                 psa_kernels=psa_kernels,
                                 psa_groups=psa_groups,
                                 res_scale=enc_res_scale,
                                 dropout_max=dropout_max,)

        self.decoder = DecoderTF(upscale=upscale, in_ch=dim,
                                 decoder_feats=decoder_feats,
                                 sft_hidden=sft_hidden,
                                 cross_blocks_per_stage=cross_blocks_per_stage,
                                 stage_res_scale=stage_res_scale,
                                 dropout_max=dropout_max,)

    def forward(self, x_lr: torch.Tensor, ele_lr: torch.Tensor, ele_hr: torch.Tensor = None):
        bridge = self.encoder(x_lr, ele_lr)
        y = self.decoder(x_lr, bridge, ele_hr)
        return y

# ========== 编码器 ==========
class EncoderTF(nn.Module):
    def __init__(
        self,
        dim: int = 64,
        depth: int = 6,                 # 总块数 N
        res_scale: float = 0.1,         # 每块残差缩放
        dem_ch: int = 16,               # DEM 编码通道
        dem_every: int = 2,             # FiLM 施放间隔；1=每块；2=每两块...
        sample_blocks: tuple = (1, 3, 5),  # 抽样到 PSA 的块索引（0-based）
        psa_kernels=(1, 3, 5),
        psa_groups=(1, 1, 1),
        dropout_max: float = 0.2
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.dem_every = dem_every
        self.sample_blocks = tuple(sample_blocks)
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(2, dim, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
        )
        self.blocks = nn.ModuleList([
            ResidualCrossFusionBlock(dim=dim, dropout_max=dropout_max, res_scale=res_scale)
            for _ in range(depth)
        ])
        # 始终使用 DEM 条件模块
        self.dem_enc = DemEncoderLR(cond_ch=dem_ch)
        self.film    = SpatialFiLM(cond_ch=dem_ch, dim=dim)
        # PSA：把多个抽样特征 cat 后投影回 dim
        inplans_psa = max(1, len(self.sample_blocks)) * dim
        self.psa = PSAModule(inplans=inplans_psa, out_planes=dim,
                             conv_kernels=list(psa_kernels), conv_groups=list(psa_groups))

        # Gate 融合：把 f_last 与 f_psa 动态融合为 bridge
        self.fuse_gate = nn.Sequential(nn.Conv2d(dim * 2, dim, 1), nn.Sigmoid())

    def forward(self, x_lr: torch.Tensor, ele_lr: torch.Tensor):
        h = self.stem(x_lr)
        # DEM 条件
        cond = self.dem_enc(ele_lr)
        sampled = []
        for i, blk in enumerate(self.blocks):
            # 按间隔施放 FiLM（例如 dem_every=2 → i=0,2,4,...）
            if i % self.dem_every == 0:
                h = self.film(h, cond)
            h = blk(h, ele_lr)  # 每块自带残差
            # 抽样点（块索引）
            if i in self.sample_blocks:
                sampled.append(h)

        f_last = h
        f_psa  = f_last if len(sampled) == 0 else self.psa(torch.cat(sampled, dim=1))  # (B,dim,H,W)

        gate   = self.fuse_gate(torch.cat([f_last, f_psa], dim=1))
        bridge = gate * f_last + (1 - gate) * f_psa
        return bridge

# ========== CrossFusion + 残差 ==========
class ResidualCrossFusionBlock(nn.Module):
    """对 CrossFusionBlock 加外层残差，深堆更稳"""
    def __init__(self, dim, res_scale=0.1, dropout_max=0.1, **kw):
        super().__init__()
        self.block = CrossFusionBlock(dim=dim, dropout_max=0.1, **kw)
        self.res_scale = nn.Parameter(torch.tensor(res_scale), requires_grad=True)

    def forward(self, x, ele):
        return x + self.res_scale * self.block(x, ele)

class CrossFusionBlock(nn.Module):
    def __init__(self, dim, num_heads=None, ffn_expansion_factor=2, LayerNorm_type='WithBias',
                 conv_layers=4, trans_layers=2, dropout_max=None):
        super().__init__()

        self.conv_branch = DetailFeatureExtraction(num_layers=conv_layers, dim=dim)

        heads = num_heads if num_heads is not None else (4 if dim >= 32 else 2)

        dp_max = dropout_max  # 例如编码器最大 0.1
        dp_list = torch.linspace(0, dp_max, steps=trans_layers).tolist()

        self.trans_branch = nn.ModuleList([
            TransformerBlock(dim, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                             bias=False, LayerNorm_type=LayerNorm_type,
                             drop_path=dp_list[i])  # 逐层分配
            for i in range(trans_layers)
        ])

        self.gate_conv = nn.Conv2d(dim * 2, dim, 1)
        self.gate_trans = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, x, ele):
        x_conv = self.conv_branch(x, ele)
        x_trans = x
        for blk in self.trans_branch:
            x_trans = blk(x_trans)
        g1 = torch.sigmoid(self.gate_conv(torch.cat([x_conv, x_trans], dim=1)))
        g2 = torch.sigmoid(self.gate_trans(torch.cat([x_trans, x_conv], dim=1)))
        x_conv = x_conv + g1 * x_trans
        x_trans = x_trans + g2 * x_conv
        return (x_conv + x_trans) / 2

# ========== Helpers ==========
def _pick_gn_groups(C: int) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if C % g == 0:
            return g
    return 1

# ========== 条件/调制模块 ==========
class DemEncoderLR(nn.Module):
    def __init__(self, cond_ch: int = 16):
        super().__init__()
        self.sobel_x = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        with torch.no_grad():
            kx = torch.tensor([[[-1,0,1],[-2,0,2],[-1,0,1]]], dtype=torch.float32).unsqueeze(0) / 8.0
            ky = torch.tensor([[[-1,-2,-1],[0,0,0],[1,2,1]]], dtype=torch.float32).unsqueeze(0) / 8.0
            self.sobel_x.weight.copy_(kx)
            self.sobel_y.weight.copy_(ky)
        for p in self.parameters():
            p.requires_grad = False

        self.stem = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1, bias=True),
            nn.GroupNorm(_pick_gn_groups(64), 64), nn.SiLU(True),
            nn.Conv2d(64, cond_ch, 3, padding=1, bias=True),
            nn.GroupNorm(_pick_gn_groups(cond_ch), cond_ch), nn.SiLU(True),
        )

    def forward(self, z_lr: torch.Tensor) -> torch.Tensor:
        gx = self.sobel_x(z_lr)
        gy = self.sobel_y(z_lr)
        slope  = torch.sqrt(gx * gx + gy * gy + 1e-6)
        aspect = torch.atan2(gy, gx)
        feat = torch.cat([z_lr, slope, torch.cos(aspect), torch.sin(aspect)], dim=1)
        return self.stem(feat)

class SpatialFiLM(nn.Module):
    """空间自适应仿射：y = (1+γ)⊙x + β，条件来自 LR DEM 编码"""
    def __init__(self, cond_ch: int = 16, dim: int = 64):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Conv2d(cond_ch, cond_ch, 3, padding=1, groups=cond_ch),
            nn.SiLU(True),
            nn.Conv2d(cond_ch, 2 * dim, 1)
        )
        nn.init.zeros_(self.gen[-1].weight); nn.init.zeros_(self.gen[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gb = self.gen(cond)
        g, b = torch.chunk(gb, 2, dim=1)
        return (1.0 + g) * x + b

class SFTModulator(nn.Module):
    """解码阶段 SFT：对空间位置逐像素生成 (γ,β)"""
    def __init__(self, cond_in: int = 1, feat_dim: int = 16, hidden: int = 32):
        super().__init__()
        self.fea = nn.Sequential(
            nn.Conv2d(cond_in, hidden, 3, padding=1), nn.SiLU(True),
            nn.Conv2d(hidden, 2 * feat_dim, 3, padding=1)
        )
        nn.init.zeros_(self.fea[-1].weight); nn.init.zeros_(self.fea[-1].bias)

    def forward(self, feat: torch.Tensor, dem: torch.Tensor) -> torch.Tensor:
        gb = self.fea(dem)
        g, b = torch.chunk(gb, 2, dim=1)
        return (1.0 + g) * feat + b

# ========== 解码器 ==========
class UpsampleConvPixelShuffle(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), 3, padding=1)
        self.ps   = nn.PixelShuffle(scale_factor)
    def forward(self, x):
        return self.ps(self.conv(x))

class DecoderTF(nn.Module):
    def __init__(self, upscale: int = 4, in_ch: int = 64, decoder_feats=None, sft_hidden: int = 64,
                 cross_blocks_per_stage: int = 4, stage_res_scale: float = 0.2, dropout_max: float = 0.1):
        super().__init__()
        self.upscale = upscale
        self.stages  = int(math.log2(upscale))

        if decoder_feats is None:
            if   self.stages == 1:
                decoder_feats = [max(8, in_ch // 4)]
            elif self.stages == 2:
                decoder_feats = [max(8, in_ch // 4), max(8, in_ch // 8)]
            elif self.stages == 3:
                decoder_feats = [max(8, in_ch // 4), max(8, in_ch // 8), max(4, in_ch // 16)]
            elif self.stages == 4:
                decoder_feats = [
                    max(8, in_ch // 4),
                    max(8, in_ch // 8),
                    max(4, in_ch // 16),
                    max(4, in_ch // 32)
                ]
            else:
                raise ValueError(f"Unsupported stages={self.stages} for decoder_feats auto config.")
        assert len(decoder_feats) == self.stages
        self.decoder_feats = decoder_feats

        mods = []
        feat_in = in_ch
        for fo in decoder_feats:
            mods.append(ReconstructionStage(
                feat_in=feat_in, feat_out=fo, sft_hidden=sft_hidden,
                cross_blocks=cross_blocks_per_stage, res_scale=stage_res_scale, dropout_max = dropout_max
            ))
            feat_in = fo * 2
        self.stages_mod = nn.ModuleList(mods)

        # 尾部映射到 2 通道 (u, v)
        self.tail = nn.Conv2d(feat_in, 2, 1)

    def forward(self, x_lr, bridge, ele_hr):
        x_lr_ini = x_lr.clone()
        B, _, H, W = x_lr.shape
        # 逐 stage 需要的 DEM 尺度
        out = bridge
        for si, stage in enumerate(self.stages_mod):
            # 当前 stage 目标空间尺度 = 2^(si+1) * (H, W)
            # 已给的是 ele_hr(=2^S)，则 dem_s 缩放为 2^(si+1)
            scale_down = self.upscale // (2 ** (si + 1))
            if scale_down == 1:
                dem_s = ele_hr
            else:
                dem_s = F.interpolate(ele_hr, scale_factor=1.0 / scale_down,
                                      mode='bilinear', align_corners=False)
            x_lr, out = stage(x_lr, out, dem_s)

        # 全局图像残差（常规 SR）
        base = F.interpolate(x_lr_ini, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        return self.tail(out) + base

class ReconstructionStage(nn.Module):
    def __init__(self, feat_in: int, feat_out: int, sft_hidden: int = 64,
                 cross_blocks: int = 4, res_scale: float = 0.2, dropout_max: float = 0.1):
        super().__init__()
        self.feat_in  = feat_in
        self.feat_out = feat_out

        self.up_feat = UpsampleConvPixelShuffle(in_channels=feat_in, out_channels=feat_out, scale_factor=2)
        self.up_img  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.sft      = SFTModulator(cond_in=1, feat_dim=feat_out, hidden=sft_hidden)
        self.gate_net = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(True),
                                      nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid())
        self.img_proj = nn.Conv2d(2, feat_out, 1)
        self.dem_proj = nn.Conv2d(1, feat_out, 3, padding=1)

        fuse_dim = feat_out * 2
        self.cross_branch = nn.ModuleList([CrossFusionBlock(dim=fuse_dim, dropout_max = dropout_max) for _ in range(cross_blocks)])
        self.res_scale = nn.Parameter(torch.tensor(res_scale), requires_grad=True)

    def forward(self, x_lr, feat_in, dem_s):
        # 高频/低频分支
        feat_up  = self.up_feat(feat_in)         # (B,feat_out,·,·)
        img_up   = self.up_img(x_lr)             # (B,2,·,·)
        feat_mod = self.sft(feat_up, dem_s)
        gate     = self.gate_net(dem_s)
        img_proj = self.img_proj(img_up)

        fusion  = gate * feat_mod + (1 - gate) * img_proj      # (B,feat_out,·,·)
        dem_c   = self.dem_proj(dem_s)                          # (B,feat_out,·,·)
        x_cat   = torch.cat([fusion, dem_c], dim=1)             # (B,2*feat_out,·,·)

        out = x_cat
        for blk in self.cross_branch:
            out = blk(out, dem_s)
        out = out + self.res_scale * x_cat                      # 单层 Stage 残差
        return img_up, out

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = WindSR_Terrain(upscale=4, dim=64, group_depth=6,
                         decoder_feats=None, cross_blocks_per_stage=4).to(device)
    B, H, W = 2, 16, 16
    x_lr   = torch.randn(B, 2, H, W, device=device)
    ele_lr = torch.randn(B, 1, H, W, device=device)
    ele_hr = torch.randn(B, 1, 4*H, 4*W, device=device)
    with torch.no_grad():
        y = net(x_lr, ele_lr, ele_hr)
    print("Input :", x_lr.shape, "LR-DEM:", ele_lr.shape, "HR-DEM:", ele_hr.shape)
    print("Output:", y.shape)
