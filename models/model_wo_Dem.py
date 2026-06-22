# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import dropout

from .utils import TransformerBlock, DetailFeatureExtraction
from .PSA import PSAModule

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
    """
    扁平化编码器（不分组）：
      - Stem: 2→dim
      - depth 个 ResidualCrossFusionBlock（每块自带残差）
      - 始终使用 DEM：FiLM(DEM) 按间隔 dem_every 施放（1=每块；2=每两块……）
      - 在 sample_blocks 指定的块索引处抽样 → PSA(输出=dim) → 与最后一块输出 Gate 融合
    前向:
      f_last, f_psa, bridge = EncoderTF(...)(x_lr, ele_lr)
      * x_lr   : (B,2,H,W)
      * ele_lr : (B,1,H,W)  —— 必须提供
    """
    def __init__(
        self,
        dim: int = 64,
        depth: int = 6,                 # 总块数 N（你现在是6层）
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
        assert dem_every >= 1, "dem_every 必须是正整数（1 表示每个块都施放 FiLM）"
        self.dem_every = dem_every
        self.sample_blocks = tuple(sample_blocks)
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(2, dim, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
        )
        # N 个块（每块自带残差）
        self.blocks = nn.ModuleList([
            ResidualCrossFusionBlock(dim=dim, dropout_max=dropout_max, res_scale=res_scale)
            for _ in range(depth)
        ])
        
        
        # PSA：把多个抽样特征 cat 后投影回 dim
        inplans_psa = max(1, len(self.sample_blocks)) * dim
        self.psa = PSAModule(inplans=inplans_psa, out_planes=dim,
                             conv_kernels=list(psa_kernels), conv_groups=list(psa_groups))

        # Gate 融合：把 f_last 与 f_psa 动态融合为 bridge
        self.fuse_gate = nn.Sequential(nn.Conv2d(dim * 2, dim, 1), nn.Sigmoid())

    def forward(self, x_lr: torch.Tensor, ele_lr: torch.Tensor):
        
        h = self.stem(x_lr)
        sampled = []
        
        for i, blk in enumerate(self.blocks):
            # 按间隔施放 FiLM（例如 dem_every=2 → i=0,2,4,...）
            h = blk(h)  # 每块自带残差
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

    def forward(self, x):
        return x + self.res_scale * self.block(x)

class CrossFusionBlock(nn.Module):
    """
    卷积分支(DetailFeatureExtraction) + Transformer 分支 互相门控交互
    返回两支平均
    """
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

    def forward(self, x):
        x_conv = self.conv_branch(x)
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

                # 默认通道日程
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

        # 逐 stage 构建
        mods = []
        feat_in = in_ch
        for fo in decoder_feats:
            mods.append(ReconstructionStage(
                feat_in=feat_in, feat_out=fo, sft_hidden=sft_hidden,
                cross_blocks=cross_blocks_per_stage, res_scale=stage_res_scale, dropout_max = dropout_max
            ))
            feat_in = fo  # 因为输出拼接了 DEM（2*feat_out）
        self.stages_mod = nn.ModuleList(mods)

        # 尾部映射到 2 通道 (u, v)
        self.tail = nn.Conv2d(feat_in, 2, 1)

    def forward(self, x_lr, bridge, ele_hr):
        """
        x_lr   : (B,2,H,W)
        bridge : (B,in_ch,H,W)
        ele_hr : (B,1,upscale*H,upscale*W)；若 None，外部应补上
        """
        x_lr_ini = x_lr.clone()
        B, _, H, W = x_lr.shape
        # 逐 stage 需要的 DEM 尺度
        out = bridge
        for si, stage in enumerate(self.stages_mod):
            # 当前 stage 目标空间尺度 = 2^(si+1) * (H, W)
            # 已给的是 ele_hr(=2^S)，则 dem_s 缩放为 2^(si+1)
            scale_down = self.upscale // (2 ** (si + 1))
            x_lr, out = stage(x_lr, out)

        # 全局图像残差（常规 SR）
        base = F.interpolate(x_lr_ini, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        return self.tail(out) + base

class ReconstructionStage(nn.Module):
    """
    单个 ×2 上采样 stage：
      - 高频支路：PixelShuffle 上采样 + SFT(DEM)
      - 低频支路：双线性插值 + 1×1 投影到 feat_out
      - DEM-aware gate 融合（逐像素控制高/低频比例）
      - 拼接 DEM 投影（1→feat_out） → (B, 2*feat_out, ·, ·)
      - CrossFusion 堆叠 + 单层 stage 残差
    输入：
      feat_in : 上一 stage 的输出通道（首 stage 为 encoder dim）
      feat_out: 本 stage 的“半数通道”（拼接 DEM 后进入 CrossFusion 的通道 = 2*feat_out）
    """
    def __init__(self, feat_in: int, feat_out: int, sft_hidden: int = 64,
                 cross_blocks: int = 4, res_scale: float = 0.2, dropout_max: float = 0.1):
        super().__init__()
        self.feat_in  = feat_in
        self.feat_out = feat_out

        self.up_feat = UpsampleConvPixelShuffle(in_channels=feat_in, out_channels=feat_out, scale_factor=2)
        self.up_img  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.gate_net = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(True),
                                      nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid())
        self.img_proj = nn.Conv2d(2, feat_out, 1)

        fuse_dim = feat_out
        self.cross_branch = nn.ModuleList([CrossFusionBlock(dim=fuse_dim, dropout_max = dropout_max) for _ in range(cross_blocks)])
        
        self.res_scale = nn.Parameter(torch.tensor(res_scale), requires_grad=True)

    def forward(self, x_lr, feat_in):
        # 高频/低频分支
        feat_up  = self.up_feat(feat_in)         # (B,feat_out,·,·)
        img_up   = self.up_img(x_lr)             # (B,2,·,·)
        
        img_proj = self.img_proj(img_up)

        fusion  = feat_up + img_proj      # (B,feat_out,·,·)
        
        x_cat   =  fusion         
        out = x_cat
        
        for blk in self.cross_branch:
            out = blk(out)
            
        out = out + self.res_scale * x_cat                      # 单层 Stage 残差
        return img_up, out


# ========== 快速自检 ==========
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
