####
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import numbers
from einops import rearrange
from modules import ConvSC
from timm.models.layers import trunc_normal_
from torchvision.ops import DeformConv2d
from timm.layers import DropPath


########
class BasicConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 upsampling=False,
                 act_norm=False,
                 act_inplace=True):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if upsampling is True:
            self.conv = nn.Sequential(*[
                nn.Conv2d(in_channels, out_channels*4, kernel_size=kernel_size,
                          stride=1, padding=padding, dilation=dilation),
                nn.PixelShuffle(2)
            ])
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation)

        self.norm = nn.GroupNorm(1, out_channels)
        self.act = nn.SiLU(inplace=act_inplace)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC(nn.Module):

    def __init__(self,
                 C_in,
                 C_out,
                 kernel_size=3,
                 downsampling=False,
                 upsampling=False,
                 act_norm=True,
                 act_inplace=True):
        super(ConvSC, self).__init__()

        stride = 2 if downsampling is True else 1
        padding = (kernel_size - stride + 1) // 2

        self.conv = BasicConv2d(C_in, C_out, kernel_size=kernel_size, stride=stride,
                                upsampling=upsampling, padding=padding,
                                act_norm=act_norm, act_inplace=act_inplace)

    def forward(self, x):
        y = self.conv(x)
        return y

class GroupConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 groups=1,
                 act_norm=False,
                 act_inplace=True):
        super(GroupConv2d, self).__init__()
        self.act_norm=act_norm
        if in_channels % groups != 0:
            groups=1
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=groups)
        self.norm = nn.GroupNorm(groups,out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=act_inplace)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y

## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out
    ##########################################################################


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        ffn_expansion_factor,
        bias,
        LayerNorm_type,
        drop_path: float = 0.0,   # << 新增：每层的 droppath 概率
    ):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn  = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn   = FeedForward(dim, ffn_expansion_factor, bias)

        # << 新增：残差路径的随机深度（训练时生效，推理自动关闭）
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        # 在“加残差”之前应用 droppath
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x

##########################################################################
class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        return self.bottleneckBlock(x)


class DetailNode(nn.Module):
    def __init__(self):
        super(DetailNode, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)

    def forward(self, z1, z2):
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 + self.theta_eta(z2)
        return z1, z2


class DetailFeatureExtraction_(nn.Module):
    def __init__(self, num_layers=1, dim=2):
        super(DetailFeatureExtraction_, self).__init__()
        INNmodules = [DetailNode() for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)

        self.shffleconv = nn.Conv2d(dim, 64, kernel_size=3, stride=1, padding=1)
        self.fusion = nn.Conv2d(64, dim, kernel_size=3, stride=1, padding=1)

    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:x.shape[1]]
        return z1, z2

    def forward(self, x):
        z1, z2 = self.separateFeature(self.shffleconv(x))
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        z_out = self.fusion(torch.cat((z1, z2), dim=1))
        return z_out

#########变形卷积
class DeformableConvBlock(nn.Module):
    """单个可变形卷积模块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, modulation=True):
        super(DeformableConvBlock, self).__init__()
        self.offset_conv = nn.Conv2d(1, 2 * kernel_size * kernel_size,
                                     kernel_size=3, padding=1)
        if modulation:
            self.mask_conv = nn.Conv2d(in_channels, kernel_size * kernel_size,
                                       kernel_size=3, padding=1)
        else:
            self.mask_conv = None

        self.deform_conv = DeformConv2d(in_channels, out_channels,
                                        kernel_size=kernel_size, padding=padding, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.modulation = modulation

    def forward(self, x, ele):
        offset = self.offset_conv(ele)  # (B, 2*k*k, H, W)
        if self.modulation:
            mask = torch.sigmoid(self.mask_conv(x))  # (B, k*k, H, W)
        else:
            mask = None
        x = self.deform_conv(x, offset, mask)
        return self.relu(x)


# class DetailFeatureExtraction(nn.Module):
#     """
#     Deformable Conv 堆叠替代原来的 INN 结构
#     - 支持多层叠加
#     """
#     def __init__(self, num_layers=6, dim=32, kernel_size=3):
#         super(DetailFeatureExtraction, self).__init__()
#         self.layers = nn.Sequential(*[
#             DeformableConvBlock(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2)
#             for _ in range(num_layers)
#         ])

#     def forward(self, x, ele=None):
#         return self.layers(x, ele)


class DetailFeatureExtraction(nn.Module):
    def __init__(self, num_layers=6, dim=32, kernel_size=3):
        super(DetailFeatureExtraction, self).__init__()
        self.layers = nn.ModuleList([
            DeformableConvBlock(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2)
            for _ in range(num_layers)
        ])

    def forward(self, x, ele):
        for layer in self.layers:
            x = layer(x, ele)
        return x
