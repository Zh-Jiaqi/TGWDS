import torch
import torch.nn as nn

class SEWeightModule(nn.Module):
    """Squeeze-and-Excitation"""
    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)
        return weight


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


class PSAModule(nn.Module):

    def __init__(self, inplans, out_planes=64,
                 conv_kernels=[1, 3, 5],
                 stride=1,
                 conv_groups=[1, 1, 1]):
        super(PSAModule, self).__init__()
        self.out_planes = out_planes
        self.split_channel = out_planes // 3  

        self.convs = nn.ModuleList([
            conv(inplans, self.split_channel, kernel_size=k, padding=k // 2, stride=stride, groups=g)
            for k, g in zip(conv_kernels, conv_groups)
        ])

        self.se_blocks = nn.ModuleList([
            SEWeightModule(self.split_channel) for _ in range(3)
        ])

        self.softmax = nn.Softmax(dim=1)


        self.proj = nn.Conv2d(self.split_channel * 3, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        B = x.size(0)


        feats = [conv_i(x) for conv_i in self.convs]  # list of (B,C,H,W)


        se_weights = [se(f) for f, se in zip(feats, self.se_blocks)]  # list of (B,C,1,1)


        se_stack = torch.stack(se_weights, dim=1)  # (B,3,C,1,1)
        attn = self.softmax(se_stack)              # (B,3,C,1,1)


        feats_stack = torch.stack(feats, dim=1)    # (B,3,C,H,W)
        feats_weighted = feats_stack * attn        # (B,3,C,H,W)


        out = feats_weighted.view(B, -1, feats[0].size(2), feats[0].size(3))  # (B,3C,H,W)


        out = self.proj(out)   # (B, out_planes, H, W)
        return out
