import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalAveragePooling(nn.Module):
    """全局平均池化 (GAP)，用于压缩空间维度并聚合全局信息"""
    
    def __init__(self):
        super(GlobalAveragePooling, self).__init__()

    def forward(self, x):
        return F.adaptive_avg_pool2d(x, (1, 1))


class FPN(nn.Module):
    """特征金字塔网络 (FPN)，用于多尺度特征融合"""
    
    def __init__(self, in_channels, out_channels, num_levels=4):
        super(FPN, self).__init__()
        self.num_levels = num_levels
        self.lateral_convs = nn.ModuleList()
        self.smooth_convs = nn.ModuleList()

        for _ in range(num_levels):
            self.lateral_convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            self.smooth_convs.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

    def forward(self, features):
        """features: List of feature maps from backbone (lowest to highest resolution)"""
        assert len(features) == self.num_levels

        # 先进行横向连接
        laterals = [lateral_conv(f) for lateral_conv, f in zip(self.lateral_convs, features)]
        
        # 自顶向下传播
        for i in range(self.num_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(laterals[i], size=laterals[i - 1].shape[2:], mode="nearest")

        # 逐层平滑
        return [smooth_conv(l) for smooth_conv, l in zip(self.smooth_convs, laterals)]


class SimpleConvNeck(nn.Module):
    """简单的卷积 Neck，使用 1x1 和 3x3 卷积进行通道压缩"""
    
    def __init__(self, in_channels, out_channels):
        super(SimpleConvNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DeformableConvNeck(nn.Module):
    """可变形卷积 Neck，提高模型对物体形变的适应能力"""
    
    def __init__(self, in_channels, out_channels):
        super(DeformableConvNeck, self).__init__()
        self.conv_offset = nn.Conv2d(in_channels, 18, kernel_size=3, padding=1)  # 计算偏移量
        self.deform_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        offset = self.conv_offset(x)
        x = self.deform_conv(x)  # 这里需要一个 Deformable Conv 实现
        x = self.bn(x)
        x = self.relu(x)
        return x


class HourglassNeck(nn.Module):
    """沙漏结构 Neck，用于高精度姿态估计"""
    
    def __init__(self, num_stacks, in_channels, out_channels):
        super(HourglassNeck, self).__init__()
        self.num_stacks = num_stacks
        self.hg_modules = nn.ModuleList([self._make_hourglass(in_channels, out_channels) for _ in range(num_stacks)])

    def _make_hourglass(self, in_channels, out_channels):
        """构建沙漏模块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        for hg in self.hg_modules:
            x = hg(x)
        return x
