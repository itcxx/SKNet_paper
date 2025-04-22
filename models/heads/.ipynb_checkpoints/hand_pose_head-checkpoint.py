import torch
import torch.nn as nn
import torch.nn.functional as F


class Heatmap3DHead(nn.Module):
    """Heatmap3DHead: 生成 3D 关键点热力图

    Args:
        in_channels (int): 输入特征通道数.
        out_channels (int): 输出通道数 (关键点数 * 深度维度).
        depth_size (int): 3D 热力图深度维度.
        deconv_out_channels (tuple[int]): 反卷积层的通道数.
        deconv_kernel_sizes (tuple[int]): 反卷积层的卷积核大小.
    """

    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 depth_size: int = 64,
                 deconv_out_channels=(256, 256, 256),
                 deconv_kernel_sizes=(4, 4, 4)):
        super().__init__()
        
        assert out_channels % depth_size == 0, "out_channels 必须是 depth_size 的倍数"
        self.depth_size = depth_size

        # 反卷积层 (用于上采样)
        deconv_layers = []
        prev_channels = in_channels
        for out_c, kernel in zip(deconv_out_channels, deconv_kernel_sizes):
            deconv_layers.append(nn.ConvTranspose2d(prev_channels, out_c, kernel_size=kernel, stride=2, padding=1, bias=False))
            deconv_layers.append(nn.BatchNorm2d(out_c))
            deconv_layers.append(nn.ReLU(inplace=True))
            prev_channels = out_c

        self.deconv_layers = nn.Sequential(*deconv_layers)

        # 最终输出层
        self.final_layer = nn.Conv2d(prev_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.deconv_layers(x)
        x = self.final_layer(x)

        N, C, H, W = x.shape
        # 重新调整形状: (N, num_joints, depth_size, H, W)
        x = x.view(N, C // self.depth_size, self.depth_size, H, W)

        return x


class Heatmap1DHead(nn.Module):
    """Heatmap1DHead: 预测相对根深度的 1D 热力图

    Args:
        in_channels (int): 输入通道数.
        heatmap_size (int): 1D 热力图大小.
        hidden_dims (tuple[int]): 全连接层的隐藏单元数.
    """

    def __init__(self, 
                 in_channels: int = 2048, 
                 heatmap_size: int = 64, 
                 hidden_dims=(512, )):
        super().__init__()

        self.in_channels = in_channels
        self.heatmap_size = heatmap_size

        # 全连接层
        layers = []
        prev_dim = in_channels
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, heatmap_size))
        self.fc = nn.Sequential(*layers)

    def soft_argmax_1d(self, heatmap1d):
        """1D Soft-Argmax 计算根深度"""
        heatmap1d = F.softmax(heatmap1d, dim=1)  # 归一化
        accu = heatmap1d * torch.arange(self.heatmap_size, dtype=heatmap1d.dtype, device=heatmap1d.device)[None, :]
        coord = accu.sum(dim=1)
        return coord

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.shape[0], -1)
        x = self.fc(x)
        return self.soft_argmax_1d(x).view(-1, 1)

    def init_weights(self):
        """初始化权重"""
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class MultilabelClassificationHead(nn.Module):
    """预测手部类型 (左手/右手)"""

    def __init__(self, in_channels, num_labels=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.ReLU(),
            nn.Linear(512, num_labels)
        )

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.shape[0], -1)
        return self.fc(x)


class HandPoseHead(nn.Module):
    """3D 手部关键点检测 Head"""

    def __init__(self, in_channels, num_joints, depth_size):
        super().__init__()
        self.heatmap_head = Heatmap3DHead(in_channels, num_joints*depth_size, depth_size)
        self.root_head = Heatmap1DHead(in_channels)
        self.hand_type_head = MultilabelClassificationHead(in_channels)

        self.loss_kpt = nn.MSELoss()
        self.loss_root = nn.L1Loss()
        self.loss_hand_type = nn.BCEWithLogitsLoss()

    def forward(self, x):
        heatmaps = self.heatmap_head(x)  # 3D 关键点热力图
        root_depth = self.root_head(x)  # 根部深度预测
        hand_type = self.hand_type_head(x)  # 手部类别预测
        return heatmaps, root_depth, hand_type

    def compute_loss(self, pred, target):
        """计算损失"""
        pred_heatmaps, pred_root, pred_hand_type = pred
        target_heatmaps, target_root, target_hand_type = target

        loss_kpt = self.loss_kpt(pred_heatmaps, target_heatmaps)
        loss_root = self.loss_root(pred_root, target_root)
        loss_hand = self.loss_hand_type(pred_hand_type, target_hand_type)

        return {"loss_kpt": loss_kpt, "loss_root": loss_root, "loss_hand_type": loss_hand}


# # ** 测试**
# if __name__ == "__main__":
#     head = HandPoseHead(in_channels=512, num_joints=21, depth_size=32)
#     print(head)
#     x = torch.randn(1, 512, 7, 7)
#     heatmaps, root_depth, hand_type = head(x)
#     print("输出形状:", heatmaps.shape, root_depth.shape, hand_type.shape)  # (B, out_channels, H, W) 输出形状: torch.Size([1, 21, 32, 56, 56]) torch.Size([1, 1]) torch.Size([1, 2])
