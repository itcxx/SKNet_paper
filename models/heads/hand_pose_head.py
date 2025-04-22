import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

# class ChannelContrastiveLoss(nn.Module):
#     """
#     通道级对比损失，使用 soft-argmax 从 pred 和 target heatmaps 中提取 peak 坐标并构建 triplet-style loss。
#     输入:
#         - pred_heatmaps: (B, K, D, H, W)
#         - target_heatmaps: (B, K, D, H, W)
#     """
#     def __init__(self, margin=4.0, temperature=0.05):
#         super().__init__()
#         self.margin = margin
#         self.temperature = temperature
#
#     def soft_argmax_3d(self, heatmap):
#         """
#         3D soft-argmax 实现。
#         输入: (K, D, H, W) 或 (1, K, D, H, W)
#         输出: (K, 3) => (x, y, z)
#         """
#         if heatmap.dim() == 4:
#             heatmap = heatmap.unsqueeze(0)  # (1, K, D, H, W)
#
#         N, K, D, H, W = heatmap.shape
#         heatmap = heatmap.view(N, K, -1)
#         heatmap = heatmap - heatmap.amax(dim=2, keepdim=True)
#         heatmap = F.softmax(heatmap / self.temperature, dim=2)
#         heatmap = heatmap.view(N, K, D, H, W)
#
#         device = heatmap.device
#         z_range = torch.linspace(0, D - 1, D, device=device)
#         y_range = torch.linspace(0, H - 1, H, device=device)
#         x_range = torch.linspace(0, W - 1, W, device=device)
#         zz, yy, xx = torch.meshgrid(z_range, y_range, x_range, indexing='ij')
#
#         xx = xx.view(1, 1, D, H, W)
#         yy = yy.view(1, 1, D, H, W)
#         zz = zz.view(1, 1, D, H, W)
#
#         x = torch.sum(heatmap * xx, dim=(2, 3, 4))
#         y = torch.sum(heatmap * yy, dim=(2, 3, 4))
#         z = torch.sum(heatmap * zz, dim=(2, 3, 4))
#
#         return torch.stack([x, y, z], dim=2).squeeze(0)  # (K, 3)
#
#     def forward(self, pred_heatmaps, target_heatmaps):
#         """
#         pred_heatmaps, target_heatmaps: (B, K, D, H, W)
#         """
#         B, K, D, H, W = pred_heatmaps.shape
#         total_loss = 0.0
#
#         for b in range(B):
#             pred = pred_heatmaps[b]      # (K, D, H, W)
#             target = target_heatmaps[b]  # (K, D, H, W)
#
#             pred_kpts = self.soft_argmax_3d(pred)     # (K, 3)
#             target_kpts = self.soft_argmax_3d(target) # (K, 3)
#
#             for k in range(K):
#                 gt_k = target_kpts[k]
#                 pred_k = pred_kpts[k]
#                 loss_pos = F.mse_loss(pred_k, gt_k)
#
#                 loss_neg = 0.0
#                 for j in range(K):
#                     if j == k:
#                         continue
#                     pred_j = pred_kpts[j]
#                     d_pos = F.mse_loss(pred_k, gt_k)
#                     d_neg = F.mse_loss(pred_j, gt_k)
#                     loss_neg += F.relu(self.margin - (d_neg - d_pos))
#
#                 total_loss += loss_pos + loss_neg / (K - 1)
#
#         return total_loss / B
class ChannelContrastiveLoss(nn.Module):
    def __init__(self, margin=4.0, temperature=0.05):
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def soft_argmax_3d(self, heatmaps):
        """
        输入: heatmaps: (B, K, D, H, W)
        输出: keypoints: (B, K, 3) => (x, y, z)
        """
        B, K, D, H, W = heatmaps.shape
        heatmaps = heatmaps.view(B, K, -1)
        heatmaps = heatmaps - heatmaps.amax(dim=2, keepdim=True)
        heatmaps = F.softmax(heatmaps / self.temperature, dim=2)
        heatmaps = heatmaps.view(B, K, D, H, W)

        device = heatmaps.device
        z_range = torch.linspace(0, D - 1, D, device=device)
        y_range = torch.linspace(0, H - 1, H, device=device)
        x_range = torch.linspace(0, W - 1, W, device=device)
        zz, yy, xx = torch.meshgrid(z_range, y_range, x_range, indexing='ij')

        zz = zz[None, None, :, :, :].to(device)
        yy = yy[None, None, :, :, :].to(device)
        xx = xx[None, None, :, :, :].to(device)

        x = torch.sum(heatmaps * xx, dim=(2, 3, 4))
        y = torch.sum(heatmaps * yy, dim=(2, 3, 4))
        z = torch.sum(heatmaps * zz, dim=(2, 3, 4))

        keypoints = torch.stack([x, y, z], dim=2)  # (B, K, 3)
        return keypoints

    def forward(self, pred_heatmaps, target_heatmaps):
        """
        pred_heatmaps, target_heatmaps: (B, K, D, H, W)
        返回: scalar loss
        """
        B, K, D, H, W = pred_heatmaps.shape
        pred_kpts = self.soft_argmax_3d(pred_heatmaps)     # (B, K, 3)
        target_kpts = self.soft_argmax_3d(target_heatmaps) # (B, K, 3)

        # 正样本距离
        loss_pos = F.mse_loss(pred_kpts, target_kpts, reduction='none')  # (B, K, 3)
        loss_pos = loss_pos.mean(dim=2)  # (B, K)

        # 负样本距离（构建对比项）
        # Expand to (B, K, K, 3)
        pred_kpts_i = pred_kpts.unsqueeze(2)  # (B, K, 1, 3)
        pred_kpts_j = pred_kpts.unsqueeze(1)  # (B, 1, K, 3)
        target_kpts_i = target_kpts.unsqueeze(2)  # (B, K, 1, 3)

        d_pos = F.mse_loss(pred_kpts_i, target_kpts_i, reduction='none').mean(dim=3)  # (B, K, 1)
        d_neg = F.mse_loss(pred_kpts_j, target_kpts_i, reduction='none').mean(dim=3)  # (B, K, K)

        # 屏蔽对角线（j == k，正样本）
        eye = torch.eye(K, device=pred_heatmaps.device).unsqueeze(0)  # (1, K, K)
        mask = 1.0 - eye  # (1, K, K)
        d_neg_masked = d_neg * mask  # (B, K, K)

        # Triplet loss: relu(margin - (d_neg - d_pos))
        triplet_loss = F.relu(self.margin - (d_neg_masked - d_pos))  # (B, K, K)
        loss_neg = triplet_loss.sum(dim=2) / (K - 1)  # 平均负样本损失 (B, K)

        total_loss = (loss_pos + loss_neg).mean()  # batch + joints 平均
        return total_loss
# class KeypointMSELoss(nn.Module):
#     """类似 Interhand 的 KeypointMSELoss, 用于热图的 MSE 回归.
#
#     Args:
#         use_target_weight (bool): 是否使用关键点权重, 默认 False
#         skip_empty_channel (bool): 是否跳过全部为0的通道, 默认 False
#         loss_weight (float): 损失整体权重, 默认 1.0
#     """
#
#     def __init__(self,
#                  use_target_weight: bool = False,
#                  skip_empty_channel: bool = False,
#                  loss_weight: float = 1.0):
#         super().__init__()
#         self.use_target_weight = use_target_weight
#         self.skip_empty_channel = skip_empty_channel
#         self.loss_weight = loss_weight
#
#     def forward(self,
#                 output: torch.Tensor,
#                 target: torch.Tensor,
#                 target_weights: torch.Tensor = None,
#                 mask: torch.Tensor = None) -> torch.Tensor:
#         """
#         计算 MSE Loss (B,K,H,W) 形状的热图.
#
#         Args:
#             output (Tensor): 模型输出的热图, shape = [B, K, H, W]
#             target (Tensor): GT热图, shape = [B, K, H, W]
#             target_weights (Tensor, optional):
#                 - 如果 shape = [B, K], 表示每个关节的权重
#                 - 如果 shape = [B, K, H, W], 表示每个像素级别的权重
#             mask (Tensor, optional):
#                 - 空间掩码, shape 可以是 [B, K, H, W] 或 [B, 1, H, W].
#                   1 表示有效, 0 表示无效.
#
#         Returns:
#             Tensor: 标量损失, shape []
#         """
#
#         # 1) 构造最终 mask
#         final_mask = self._get_mask(target, target_weights, mask)
#
#         # 2) 若无任何 mask, 直接用 mse_loss
#         if final_mask is None:
#             loss = F.mse_loss(output, target, reduction='mean')
#         else:
#             # 否则 先 element-wise 计算 mse, 再乘 mask 后 mean
#             _loss = F.mse_loss(output, target, reduction='none')  # 保留 [B,K,H,W]
#             loss = (_loss * final_mask).mean()  # 再做均值
#
#         return loss * self.loss_weight
#
#     def _get_mask(self,
#                   target: torch.Tensor,
#                   target_weights: torch.Tensor = None,
#                   mask: torch.Tensor = None) -> torch.Tensor:
#         """
#         生成最终的 mask, 结合:
#           - 输入的 mask
#           - target_weights
#           - skip_empty_channel
#         若最终没有任何有效屏蔽, 则返回 None.
#         """
#
#         # ------------------------
#         # a) 和 target shape 对齐
#         # ------------------------
#         final_mask = mask
#         # 如果 mask 不为空, 检查它维度是否能 broadcast 到 target
#         if final_mask is not None:
#             assert final_mask.dim() == target.dim(), (
#                 f'Mask形状与target不匹配: mask={final_mask.shape}, target={target.shape}')
#
#         # ------------------------
#         # b) target_weights
#         # 如果 use_target_weight=True 或 训练过程中需要可见性加权
#         # ------------------------
#         if self.use_target_weight and (target_weights is not None):
#             # target_weights 的 shape 可能是 (B,K) 或 (B,K,H,W)
#             if target_weights.dim() == 2:
#                 # (B,K) -> (B,K,1,1)
#                 target_weights = target_weights[..., None, None]
#             # 广播到 (B,K,H,W)
#             if final_mask is None:
#                 final_mask = target_weights
#             else:
#                 final_mask = final_mask * target_weights
#
#         # ------------------------
#         # c) skip_empty_channel
#         # 如果启用, 则对于一个通道里全是0的GT, 直接跳过.
#         # ------------------------
#         if self.skip_empty_channel:
#             # 找到 non-zero 的通道
#             # shape (B,K,H,W) -> flatten(2) => (B,K, H*W)
#             # any(dim=2) => (B,K)
#             non_zero_map = (target != 0).flatten(2).any(dim=2)  # bool, shape=(B,K)
#             # 再 reshape 回( B,K,1,1 ) 以便广播
#             non_zero_map = non_zero_map[..., None, None]  # (B,K,1,1)
#
#             if final_mask is None:
#                 final_mask = non_zero_map.float()
#             else:
#                 final_mask = final_mask * non_zero_map.float()
#
#         # 如果最终没有得到任何 mask, 则返回 None
#         if final_mask is None:
#             return None
#
#         return final_mask
class Heatmap3DHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth_size: int = 64,
        deconv_out_channels=(256, 256, 256),
        deconv_kernel_sizes=(4, 4, 4),
        add_last_bnrelu: bool = False  # ✅ 控制最后一层是否加BN+ReLU
    ):
        super().__init__()
        assert out_channels % depth_size == 0, "out_channels 必须是 depth_size 的倍数"
        self.depth_size = depth_size

        deconv_layers = []
        prev_channels = in_channels

        for i, (out_c, kernel) in enumerate(zip(deconv_out_channels, deconv_kernel_sizes)):
            deconv_layers.append(
                nn.ConvTranspose2d(
                    prev_channels,
                    out_c,
                    kernel_size=kernel,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False
                )
            )
            is_last = (i == len(deconv_out_channels) - 1)
            if not is_last or (is_last and add_last_bnrelu):
                deconv_layers.append(nn.BatchNorm2d(out_c))
                deconv_layers.append(nn.ReLU(inplace=True))

            prev_channels = out_c

        self.deconv_layers = nn.Sequential(*deconv_layers)

        # final conv to output heatmaps
        self.final_layer = nn.Conv2d(prev_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv_layers(x)  # upsample
        x = self.final_layer(x)    # channel -> out_channels
        N, C, H, W = x.shape
        x = x.view(N, C // self.depth_size, self.depth_size, H, W)  # (N, K, D, H, W)

        # x = torch.sigmoid(x)  # optional, depends on your label range (0~1)  , 2025/04/08-2 Test01
        # x = F.softplus(x) / (F.softplus(x).max() + 1e-6) # 2025/04/08-2 Test02

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

    def __init__(self, in_channels, num_joints, depth_size, heatmap_size):
        super().__init__()
        self.heatmap_head = Heatmap3DHead(in_channels, num_joints*depth_size, depth_size)
        self.root_head = Heatmap1DHead(in_channels)
        self.hand_type_head = MultilabelClassificationHead(in_channels)
        self.heatmap_size = heatmap_size  # (D, H, W)

        self.loss_kpt = nn.MSELoss(reduction="none")
        # self.loss_kpt = KeypointMSELoss(
        #     use_target_weight=False,
        #     skip_empty_channel=True,
        #     loss_weight=1.0
        # )
        self.loss_kpt_xy = nn.MSELoss()  # 额外的 (x, y) 位置约束损失
        self.loss_root = nn.L1Loss()
        self.loss_hand_type = nn.BCEWithLogitsLoss()

    def forward(self, x):
        heatmaps = self.heatmap_head(x)  # 3D 关键点热力图
        root_depth = self.root_head(x)  # 根部深度预测
        hand_type = self.hand_type_head(x)  # 手部类别预测
        return heatmaps, root_depth, hand_type

    @staticmethod
    def get_heatmap_3d_maximum(heatmaps: torch.Tensor, image_size=None, depth_bound=400):
        """
        从 3D 热图中提取最大响应位置，作为预测关键点的坐标。

        输入:
          - heatmaps: (N, K, D, H, W) - 3D 关键点热图
          - image_size: tuple (width, height)，若为 None，则返回热图尺度下的点位置

        输出:
          - keypoints: (N, K, 3) -> (x, y, z)
          - scores: (N, K) -> 对应关键点的置信度
        """
        N, K, D, H, W = heatmaps.shape
        scores, indices = torch.max(heatmaps.view(N, K, -1), dim=2)
        depth_bound = depth_bound  # 设定 z 归一化范围
        # 转换 indices 为三维坐标 (x, y, z)
        x_heat = indices % W  # 热图 x 坐标
        y_heat = (indices // W) % H  # 热图 y 坐标
        z_heat = indices // (W * H)  # 热图 z 坐标

        # 若不提供原图尺寸，则返回热图坐标
        if image_size is None:
            keypoints = torch.stack((x_heat, y_heat, z_heat), dim=-1).float()
            return keypoints, scores

        # 若提供 image_size，则转换到原始图像坐标
        img_w, img_h = image_size

        keypoints = torch.zeros((N, K, 3), dtype=torch.float32, device=heatmaps.device)
        keypoints[:, :, 0] = x_heat.float() * img_w / W
        keypoints[:, :, 1] = y_heat.float() * img_h / H
        keypoints[:, :, 2] = ((z_heat.float() / D) - 0.5) * depth_bound

        return keypoints, scores

    def soft_argmax_2d(self,heatmap_3d , T = 0.001):
        # heatmap_3d: (N, K, D, H, W)
        N, K, D, H, W = heatmap_3d.shape

        # 求 z 方向上的最大值（聚合）
        xy_heatmap = torch.sum(heatmap_3d, dim=2)  # (N, K, H, W)
        xy_heatmap = xy_heatmap.view(N, K, -1)
        xy_heatmap = xy_heatmap - xy_heatmap.amax(dim=2, keepdim=True)  # ✅ 稳定 ############
        xy_heatmap = F.softmax(xy_heatmap / T, dim=2)  # ✅ dim=2 是 (H×W) 内部
        xy_heatmap = xy_heatmap.view(N, K, H, W)

        # 构造坐标网格
        device = heatmap_3d.device
        x_coords = torch.arange(W, device=device).float()
        y_coords = torch.arange(H, device=device).float()
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')  # (H, W)

        x_grid = x_grid.view(1, 1, H, W)
        y_grid = y_grid.view(1, 1, H, W)

        x = (xy_heatmap * x_grid).sum(dim=(2, 3))  # (N, K)
        y = (xy_heatmap * y_grid).sum(dim=(2, 3))  # (N, K)

        return torch.stack([x, y], dim=-1)  # (N, K, 2)

    @staticmethod
    def soft_argmax_3d(heatmap_3d , T = 0.05):
        """
        支持输入为 np.ndarray 或 torch.Tensor 的 soft-argmax 3D 实现。

        Args:
            heatmap_3d (Union[np.ndarray, Tensor]): (N, K, D, H, W) 或 (K, D, H, W)

        Returns:
            keypoints_3d (Tensor): (N, K, 3) -> (x, y, z)
            scores (Tensor): (N, K) -> 每个关键点的置信度
        """
        import numpy as np
        import torch
        import torch.nn.functional as F

        # 转为 Tensor
        if isinstance(heatmap_3d, np.ndarray):
            heatmap_3d = torch.from_numpy(heatmap_3d).float()

        # 补充 batch 维度
        if heatmap_3d.dim() == 4:  # (K, D, H, W) → (1, K, D, H, W)
            heatmap_3d = heatmap_3d.unsqueeze(0)

        N, K, D, H, W = heatmap_3d.shape

        # Softmax
        # heatmap = heatmap_3d.view(N * K, -1)
        # heatmap = F.softmax(heatmap / T, dim=1)
        # heatmap = heatmap.view(N, K, D, H, W)
        heatmap = heatmap_3d.view(N, K, -1)
        heatmap = heatmap - heatmap.amax(dim=2, keepdim=True)   # ✅ 稳定
        heatmap = F.softmax(heatmap / T, dim=2)  # ✅ 每个关键点内部做 softmax
        heatmap = heatmap.view(N, K, D, H, W)

        device = heatmap.device
        z_range = torch.linspace(0, D - 1, D, device=device)
        y_range = torch.linspace(0, H - 1, H, device=device)
        x_range = torch.linspace(0, W - 1, W, device=device)
        zz, yy, xx = torch.meshgrid(z_range, y_range, x_range, indexing="ij")

        xx = xx.view(1, 1, D, H, W)
        yy = yy.view(1, 1, D, H, W)
        zz = zz.view(1, 1, D, H, W)

        x = torch.sum(heatmap * xx, dim=(2, 3, 4))
        y = torch.sum(heatmap * yy, dim=(2, 3, 4))
        z = torch.sum(heatmap * zz, dim=(2, 3, 4))

        keypoints = torch.stack([x, y, z], dim=2)  # (N, K, 3)
        scores = torch.amax(heatmap, dim=(2, 3, 4))  # (N, K)

        return keypoints, scores

    def compute_loss(self, pred, target):
        """计算损失"""
        pred_heatmaps, pred_root, pred_hand_type = pred
        target_heatmaps, target_root, target_hand_type = target

        ######################################################
        # 2025/03/24 loss
        ######################################################
        # # print(f"heatmaps:{target_heatmaps}")
        #
        # # 获取 3D 热图的峰值 mask（只对 target 中高于某阈值的区域进行监督）
        # heatmap_mask = (target_heatmaps > 0.005).float()  # ✅ 你也可以尝试 0.01, 0.1 进行比较
        # # print(f"heatmap_mask:{heatmap_mask}")
        #
        # # 在计算损失前，对 target_hand_type 做 squeeze 操作
        # if target_hand_type.dim() == 3:
        #     target_hand_type = target_hand_type.squeeze(1)
        #
        # # 计算 3D heatmap loss（仅在 mask 内计算）
        # loss_kpt = (self.loss_kpt(pred_heatmaps, target_heatmaps) * heatmap_mask).mean()
        # loss_root = self.loss_root(pred_root, target_root)
        # loss_hand = self.loss_hand_type(pred_hand_type, target_hand_type)
        # # ========== (x, y) 位置监督：soft-argmax 2D  ==========
        # pred_xy = self.soft_argmax_2d(pred_heatmaps)  # (B, K, 2)
        # target_xy = self.soft_argmax_2d(target_heatmaps)  # (B, K, 2) from GT heatmap
        #
        # loss_kpt_xy = self.loss_kpt_xy(pred_xy, target_xy)
        ######################################################
        # 2025/03/25 loss
        ######################################################
        # # -------------------
        # # ✅ soft mask 替代 binary mask：更平滑、更结构化监督
        # soft_mask = target_heatmaps  # (N, K, D, H, W)，高斯结构就是 weight
        # # -------------------
        # # ✅ 防止 hand_type dim 不一致
        # if target_hand_type.dim() == 3:
        #     target_hand_type = target_hand_type.squeeze(1)
        # # -------------------
        # # ✅ 3D heatmap MSE loss（结构损失）
        # loss_kpt = (self.loss_kpt(pred_heatmaps, target_heatmaps) * soft_mask).mean()
        # # -------------------
        # # ✅ 位置级别精细监督（可选 soft-argmax）, 在train中选择什么时候使用这个loss
        # pred_xy = self.soft_argmax_2d(pred_heatmaps)
        # target_xy = self.soft_argmax_2d(target_heatmaps)
        # loss_kpt_xy = self.loss_kpt_xy(pred_xy, target_xy)
        # # -------------------
        # # ✅ Root depth L1 loss
        # loss_root = self.loss_root(pred_root, target_root)
        # # -------------------
        # # ✅ Hand type BCE loss
        # loss_hand = self.loss_hand_type(pred_hand_type, target_hand_type)
        ######################################################
        # 2025/04/08 loss 在之前的loss的基础上增加了一个信息熵来进行正则化，防止模型学习到的热图，因为loss_xy的调节坍塌成一个点。
        ######################################################
        # === soft mask 替代 binary mask：更平滑、更结构化监督
        soft_mask = target_heatmaps  # (N, K, D, H, W)

        #####################################################
        # 2025/04/10 soft_mask 可能是导致模型学习到的K维度的热图学习到一快的原因，直接使用binary mask 进行测试
        #####################################################
        # binary_mask = (target_heatmaps > 1e-3).float()

        #####################################################
        # 2025/04/10-1 soft_mask 可能是导致模型学习到的K维度的热图学习到一快的原因，直接不用mask 进行训练测试
        #####################################################
        #

        #####################################################
        # 2025/04/11 按照 batch 中不同样本，轮流监督不同的 joint channel
        #####################################################

        #####################################################
        # 2025/04/13 比较学习方式的通道层次监督
        #####################################################
        # contrastive_loss = ChannelContrastiveLoss(margin=4.0)
        # loss_contrastive = contrastive_loss(pred_heatmaps, target_heatmaps)

        #####################################################
        # 2025/04/14  soft_mask 容易导致模型学到整图耦合特征, 加入 tail regularization（尾部正则）
        #####################################################
        tail_mask = 1 - soft_mask


        # pred_heatmaps: (N, K, D, H, W)
        N, K, D, H, W = pred_heatmaps.shape
        # === Soft mask（就是原始 GT heatmap）
        # 初始化 mask，全 0
        joint_mask = torch.zeros_like(pred_heatmaps)
        # 为每个样本激活一个 joint channel
        for b in range(N):
            joint_idx = b % K
            joint_mask[b, joint_idx] = 1.0  # 只监督该 joint


        # === 防止 hand_type dim 不一致
        if target_hand_type.dim() == 3:
            target_hand_type = target_hand_type.squeeze(1)

        # === 3D heatmap MSE loss（结构监督）
        # loss_kpt = (self.loss_kpt(pred_heatmaps, target_heatmaps) * soft_mask).mean()     # (1) 2025/04/08
        # loss_kpt = (self.loss_kpt(pred_heatmaps, target_heatmaps) * binary_mask).mean()   # (2) 2025/04/10
        # loss_kpt = (self.loss_kpt(pred_heatmaps, target_heatmaps)).mean()                   # (3) 2025/04/10-1
        # loss_kpt = (self.loss_kpt(pred_heatmaps,target_heatmaps) * 250.0 * joint_mask).mean()   # (4) 2025/04/11
        # (5) 通道对比损失  , 2025/04/13
        # loss_kpt = loss_contrastive #通道对比损失 2025/04/13

        # (6) 组合 : 通道对比损失 + soft_mask   , 2025/04/13-1
        # loss_soft = (self.loss_kpt(pred_heatmaps, target_heatmaps) * soft_mask).mean()
        # loss_contrastive = contrastive_loss(pred_heatmaps, target_heatmaps)
        # loss_kpt = loss_soft + 0.01 * loss_contrastive

        # (7) tail regularization （尾部正则） , 2025/04/14
        loss_center = (self.loss_kpt(pred_heatmaps, target_heatmaps) * soft_mask).mean()
        # normalized_mask = soft_mask / 255.0
        # tail_mask = 1.0 - normalized_mask
        tail_mask = 255 - soft_mask
        loss_tail = (self.loss_kpt(pred_heatmaps, target_heatmaps) * tail_mask).mean()
        # α 是尾部区域的惩罚因子，建议 0.05 ~ 0.2
        alpha = 0.2
        loss_kpt = loss_center + alpha * loss_tail

        # === (x, y) soft-argmax 位置监督
        pred_xy = self.soft_argmax_2d(pred_heatmaps)
        target_xy = self.soft_argmax_2d(target_heatmaps)
        loss_kpt_xy = self.loss_kpt_xy(pred_xy, target_xy)

        # === 根深度监督
        # loss_root = self.loss_root(pred_root, target_root)

        # === 手类型监督
        # loss_hand = self.loss_hand_type(pred_hand_type, target_hand_type)

        # === 💡【新增】信息熵正则项（鼓励热图分布结构，不要塌缩）
        # 添加 softmax 归一化，防止直接用 raw logits
        # N, K, D, H, W = pred_heatmaps.shape
        # pred_softmax = F.softmax(pred_heatmaps.view(N * K, -1), dim=1).view(N, K, D, H, W)
        # entropy = -(pred_softmax * torch.log(pred_softmax + 1e-6)).mean()  # batch mean

        # print({"loss_kpt": loss_kpt, "loss_kpt_xy": loss_kpt_xy ,"loss_root": loss_root, "loss_hand_type": loss_hand})
        # return {"loss_kpt": loss_kpt, "loss_kpt_xy": loss_kpt_xy , "loss_root": loss_root, "loss_hand_type": loss_hand}

        return {
        "loss_center": loss_center,
        "loss_tail": loss_tail,
        "loss_kpt": loss_kpt,
        "loss_kpt_xy": loss_kpt_xy,
        # "loss_root": loss_root,
        # "loss_hand_type": loss_hand,
        # "loss_entropy": entropy,
        # "loss_contrast": loss_contrastive,
        # "loss_soft": loss_soft
    }


# #
# ** 测试**
# if __name__ == "__main__":
#     head = HandPoseHead(in_channels=512, num_joints=21, depth_size=64,heatmap_size=[64,64,64])
#     print(head)
#     x = torch.randn(5, 512, 8, 8)
#     heatmaps, root_depth, hand_type = head(x)
#     # print("输出形状:", heatmaps.shape, root_depth.shape, hand_type.shape)  # (B, out_channels, H, W) 输出形状: torch.Size([1, 21, 32, 56, 56]) torch.Size([1, 1]) torch.Size([1, 2])
#     # print(head.predict(x))torch.tensor(np.expand_dims(heatmaps_3d, axis=0))
#     y = torch.randn(5,512,8,8)
#     heatmaps2,_,_ = head(y)
#
#     # print(f"hand_type:{hand_type}")
#     hand_type=torch.tensor([[1.,0.]])
#     head.compute_loss(pred = [heatmaps, root_depth, hand_type ],
#                       target = [heatmaps, root_depth, hand_type ])
