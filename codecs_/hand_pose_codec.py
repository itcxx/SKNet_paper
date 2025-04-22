import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch
import torch.nn.functional as F
import numpy
# from .hand_3d_heatmap import *

# class HandPoseCodec:
#     """3D 手部关键点编码 & 解码"""

#     def __init__(self, 
#                  image_size=(256, 256), 
#                  heatmap_size=(64, 64, 64), 
#                  root_heatmap_size=64,
#                  depth_size=64,
#                  heatmap3d_depth_bound=400.0,
#                  root_depth_bound=400.0,
#                  sigma=2, 
#                  max_bound=1.0):
#         """
#         3D 关键点编码器 (生成 3D Heatmap) 和解码器 (Heatmap -> 3D 关键点)

#         Args:
#             image_size (tuple): 原始输入图像大小 (H, W)
#             heatmap_size (tuple): 3D 热图大小 (D, H, W)
#             root_heatmap_size (int): 根部深度的 Heatmap 大小
#             depth_size (int): 深度离散化的尺寸 (默认为 64)
#             heatmap3d_depth_bound (float): 3D 热图的深度边界
#             root_depth_bound (float): 根部深度边界
#             sigma (int): 生成高斯热图的标准差
#             max_bound (float): 3D 热图最大值
#         """
#         self.image_size = np.array(image_size)
#         self.heatmap_size = np.array(heatmap_size)
#         self.root_heatmap_size = root_heatmap_size
#         self.depth_size = depth_size
#         self.heatmap3d_depth_bound = heatmap3d_depth_bound
#         self.root_depth_bound = root_depth_bound
#         self.sigma = sigma
#         self.max_bound = max_bound
#         self.scale_factor = (np.array(image_size) / heatmap_size[:-1]).astype(np.float32)

#     def encode(self, keypoints, keypoints_visible, rel_root_depth, rel_root_valid, hand_type, hand_type_valid):
#         """
#         **编码 (Encode): 3D 关键点 -> 3D 热图**
        
#         Args:
#             keypoints (Tensor): 3D 关键点 (N, K, 3)
#             keypoints_visible (Tensor): 关键点可见性 (N, K)
#             rel_root_depth (float): 根部相对深度
#             rel_root_valid (float): 根部深度的有效性
#             hand_type (Tensor): 手部类型 (左: [1, 0], 右: [0, 1])
#             hand_type_valid (float): 手部类型的有效性

#         Returns:
#             dict: 包含 3D 热图、深度权重、手部类型等
#         """
#         # **生成 3D 高斯热图**
#         heatmaps, keypoint_weights = self.generate_3d_gaussian_heatmaps(
#             keypoints, keypoints_visible)

#         # **处理根部深度**
#         rel_root_depth = (rel_root_depth / self.root_depth_bound + 0.5) * self.root_heatmap_size
#         rel_root_valid = rel_root_valid * (rel_root_depth >= 0) * (rel_root_depth <= self.root_heatmap_size)

#         return {
#             "heatmaps": heatmaps,  # 3D 关键点热图
#             "keypoint_weights": keypoint_weights,  # 关键点的权重
#             "root_depth": rel_root_depth * np.ones(1, dtype=np.float32),
#             "root_depth_weight": rel_root_valid * np.ones(1, dtype=np.float32),
#             "type": hand_type,
#             "type_weight": hand_type_valid
#         }

#     def decode(self, heatmaps, root_depth, hand_type):
#         """
#         **解码 (Decode): 3D 热图 -> 3D 关键点**
        
#         Args:
#             heatmaps (Tensor): 3D 热图 (K, D, H, W)
#             root_depth (Tensor): 根部深度预测
#             hand_type (Tensor): 预测手部类型
        
#         Returns:
#             tuple:
#             - keypoints (Tensor): 预测 3D 关键点 (N, K, D)
#             - scores (Tensor): 关键点置信度 (N, K)
#             - rel_root_depth (Tensor): 预测的根部深度
#             - hand_type (Tensor): 预测的手部类型
#         """
#         keypoints, scores = self.get_heatmap_3d_maximum(heatmaps)

#         # **恢复深度坐标**
#         keypoints[..., 2] = (keypoints[..., 2] / self.depth_size - 0.5) * self.heatmap3d_depth_bound
#         keypoints[..., :2] = keypoints[..., :2] * self.scale_factor  # 归一化
        
#         # **解码根部深度**
#         rel_root_depth = (root_depth / self.root_heatmap_size - 0.5) * self.root_depth_bound

#         # **解码手部类型**
#         hand_type = (hand_type > 0).astype(int)

#         return keypoints, scores, rel_root_depth, hand_type

#     def generate_3d_gaussian_heatmaps(self, keypoints, keypoints_visible):
#         """
#         生成 3D 高斯热图
        
#         Args:
#             keypoints (Tensor): 3D 关键点 (N, K, 3)
#             keypoints_visible (Tensor): 关键点可见性 (N, K)

#         Returns:
#             heatmaps (Tensor): 生成的 3D 热图 (K, D, H, W)
#             keypoint_weights (Tensor): 关键点的置信度权重
#         """
#         print(f"keypoints: {keypoints.shape}")
#         N, K, _ = keypoints.shape
#         D, H, W = self.heatmap_size

#         heatmaps = np.zeros((K, D, H, W), dtype=np.float32)
#         keypoint_weights = np.zeros((N, K), dtype=np.float32)

#         for i in range(K):
#             if keypoints_visible[:, i].sum() == 0:
#                 continue
#             x, y, z = keypoints[0, i]
#             x, y = x / self.scale_factor[0], y / self.scale_factor[1]  # 归一化
#             z = (z / self.heatmap3d_depth_bound + 0.5) * D

#             # 生成高斯分布
#             xx, yy, zz = np.meshgrid(np.arange(W), np.arange(H), np.arange(D), indexing="ij")
#             heatmaps[i] = np.exp(-((xx - x) ** 2 + (yy - y) ** 2 + (zz - z) ** 2) / (2 * self.sigma ** 2))

#             keypoint_weights[:, i] = keypoints_visible[:, i]

#         return heatmaps, keypoint_weights

#     def get_heatmap_3d_maximum(self, heatmap):
#         """
#         **从 3D 热图中提取最大值的坐标**
        
#         Args:
#             heatmap (Tensor): (K, D, H, W) 形状的 3D 热图
        
#         Returns:
#             tuple:
#             - keypoints (Tensor): (K, 3) 关键点的 (x, y, z) 坐标
#             - scores (Tensor): (K) 关键点的置信度
#         """
#         K, D, H, W = heatmap.shape
#         reshaped_heatmap = heatmap.reshape(K, -1)
    
#         # NumPy 版本的最大值
#         scores = np.max(reshaped_heatmap, axis=-1)  
#         indices = np.argmax(reshaped_heatmap, axis=-1)
    
#         # 计算 (x, y, z) 坐标
#         keypoints = np.zeros((K, 3), dtype=np.float32)
#         keypoints[:, 0] = indices % W  # x
#         keypoints[:, 1] = (indices // W) % H  # y
#         keypoints[:, 2] = indices // (W * H)  # z
    
#         return keypoints, scores


import numpy as np
import torch
import torch.nn.functional as F

def softmax_np(x):
    """NumPy 实现的 softmax，用于归一化高斯分布"""
    x = x - np.max(x)
    print(f"max heatmap x: {x.max()}")
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

class HandPoseCodec:
    """3D 手部关键点编码 & 解码

    生成 3D 高斯热图，并可从热图中解码出 3D 关键点。

    Args:
        image_size (tuple): 原始输入图像大小 (width, height)
        heatmap_size (tuple): 3D 热图大小 (D, H, W)
        root_heatmap_size (int): 根部深度的 Heatmap 大小
        depth_size (int): 深度离散化的尺寸（通常与 heatmap 的 D 相同）
        heatmap3d_depth_bound (float): 3D 热图的深度边界（相机坐标下的最大深度范围）
        root_depth_bound (float): 根部深度边界
        sigma (int): 生成高斯热图的标准差
        max_bound (float): 热图最大值（用于归一化）
    """
    def __init__(self, 
                 image_size=(256, 256), 
                 heatmap_size=(64, 64, 64), 
                 root_heatmap_size=64,
                 depth_size=64,
                 heatmap3d_depth_bound=400.0,
                 root_depth_bound=400.0,
                 sigma=4,
                 max_bound=1.0,
                 offset = 0.125):
        self.image_size = np.array(image_size)  # (width, height)
        self.heatmap_size = np.array(heatmap_size)  # (D, H, W)
        self.root_heatmap_size = root_heatmap_size
        self.depth_size = depth_size
        self.heatmap3d_depth_bound = heatmap3d_depth_bound
        self.root_depth_bound = root_depth_bound
        self.sigma = sigma
        self.offset = offset #  深度值的偏移量，0.5表示深度为0的时候在中间的区域
        self.max_bound = max_bound
        # 计算 x, y 方向的缩放因子：image_size[0]对应宽度, heatmap_size[2]对应热图宽度;
        # image_size[1]对应高度, heatmap_size[1]对应热图高度
        self.scale_factor = (np.array([self.image_size[0], self.image_size[1]]) / self.heatmap_size[[2,1]]).astype(np.float32)

    def encode(self, keypoints, keypoints_visible, rel_root_depth, rel_root_valid, hand_type, hand_type_valid, focal=None, principal_pt=None):
        """
        编码 (Encode): 3D 关键点 -> 3D 热图

        如果提供相机内参，则先将相机坐标系下的关键点投影到图像坐标系，
        然后只取投影后的 (x, y) 作为 2D 关键点用于生成热图，
        同时用原始 keypoints 的深度作为 z。

        Args:
            keypoints (np.ndarray): 3D 关键点 (N, K, 3)
            keypoints_visible (np.ndarray): 关键点可见性 (N, K)
            rel_root_depth (np.ndarray or float): 根部相对深度
            rel_root_valid (np.ndarray or float): 根部深度有效性
            hand_type (np.ndarray): 手部类型 (例如 [1, 0] 表示左手)
            hand_type_valid (np.ndarray or float): 手部类型有效性
            focal (tuple or None): (fx, fy) 相机焦距
            principal_pt (tuple or None): (cx, cy) 相机主点
        Returns:
            dict: 包含生成的 3D 热图及相关权重信息
        """
        
        # print(f"original keypoints:{keypoints}")
        # print(f"rel_root depgh: {rel_root_depth}")
        
        if focal is not None and principal_pt is not None:
            # 对 batch 中第一个样本进行投影
            X = keypoints[0, :, 0]
            Y = keypoints[0, :, 1]
            Z = keypoints[0, :, 2]
            proj_x = (X / Z) * focal[0] + principal_pt[0]
            proj_y = (Y / Z) * focal[1] + principal_pt[1]
            keypoints_proj = keypoints.copy()
            keypoints_proj[0, :, 0] = proj_x
            keypoints_proj[0, :, 1] = proj_y
            # 仅取 2D 部分用于热图生成
            keypoints_2d = keypoints_proj[..., :2]
        else:
            keypoints_2d = keypoints[..., :2]
        # 分离深度信息
        keypoints_depth = keypoints[..., 2]  # shape: (N, K)

        # 生成 3D 高斯热图：使用 2D (x,y) 和深度 z 生成热图
        heatmaps, keypoint_weights = self.generate_3d_gaussian_heatmaps(keypoints_2d, keypoints_depth, keypoints_visible)

        # 处理根部深度，将真实深度映射到 [0, root_heatmap_size] 区间
        rel_root_depth_encoded = (rel_root_depth / self.root_depth_bound + self.offset) * self.root_heatmap_size
        rel_root_valid = rel_root_valid * (rel_root_depth_encoded >= 0) * (rel_root_depth_encoded <= self.root_heatmap_size)

        return {
            "heatmaps": heatmaps,            # 3D 关键点热图, shape: (K, D, H, W)
            "keypoint_weights": keypoint_weights,  # (N, K)
            "root_depth": rel_root_depth_encoded * np.ones(1, dtype=np.float32),
            "root_depth_weight": rel_root_valid * np.ones(1, dtype=np.float32),
            "type": hand_type,
            "type_weight": hand_type_valid
        }

    def decode(self, heatmaps, root_depth, hand_type):
        """
        解码 (Decode): 3D 热图 -> 3D 关键点

        Args:
            heatmaps (np.ndarray): 3D 热图 (K, D, H, W)
            root_depth (np.ndarray): 根部深度预测
            hand_type (np.ndarray): 预测手部类型

        Returns:
            tuple: (keypoints, scores, rel_root_depth, hand_type)
                keypoints: (K, 3) 预测 3D 关键点（图像坐标和深度）
                scores: (K,) 关键点置信度
                rel_root_depth: 预测的根部深度
                hand_type: 预测的手部类型
        """
        # keypoints, scores = self.get_heatmap_3d_maximum(heatmaps)
        keypoints, scores = self.soft_argmax_3d(heatmaps)
        # 恢复 x, y 坐标：将热图坐标还原为图像坐标
        keypoints[..., 0] = (keypoints[..., 0] / self.heatmap_size[2]) * self.image_size[0]
        keypoints[..., 1] = (keypoints[..., 1] / self.heatmap_size[1]) * self.image_size[1]
        # 恢复 z 坐标：反向映射 z: 先除以 D，然后减去 0.5，再乘以深度边界
        keypoints[..., 2] = ((keypoints[..., 2] / self.heatmap_size[0]) - self.offset) * self.heatmap3d_depth_bound

        # 解码根部深度
        rel_root_depth_decoded = (root_depth / self.root_heatmap_size - self.offset) * self.root_depth_bound
        # 解码手部类型 (阈值为 0)
        hand_type_decoded = (hand_type > 0).astype(int)

        return keypoints, scores, rel_root_depth_decoded, hand_type_decoded

    ##################################################################
    ## 采用的是热图图像像素坐标的相对位置进行热图生成（64，64），热图生成时过于稀疏
    ## 实验 2025/03/24， 2025/03/25 采用的是这个方法中的 01, 02
    ##################################################################
    # def generate_3d_gaussian_heatmaps(self, keypoints_2d, keypoints_depth, keypoints_visible):
    #     """
    #     生成 3D 高斯热图
    #
    #     Args:
    #         keypoints_2d (np.ndarray): 2D 像素关键点, shape: (N, K, 2)
    #         keypoints_depth (np.ndarray): 关键点深度, shape: (N, K)
    #         keypoints_visible (np.ndarray): 关键点可见性, shape: (N, K)
    #
    #     Returns:
    #         tuple:
    #             heatmaps (np.ndarray): 生成的 3D 热图, shape: (K, D, H, W)
    #             keypoint_weights (np.ndarray): 关键点的置信度权重, shape: (N, K)
    #     """
    #     N, K, _ = keypoints_2d.shape
    #     D, H, W = self.heatmap_size  # D: depth, H: height, W: width
    #
    #     heatmaps = np.zeros((K, D, H, W), dtype=np.float32)
    #     keypoint_weights = np.zeros((N, K), dtype=np.float32)
    #
    #     # 生成网格，顺序为 (D, H, W)
    #     zz, yy, xx = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing="ij")
    #     for i in range(K):
    #         if keypoints_visible[0, i] == 0:
    #             continue
    #
    #         x_img, y_img = keypoints_2d[0, i]
    #         z_cam = keypoints_depth[0, i]
    #
    #         # 坐标转换， 这里转换成了热图像素级别的坐标（<64)
    #         x = x_img * W / self.image_size[0]
    #         y = y_img * H / self.image_size[1]
    #         z = (z_cam / self.heatmap3d_depth_bound + 0.5) * D
    #
    #         # 裁剪防止越界
    #         x = np.clip(x, 0, W - 1)
    #         y = np.clip(y, 0, H - 1)
    #         z = np.clip(z, 0, D - 1)
    #
    #         heatmap = np.exp(-(((xx - x) ** 2 + (yy - y) ** 2 + (zz - z) ** 2) / (2 * self.sigma ** 2)))
    #
    #         # 高斯生成, method 01
    #         # heatmap /= np.sum(heatmap) + 1e-8  # 避免除0  ✅ Normalize
    #         # heatmap *= 255  # ✅ 增强峰值对比度
    #         # heatmap = np.clip(heatmap, 0, 1)  # 限制范围避免数值爆炸
    #
    #         # 高斯生成, method 02
    #         heatmap = heatmap / np.max(heatmap + 1e-6)  # 🌟 最大值归一为 1.0
    #
    #
    #         # print(f"max heatmap: {heatmap.max()}")
    #         heatmaps[i] = heatmap
    #         keypoint_weights[0, i] = 1.0
    #     return heatmaps, keypoint_weights

    ##################################################################
    ## 采用的图像比率的数值进行热图生成，防止热图过于稀疏
    ## 实验 2025/03/25-1
    ##################################################################
    # def generate_3d_gaussian_heatmaps(self, keypoints_2d, keypoints_depth, keypoints_visible):
    #     """
    #     生成 3D 高斯热图（使用归一化比例坐标，防止稀疏，提升训练稳定性）
    #
    #     Args:
    #         keypoints_2d (np.ndarray): 2D 像素关键点, shape: (N, K, 2)
    #         keypoints_depth (np.ndarray): 关键点深度, shape: (N, K)
    #         keypoints_visible (np.ndarray): 关键点可见性, shape: (N, K)
    #
    #     Returns:
    #         tuple:
    #             heatmaps (np.ndarray): 生成的 3D 热图, shape: (K, D, H, W)
    #             keypoint_weights (np.ndarray): 关键点的置信度权重, shape: (N, K)
    #     """
    #     N, K, _ = keypoints_2d.shape
    #     D, H, W = self.heatmap_size  # D: depth, H: height, W: width
    #
    #     heatmaps = np.zeros((K, D, H, W), dtype=np.float32)
    #     keypoint_weights = np.zeros((N, K), dtype=np.float32)
    #
    #     # 使用归一化网格 [0, 1] 坐标
    #     zz, yy, xx = np.meshgrid(
    #         np.linspace(0, 1, D),
    #         np.linspace(0, 1, H),
    #         np.linspace(0, 1, W),
    #         indexing="ij"
    #     )
    #
    #     # self.sigma = 0.05  # 在归一化坐标系下的标准差（覆盖大约 5% 空间）
    #     min_value_clip = 0.01  # 防止过于稀疏
    #
    #     for i in range(K):
    #         if keypoints_visible[0, i] == 0:
    #             continue
    #
    #         x_img, y_img = keypoints_2d[0, i]
    #         z_cam = keypoints_depth[0, i]
    #
    #         # === 映射到归一化比例空间 [0, 1] ===
    #         x = x_img / self.image_size[0]
    #         y = y_img / self.image_size[1]
    #         z = (z_cam / self.heatmap3d_depth_bound + self.offset)
    #
    #         x = np.clip(x, 0, 1)
    #         y = np.clip(y, 0, 1)
    #         z = np.clip(z, 0, 1)
    #
    #         # === 高斯生成（在比例空间下计算） ===
    #         heatmap = np.exp(-(((xx - x) ** 2 + (yy - y) ** 2 + (zz - z) ** 2) / (2 * self.sigma ** 2)))
    #
    #         # === 防止梯度消失：最低值裁剪（提升尾部响应） ===
    #         heatmap = np.clip(heatmap, min_value_clip, 1.0)
    #
    #         # === 最大值归一化（控制数值范围） ===
    #         heatmap = heatmap / (np.max(heatmap) + 1e-6)
    #
    #         heatmaps[i] = heatmap
    #         keypoint_weights[0, i] = 1.0
    #
    #     return heatmaps, keypoint_weights
    #
    ##################################################################
    ## 按照Interhand论文中的形式生成热图直接 * 255
    ## 实验 2025/04/09
    ##################################################################
    def generate_3d_gaussian_heatmaps(self, keypoints_2d, keypoints_depth, keypoints_visible):
        """
        生成 InterHand 论文风格的 3D 高斯热图（在体素坐标系中生成，并乘以255）

        Args:
            keypoints_2d (np.ndarray): 2D 像素关键点, shape: (N, K, 2)
            keypoints_depth (np.ndarray): 关键点深度, shape: (N, K)
            keypoints_visible (np.ndarray): 关键点可见性, shape: (N, K)

        Returns:
            tuple:
                heatmaps (np.ndarray): 生成的 3D 热图, shape: (K, D, H, W)
                keypoint_weights (np.ndarray): 关键点的置信度权重, shape: (N, K)
        """
        N, K, _ = keypoints_2d.shape
        D, H, W = self.heatmap_size  # (depth, height, width)

        heatmaps = np.zeros((K, D, H, W), dtype=np.float32)
        keypoint_weights = np.zeros((N, K), dtype=np.float32)

        # 创建 3D 网格 (voxel 坐标)
        zz, yy, xx = np.meshgrid(
            np.arange(D), np.arange(H), np.arange(W),
            indexing='ij'
        )

        for i in range(K):
            if keypoints_visible[0, i] == 0:
                continue

            # 将关键点坐标映射到 voxel 空间
            x = keypoints_2d[0, i, 0] * W / self.image_size[0]
            y = keypoints_2d[0, i, 1] * H / self.image_size[1]
            z = (keypoints_depth[0, i] / self.heatmap3d_depth_bound + self.offset) * D

            if not (0 <= x < W and 0 <= y < H and 0 <= z < D):
                continue

            sigma = 2.5
            # 生成高斯热图 (注意 sigma 是以 voxel 为单位)
            heatmap = np.exp(-((xx - x) ** 2 + (yy - y) ** 2 + (zz - z) ** 2) / (sigma ** 2))

            # 按照论文做法乘以 255
            heatmap = heatmap * 255.0

            heatmaps[i] = heatmap
            keypoint_weights[0, i] = 1.0

        return heatmaps, keypoint_weights

    def soft_argmax_3d(self, heatmap_3d):
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
        T = 0.01

        # Softmax
        heatmap = heatmap_3d.view(N * K, -1)
        heatmap = F.softmax(heatmap / T, dim=1)
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

    # def get_heatmap_3d_maximum(self, heatmap):
    #     """
    #     从 3D 热图中提取最大值的坐标
    #
    #     Args:
    #         heatmap (np.ndarray): 3D 热图, shape: (K, D, H, W)
    #
    #     Returns:
    #         tuple:
    #             keypoints (np.ndarray): (K, 3) 关键点的 (x, y, z) 坐标（热图坐标）
    #             scores (np.ndarray): (K,) 关键点的置信度
    #     """
    #     K, D, H, W = heatmap.shape
    #     reshaped = heatmap.reshape(K, -1)
    #     scores = np.max(reshaped, axis=-1)
    #     indices = np.argmax(reshaped, axis=-1)
    #     keypoints = np.zeros((K, 3), dtype=np.float32)
    #     # 网格顺序为 (D, H, W)：
    #     keypoints[:, 0] = indices % W                   # x 坐标
    #     keypoints[:, 1] = (indices // W) % H              # y 坐标
    #     keypoints[:, 2] = indices // (W * H)              # z 坐标
    #     return keypoints, scores

# if __name__ == "__main__":
#     # 测试代码
#     N, K, _ = 2, 21, 3
#     # 随机生成 3D 关键点，其中 x,y 为图像像素坐标，z 为深度（范围 -200 到 200）
#     keypoints_3d = np.random.uniform(0, 480, size=(N, K, 3)).astype(np.float32)
#     keypoints_3d[..., 2] = np.random.uniform(-200, 200, size=(N, K)).astype(np.float32)
#     print(f"keypoints_3d:{keypoints_3d}")
#     keypoints_visible = np.ones((N, K), dtype=np.float32)
#     rel_root_depth = np.array([100.0], dtype=np.float32)
#     rel_root_valid = np.array([1.0], dtype=np.float32)
#     hand_type = np.array([[1, 0]], dtype=np.float32)
#     hand_type_valid = np.array([[1.0]], dtype=np.float32)
#
#     # 测试时设置 image_size 与 heatmap_size
#     codec = HandPoseCodec(image_size=(640,480), heatmap_size=(64,64,64))
#     focal = (800,800)
#     principal_pt = (320,240)
#     encoded = codec.encode(keypoints_3d, keypoints_visible, rel_root_depth, rel_root_valid, hand_type, hand_type_valid)
#     heatmaps = encoded["heatmaps"]
#     print("Heatmaps shape:", heatmaps.shape)
#
#     decoded = codec.decode(heatmaps, np.expand_dims(rel_root_depth, axis=0), np.expand_dims(hand_type, axis=0))
#     decode_keypoints, decode_scores, decode_rel_root_depth, decode_hand_type = decoded
#     print("Decoded keypoints:\n", decode_keypoints)
#     print("Scores:\n", decode_scores)
#     print("Root Depth:\n", decode_rel_root_depth)
#     print("Hand Type:\n", decode_hand_type)
