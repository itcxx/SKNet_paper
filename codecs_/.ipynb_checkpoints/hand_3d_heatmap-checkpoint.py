import numpy as np
import torch
import torch.nn.functional as F

def generate_3d_gaussian_heatmaps(heatmap_size, keypoints, sigma=2, max_bound=1.0):
    """
    生成 3D 关键点的热图（高斯分布）

    Args:
        heatmap_size (tuple): (D, H, W) 3D 热图大小
        keypoints (Tensor): (N, K, 3) 关键点坐标
        sigma (float): 高斯标准差
        max_bound (float): 热图最大值

    Returns:
        Tensor: (N, K, D, H, W) 3D 热图
    """
    N, K, _ = keypoints.shape
    D, H, W = heatmap_size
    heatmaps = torch.zeros((N, K, D, H, W), dtype=torch.float32)

    for n in range(N):
        for k in range(K):
            x, y, z = keypoints[n, k]
            x = int((x + 1) / 2 * (W - 1))  # 映射到 (0, W-1)
            y = int((y + 1) / 2 * (H - 1))
            z = int((z + 1) / 2 * (D - 1))

            for dz in range(-sigma, sigma + 1):
                for dy in range(-sigma, sigma + 1):
                    for dx in range(-sigma, sigma + 1):
                        xi, yi, zi = x + dx, y + dy, z + dz
                        if 0 <= xi < W and 0 <= yi < H and 0 <= zi < D:
                            heatmaps[n, k, zi, yi, xi] = np.exp(
                                -(dx**2 + dy**2 + dz**2) / (2 * sigma**2))

    heatmaps = heatmaps / heatmaps.max() * max_bound  # 归一化
    return heatmaps

def decode_3d_heatmap(heatmaps):
    """
    3D 热图解码 3D 关键点坐标

    Args:
        heatmaps (Tensor): (N, K, D, H, W) 3D 热图

    Returns:
        Tensor: (N, K, 3) 关键点 (x, y, z) 坐标
    """
    N, K, D, H, W = heatmaps.shape
    joints_3d = torch.zeros((N, K, 3))

    for n in range(N):
        for k in range(K):
            heatmap = heatmaps[n, k]  # (D, H, W)
            max_idx = torch.argmax(heatmap)  # 找最大值索引
            z, y, x = np.unravel_index(max_idx.cpu().numpy(), (D, H, W))

            # Soft-argmax 平滑坐标
            dz, dy, dx = torch.meshgrid(
                torch.linspace(-1, 1, D),
                torch.linspace(-1, 1, H),
                torch.linspace(-1, 1, W)
            )
            weighted_x = (heatmap * dx).sum() / heatmap.sum()
            weighted_y = (heatmap * dy).sum() / heatmap.sum()
            weighted_z = (heatmap * dz).sum() / heatmap.sum()

            joints_3d[n, k] = torch.tensor([weighted_x, weighted_y, weighted_z])

    return joints_3d
