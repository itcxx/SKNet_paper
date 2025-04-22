import os
import sys
# 添加项目根目录到 PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image
import io
import cv2
from models.heads.hand_pose_head import *
import torch


# 以下辅助函数用于反归一化和绘制关键点/热图
# def unnormalize(img, mean, std):
#     """反归一化，将 [0,1] 的图像还原到 [0,255]"""
#     img = img * std + mean
#     img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
#     return img


def unnormalize(img, mean, std):
    """
    自动兼容3通道或6通道图像的反归一化。

    参数:
        img: np.ndarray 或 torch.Tensor，shape 可以是 (C, H, W) 或 (H, W, C)
        mean: list 或 1D tensor，例如 [0.485, 0.456, 0.406]
        std: list 或 1D tensor，例如 [0.229, 0.224, 0.225]
    返回:
        反归一化后的 np.uint8 图像，shape 保持不变。
    """
    # 转成 numpy
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    img = img.astype(np.float32)

    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    # 如果 img 是 6 通道（cat 了两个 3 通道图），则扩展 mean/std
    if img.shape[0] == 6 or img.shape[-1] == 6:
        if len(mean) == 3:
            mean = np.concatenate([mean, mean])
            std = np.concatenate([std, std])

    # 检查通道位置
    if img.shape[0] == len(mean):  # (C, H, W)
        for c in range(len(mean)):
            img[c] = img[c] * std[c] + mean[c]
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    elif img.shape[-1] == len(mean):  # (H, W, C)
        for c in range(len(mean)):
            img[..., c] = img[..., c] * std[c] + mean[c]
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Image channels ({img.shape}) do not match mean/std shape ({len(mean)})")

    return img

# def draw_keypoints(img, keypoints):
#     """在图像上绘制 2D 关键点，keypoints shape: (K,3)"""
#     for x, y, v in keypoints:
#         if v > 0:  # 仅绘制可见关键点
#             cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)
#     return img

def draw_keypoints(img, keypoints):
    """在图像上绘制 2D 关键点，keypoints shape: (K, 3) or (K, 2)"""
    h, w = img.shape[:2]
    for kp in keypoints:
        if len(kp) == 3:
            x, y, _ = kp
        else:
            x, y = kp
        if 0 <= int(x) < w and 0 <= int(y) < h:
            cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)
    return img

def visualize_heatmaps(img, heatmaps_3d):
    """
    生成 3D 热图的 2D 最大投影，并将其伪彩色叠加到图像上
    Args:
        img (numpy.ndarray): 原图 (H, W, 3)
        heatmaps_3d (np.ndarray): 3D 热图 (K, D, H, W)
    Returns:
        img_overlay (numpy.ndarray): 叠加后的可视化图像
    """
    K, D, H, W = heatmaps_3d.shape
    # 沿深度维度取最大值 (得到 2D 热图，形状 (K, H, W))
    heatmaps_2d = np.max(heatmaps_3d, axis=1)
    # 对所有关键点的热图求均值 (H, W)
    heatmaps_2d = np.mean(heatmaps_2d, axis=0)
    # 归一化并转换为 uint8
    heatmaps_2d = (heatmaps_2d - heatmaps_2d.min()) / (heatmaps_2d.max() - heatmaps_2d.min() + 1e-5) * 255
    heatmaps_2d = heatmaps_2d.astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmaps_2d, cv2.COLORMAP_JET)
    heatmap_color = cv2.resize(heatmap_color, (img.shape[1], img.shape[0]))
    img_overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
    return img_overlay

def plot_3d_keypoints_to_image(keypoints_3d, title="3D Keypoints", fig_size=(4,4)):
    """
    用 matplotlib 绘制 3D 散点图，然后把结果转换成 (H, W, 3) 的BGR图像返回。
    keypoints_3d: (K, 3) 的 numpy 数组，可能是 GT 也可能是 Pred
    title: 绘图标题
    fig_size: 绘图 figure 的大小 (宽, 高)，可根据需求调整
    """
    K, dims = keypoints_3d.shape
    assert dims == 3, f"keypoints_3d.shape must be (K,3), got {keypoints_3d.shape}"

    # 1) 简单的 min-max 归一化，防止数值跨度太大导致图像不好看
    min_vals = np.min(keypoints_3d, axis=0)
    max_vals = np.max(keypoints_3d, axis=0)
    ranges = (max_vals - min_vals) + 1e-9
    norm_pts = (keypoints_3d - min_vals) / ranges  # -> [0,1]

    # 2) Matplotlib 画 3D 散点图
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(norm_pts[:,0], norm_pts[:,1], norm_pts[:,2], c='r', marker='o', s=30)
    ax.set_title(title)
    ax.set_xlim([0,1]); ax.set_ylim([0,1]); ax.set_zlim([0,1])
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    # 3) 保存到内存中，再转成 numpy BGR
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)

    # 用 PIL Image 打开，再转成 numpy
    buf.seek(0)
    pil_img = Image.open(buf).convert('RGB')
    plot_np = np.array(pil_img)  # (H, W, 3), RGB
    plot_bgr = cv2.cvtColor(plot_np, cv2.COLOR_RGB2BGR)
    return plot_bgr

def save_visualization_results(imgs, batch, epoch, iteration_count, vis_save_path, config, model):
    """
    在原有2D可视化(左:GT+热图, 右:Pred+热图)的右边，再拼接两个3D散点图:
      - 第一个3D图: GT 3D关键点
      - 第二个3D图: Pred 3D关键点
    并将最终大图一起保存.

    imgs (Tensor): 当前 batch 的输入图像，(B,C,H,W)
    batch (dict): 数据字典，其中 "meta_info" 中存有 "heatmaps", "root_depth" 等
    epoch, iteration_count: 标记当前训练进度
    vis_save_path (str): 保存结果的目录
    config (dict): 配置，含 data_preprocessor
    model (nn.Module): 当前训练模型
    """
    # 1. 还原第一个样本的原图
    sample_img = imgs[0].detach().cpu()  # (C, H, W)
    sample_img_np = sample_img.permute(1,2,0).numpy()  # (H, W, C)

    mean = np.array(config["data_preprocessor"]["mean"]) / 255.0
    std  = np.array(config["data_preprocessor"]["std"]) / 255.0
    sample_img_np = unnormalize(sample_img_np, mean, std)
    sample_img_np = sample_img_np[...,:3] #防止是融合的图片有6通道只截取前3通道
    sample_img_np = cv2.cvtColor(sample_img_np, cv2.COLOR_RGB2BGR)

    # 2. 绘制GT 2D Keypoints & (如有)GT 3D heatmap投影
    gt_keypoints_2d = batch["keypoints_2d"][0].detach().cpu().numpy()  # (K, 3)
    vis_gt = sample_img_np.copy()
    vis_gt = draw_keypoints(vis_gt, gt_keypoints_2d)

    gt_heatmaps = batch["meta_info"].get("heatmaps", None)
    if gt_heatmaps is not None:
        gt_hm_3d = gt_heatmaps[0].detach().cpu().numpy()  # (K, D, H, W)
        vis_gt_heatmap = visualize_heatmaps(sample_img_np.copy(), gt_hm_3d)
    else:
        vis_gt_heatmap = vis_gt

    # 3. 预测结果 (2D)
    pred_heatmaps_batch, pred_root, pred_hand_type = model(imgs)
    heatmaps_pred = pred_heatmaps_batch[0].detach().cpu()  # (K, D, H, W)

    # pred_keypoints_3d, _ = HandPoseHead.get_heatmap_3d_maximum(heatmaps_pred.unsqueeze(0),
    #                                                            image_size=[256,256])
    pred_keypoints_3d, _ = HandPoseHead.soft_argmax_3d(heatmaps_pred)

    # 原始图像尺寸
    orig_H, orig_W = sample_img_np.shape[:2]  # (H, W)
    # 热图尺寸（你可以直接从 heatmaps_pred 的 shape 获取）
    _, D, H, W = heatmaps_pred.shape
    # 缩放系数 (注意：这里只是 2D 图像方向)
    scale_x = orig_W / W
    scale_y = orig_H / H
    scale_z = 1.0  # 深度方向一般不缩放，或看你要不要做投影/重建

    # 缩放3D关键点坐标（默认 keypoints 为 [x, y, z]）
    pred_keypoints_3d[..., 0] *= scale_x  # x
    pred_keypoints_3d[..., 1] *= scale_y  # y
    # pred_keypoints_3d[..., 2] *= scale_z  # z：如果你有需要映射 z，可以自定义 scale_z

    pred_keypoints_3d = pred_keypoints_3d[0].numpy()  # (K,3)

    # 在2D图上画预测关节 (x,y)
    vis_pred = sample_img_np.copy()
    vis_pred = draw_keypoints(vis_pred, pred_keypoints_3d)
    vis_pred = visualize_heatmaps(vis_pred, heatmaps_pred.numpy())

    # 将GT可视图和Pred可视图拼接 (左右)
    combined_vis_2d = cv2.hconcat([vis_gt_heatmap, vis_pred])  # shape (H, 2W, 3)

    # 4. 如果 batch 里有 GT 3D keypoints，就可单独可视化
    gt_keypoints_3d = batch.get("keypoints_3d", None)
    if gt_keypoints_3d is not None:
        gt_keypoints_3d = gt_keypoints_3d[0].cpu().numpy()  # (K,3)
        img_3d_gt = plot_3d_keypoints_to_image(gt_keypoints_3d, title="GT 3D Points")
    else:
        # 如果没有 GT 3D，则用空图替代 (或直接不拼接)
        img_3d_gt = np.zeros((combined_vis_2d.shape[0], combined_vis_2d.shape[0], 3), dtype=np.uint8)

    # 5. 预测 3D 可视化
    img_3d_pred = plot_3d_keypoints_to_image(pred_keypoints_3d, title="Pred 3D Points")

    # 6. 再把两个3D图拼到2D图右边: [2D | GT 3D | Pred 3D]
    #    注意：不同图像可能 H 不同，需要先把 3D 图的高度 resize 到和 combined_vis_2d 一样
    h_2d = combined_vis_2d.shape[0]
    # 保持 3D 图的纵横比，只 resize 高度
    def resize_height_keep_ratio(img, target_h):
        h, w = img.shape[:2]
        ratio = target_h / h
        new_w = int(w * ratio)
        return cv2.resize(img, (new_w, target_h))

    img_3d_gt_resized   = resize_height_keep_ratio(img_3d_gt,   h_2d)
    img_3d_pred_resized = resize_height_keep_ratio(img_3d_pred, h_2d)

    combined_final = cv2.hconcat([combined_vis_2d, img_3d_gt_resized, img_3d_pred_resized])

    # 7. 保存最终图像
    save_img_path = os.path.join(vis_save_path, f"epoch_{epoch+1}_iter_{iteration_count}.jpg")
    cv2.imwrite(save_img_path, combined_final)
    # print(f"可视化结果已保存: {save_img_path}")