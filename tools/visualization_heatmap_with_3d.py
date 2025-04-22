import os
import sys
import argparse
import yaml
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# 添加项目根目录到 PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from datasets import HandPoseDataset,LightHandDataset
from utils.build_pipeline import build_pipeline  # 根据配置文件构建 pipeline
from models.build_model import build_model
from utils.collate import custom_collate_fn  # 自定义 collate_fn
from models.heads.hand_pose_head import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import io

# visualize_gt_only.py
import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

from datasets import HandPoseDataset, LightHandDataset
from utils.build_pipeline import build_pipeline
from utils.collate import custom_collate_fn

def parse_args():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")
    parser = argparse.ArgumentParser(description="Visualize Ground Truth only")
    parser.add_argument("--config", type=str, default=CONFIG_PATH)
    return parser.parse_args()

def unnormalize(img, mean, std):
    img = img * std + mean
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img

def draw_keypoints(img, keypoints):
    for x, y, v in keypoints:
        if v > 0:
            cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)
    return img

def visualize_heatmaps(img, heatmaps_3d):
    K, D, H, W = heatmaps_3d.shape
    heatmaps_2d = np.max(heatmaps_3d, axis=1)
    heatmaps_2d = np.mean(heatmaps_2d, axis=0)
    heatmaps_2d = (heatmaps_2d - heatmaps_2d.min()) / (heatmaps_2d.max() - heatmaps_2d.min() + 1e-5) * 255
    heatmaps_2d = heatmaps_2d.astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmaps_2d, cv2.COLORMAP_JET)
    heatmap_color = cv2.resize(heatmap_color, (img.shape[1], img.shape[0]))
    return cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

def plot_3d_keypoints_to_image(keypoints_3d, title="3D Keypoints", fig_size=(4,4)):
    min_vals = np.min(keypoints_3d, axis=0)
    max_vals = np.max(keypoints_3d, axis=0)
    norm_pts = (keypoints_3d - min_vals) / (max_vals - min_vals + 1e-9)

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(norm_pts[:,0], norm_pts[:,1], norm_pts[:,2], c='r', marker='o', s=30)
    ax.set_title(title)
    ax.set_xlim([0,1]); ax.set_ylim([0,1]); ax.set_zlim([0,1])
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    ax.view_init(elev=10, azim=-90)  # ✅ 设置视角方向
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    pil_img = Image.open(buf).convert('RGB')
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# def visualize_gt_and_mock_pred(img_tensor, batch, config):
#     img = img_tensor.permute(1,2,0).numpy()
#     mean = np.array(config["data_preprocessor"]["mean"]) / 255.0
#     std = np.array(config["data_preprocessor"]["std"]) / 255.0
#     img = unnormalize(img, mean, std)
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#
#     keypoints_2d = batch["keypoints_2d"][0].cpu().numpy()
#     keypoints_3d = batch["keypoints_2d"][0].cpu().numpy()
#     heatmaps_3d = batch["meta_info"]["heatmaps"][0].cpu().numpy()
#
#     gt_2d = draw_keypoints(img.copy(), keypoints_2d)
#     gt_heatmap = visualize_heatmaps(img.copy(), heatmaps_3d)
#     gt_3d = plot_3d_keypoints_to_image(keypoints_3d, title="GT 3D")
#
#     # mock pred with GT
#     pred_2d = gt_2d.copy()
#     pred_heatmap = gt_heatmap.copy()
#     pred_3d = gt_3d.copy()
#
#     # resize all to same height
#     h = img.shape[0]
#     def resize_keep_height(im):
#         return cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))
#
#     concat = cv2.hconcat([
#         resize_keep_height(gt_heatmap),
#         resize_keep_height(pred_heatmap),
#         resize_keep_height(gt_3d),
#         resize_keep_height(pred_3d)
#     ])
#
#     cv2.imshow("[GT Heatmap | Pred Heatmap | GT 3D | Pred 3D]", concat)
#     cv2.waitKey(0)

def visualize_gt_and_mock_pred(img_tensor, batch, config):
    img = img_tensor.permute(1,2,0).numpy()
    mean = np.array(config["data_preprocessor"]["mean"]) / 255.0
    std = np.array(config["data_preprocessor"]["std"]) / 255.0
    img = unnormalize(img, mean, std)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    keypoints_2d = batch["keypoints_2d"][0].cpu().numpy()  # (K, 3)
    keypoints_3d = batch["keypoints_3d"][0].cpu().numpy()  # (K, 3)
    heatmaps_3d = batch["meta_info"]["heatmaps"][0].cpu().numpy()  # (K, D, H, W)

    # --- Ground Truth 可视化 ---
    gt_2d = draw_keypoints(img.copy(), keypoints_2d)
    gt_heatmap = visualize_heatmaps(img.copy(), heatmaps_3d)
    gt_3d = plot_3d_keypoints_to_image(keypoints_3d, title="GT 3D")

    # --- 模拟预测值（加噪声） ---
    noisy_2d = keypoints_2d.copy()
    noisy_3d = keypoints_3d.copy()

    # 对可见点添加噪声
    visible_mask = noisy_2d[:, 2] > 0
    noisy_2d[visible_mask, 0:2] += np.random.normal(scale=2.0, size=(visible_mask.sum(), 2))
    noisy_3d += np.random.normal(scale=0.0002, size=noisy_3d.shape)

    # 模拟热图扰动
    noisy_heatmaps = heatmaps_3d + np.random.normal(scale=0.05, size=heatmaps_3d.shape)
    noisy_heatmaps = np.clip(noisy_heatmaps, 0.0, 1.0)

    pred_2d = draw_keypoints(img.copy(), noisy_2d)
    pred_heatmap = visualize_heatmaps(img.copy(), noisy_heatmaps)
    pred_3d = plot_3d_keypoints_to_image(noisy_3d, title="Pred 3D ")

    # --- 拼接可视化图像 ---
    h = img.shape[0]
    def resize_keep_height(im):
        return cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))

    concat = cv2.hconcat([
        resize_keep_height(gt_heatmap),
        resize_keep_height(pred_heatmap),
        resize_keep_height(gt_3d),
        resize_keep_height(pred_3d)
    ])

    cv2.imshow("[GT Heatmap | Pred Heatmap | GT 3D | Pred 3D ]", concat)
    cv2.waitKey(0)

def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    transform = build_pipeline(config, pipeline_key="train_pipeline")
    dataset_name = config.get("dataset_name", "Interhand99k")

    if dataset_name == "Interhand99k":
        dataset = LightHandDataset(config_path=args.config, mode="val", transform=transform)
    elif dataset_name == "HandPose":
        dataset = HandPoseDataset(config_path=args.config, mode="val", transform=transform)
    else:
        raise ValueError("Unsupported dataset")

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=custom_collate_fn)

    for idx, batch in enumerate(loader):
        img_tensor = batch["img"][0]
        visualize_gt_and_mock_pred(img_tensor, batch, config)

if __name__ == "__main__":
    main()
