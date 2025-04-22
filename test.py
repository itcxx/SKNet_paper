import os
import sys
import argparse
import yaml
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# 添加项目根目录到 PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from datasets import HandPoseDataset, LightHandDataset, HandPoseDatasetTw
from utils.build_pipeline import build_pipeline  # 根据配置文件构建 pipeline
from models.build_model import build_model
from utils.collate import custom_collate_fn  # 自定义 collate_fn
from models.heads.hand_pose_head import HandPoseHead
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import io

def visualize_keypoint_maps(
    heatmaps_3d: np.ndarray,
    num_joints_to_show: int = 5,
    method: str = "max",  # "max" 或 "sum"
    alpha: float = 0.7,
    save: bool = False,
    save_path: str = "keypoint_maps.png"
):
    """
    可视化每个关键点在 H×W 平面的响应图（聚合深度维度）

    Args:
        heatmaps_3d (np.ndarray): (K, D, H, W)
        num_joints_to_show (int): 显示的关键点数量
        method (str): 聚合方法（"max" 或 "sum"）
        alpha (float): 图像透明度
        save (bool): 是否保存图像
        save_path (str): 保存路径
    """
    K, D, H, W = heatmaps_3d.shape
    num_show = min(K, num_joints_to_show)

    if method == "max":
        proj_maps = np.max(heatmaps_3d, axis=1)  # (K, H, W)
    elif method == "sum":
        proj_maps = np.sum(heatmaps_3d, axis=1)
    else:
        raise ValueError("method 仅支持 'max' 或 'sum'")

    # 归一化
    proj_maps = proj_maps - proj_maps.min(axis=(1, 2), keepdims=True)
    proj_maps = proj_maps / (proj_maps.max(axis=(1, 2), keepdims=True) + 1e-6)

    # 可视化
    fig, axes = plt.subplots(1, num_show, figsize=(4 * num_show, 4))
    if num_show == 1:
        axes = [axes]

    for i in range(num_show):
        axes[i].imshow(proj_maps[i], cmap='jet', alpha=alpha)
        axes[i].set_title(f"Keypoint {i}")
        axes[i].axis("off")

    plt.suptitle(f"Per-Keypoint Heatmaps ({method}-projection over depth)", fontsize=16)
    plt.tight_layout()

    if save:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Keypoint map visualization saved to: {save_path}")
    plt.show()


def soft_argmax_3d(heatmap_3d):
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
    # heatmap = heatmap_3d.view(N * K, -1)
    # heatmap = F.softmax(heatmap / T, dim=1)
    # heatmap = heatmap.view(N, K, D, H, W)
    heatmap = heatmap_3d.view(N, K, -1)
    heatmap = heatmap - heatmap.amax(dim=2, keepdim=True)  # ✅ 稳定
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

# def parse_args():
#     parser = argparse.ArgumentParser(description="3D Hand Pose Inference")
#     parser.add_argument("--config", type=str, default="./config/config_Interhand99k_model01.yaml", help="Path to config file")
#     parser.add_argument("--resume", type=str, default=None, help="Path to pre-trained checkpoint")
#     # distributed training 参数
#     parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed inference")
#     # 新增参数：是否直接显示结果，以及每次最多显示多少张图片
#     parser.add_argument("--show", default=True, help="如果设置，则直接在窗口中显示推测结果，而不是仅保存")
#     parser.add_argument("--max_show", type=int, default=10, help="最多显示的图片数量")
#     return parser.parse_args()

def parse_args():
    parser = argparse.ArgumentParser(description="3D Hand Pose Inference")
    parser.add_argument("--config", type=str, default="./config/config_HandPose_TwoInput_res18_model06.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to pre-trained checkpoint")
    # distributed training 参数
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed inference")
    # 新增参数：是否直接显示结果，以及每次最多显示多少张图片
    parser.add_argument("--show", default=True, help="如果设置，则直接在窗口中显示推测结果，而不是仅保存")
    parser.add_argument("--max_show", type=int, default=10, help="最多显示的图片数量")
    return parser.parse_args()

def merge_config_args(config, args):
    # 这里只处理 resume 参数，其它训练参数可忽略
    if args.resume is not None:
        config["train"]["resume"] = args.resume
    return config


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


def draw_keypoints(img, keypoints):
    """在图像上绘制 2D 关键点，keypoints shape: (K,3)"""
    # print(f"keypoints:{keypoints}")
    for x, y, v in keypoints:
        # if v > 0:  # 仅绘制可见关键点
        cv2.circle(img, (int(x), int(y)), 3, (0, 0, 0), -1)
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
    heatmaps_2d = np.max(heatmaps_3d, axis=1)  # (K, H, W)
    heatmaps_2d = np.mean(heatmaps_2d, axis=0)  # (H, W)
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


def visualize_heatmap_entropy(
    heatmap_3d: torch.Tensor,
    show_topk: int = 5,
    slice_axis: int = 0,  # 0=depth, 1=height, 2=width
    colormap: str = 'jet',
    normalize_entropy: bool = True  # 是否归一化熵（按 log(N_voxel)）
):
    """
    可视化每个关键点热图的熵，并显示 top-K 的热图切片。

    Args:
        heatmap_3d (Tensor): (K, D, H, W) 或 (1, K, D, H, W)
        show_topk (int): 显示前 K 个关键点的热图切片
        slice_axis (int): 切片维度：0=depth, 1=height, 2=width
        colormap (str): 用于显示热图的颜色映射
        normalize_entropy (bool): 是否将熵归一化到 [0, 1]（除以 log(N_voxel)）
    """
    if heatmap_3d.dim() == 5:
        heatmap_3d = heatmap_3d[0]  # 去除 batch 维度

    K, D, H, W = heatmap_3d.shape
    entropies = []
    heatmap_soft_list = []

    for i in range(K):
        hmap = heatmap_3d[i]  # (D, H, W)
        hmap_soft = F.softmax(hmap.view(-1), dim=0).view(D, H, W)  # 每个关键点独立 softmax
        heatmap_soft_list.append(hmap_soft)

        entropy = -(hmap_soft * torch.log(hmap_soft + 1e-6)).sum().item()

        if normalize_entropy:
            N_voxel = D * H * W
            entropy /= np.log(N_voxel)

        entropies.append(entropy)

    # 转成 Tensor 方便可视化
    entropies = np.array(entropies)
    heatmap_soft = torch.stack(heatmap_soft_list)  # (K, D, H, W)

    # === 画图（可选 top-K）
    topk = min(K, show_topk)
    fig, axs = plt.subplots(2, topk, figsize=(topk * 3, 5))

    for i in range(topk):
        hmap = heatmap_soft[i].cpu().numpy()
        entropy_val = entropies[i]

        # 按选定维度切一个中间面
        if slice_axis == 0:
            hmap_slice = hmap[D // 2]
        elif slice_axis == 1:
            hmap_slice = hmap[:, H // 2, :]
        else:
            hmap_slice = hmap[:, :, W // 2]

        axs[0, i].imshow(hmap_slice, cmap=colormap)
        axs[0, i].set_title(f"Kpt {i} | Entropy: {entropy_val:.3f}")
        axs[0, i].axis("off")

    # === 画柱状图
    axs[1, 0].bar(np.arange(K), entropies, color='skyblue')
    axs[1, 0].set_title("Entropy per Keypoint")
    axs[1, 0].set_xlabel("Joint Index")
    axs[1, 0].set_ylabel("Entropy")
    axs[1, 0].set_ylim(0, np.max(entropies) * 1.2)

    for j in range(1, topk):
        axs[1, j].axis("off")

    plt.tight_layout()
    plt.show()

    return entropies

def save_visualization_results(imgs, batch, index, vis_save_path, config, model, show=False):
    """
    可视化单个样本的推测结果:
      - 左侧显示原图与预测的 2D 关键点/热图叠加
      - 右侧显示预测的 3D 关键点散点图
    如果 show 为 True，则直接显示图像；否则将图像保存到指定目录。
    """
    # 还原图像
    sample_img = imgs[index].detach().cpu()  # (C, H, W)
    sample_img_np = sample_img.permute(1, 2, 0).numpy()  # (H, W, C)
    mean = np.array(config["data_preprocessor"]["mean"]) / 255.0
    std  = np.array(config["data_preprocessor"]["std"]) / 255.0
    sample_img_np = unnormalize(sample_img_np, mean, std)
    sample_img_np = sample_img_np[..., :3]  # 防止是融合的图片有6通道只截取前3通道
    sample_img_np = cv2.cvtColor(sample_img_np, cv2.COLOR_RGB2BGR)

    # 预测 2D 和 3D 关键点
    pred_heatmaps_batch, pred_root, pred_hand_type = model(imgs)
    heatmaps_pred = pred_heatmaps_batch[index].detach().cpu()  # (K, D, H, W)
    # 获取预测的 3D 关键点
    # pred_keypoints_3d, _ = HandPoseHead.get_heatmap_3d_maximum(
    #     heatmaps_pred.unsqueeze(0),
    #     image_size=[256,256]
    # )

    # 绘制绘制热图的熵图
    # visualize_heatmap_entropy(heatmaps_pred, show_topk=6, slice_axis=0)

    # 在所有深度上聚合（如求和或最大），然后显示每个关键点在 H × W 平面上的响应分布
    print(f"heatmaps_pred shape: {heatmaps_pred.shape}")
    visualize_keypoint_maps(
        heatmaps_3d=heatmaps_pred.cpu().numpy(),  # 取第一张样本 (K, D, H, W)
        num_joints_to_show=15,
        method="max",  # 或者 "sum"
        alpha=0.8
    )

    pred_keypoints_3d, _ = soft_argmax_3d(
        heatmap_3d=heatmaps_pred.unsqueeze(0),
    )

    pred_keypoints_3d = pred_keypoints_3d * 4


    pred_keypoints_3d = pred_keypoints_3d[0].numpy()  # (K,3)

    print(f"pred_keypoints_3d:{pred_keypoints_3d}")
    # 在原图上绘制预测的 2D 关键点和热图
    vis_pred = sample_img_np.copy()
    vis_pred = draw_keypoints(vis_pred, pred_keypoints_3d)
    vis_pred = visualize_heatmaps(vis_pred, heatmaps_pred.numpy())

    # 生成预测的 3D 关键点散点图
    img_3d_pred = plot_3d_keypoints_to_image(pred_keypoints_3d, title="Pred 3D Points")

    # 调整 3D 图像的高度与 2D 图像一致
    target_h = vis_pred.shape[0]
    h, w = img_3d_pred.shape[:2]
    ratio = target_h / h
    new_w = int(w * ratio)
    img_3d_pred_resized = cv2.resize(img_3d_pred, (new_w, target_h))

    # 拼接：左边显示 2D 预测，右边显示 3D 预测
    combined_vis = cv2.hconcat([vis_pred, img_3d_pred_resized])
    if show:
        cv2.imshow(f"Inference Result {index}", combined_vis)
        cv2.waitKey(0)
    else:
        save_img_path = os.path.join(vis_save_path, f"inference_result_{index}.jpg")
        cv2.imwrite(save_img_path, combined_vis)
        print(f"推测结果已保存: {save_img_path}")


def inference():
    args = parse_args()

    # 读取 YAML 配置
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = merge_config_args(config, args)

    # 多GPU / 分布式设置（若有）
    distributed = config.get("distributed", False)
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = args.local_rank
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(config.get("device", "cpu"))

    # 构建数据增强 pipeline
    # 推断时建议使用 val_pipeline
    if "val_pipeline" in config:
        val_transform = build_pipeline(config, pipeline_key="val_pipeline")
    else:
        val_transform = build_pipeline(config, pipeline_key="train_pipeline")

    # 构建测试/验证数据集（这里以验证集为例）
    views = config.get("input_views", None)
    dataset_name = config.get("dataset_name", 'Interhand99k')
    if dataset_name == 'Interhand99k':
        test_dataset = LightHandDataset(config_path=args.config, mode="val", transform=val_transform)
    elif dataset_name == 'HandPose':
        # test_dataset = HandPoseDataset(config_path=args.config, mode="val", transform=val_transform)
        DatasetClass = HandPoseDatasetTw if views == 2 else HandPoseDataset
        # train_dataset = DatasetClass(config_path, mode="train", transform=train_transform)
        test_dataset = DatasetClass(args.config, mode="val", transform=val_transform)

    else:
        print("⚠️ 请在配置文件中选择正确的数据集名称")
        return

    test_loader = DataLoader(
        test_dataset,
        batch_size=10,  # 单张图片进行推断，便于可视化
        shuffle=False,
        num_workers=config["train"]["num_workers"],
        collate_fn=custom_collate_fn
    )

    # 构建模型
    model = build_model(config)
    model.to(device)
    print(f"加载模型: {model}")

    # 加载预训练模型权重
    # if config["train"].get("resume", None) is None:
    #     print("请提供预训练模型的 checkpoint 路径（--resume）")
    #     return
    # resume_path = config["train"]["resume"]
    resume_path = "./work_dir/config_HandPose_TwoInput_res18_model06_04_16/checkpoints/model_final.pth"
    # print(f"加载预训练模型: {resume_path}")
    checkpoint = torch.load(resume_path, map_location=device)

    # 获取 state_dict
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    # 去掉 module. 前缀（如果存在）
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v



    # 加载参数
    model.load_state_dict(new_state_dict)
    model.eval()

    # 可视化结果保存目录（仅在不直接显示时使用）
    vis_save_path = config["train"].get("visualization_save_path", "./inference_results")
    os.makedirs(vis_save_path, exist_ok=True)

    shown_count = 0  # 用于记录已显示的图片数量
    # 推断循环：对测试集中的每个样本进行前向传播并保存或显示可视化结果
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            views = config.get("input_views", None)
            dataset_name = config.get("dataset_name", 'Interhand99k')
            if views==2 and dataset_name== "HandPose":
                if "fusion_img" not in batch:
                    print("Warning: 'fusion_img' not found in batch. Skipping this sample.")
                    continue
                imgs = batch["fusion_img"].to(device)  # (B, C, H, W)
            else:
                imgs = batch['img'].to(device)
            # 若设置直接显示结果，则每次最多显示 args.max_show 张图片
            if args.show:
                if shown_count >= args.max_show:
                    print(f"已显示 {args.max_show} 张图片，退出显示。")
                    break
                save_visualization_results(imgs, batch, index=0, vis_save_path=vis_save_path, config=config,
                                           model=model, show=True)
                shown_count += 1
            else:
                save_visualization_results(imgs, batch, index=0, vis_save_path=vis_save_path, config=config,
                                           model=model, show=False)


if __name__ == "__main__":
    inference()
