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

def parse_args():
    parser = argparse.ArgumentParser(description="3D Hand Pose Training")
    parser.add_argument("--config", type=str, default="config/config_Interhand99k.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (override config)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (override config)")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of DataLoader workers (override config)")
    parser.add_argument("--resume", type=str, default=None, help="Path to resume checkpoint (override config)")
    # distributed training 参数，torch.distributed.launch 或 torchrun 会自动传入 --local_rank
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    return parser.parse_args()

def merge_config_args(config, args):
    """
    将命令行参数合并到配置文件中：
    如果命令行参数不为 None，则覆盖配置文件中的对应值。
    """
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["train"]["num_workers"] = args.num_workers
    if args.resume is not None:
        config["train"]["resume"] = args.resume
    return config

# 以下辅助函数用于反归一化和绘制关键点/热图
def unnormalize(img, mean, std):
    """反归一化，将 [0,1] 的图像还原到 [0,255]"""
    img = img * std + mean
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img

def draw_keypoints(img, keypoints):
    """在图像上绘制 2D 关键点，keypoints shape: (K,3)"""
    for x, y, v in keypoints:
        if v > 0:  # 仅绘制可见关键点
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

    pred_keypoints_3d, _ = HandPoseHead.get_heatmap_3d_maximum(heatmaps_pred.unsqueeze(0),
                                                               image_size=[256,256])
    # pred_keypoints_3d, _ = HandPoseHead.soft_argmax_3d(heatmaps_pred)
    # pred_keypoints_3d = pred_keypoints_3d * 4.0

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
    print(f"可视化结果已保存: {save_img_path}")

def main():
    args = parse_args()

    # 读取 YAML 配置
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = merge_config_args(config, args)

    # 多GPU / 分布式设置
    distributed = config.get("distributed", False)
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = args.local_rank
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(config.get("device", "cpu"))

    # 构建数据增强 pipeline
    # 对于训练集，可以使用 "train_pipeline"，对于验证集建议单独配置 "val_pipeline"（如果没有，则可以使用 train_pipeline，但通常验证不应使用随机变换）
    train_transform = build_pipeline(config, pipeline_key="train_pipeline")
    val_transform = build_pipeline(config, pipeline_key="val_pipeline") if "val_pipeline" in config else None

    # 构建训练数据集
    dataset_name = config.get("dataset_name",'Interhand99k')
    if dataset_name== 'Interhand99k':
        train_dataset = LightHandDataset(config_path=args.config, mode="train", transform=train_transform)
        val_dataset = LightHandDataset(config_path=args.config, mode="val", transform=val_transform)
    elif dataset_name== 'HandPose':
        train_dataset = HandPoseDataset(config_path=args.config, mode="train", transform=train_transform)
        val_dataset = HandPoseDataset(config_path=args.config, mode="val", transform=val_transform)
    else:
        train_dataset = None
        val_dataset = None
        print(f"⚠️ Please select dataset name in config file...")
        return None

    # train_dataset = HandPoseDataset(config_path=args.config, mode="train", transform=train_transform)
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=(train_sampler is None),
        num_workers=config["train"]["num_workers"],
        collate_fn=custom_collate_fn,
        sampler=train_sampler
    )

    # 构建验证数据集
    # val_dataset = HandPoseDataset(config_path=args.config, mode="val", transform=val_transform)
    if distributed:
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        val_sampler = None

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config["train"]["num_workers"],
        collate_fn=custom_collate_fn,
        sampler=val_sampler
    )

    # 构建模型
    model = build_model(config)
    model.to(device)
    print(f"model:{model}")
    if distributed:
        model = DDP(model, device_ids=[args.local_rank])
    model.train()

    # 构建优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"].get("weight_decay", 0.0)
    )

    # Resume checkpoint if provided
    start_epoch = 0
    if config["train"].get("resume", None) is not None:
        resume_path = config["train"]["resume"]
        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint.get("epoch", 0)
        else:
            # 如果仅保存了 state_dict，则直接加载
            model.load_state_dict(checkpoint)
        print(f"Resumed at epoch {start_epoch}")

    # 可视化间隔（迭代数）
    vis_interval = config["train"].get("visualization_interval", 100)
    vis_save_path = config["train"].get("visualization_save_path", "./vis_results")
    os.makedirs(vis_save_path, exist_ok=True)
    iteration_count = 0

    best_val_loss = float("inf")
    best_epoch = 0

    # 主训练循环
    for epoch in range(start_epoch, config["train"]["epochs"]):
        if distributed:
            train_sampler.set_epoch(epoch)
        epoch_loss = 0.0
        for batch in train_loader:
            if batch is None:
                continue
            imgs = batch["img"].to(device)  # (B, C, H, W)
            targets = batch["meta_info"]
            # target_heatmaps = torch.tensor(targets["heatmaps"], dtype=torch.float).to(device)
            target_heatmaps = targets["heatmaps"].clone().detach().to(torch.float).to(device)
            # target_root = torch.tensor(targets["root_depth"], dtype=torch.float).to(device)
            target_root = targets["root_depth"].clone().detach().to(torch.float).to(device)
            # target_hand_type = torch.tensor(targets["hand_typ"], dtype=torch.float).to(device)
            target_hand_type = targets["hand_typ"].clone().detach().to(torch.float).to(device)

            pred_heatmaps, pred_root, pred_hand_type = model(imgs)
            loss_dict = model.head.compute_loss(
                (pred_heatmaps, pred_root, pred_hand_type),
                (target_heatmaps, target_root, target_hand_type)
            )
            ##########################
            # 2025/03/24 loss
            ##########################
            # total_loss = loss_dict["loss_kpt"] + 0.2*loss_dict["loss_kpt_xy"]+ loss_dict["loss_root"] + loss_dict["loss_hand_type"]

            ##########################
            # 2025/03/25 loss
            ##########################
            if epoch > 40:
                total_loss = 10000 * loss_dict["loss_kpt"] + 0.0001 * loss_dict["loss_kpt_xy"] + loss_dict["loss_root"] + 0.5* loss_dict[
                    "loss_hand_type"]
            else:
                total_loss = 10000 * loss_dict["loss_kpt"]  + loss_dict["loss_root"] + 0.5 * loss_dict["loss_hand_type"]

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            iteration_count += 1

            # 在每 vis_interval 次迭代后进行可视化保存（仅主进程）
            if iteration_count % vis_interval == 0 and (not distributed or dist.get_rank() == 0):
                # 使用当前 batch 第一个样本进行可视化
                save_visualization_results(imgs, batch, epoch, iteration_count, vis_save_path, config, model)

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{config['train']['epochs']}], Train Loss: {avg_train_loss:.4f}")

        # 验证阶段
        model.eval()
        val_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                imgs_val = batch["img"].to(device)
                targets_val = batch["meta_info"]
                # target_heatmaps_val = torch.tensor(targets_val["heatmaps"], dtype=torch.float).to(device)
                target_heatmaps_val = targets_val["heatmaps"].clone().detach().to(torch.float).to(device)
                # target_root_val = torch.tensor(targets_val["root_depth"], dtype=torch.float).to(device)
                target_root_val = targets_val["root_depth"].clone().detach().to(torch.float).to(device)
                # target_hand_type_val = torch.tensor(targets_val["hand_typ"], dtype=torch.float).to(device)
                target_hand_type_val = targets_val["hand_typ"].clone().detach().to(torch.float).to(device)

                pred_heatmaps_val, pred_root_val, pred_hand_type_val = model(imgs_val)
                loss_dict_val = model.head.compute_loss(
                    (pred_heatmaps_val, pred_root_val, pred_hand_type_val),
                    (target_heatmaps_val, target_root_val, target_hand_type_val)
                )
                total_val_loss = loss_dict_val["loss_kpt"] + loss_dict_val["loss_root"] + loss_dict_val["loss_hand_type"] + loss_dict_val["loss_kpt_xy"]
                val_loss += total_val_loss.item()
                count += 1

        avg_val_loss = val_loss / count if count > 0 else float("inf")
        print(f"Epoch [{epoch+1}/{config['train']['epochs']}], Val Loss: {avg_val_loss:.4f}")
        model.train()

        # 如果当前验证损失更好，则保存最佳 checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            best_checkpoint_path = os.path.join(config.get("checkpoint_dir", "./checkpoints"), "best_model.pth")
            os.makedirs(os.path.dirname(best_checkpoint_path), exist_ok=True)
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, best_checkpoint_path)
            print(f"New best model saved at epoch {epoch+1} with val loss {avg_val_loss:.4f}")

    # 最后保存最终模型（仅主进程保存）
    if not distributed or (distributed and dist.get_rank() == 0):
        final_checkpoint_path = os.path.join(config.get("checkpoint_dir", "./checkpoints"), "model_final.pth")
        torch.save(model.state_dict(), final_checkpoint_path)
        print(f"Final model saved at {final_checkpoint_path}")

    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
