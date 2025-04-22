import os
import sys
import argparse
import yaml
import torch
import shutil
import torch.optim as optim
from torch.nn.parallel import DataParallel

from test import soft_argmax_3d

# 添加项目根目录到 PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from datasets import HandPoseDataset,LightHandDataset,HandPoseDatasetTw
from utils.build_pipeline import build_pipeline  # 根据配置文件构建 pipeline
from codecs_.hand_pose_codec import *
from models.build_model import build_model
from utils.collate import custom_collate_fn  # 自定义 collate_fn
from utils.visualize import *
from utils.metrics import evaluate_relative_pose
from tools.visualization_heatmap import visualize_keypoint_maps
# 初始化 TensorBoard 日志写入器
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import ConcatDataset, DataLoader

"""
    这个代码主要是用来训练双图片输入的模型
"""

def parse_args():
    parser = argparse.ArgumentParser(description="3D Hand Pose Training")
    parser.add_argument("--config", type=str, default="config/config_HandPose_TwoInput_res18_model06.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (override config)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (override config)")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of DataLoader workers (override config)")
    parser.add_argument("--resume", type=str, default=None, help="Path to resume checkpoint (override config)")
    # distributed training 参数，torch.distributed.launch 或 torchrun 会自动传入 --local_rank
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    # 新增工作文件夹配置参数
    parser.add_argument("--work_dir", type=str, default="./work_dir/config_HandPose_TwoInput_res18_model06_04_22", help="Working folder path, used to save the configuration files and logs used in this training")
    return parser.parse_args()

# def parse_args():
#     parser = argparse.ArgumentParser(description="3D Hand Pose Training")
#     parser.add_argument("--config", type=str, default="config/config_Interhand99k_model01.yaml", help="Path to config file")
#     parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (override config)")
#     parser.add_argument("--batch_size", type=int, default=None, help="Batch size (override config)")
#     parser.add_argument("--num_workers", type=int, default=None, help="Number of DataLoader workers (override config)")
#     parser.add_argument("--resume", type=str, default=None, help="Path to resume checkpoint (override config)")
#     # distributed training 参数，torch.distributed.launch 或 torchrun 会自动传入 --local_rank
#     parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
#     # 新增工作文件夹配置参数
#     parser.add_argument("--work_dir", type=str, default="./work_dir/config_Interhand99k_test_04_14", help="工作文件夹路径，用于保存本次训练使用的配置文件及日志")
#     return parser.parse_args()

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

# ==============================
# ✅ 每轮训练或验证
# ==============================
# def run_one_epoch(model, dataloader, optimizer, device, writer, epoch, config, iteration_count=0,
#                   vis_interval=0, vis_save_path=None, is_train=True, desc="Train"):
#     phase = "Train" if is_train else "Val"
#     bar = tqdm(dataloader, desc=f"{desc} Epoch {epoch+1}", unit="batch")
#     total_loss = 0.0
#     count = 0
#     for batch in bar:
#         if batch is None: continue
#         imgs = batch["fusion_img"].to(device)
#         targets = batch["meta_info"]
#         target_heatmaps = targets["heatmaps"].clone().detach().float().to(device)
#         target_root = targets["root_depth"].clone().detach().float().to(device)
#         target_hand_type = targets["hand_typ"].clone().detach().float().to(device)
#
#         if is_train: optimizer.zero_grad()
#         pred_heatmaps, pred_root, pred_hand_type = model(imgs)
#         loss_dict = model.module.head.compute_loss(
#             (pred_heatmaps, pred_root, pred_hand_type),
#             (target_heatmaps, target_root, target_hand_type))
#
#         if epoch > 40:
#             loss = 10000 * loss_dict["loss_kpt"] + 0.0001 * loss_dict["loss_kpt_xy"] + loss_dict["loss_root"] + 0.5 * loss_dict["loss_hand_type"]
#         else:
#             loss = 10000 * loss_dict["loss_kpt"] + loss_dict["loss_root"] + 0.5 * loss_dict["loss_hand_type"]
#
#         if is_train:
#             loss.backward()
#             optimizer.step()
#             iteration_count += 1
#
#         total_loss += loss.item()
#         count += 1
#         bar.set_postfix(loss=f"{loss.item():.4f}")
#
#         if is_train and vis_interval > 0 and iteration_count % vis_interval == 0 and vis_save_path is not None:
#             save_visualization_results(imgs, batch, epoch, iteration_count, vis_save_path, config, model)
#
#     avg_loss = total_loss / count if count > 0 else float("inf")
#     writer.add_scalar(f"Loss/{phase}", avg_loss, epoch+1)
#     return avg_loss, iteration_count


def run_one_epoch(model, dataloader, optimizer, device, writer, epoch, config, iteration_count=0,
                  vis_interval=0, vis_save_path=None, is_train=True, desc="Train"):
    phase = "Train" if is_train else "Val"
    bar = tqdm(dataloader, desc=f"{desc} Epoch {epoch+1}", unit="batch")
    total_loss = 0.0
    count = 0

    # 验证阶段收集
    all_pred_2d = []
    all_pred_z = []
    all_gt_2d = []
    all_gt_z = []
    all_mask = []

    input_views = config.get("input_views", 1)

    for batch in bar:
        if batch is None: continue
        if input_views == 2:
            imgs = batch["fusion_img"].to(device)
        else:
            imgs = batch["img"].to(device)
        targets = batch["meta_info"]
        target_heatmaps = targets["heatmaps"].clone().detach().float().to(device)
        target_root = targets["root_depth"].clone().detach().float().to(device)
        target_hand_type = targets["hand_typ"].clone().detach().float().to(device)

        if is_train: optimizer.zero_grad()
        pred_heatmaps, pred_root, pred_hand_type = model(imgs)

        loss_dict = model.module.head.compute_loss(
            (pred_heatmaps, pred_root, pred_hand_type),
            (target_heatmaps, target_root, target_hand_type))
        print(f"loss_dict:{loss_dict}")

        # if epoch > 40:
        #     # loss = 10000 * loss_dict["loss_kpt"] + 0.001 * loss_dict["loss_kpt_xy"] + loss_dict["loss_root"] + 0.5 * loss_dict["loss_hand_type"]
        #     loss = 10000 * loss_dict["loss_kpt"] + 0.0001 * loss_dict["loss_kpt_xy"]
        # else:
        #     # loss = 10000 * loss_dict["loss_kpt"] + loss_dict["loss_root"] + 0.5 * loss_dict["loss_hand_type"]
        #     loss = 10000 * loss_dict["loss_kpt"]

        loss = loss_dict["loss_kpt"]

        if is_train:
            loss.backward()
            optimizer.step()
            iteration_count += 1
        else:
            # === 获取预测和GT关键点（假设你有argmax转为坐标的函数）

            # pred_kpt_2d = heatmap_to_coord(pred_heatmaps.detach().cpu().numpy())  # (B, J, 2)
            # gt_kpt_2d = heatmap_to_coord(target_heatmaps.detach().cpu().numpy())

            pred_kpt_3d, _ = model.module.head.soft_argmax_3d(pred_heatmaps.detach())  # (B, J, 3)
            gt_kpt_3d, _ = model.module.head.soft_argmax_3d(target_heatmaps.detach())  # (B, J, 3)
            # 热图坐标 → 图像坐标（按比例缩放）
            D, H, W = model.module.head.heatmap_size
            # img_w, img_h = batch["fusion_img"].shape[-1], batch["fusion_img"].shape[-2]
            img_w, img_h = imgs.shape[-1], imgs.shape[-2]
            depth_bound = 400  # mm

            def scale_coords(kpt_3d):
                out = kpt_3d.clone()
                out[:, :, 0] = out[:, :, 0] * img_w / W
                out[:, :, 1] = out[:, :, 1] * img_h / H
                out[:, :, 2] = (out[:, :, 2] / D - 0.5) * depth_bound
                return out

            pred_3d_scaled = scale_coords(pred_kpt_3d).cpu().numpy()
            gt_3d_scaled = scale_coords(gt_kpt_3d).cpu().numpy()
            # print(f"pred_3d scaled:{pred_3d_scaled}")
            # print(f"gt_3d   scaled:{gt_3d_scaled}")

            # 拆分 xy + z
            pred_2d = pred_3d_scaled[:, :, :2]
            gt_2d = gt_3d_scaled[:, :, :2]
            pred_z = pred_3d_scaled[:, :, 2]
            gt_z = gt_3d_scaled[:, :, 2]

            # 你可能还需要 batch["valid_mask"] 或自己构造 mask
            mask = np.ones(pred_2d.shape[:2], dtype=bool)

            all_pred_2d.append(pred_2d)
            all_gt_2d.append(gt_2d)
            all_pred_z.append(pred_z)
            all_gt_z.append(gt_z)
            all_mask.append(mask)

        total_loss += loss.item()
        count += 1
        bar.set_postfix(loss=f"{loss.item():.4f}")

        if is_train and vis_interval > 0 and iteration_count % vis_interval == 0 and vis_save_path is not None:
            save_visualization_results(imgs, batch, epoch, iteration_count, vis_save_path, config, model)

    avg_loss = total_loss / count if count > 0 else float("inf")
    writer.add_scalar(f"Loss/{phase}", avg_loss, epoch+1)

    # ======= 验证阶段计算评价指标 =======
    if not is_train and len(all_pred_2d) > 0:
        pred_2d_all = np.concatenate(all_pred_2d, axis=0)
        pred_z_all = np.concatenate(all_pred_z, axis=0)
        gt_2d_all = np.concatenate(all_gt_2d, axis=0)
        gt_z_all = np.concatenate(all_gt_z, axis=0)
        mask_all = np.concatenate(all_mask, axis=0)

        eval_result = evaluate_relative_pose(
            pred_2d_all, pred_z_all, gt_2d_all, gt_z_all, mask_all,
            threshold_mm=20.0, threshold_px=10.0
        )
        print(f"{eval_result}")

        for k, v in eval_result.items():
            writer.add_scalar(f"Metric/{phase}_{k}", v, epoch+1)
            bar.write(f"{phase} {k}: {v:.4f}")

    return avg_loss, iteration_count


# ==============================
# ✅ 初始化模型与优化器
# ==============================
def build_model_and_optimizer(config, device):
    model = build_model(config).to(device)
    gpu_ids = config["train"].get("gpu", None)
    if gpu_ids and len(gpu_ids) > 1:
        model = DataParallel(model, device_ids=gpu_ids)

    optimizer = optim.Adam(model.parameters(), lr=config["train"]["learning_rate"],
                           weight_decay=config["train"].get("weight_decay", 0.0))
    start_epoch = 0
    resume_path = config["train"].get("resume", None)
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        new_state_dict = {k if k.startswith("module.") else "module."+k: v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        print(f"Resuming from checkpoint: {resume_path}, start_epoch : {start_epoch}")
    return model, optimizer, start_epoch


# ==============================
# ✅ 加载数据集
# ==============================
# def build_dataloaders(config, config_path):
#     train_transform = build_pipeline(config, pipeline_key="train_pipeline")
#     val_transform = build_pipeline(config, pipeline_key="val_pipeline") if "val_pipeline" in config else None
#     name = config.get("dataset_name", 'Interhand99k')
#     views = config.get("input_views", None)
#
#     if name == 'Interhand99k':
#         train_dataset = LightHandDataset(config_path, mode="train", transform=train_transform)
#         val_dataset = LightHandDataset(config_path, mode="val", transform=val_transform)
#     elif name == 'HandPose':
#         DatasetClass = HandPoseDatasetTw if views == 2 else HandPoseDataset
#         # train_dataset = DatasetClass(config_path, mode="train", transform=train_transform)
#         train_dataset = DatasetClass(config_path, mode="train", transform=val_transform)
#         val_dataset = DatasetClass(config_path, mode="val", transform=val_transform)
#     else:
#         raise ValueError("Unknown dataset name. Please check config.")
#
#     train_loader = torch.utils.data.DataLoader(train_dataset,
#         batch_size=config["train"]["batch_size"], shuffle=True,
#         num_workers=config["train"]["num_workers"], collate_fn=custom_collate_fn)
#
#     val_loader = torch.utils.data.DataLoader(val_dataset,
#         batch_size=config["train"]["batch_size"], shuffle=False,
#         num_workers=config["train"]["num_workers"], collate_fn=custom_collate_fn)
#
#     return train_loader, val_loader

def build_dataloaders(config, config_path):
    train_transform = build_pipeline(config, pipeline_key="train_pipeline")
    val_transform = build_pipeline(config, pipeline_key="val_pipeline") if "val_pipeline" in config else None
    name = config.get("dataset_name", 'HandPose')
    views = config.get("input_views", None)

    def load_dataset_group(mode, transform):
        dataset_cfgs = config["dataset"][mode]
        if not isinstance(dataset_cfgs, list):  # 兼容单项格式
            dataset_cfgs = [dataset_cfgs]

        datasets = []
        total_count = 0
        print(f"\n📦 Loading {mode} dataset(s):")
        for ds_cfg in dataset_cfgs:
            json_file = ds_cfg["json_file"]
            img_dir = ds_cfg["img_dir"]

            DatasetClass = HandPoseDatasetTw if views == 2 else HandPoseDataset
            dataset = DatasetClass(
                json_file=json_file,
                img_dir=img_dir,
                transform=transform,
                config=config
            )
            count = len(dataset)
            print(f"✅ {DatasetClass.__name__} from: {img_dir}  ({count} samples)")
            total_count += count
            datasets.append(dataset)

        print(f"🔢 Total {mode} samples: {total_count}")
        return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    if name == 'HandPose':
        train_dataset = load_dataset_group("train", train_transform)
        val_dataset = load_dataset_group("val", val_transform)
    elif name == 'Interhand99k':
        train_dataset = LightHandDataset(config_path, mode="train", transform=train_transform)
        val_dataset = LightHandDataset(config_path, mode="val", transform=val_transform)
    else:
        raise NotImplementedError(f"不支持的数据集类型：{name}")

    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=config["train"]["batch_size"], shuffle=True,
        num_workers=config["train"]["num_workers"], collate_fn=custom_collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=config["train"]["batch_size"], shuffle=False,
        num_workers=config["train"]["num_workers"], collate_fn=custom_collate_fn)
    return train_loader, val_loader

# ==============================
# ✅ 保存模型
# ==============================
def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, path)
    print(f"✅ Model checkpoint saved to: {path}")
# ==============================
# ✅ 初始化文件夹、设备等
# ==============================
def setup_environment_and_dirs(args, config):
    os.makedirs(args.work_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(args.work_dir, os.path.basename(args.config)))

    os.makedirs(os.path.join(args.work_dir, config["train"].get("visualization_save_path", "vis_results")), exist_ok=True)
    os.makedirs(os.path.join(args.work_dir, config.get("checkpoint_dir", "checkpoints")), exist_ok=True)
    os.makedirs(os.path.join(args.work_dir, "logs"), exist_ok=True)

    print(f"配置文件已保存到: {os.path.join(args.work_dir, os.path.basename(args.config))}")
    print(f"TensorBoard 日志保存在: {os.path.join(args.work_dir, 'logs')}")

    gpu_ids = config["train"].get("gpu", None)
    return torch.device("cuda") if gpu_ids and torch.cuda.is_available() else torch.device("cpu")


# ==============================
# ✅ 主训练函数（重构）
# ==============================
def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = merge_config_args(config, args)

    device = setup_environment_and_dirs(args, config)
    writer = SummaryWriter(log_dir=os.path.join(args.work_dir, "logs"))

    train_loader, val_loader = build_dataloaders(config, args.config)
    model, optimizer, start_epoch = build_model_and_optimizer(config, device)

    vis_save_path = os.path.join(args.work_dir, config["train"].get("visualization_save_path", "vis_results"))
    os.makedirs(vis_save_path, exist_ok=True)
    checkpoint_dir = os.path.join(args.work_dir, config.get("checkpoint_dir", "checkpoints"))

    vis_interval = config["train"].get("visualization_interval", 100)
    iteration_count = 0
    best_val_loss = float("inf")

    for epoch in range(start_epoch, config["train"]["epochs"]):
        # ==== Train ====
        model.train()
        train_loss, iteration_count = run_one_epoch(
            model, train_loader, optimizer, device, writer, epoch, config, iteration_count,
            vis_interval, vis_save_path, desc="Train")

        # ==== Validate ====
        model.eval()
        with torch.no_grad():
            val_loss, _ = run_one_epoch(
                model, val_loader, optimizer, device, writer, epoch, config, is_train=False,
                vis_interval=0, vis_save_path=None, desc="Val")

        # ==== Save best model ====
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, os.path.join(checkpoint_dir, "best_model.pth"))
            writer.add_text("Checkpoint", f"Best model at epoch {epoch+1} with val loss {val_loss:.4f}", epoch+1)

    # ==== Save final model ====
    save_checkpoint(model, optimizer, config["train"]["epochs"], os.path.join(checkpoint_dir, "model_final.pth"))
    writer.close()


if __name__ == "__main__":
    main()