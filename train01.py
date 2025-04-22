import os
import sys
import argparse
import yaml
import torch
import shutil
import torch.optim as optim
from torch.nn.parallel import DataParallel
# 添加项目根目录到 PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from datasets import HandPoseDataset,LightHandDataset,HandPoseDatasetTw
from utils.build_pipeline import build_pipeline  # 根据配置文件构建 pipeline
from models.build_model import build_model
from utils.collate import custom_collate_fn  # 自定义 collate_fn
from utils.visualize import *
# 初始化 TensorBoard 日志写入器
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser(description="3D Hand Pose Training")
    parser.add_argument("--config", type=str, default="config/config_Interhand99k_model01.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (override config)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (override config)")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of DataLoader workers (override config)")
    parser.add_argument("--resume", type=str, default=None, help="Path to resume checkpoint (override config)")
    # distributed training 参数，torch.distributed.launch 或 torchrun 会自动传入 --local_rank
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    # 新增工作文件夹配置参数
    parser.add_argument("--work_dir", type=str, default="./work_dir/config_Interhand99k_model01_test", help="工作文件夹路径，用于保存本次训练使用的配置文件及日志")
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

# def main():
#     args = parse_args()
#     with open(args.config, "r") as f:
#         config = yaml.safe_load(f)
#     config = merge_config_args(config, args)
#
#     # 工作文件夹及子目录
#     work_dir = args.work_dir
#     os.makedirs(work_dir, exist_ok=True)
#     config_copy_path = os.path.join(work_dir, os.path.basename(args.config))
#     shutil.copy(args.config, config_copy_path)
#     print(f"配置文件已保存到: {config_copy_path}")
#
#     vis_save_path = os.path.join(work_dir, config["train"].get("visualization_save_path", "vis_results"))
#     os.makedirs(vis_save_path, exist_ok=True)
#     checkpoint_dir = os.path.join(work_dir, config.get("checkpoint_dir", "checkpoints"))
#     os.makedirs(checkpoint_dir, exist_ok=True)
#     log_dir = os.path.join(work_dir, "logs")
#     os.makedirs(log_dir, exist_ok=True)
#
#     # 获取 GPU 列表（例如 [0,1,2]）从配置文件中配置
#     gpu_ids = config["train"].get("gpu", None)
#     if gpu_ids is not None and torch.cuda.is_available():
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")
#
#     train_transform = build_pipeline(config, pipeline_key="train_pipeline")
#     val_transform = build_pipeline(config, pipeline_key="val_pipeline") if "val_pipeline" in config else None
#
#     dataset_name = config.get("dataset_name", 'Interhand99k')
#     if dataset_name == 'Interhand99k':
#         train_dataset = LightHandDataset(config_path=args.config, mode="train", transform=train_transform)
#         val_dataset = LightHandDataset(config_path=args.config, mode="val", transform=val_transform)
#     elif dataset_name == 'HandPose':
#         train_dataset = HandPoseDataset(config_path=args.config, mode="train", transform=train_transform)
#         val_dataset = HandPoseDataset(config_path=args.config, mode="val", transform=val_transform)
#     else:
#         print("⚠️ Please select dataset name in config file...")
#         return None
#
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=config["train"]["batch_size"],
#         shuffle=True,
#         num_workers=config["train"]["num_workers"],
#         collate_fn=custom_collate_fn
#     )
#     val_loader = torch.utils.data.DataLoader(
#         val_dataset,
#         batch_size=config["train"]["batch_size"],
#         shuffle=False,
#         num_workers=config["train"]["num_workers"],
#         collate_fn=custom_collate_fn
#     )
#
#     model = build_model(config)
#     model.to(device)
#     print(f"model: {model}")
#     if gpu_ids is not None and len(gpu_ids) > 1:
#         print(f"检测到多个 GPU，使用 DataParallel: {gpu_ids}")
#         model = DataParallel(model, device_ids=gpu_ids)
#     model.train()
#
#     optimizer = optim.Adam(
#         model.parameters(),
#         lr=config["train"]["learning_rate"],
#         weight_decay=config["train"].get("weight_decay", 0.0)
#     )
#
#     start_epoch = 0
#     if config["train"].get("resume", None) is not None:
#         resume_path = config["train"]["resume"]
#         print(f"Resuming from checkpoint: {resume_path}")
#         checkpoint = torch.load(resume_path, map_location=device)
#         state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
#         new_state_dict = {}
#         for k, v in state_dict.items():
#             if not k.startswith("module."):
#                 new_state_dict["module." + k] = v
#             else:
#                 new_state_dict[k] = v
#         model.load_state_dict(new_state_dict)
#         if "optimizer_state_dict" in checkpoint:
#             optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
#         start_epoch = checkpoint.get("epoch", 0)
#         print(f"Resumed at epoch {start_epoch}")
#
#     vis_interval = config["train"].get("visualization_interval", 100)
#     iteration_count = 0
#     best_val_loss = float("inf")
#     best_epoch = 0
#
#     # 主训练循环
#     for epoch in range(start_epoch, config["train"]["epochs"]):
#         epoch_loss = 0.0
#         for batch in train_loader:
#             if batch is None:
#                 continue
#             imgs = batch["img"].to(device)
#             targets = batch["meta_info"]
#             target_heatmaps = targets["heatmaps"].clone().detach().to(torch.float).to(device)
#             target_root = targets["root_depth"].clone().detach().to(torch.float).to(device)
#             target_hand_type = targets["hand_typ"].clone().detach().to(torch.float).to(device)
#             pred_heatmaps, pred_root, pred_hand_type = model(imgs)
#             loss_dict = model.module.head.compute_loss(
#                 (pred_heatmaps, pred_root, pred_hand_type),
#                 (target_heatmaps, target_root, target_hand_type)
#             )
#             if epoch > 40:
#                 total_loss = 10000 * loss_dict["loss_kpt"] + 0.0001 * loss_dict["loss_kpt_xy"] + loss_dict["loss_root"] + 0.5 * loss_dict["loss_hand_type"]
#             else:
#                 total_loss = 10000 * loss_dict["loss_kpt"] + loss_dict["loss_root"] + 0.5 * loss_dict["loss_hand_type"]
#             optimizer.zero_grad()
#             total_loss.backward()
#             optimizer.step()
#             epoch_loss += total_loss.item()
#             iteration_count += 1
#             if iteration_count % vis_interval == 0:
#                 save_visualization_results(imgs, batch, epoch, iteration_count, vis_save_path, config, model)
#         avg_train_loss = epoch_loss / len(train_loader)
#         print(f"Epoch [{epoch+1}/{config['train']['epochs']}], Train Loss: {avg_train_loss:.4f}")
#
#         model.eval()
#         val_loss = 0.0
#         count = 0
#         with torch.no_grad():
#             for batch in val_loader:
#                 if batch is None:
#                     continue
#                 imgs_val = batch["img"].to(device)
#                 targets_val = batch["meta_info"]
#                 target_heatmaps_val = targets_val["heatmaps"].clone().detach().to(torch.float).to(device)
#                 target_root_val = targets_val["root_depth"].clone().detach().to(torch.float).to(device)
#                 target_hand_type_val = targets_val["hand_typ"].clone().detach().to(torch.float).to(device)
#                 pred_heatmaps_val, pred_root_val, pred_hand_type_val = model(imgs_val)
#                 loss_dict_val = model.module.head.compute_loss(
#                     (pred_heatmaps_val, pred_root_val, pred_hand_type_val),
#                     (target_heatmaps_val, target_root_val, target_hand_type_val)
#                 )
#                 total_val_loss = (loss_dict_val["loss_kpt"] + loss_dict_val["loss_root"] +
#                                   loss_dict_val["loss_hand_type"] + loss_dict_val["loss_kpt_xy"])
#                 val_loss += total_val_loss.item()
#                 count += 1
#         avg_val_loss = val_loss / count if count > 0 else float("inf")
#         print(f"Epoch [{epoch+1}/{config['train']['epochs']}], Val Loss: {avg_val_loss:.4f}")
#         model.train()
#
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             best_epoch = epoch + 1
#             best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
#             torch.save({
#                 "epoch": epoch + 1,
#                 "model_state_dict": model.state_dict(),
#                 "optimizer_state_dict": optimizer.state_dict()
#             }, best_checkpoint_path)
#             print(f"New best model saved at epoch {epoch+1} with val loss {avg_val_loss:.4f}")
#
#     final_checkpoint_path = os.path.join(checkpoint_dir, "model_final.pth")
#     torch.save(model.state_dict(), final_checkpoint_path)
#     print(f"Final model saved at {final_checkpoint_path}")
#
# if __name__ == "__main__":
#     main()
def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = merge_config_args(config, args)

    # 工作文件夹及子目录
    work_dir = args.work_dir
    os.makedirs(work_dir, exist_ok=True)
    config_copy_path = os.path.join(work_dir, os.path.basename(args.config))
    shutil.copy(args.config, config_copy_path)
    print(f"配置文件已保存到: {config_copy_path}")

    vis_save_path = os.path.join(work_dir, config["train"].get("visualization_save_path", "vis_results"))
    os.makedirs(vis_save_path, exist_ok=True)
    checkpoint_dir = os.path.join(work_dir, config.get("checkpoint_dir", "checkpoints"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_dir = os.path.join(work_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard 日志保存在: {log_dir}")

    # 从配置文件中读取 GPU 列表，例如 "GPU": [0, 1, 2]
    gpu_ids = config["train"].get("gpu", None)
    if gpu_ids is not None and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_transform = build_pipeline(config, pipeline_key="train_pipeline")
    val_transform = build_pipeline(config, pipeline_key="val_pipeline") if "val_pipeline" in config else None

    dataset_name = config.get("dataset_name", 'Interhand99k')
    input_views = config.get("input_views", None)
    if dataset_name == 'Interhand99k':
        if input_views != 2:
            train_dataset = LightHandDataset(config_path=args.config, mode="train", transform=train_transform)
            val_dataset = LightHandDataset(config_path=args.config, mode="val", transform=val_transform)
        else:
            ...
    elif dataset_name == 'HandPose':
        train_dataset = HandPoseDataset(config_path=args.config, mode="train", transform=train_transform)
        val_dataset = HandPoseDataset(config_path=args.config, mode="val", transform=val_transform)
    else:
        print("⚠️ Please select dataset name in config file...")
        return None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["train"]["num_workers"],
        collate_fn=custom_collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config["train"]["num_workers"],
        collate_fn=custom_collate_fn
    )

    model = build_model(config)
    model.to(device)
    print(f"model: {model}")
    if gpu_ids is not None and len(gpu_ids) > 1:
        print(f"检测到多个 GPU，使用 DataParallel: {gpu_ids}")
        model = DataParallel(model, device_ids=gpu_ids)
    model.train()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"].get("weight_decay", 0.0)
    )

    start_epoch = 0
    if config["train"].get("resume", None) is not None:
        resume_path = config["train"]["resume"]
        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        new_state_dict = {}
        for k, v in state_dict.items():
            if not k.startswith("module."):
                new_state_dict["module." + k] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        print(f"Resumed at epoch {start_epoch}")

    vis_interval = config["train"].get("visualization_interval", 100)
    iteration_count = 0
    best_val_loss = float("inf")
    best_epoch = 0

    # 导入 time 和 tqdm 用于记录时间和显示进度条
    import time
    from tqdm import tqdm

    # 主训练循环
    for epoch in range(start_epoch, config["train"]["epochs"]):
        epoch_start_time = time.time()
        epoch_loss = 0.0

        # 用 tqdm 包装训练数据迭代器
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['train']['epochs']}", unit="batch")
        for batch in train_bar:
            if batch is None:
                continue
            imgs = batch["img"].to(device)
            targets = batch["meta_info"]
            target_heatmaps = targets["heatmaps"].clone().detach().to(torch.float).to(device)
            target_root = targets["root_depth"].clone().detach().to(torch.float).to(device)
            target_hand_type = targets["hand_typ"].clone().detach().to(torch.float).to(device)
            pred_heatmaps, pred_root, pred_hand_type = model(imgs)
            loss_dict = model.module.head.compute_loss(
                (pred_heatmaps, pred_root, pred_hand_type),
                (target_heatmaps, target_root, target_hand_type)
            )
            if epoch > 40:
                total_loss = 10000 * loss_dict["loss_kpt"] + 0.0001 * loss_dict["loss_kpt_xy"] + loss_dict["loss_root"] + 0.5 * loss_dict["loss_hand_type"]
            else:
                total_loss = 10000 * loss_dict["loss_kpt"] + loss_dict["loss_root"] + 0.5 * loss_dict["loss_hand_type"]

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            iteration_count += 1

            # 更新进度条显示，展示当前 batch 的损失
            train_bar.set_postfix(loss=f"{total_loss.item():.4f}")

            if iteration_count % vis_interval == 0:
                save_visualization_results(imgs, batch, epoch, iteration_count, vis_save_path, config, model)
        epoch_duration = time.time() - epoch_start_time
        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{config['train']['epochs']}], Train Loss: {avg_train_loss:.4f}, Duration: {epoch_duration:.2f}s")
        writer.add_scalar("Loss/Train", avg_train_loss, epoch+1)
        writer.add_scalar("Time/Epoch", epoch_duration, epoch+1)

        model.eval()
        val_loss = 0.0
        count = 0
        val_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}", unit="batch")
        with torch.no_grad():
            for batch in val_bar:
                if batch is None:
                    continue
                imgs_val = batch["img"].to(device)
                targets_val = batch["meta_info"]
                target_heatmaps_val = targets_val["heatmaps"].clone().detach().to(torch.float).to(device)
                target_root_val = targets_val["root_depth"].clone().detach().to(torch.float).to(device)
                target_hand_type_val = targets_val["hand_typ"].clone().detach().to(torch.float).to(device)
                pred_heatmaps_val, pred_root_val, pred_hand_type_val = model(imgs_val)
                loss_dict_val = model.module.head.compute_loss(
                    (pred_heatmaps_val, pred_root_val, pred_hand_type_val),
                    (target_heatmaps_val, target_root_val, target_hand_type_val)
                )
                total_val_loss = (loss_dict_val["loss_kpt"] + loss_dict_val["loss_root"] +
                                  loss_dict_val["loss_hand_type"] + loss_dict_val["loss_kpt_xy"])
                val_loss += total_val_loss.item()
                count += 1
                val_bar.set_postfix(loss=f"{total_val_loss.item():.4f}")
        avg_val_loss = val_loss / count if count > 0 else float("inf")
        print(f"Epoch [{epoch+1}/{config['train']['epochs']}], Val Loss: {avg_val_loss:.4f}")
        writer.add_scalar("Loss/Val", avg_val_loss, epoch+1)
        model.train()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, best_checkpoint_path)
            print(f"New best model saved at epoch {epoch+1} with val loss {avg_val_loss:.4f}")
            writer.add_text("Checkpoint", f"Best model at epoch {epoch+1} with val loss {avg_val_loss:.4f}", epoch+1)

    final_checkpoint_path = os.path.join(checkpoint_dir, "model_final.pth")
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"Final model saved at {final_checkpoint_path}")
    writer.close()

if __name__ == "__main__":
    main()