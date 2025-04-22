import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2
import argparse
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from datasets import HandPoseDataset, Compose
from codecs_ import HandPoseCodec
from utils.build_pipeline import *

def parse_args():
    """解析命令行参数"""
    # 获取当前脚本的目录
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # 拼接 config.yaml 绝对路径
    CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")
    parser = argparse.ArgumentParser(description="可视化手部数据集")
    # parser.add_argument("--config", type=str, default="../config/config.yaml", help="YAML配置文件路径")
    parser.add_argument("--config", type=str, default=CONFIG_PATH, help="YAML配置文件路径")
    parser.add_argument("--num_images", type=int, default=10, help="可视化的图片数量")
    parser.add_argument("--save", action="store_true", default=False, help="是否保存可视化结果")
    parser.add_argument("--save_path", type=str, default="vis_results", help="保存路径")
    return parser.parse_args()


def unnormalize(img, mean, std):
    """反归一化"""
    img = img * std + mean  # 还原标准化
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)  # 还原归一化
    return img

def draw_keypoints(img, keypoints):
    """绘制 2D 关键点"""
    for x, y, v in keypoints:
        if v > 0:  # 仅绘制可见关键点
            cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
    return img

def visualize_heatmaps(img, heatmaps_3d):
    """
    可视化 3D Heatmap 的 2D 投影 (最大投影)

    Args:
        img (numpy.ndarray): 原始图像 (H, W, 3)
        heatmaps_3d (numpy.ndarray): 3D 热图 (K, D, H, W)

    Returns:
        numpy.ndarray: 叠加 3D 热图的可视化图像
    """
    K, D, H, W = heatmaps_3d.shape

    # **沿深度维度 D 取最大投影 (生成 2D Heatmap)**
    heatmaps_2d = np.max(heatmaps_3d, axis=1)  # (K, H, W)

    # **确保 heatmaps_2d 形状匹配 img**
    heatmaps_2d = np.mean(heatmaps_2d, axis=0)  # (H, W)
    
    # **归一化到 0-255 并转换为 uint8**
    heatmaps_2d = (heatmaps_2d - heatmaps_2d.min()) / (heatmaps_2d.max() - heatmaps_2d.min() + 1e-5) * 255
    heatmaps_2d = heatmaps_2d.astype(np.uint8)  # 转换为 uint8

    # **使用 OpenCV 伪彩色 (转换为 3 通道)**
    heatmap_color = cv2.applyColorMap(heatmaps_2d, cv2.COLORMAP_JET)  # (H, W, 3)

    # **调整 heatmap_color 大小，以匹配 img**
    heatmap_color = cv2.resize(heatmap_color, (img.shape[1], img.shape[0]))

    # **确保 heatmap_color 也是 3 通道**
    if len(img.shape) == 2:  # 如果是灰度图，转换成 3 通道
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # **融合 Heatmap 和 原始图像**
    img_overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    return img_overlay

def visualize(dataset, codec, num_images, save, save_path, mean, std):
    """可视化数据集"""
    os.makedirs(save_path, exist_ok=True)
    index = 0  # 初始化索引

    while index < num_images:
        sample = dataset[index]
        img = sample["img"].permute(1, 2, 0).numpy()  # **反归一化**
        img = unnormalize(img, mean, std)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # **转换为 BGR 以供 OpenCV 显示**
        
        keypoints_2d = sample["keypoints_2d"].numpy()
        # **从 keypoints_2d[:, 2] 取出可见性**
        keypoints_visible = keypoints_2d[:, 2].astype(np.float32)
        keypoints_3d = np.expand_dims(sample["keypoints_3d"].numpy(), axis=0)  # (1, 21, 3)
        keypoints_2d = np.expand_dims(keypoints_2d, axis=0) # (1,21,3)
        # 将真实深度值传入到2d点第三个维度
        keypoints_2d[:,:,2] = keypoints_3d[:,:,2] * 1000 # 换成毫米单位然后传入

        # **使用 HandPoseCodec 生成 3D Heatmaps**
        encoded = codec.encode(keypoints_2d, 
                               np.expand_dims(keypoints_visible, axis=0),  # 变为 (1, 21)
                               rel_root_depth=np.expand_dims(sample["rel_root_depth"].numpy(), axis=0), 
                               rel_root_valid=np.expand_dims(sample["rel_root_valid"].numpy(), axis=0), 
                               hand_type=np.expand_dims(sample["hand_type"].numpy(), axis=0), 
                               hand_type_valid=np.expand_dims(sample["hand_type_valid"].numpy(), axis=0))
        heatmaps_3d = encoded["heatmaps"]
        keypoints_weights=encoded["keypoint_weights"]

        ##########不用上面的encode生成热图， 增强方法中设置GenerateTarget以后就生成了热图############
        
        # heatmaps_3d = sample['result']["heatmap_target"]["heatmaps"]  # 提取编码后的 3D Heatmap
        # keypoints_weights = sample['result']["heatmap_target"]['keypoint_weights']
        
        print(f"encoded keypoints_wights:{keypoints_weights.shape}")
        print(f"encoded heatmaps_3d :{heatmaps_3d.shape}")

        decoded = codec.decode(heatmaps = heatmaps_3d, 
                               root_depth = np.expand_dims(sample["rel_root_depth"].numpy(), axis=0),
                               hand_type = np.expand_dims(sample["hand_type"].numpy(), axis=0))
        decode_keypoints, decode_scores, decode_rel_root_depth, decode_hand_type = decoded
        print(f"{ decode_keypoints} ,{decode_scores}, {decode_rel_root_depth}, {decode_hand_type}")

        # **可视化**
        img_with_kpts = draw_keypoints(img.copy(), keypoints_2d[0])
        img_with_heatmap = visualize_heatmaps(img_with_kpts, heatmaps_3d)

        # **显示图像**
        cv2.imshow(f"Image {index + 1}/{num_images}", img_with_heatmap)
        key = cv2.waitKey(0)  # 等待按键
        if key == 27:  # ESC 退出
            break
        elif key == 81:  # 左箭头
            index = max(0, index - 1)
        elif key == 83:  # 右箭头
            index = min(len(dataset) - 1, index + 1)
        else:
            index += 1  # 默认下一张

        # **保存图片**
        if save:
            save_img_path = os.path.join(save_path, f"sample_{index}.jpg")
            cv2.imwrite(save_img_path, img_with_heatmap)
            print(f"✅ 已保存: {save_img_path}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    # **读取 YAML 配置**
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    mean = np.array(config["data_preprocessor"]["mean"]) / 255.0
    std = np.array(config["data_preprocessor"]["std"]) / 255.0
    
    transf = build_pipeline(config, pipeline_key = 'train_pipeline')
    dataset = HandPoseDataset(config_path=args.config, mode="train", transform=transf)

    # **创建 HandPoseCodec**
    codec = HandPoseCodec(image_size=[640,480],
                          heatmap_size=[64,64,64],  #D , H, W
                          heatmap3d_depth_bound=1500.0,
                          root_depth_bound=300.0,
                          sigma=2, 
                          max_bound=1.0
                         )

    visualize(dataset, codec, args.num_images, args.save, args.save_path, mean, std)
