import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2
import argparse
import numpy as np
import torch
import yaml
from datasets import HandPoseDataset

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
    parser.add_argument("--save", action="store_true", default = False, help="是否保存可视化结果")
    parser.add_argument("--save_path", type=str, default="vis_results", help="保存路径")
    return parser.parse_args()

def unnormalize(img, mean, std):
    """反归一化"""
    img = img * std + mean  # 还原标准化
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)  # 还原归一化
    return img

def draw_keypoints(img, keypoints):
    """绘制2D关键点"""
    for x, y, v in keypoints:
        if v > 0:  # 仅绘制可见关键点
            cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
    return img

def visualize(dataset, num_images, save, save_path, mean, std):
    """可视化数据集"""
    os.makedirs(save_path, exist_ok=True)
    index = 0  # 初始化索引

    while index < num_images:
        sample = dataset[index]
        img = sample["img"].permute(1, 2, 0).numpy() # 反归一化
        # **反归一化**
        img = unnormalize(img, mean, std)
        # **转换为 BGR 以供 OpenCV 显示**
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # img = sample["img"]
        keypoints_2d = sample["keypoints_2d"].numpy()

        img_with_kpts = draw_keypoints(img.copy(), keypoints_2d)

        # 显示图像
        cv2.imshow(f"Image {index + 1}/{num_images}", img_with_kpts)
        key = cv2.waitKey(0)  # 等待按键
        if key == 27:  # ESC 退出
            break
        elif key == 81:  # 左箭头
            index = max(0, index - 1)
        elif key == 83:  # 右箭头
            index = min(len(dataset) - 1, index + 1)
        else:
            index += 1  # 默认下一张

        # 保存图片
        if save:
            save_img_path = os.path.join(save_path, f"sample_{index}.jpg")
            cv2.imwrite(save_img_path, img_with_kpts)
            print(f"✅ 已保存: {save_img_path}")

    cv2.destroyAllWindows()



if __name__ == "__main__":
    args = parse_args()
    # 读取 YAML 配置
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    mean = np.array(config["data_preprocessor"]["mean"]) / 255.0
    std = np.array(config["data_preprocessor"]["std"]) / 255.0

    dataset = HandPoseDataset(config_path=args.config, mode="val")
    visualize(dataset, args.num_images, args.save, args.save_path, mean, std)
