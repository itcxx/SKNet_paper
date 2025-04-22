import os
import torch
import numpy as np
import cv2
import yaml
from torch.utils.data import Dataset
from pycocotools.coco import COCO


class HandPoseDataset(Dataset):
    def __init__(self, config_path, mode="train", transform=None ,):
        """
        3D 手部关键点数据集 (COCO 格式)

        Args:
            config_path (str): 配置文件路径 (YAML)
            mode (str): 'train' or 'val'
            transform (callable, optional): 数据增强
        """
        # **检查 YAML 配置文件**
        assert os.path.exists(config_path), f"❌ 配置文件 {config_path} 不存在!"
        
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        assert mode in ["train", "val"], "❌ mode 必须是 'train' 或 'val'"
        
        self.data_root = self.config["dataset"][mode]["img_dir"]
        ann_file = self.config["dataset"][mode]["json_file"]
        self.transform = transform

        # **检查数据集文件是否存在**
        assert os.path.exists(ann_file), f"❌ 标注文件 {ann_file} 不存在!"
        assert os.path.exists(self.data_root), f"❌ 图片目录 {self.data_root} 不存在!"

        # **加载 COCO 数据**
        self.coco = COCO(ann_file)
        self.img_ids = list(self.coco.imgs.keys())  # 获取所有图像 ID
        assert len(self.img_ids) > 0, "❌ 数据集为空！请检查 JSON 文件！"

        # **读取归一化参数**
        self.mean = np.array(self.config["data_preprocessor"]["mean"]) / 255.0
        self.std = np.array(self.config["data_preprocessor"]["std"]) / 255.0
        self.normalize = self.config["transform"].get("normalize", False)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]  # 获取图像信息
        img_path = os.path.join(self.data_root, img_info["file_name"])

        # **检查图片是否存在**
        if not os.path.exists(img_path):
            print(f"⚠️ 警告: 图像 {img_path} 不存在, 跳过该样本!")
            return None

        # **读取图像**
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ 警告: 读取 {img_path} 失败, 跳过该样本!")
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB

        # **加载 Annotation**
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # **COCO 可能有多个 annotation，默认取第一个**
        if len(anns) == 0:
            print(f"⚠️ 警告: 图像 {img_id} 没有 annotation, 跳过!")
            return None
        ann = anns[0]

        # **解析 Annotation**
        data_info = self.parse_data_info(ann, img_info, img_path)

        # **跳过标注缺失的样本**
        if data_info is None:
            return None
            
        # 构建一个 results 字典，将所有增强需要的信息统一放在一起
        results = {
            "img": img,
            "keypoints_2d": data_info["keypoints_2d"],
            "keypoints_3d": data_info["keypoints_3d"],
            "hand_typ": data_info["hand_type"],  # 统一键名为 "hand_typ"
            "hand_typ_valid": data_info["hand_type_valid"],
            "bbox": data_info["bbox"],
            "rel_root_depth": data_info["rel_root_depth"],
            "rel_root_valid": data_info["rel_root_valid"],
            # 初始化 meta_info 用于存放额外信息
            "meta_info": {}
        }
        
        # **数据增强**
        # 如果定义了 transform，则调用 transform( img, results )，要求每个 transform 都接收并返回 (img, results)
        if self.transform is not None:
            img, results = self.transform(img, results)
        results['img'] = img

        # 如果增强过程中产生了额外信息（例如 "heatmap_target"），将它移动到 meta_info 中
        if "heatmaps" in results.get("heatmap_target",{}):
            results["meta_info"]["heatmaps"] = results['heatmap_target']['heatmaps']
        if "keypoint_weights" in results.get("heatmap_target",{}):
            results['meta_info']['keypoint_weights'] = results['heatmap_target']['keypoint_weights']
        if "root_depth" in results.get("heatmap_target",{}):
            results['meta_info']['root_depth'] = results['heatmap_target']['root_depth']
        if "root_depth_weight" in results.get("heatmap_target",{}):
            results['meta_info']['root_depth_weight'] = results['heatmap_target']['root_depth_weight']
        if "type_weight" in results.get("heatmap_target",{}):
            results['meta_info']['type_weight'] = results['heatmap_target']['type_weight']
        if 'hand_typ' in results:
            results['meta_info']['hand_typ'] = results['hand_typ']
        if 'hand_typ_valid' in results:
            results['meta_info']['hand_typ_valid'] = results.pop('hand_typ_valid')
        ## TODO

        # 归一化（如果需要）
        if self.normalize:
            results["img"] = (results["img"] / 255.0 - self.mean) / self.std

        # 返回样本字典（转换为 PyTorch Tensor）
        return {
            "img": torch.tensor(results["img"], dtype=torch.float).permute(2, 0, 1),
            "keypoints_2d": torch.tensor(results["keypoints_2d"], dtype=torch.float),
            "keypoints_3d": torch.tensor(results["keypoints_3d"], dtype=torch.float),
            "hand_type": torch.tensor(results["hand_typ"], dtype=torch.float),
            # "hand_type_valid": torch.tensor(results["hand_typ_valid"], dtype=torch.float),
            "bbox": torch.tensor(results["bbox"], dtype=torch.float),
            "rel_root_depth": torch.tensor(results["rel_root_depth"], dtype=torch.float),
            "rel_root_valid": torch.tensor(results["rel_root_valid"], dtype=torch.float),
            "meta_info": results["meta_info"],
        }

    def parse_data_info(self, ann, img_info, img_path):
        """解析 COCO 标注数据，并将关键点深度转换为相对深度"""
        # **确保关键字段存在**
        required_keys = ["keypoints", "keypoints_3d", "bbox"]
        for key in required_keys:
            if key not in ann:
                print(f"⚠️ 警告: {img_path} 缺少 '{key}'，跳过该样本!")
                return None

        # **解析关键点**
        keypoints_2d = np.array(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
        keypoints_3d = np.array(ann["keypoints_3d"], dtype=np.float32).reshape(-1, 3)

        # **关键点可见性**
        joints_3d_visible = np.ones((keypoints_3d.shape[0],), dtype=np.float32)

        # **COCO 的 bbox (x, y, w, h) -> (x1, y1, x2, y2)**
        bbox = np.array(ann["bbox"], dtype=np.float32)
        bbox_xyxy = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], dtype=np.float32)

        # **解析 Hand Type**
        hand_type = img_info.get("hand_side", None)
        if hand_type == "right":
            hand_type = np.array([0, 1], dtype=np.float32)  # (left, right)
        elif hand_type == "left":
            hand_type = np.array([1, 0], dtype=np.float32)
        else:
            hand_type = np.array([0, 0], dtype=np.float32)  # 未知类型

        hand_type_valid = float(hand_type.sum() > 0)

        # **计算相对深度**
        wrist_idx = 0  # 假设 wrist 关键点索引为 0
        if keypoints_3d.shape[0] > wrist_idx:
            wrist_depth = keypoints_3d[wrist_idx, 2]
        else:
            wrist_depth = 0.0

        # 将每个关键点的深度转换为相对于 wrist 的深度
        keypoints_3d[:, 2] = keypoints_3d[:, 2] - wrist_depth

        # 此时 wrist 的相对深度即为 0
        rel_root_depth = 0.0
        rel_root_valid = float(joints_3d_visible[wrist_idx] > 0)
        # print(f"wrist_depth:{wrist_depth}")
        return {
            "img_id": ann["image_id"],
            "img_path": img_path,
            "keypoints_2d": keypoints_2d,
            "keypoints_3d": keypoints_3d,
            "keypoints_visible": joints_3d_visible,
            "bbox": bbox_xyxy,
            "bbox_score": np.ones(1, dtype=np.float32),
            "hand_type": hand_type[np.newaxis, :],
            "hand_type_valid": hand_type_valid,
            "rel_root_depth": np.array(wrist_depth),
            "rel_root_valid": np.array(rel_root_valid),
        }

# class HandPoseDatasetTw(Dataset):
#     def __init__(self, config_path, mode="train", transform=None, ):
#         """
#         3D 手部关键点数据集 (COCO 格式), 该方法加载双视角输入的数据，只有单视角的图片标注将被丢弃
#
#         Args:
#             config_path (str): 配置文件路径 (YAML)
#             mode (str): 'train' or 'val'
#             transform (callable, optional): 数据增强
#         """
#         # **检查 YAML 配置文件**
#         assert os.path.exists(config_path), f"❌ 配置文件 {config_path} 不存在!"
#
#         with open(config_path, "r") as f:
#             self.config = yaml.safe_load(f)
#
#         assert mode in ["train", "val"], "❌ mode 必须是 'train' 或 'val'"
#
#         self.data_root = self.config["dataset"][mode]["img_dir"]
#         ann_file = self.config["dataset"][mode]["json_file"]
#         self.transform = transform
#
#         # **检查数据集文件是否存在**
#         assert os.path.exists(ann_file), f"❌ 标注文件 {ann_file} 不存在!"
#         assert os.path.exists(self.data_root), f"❌ 图片目录 {self.data_root} 不存在!"
#
#         # **加载 COCO 数据**
#         self.coco = COCO(ann_file)
#         self.img_ids = list(self.coco.imgs.keys())  # 获取所有图像 ID
#         assert len(self.img_ids) > 0, "❌ 数据集为空！请检查 JSON 文件！"
#
#         # **读取归一化参数**
#         self.mean = np.array(self.config["data_preprocessor"]["mean"]) / 255.0
#         self.std = np.array(self.config["data_preprocessor"]["std"]) / 255.0
#         self.normalize = self.config["transform"].get("normalize", True)
#
#     def __len__(self):
#         return len(self.img_ids)
#
#     def __getitem__(self, idx):
#         img_id = self.img_ids[idx]
#         img_info = self.coco.loadImgs(img_id)[0]  # 获取图像信息
#         if img_info['wrist_view'] == False:
#             return None
#
#         img_path = os.path.join(self.data_root, img_info["file_name"])
#         assis_img_path = os.path.join(self.data_root, img_info["assis_file_name"])
#
#         # **检测双视角的图片是否存在**
#         if not os.path.exists(img_path) or not os.path.exists(assis_img_path):
#             print(f"⚠️ 警告: 图像 {img_path} or  {assis_img_path}不存在, 跳过该样本!")
#             return None
#         # **读取图像**
#         img = cv2.imread(img_path)
#         assis_img = cv2.imread(assis_img_path)
#         if img is None or assis_img is None:
#             print(f"⚠️ 警告: 读取 {img_path} ,or {assis_img_path} 失败, 跳过该样本!")
#             return None
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB
#         assis_img = cv2.cvtColor(assis_img, cv2.COLOR_BGR2RGB)
#
#         # **加载 Annotation**
#         ann_ids = self.coco.getAnnIds(imgIds=img_id)
#         anns = self.coco.loadAnns(ann_ids)
#
#         # **COCO 可能有多个 annotation，默认取第一个**
#         if len(anns) == 0:
#             print(f"⚠️ 警告: 图像 {img_id} 没有 annotation, 跳过!")
#             return None
#         ann = anns[0]
#
#         # **解析 Annotation**
#         data_info = self.parse_data_info(ann, img_info, img_path)
#
#         # **跳过标注缺失的样本**
#         if data_info is None:
#             return None
#         # print(f"img_info:{img_info}")
#
#         # 构建一个 results 字典，将所有增强需要的信息统一放在一起
#         results = {
#             "img": img,
#             "assis_img": assis_img,
#             "keypoints_2d": data_info["keypoints_2d"],
#             "keypoints_3d": data_info["keypoints_3d"],
#             "hand_typ": data_info["hand_type"],  # 统一键名为 "hand_typ"
#             "image_type": data_info["image_type"],
#             "hand_typ_valid": data_info["hand_type_valid"],
#             "bbox": data_info["bbox"],
#             "rel_root_depth": data_info["rel_root_depth"],
#             "rel_root_valid": data_info["rel_root_valid"],
#             # 初始化 meta_info 用于存放额外信息
#             "meta_info": {}
#         }
#
#         # **数据增强**
#         # 如果定义了 transform，则调用 transform( img, results )，要求每个 transform 都接收并返回 (img, results)
#         if self.transform is not None:
#             img, results = self.transform(img, results)
#         results['img'] = img
#
#         # 如果增强过程中产生了额外信息（例如 "heatmap_target"），将它移动到 meta_info 中
#         if "heatmaps" in results.get("heatmap_target", {}):
#             results["meta_info"]["heatmaps"] = results['heatmap_target']['heatmaps']
#         if "keypoint_weights" in results.get("heatmap_target", {}):
#             results['meta_info']['keypoint_weights'] = results['heatmap_target']['keypoint_weights']
#         if "root_depth" in results.get("heatmap_target", {}):
#             results['meta_info']['root_depth'] = results['heatmap_target']['root_depth']
#         if "root_depth_weight" in results.get("heatmap_target", {}):
#             results['meta_info']['root_depth_weight'] = results['heatmap_target']['root_depth_weight']
#         if "type_weight" in results.get("heatmap_target", {}):
#             results['meta_info']['type_weight'] = results['heatmap_target']['type_weight']
#         if 'hand_typ' in results:
#             results['meta_info']['hand_typ'] = results['hand_typ']
#         if 'hand_typ_valid' in results:
#             results['meta_info']['hand_typ_valid'] = results.pop('hand_typ_valid')
#         ## TODO
#         # 确保两图像尺寸一致
#         if img.shape[:2] != assis_img.shape[:2]:
#             assis_img = cv2.resize(assis_img, (img.shape[1], img.shape[0]))
#
#         # 归一化（如果需要）
#         if self.normalize:
#             results["img"] = (results["img"] / 255.0 - self.mean) / self.std
#             results["assis_img"] = (assis_img / 255.0 - self.mean) / self.std
#
#         img = torch.tensor(results["img"], dtype=torch.float).permute(2, 0, 1)
#         assis_img = torch.tensor(results["assis_img"], dtype=torch.float).permute(2, 0, 1)
#
#         # 返回样本字典（转换为 PyTorch Tensor）
#         return {
#             "img": img,
#             "assis_img": assis_img,
#             "fusion_img": torch.cat([img,assis_img], dim=0),
#             "keypoints_2d": torch.tensor(results["keypoints_2d"], dtype=torch.float),
#             "keypoints_3d": torch.tensor(results["keypoints_3d"], dtype=torch.float),
#             "hand_type": torch.tensor(results["hand_typ"], dtype=torch.float),
#             "image_type": torch.tensor(results["image_type"], dtype=torch.float),
#             # "hand_type_valid": torch.tensor(results["hand_typ_valid"], dtype=torch.float),
#             "bbox": torch.tensor(results["bbox"], dtype=torch.float),
#             "rel_root_depth": torch.tensor(results["rel_root_depth"], dtype=torch.float),
#             "rel_root_valid": torch.tensor(results["rel_root_valid"], dtype=torch.float),
#             "meta_info": results["meta_info"],
#         }
#
#     def parse_data_info(self, ann, img_info, img_path):
#         """解析 COCO 标注数据，并将关键点深度转换为相对深度"""
#         # **确保关键字段存在**
#         required_keys = ["keypoints", "keypoints_3d", "bbox"]
#         for key in required_keys:
#             if key not in ann:
#                 print(f"⚠️ 警告: {img_path} 缺少 '{key}'，跳过该样本!")
#                 return None
#
#         # **解析关键点**
#         keypoints_2d = np.array(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
#         keypoints_3d = np.array(ann["keypoints_3d"], dtype=np.float32).reshape(-1, 3)
#
#         # **关键点可见性**
#         joints_3d_visible = np.ones((keypoints_3d.shape[0],), dtype=np.float32)
#
#         # **COCO 的 bbox (x, y, w, h) -> (x1, y1, x2, y2)**
#         bbox = np.array(ann["bbox"], dtype=np.float32)
#         bbox_xyxy = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], dtype=np.float32)
#
#         # **解析 Hand Type**
#         hand_type = img_info.get("hand_side", None)
#         if hand_type == "right":
#             hand_type = np.array([0, 1], dtype=np.float32)  # (left, right)
#         elif hand_type == "left":
#             hand_type = np.array([1, 0], dtype=np.float32)
#         else:
#             hand_type = np.array([0, 0], dtype=np.float32)  # 未知类型
#
#         hand_type_valid = float(hand_type.sum() > 0)
#         # **解析 image type , top or down**
#         image_type = img_info.get("image_type", None)
#         if image_type == "top":
#             image_type = np.array([0, 1], dtype=np.float32) #(top, down)
#         elif image_type == "down":
#             image_type = np.array([1,0], dtype=np.float32)
#         else:
#             image_type = np.array([0,0], dtype=np.float32)
#
#
#         # **计算相对深度**
#         wrist_idx = 0  # 假设 wrist 关键点索引为 0
#         if keypoints_3d.shape[0] > wrist_idx:
#             wrist_depth = keypoints_3d[wrist_idx, 2]
#         else:
#             wrist_depth = 0.0
#
#         # 将每个关键点的深度转换为相对于 wrist 的深度
#         keypoints_3d[:, 2] = keypoints_3d[:, 2] - wrist_depth
#
#         # 此时 wrist 的相对深度即为 0
#         rel_root_depth = 0.0
#         rel_root_valid = float(joints_3d_visible[wrist_idx] > 0)
#         # print(f"wrist_depth:{wrist_depth}")
#         return {
#             "img_id": ann["image_id"],
#             "img_path": img_path,
#             "keypoints_2d": keypoints_2d,
#             "keypoints_3d": keypoints_3d,
#             "keypoints_visible": joints_3d_visible,
#             "bbox": bbox_xyxy,
#             "bbox_score": np.ones(1, dtype=np.float32),
#             "hand_type": hand_type[np.newaxis, :],  # left , down
#             "hand_type_valid": hand_type_valid,
#             "rel_root_depth": np.array(wrist_depth),
#             "rel_root_valid": np.array(rel_root_valid),
#             "image_type": image_type, # top, down
#         }

# 2025/04/22
class HandPoseDatasetTw(Dataset):
    def __init__(self, config_path=None, mode=None, transform=None, json_file=None, img_dir=None, config=None):
        """
        支持两种初始化方式：
        1）传入 config_path + mode（旧方式）
        2）直接传入 json_file + img_dir（推荐用于多数据集组合）
        """
        if config_path:
            assert os.path.exists(config_path), f"❌ 配置文件 {config_path} 不存在!"
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
            assert mode in ["train", "val"], "❌ mode 必须是 'train' 或 'val'"
            self.data_root = self.config["dataset"][mode]["img_dir"]
            ann_file = self.config["dataset"][mode]["json_file"]
            self.transform = transform
            self.mean = np.array(self.config["data_preprocessor"]["mean"]) / 255.0
            self.std = np.array(self.config["data_preprocessor"]["std"]) / 255.0
            self.normalize = self.config["transform"].get("normalize", True)
        else:
            assert json_file is not None and img_dir is not None, "❌ 需提供 json_file 与 img_dir"
            assert os.path.exists(json_file), f"❌ 标注文件 {json_file} 不存在!"
            assert os.path.exists(img_dir), f"❌ 图片目录 {img_dir} 不存在!"
            self.data_root = img_dir
            ann_file = json_file
            self.transform = transform
            self.config = config or {}
            self.mean = np.array(self.config.get("data_preprocessor", {}).get("mean", [0.5, 0.5, 0.5])) / 255.0
            self.std = np.array(self.config.get("data_preprocessor", {}).get("std", [0.5, 0.5, 0.5])) / 255.0
            self.normalize = self.config.get("transform", {}).get("normalize", True)

        self.coco = COCO(ann_file)
        self.img_ids = list(self.coco.imgs.keys())
        assert len(self.img_ids) > 0, "❌ 数据集为空！"

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]  # 获取图像信息
        if img_info['wrist_view'] == False:
            return None

        img_path = os.path.join(self.data_root, img_info["file_name"])
        assis_img_path = os.path.join(self.data_root, img_info["assis_file_name"])

        # **检测双视角的图片是否存在**
        if not os.path.exists(img_path) or not os.path.exists(assis_img_path):
            print(f"⚠️ 警告: 图像 {img_path} or  {assis_img_path}不存在, 跳过该样本!")
            return None
        # **读取图像**
        img = cv2.imread(img_path)
        assis_img = cv2.imread(assis_img_path)
        if img is None or assis_img is None:
            print(f"⚠️ 警告: 读取 {img_path} ,or {assis_img_path} 失败, 跳过该样本!")
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB
        assis_img = cv2.cvtColor(assis_img, cv2.COLOR_BGR2RGB)

        # **加载 Annotation**
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # **COCO 可能有多个 annotation，默认取第一个**
        if len(anns) == 0:
            print(f"⚠️ 警告: 图像 {img_id} 没有 annotation, 跳过!")
            return None
        ann = anns[0]

        # **解析 Annotation**
        data_info = self.parse_data_info(ann, img_info, img_path)

        # **跳过标注缺失的样本**
        if data_info is None:
            return None
        # print(f"img_info:{img_info}")

        # 构建一个 results 字典，将所有增强需要的信息统一放在一起
        results = {
            "img": img,
            "assis_img": assis_img,
            "keypoints_2d": data_info["keypoints_2d"],
            "keypoints_3d": data_info["keypoints_3d"],
            "hand_typ": data_info["hand_type"],  # 统一键名为 "hand_typ"
            "image_type": data_info["image_type"],
            "hand_typ_valid": data_info["hand_type_valid"],
            "bbox": data_info["bbox"],
            "rel_root_depth": data_info["rel_root_depth"],
            "rel_root_valid": data_info["rel_root_valid"],
            # 初始化 meta_info 用于存放额外信息
            "meta_info": {}
        }

        # **数据增强**
        # 如果定义了 transform，则调用 transform( img, results )，要求每个 transform 都接收并返回 (img, results)
        if self.transform is not None:
            img, results = self.transform(img, results)
        results['img'] = img

        # 如果增强过程中产生了额外信息（例如 "heatmap_target"），将它移动到 meta_info 中
        if "heatmaps" in results.get("heatmap_target", {}):
            results["meta_info"]["heatmaps"] = results['heatmap_target']['heatmaps']
        if "keypoint_weights" in results.get("heatmap_target", {}):
            results['meta_info']['keypoint_weights'] = results['heatmap_target']['keypoint_weights']
        if "root_depth" in results.get("heatmap_target", {}):
            results['meta_info']['root_depth'] = results['heatmap_target']['root_depth']
        if "root_depth_weight" in results.get("heatmap_target", {}):
            results['meta_info']['root_depth_weight'] = results['heatmap_target']['root_depth_weight']
        if "type_weight" in results.get("heatmap_target", {}):
            results['meta_info']['type_weight'] = results['heatmap_target']['type_weight']
        if 'hand_typ' in results:
            results['meta_info']['hand_typ'] = results['hand_typ']
        if 'hand_typ_valid' in results:
            results['meta_info']['hand_typ_valid'] = results.pop('hand_typ_valid')
        ## TODO
        # 确保两图像尺寸一致
        if img.shape[:2] != assis_img.shape[:2]:
            assis_img = cv2.resize(assis_img, (img.shape[1], img.shape[0]))

        # 归一化（如果需要）
        if self.normalize:
            results["img"] = (results["img"] / 255.0 - self.mean) / self.std
            results["assis_img"] = (assis_img / 255.0 - self.mean) / self.std

        img = torch.tensor(results["img"], dtype=torch.float).permute(2, 0, 1)
        assis_img = torch.tensor(results["assis_img"], dtype=torch.float).permute(2, 0, 1)

        # 返回样本字典（转换为 PyTorch Tensor）
        return {
            "img": img,
            "assis_img": assis_img,
            "fusion_img": torch.cat([img,assis_img], dim=0),
            "keypoints_2d": torch.tensor(results["keypoints_2d"], dtype=torch.float),
            "keypoints_3d": torch.tensor(results["keypoints_3d"], dtype=torch.float),
            "hand_type": torch.tensor(results["hand_typ"], dtype=torch.float),
            "image_type": torch.tensor(results["image_type"], dtype=torch.float),
            # "hand_type_valid": torch.tensor(results["hand_typ_valid"], dtype=torch.float),
            "bbox": torch.tensor(results["bbox"], dtype=torch.float),
            "rel_root_depth": torch.tensor(results["rel_root_depth"], dtype=torch.float),
            "rel_root_valid": torch.tensor(results["rel_root_valid"], dtype=torch.float),
            "meta_info": results["meta_info"],
        }

    def parse_data_info(self, ann, img_info, img_path):
        """解析 COCO 标注数据，并将关键点深度转换为相对深度"""
        # **确保关键字段存在**
        required_keys = ["keypoints", "keypoints_3d", "bbox"]
        for key in required_keys:
            if key not in ann:
                print(f"⚠️ 警告: {img_path} 缺少 '{key}'，跳过该样本!")
                return None

        # **解析关键点**
        keypoints_2d = np.array(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
        keypoints_3d = np.array(ann["keypoints_3d"], dtype=np.float32).reshape(-1, 3)

        # **关键点可见性**
        joints_3d_visible = np.ones((keypoints_3d.shape[0],), dtype=np.float32)

        # **COCO 的 bbox (x, y, w, h) -> (x1, y1, x2, y2)**
        bbox = np.array(ann["bbox"], dtype=np.float32)
        bbox_xyxy = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], dtype=np.float32)

        # **解析 Hand Type**
        hand_type = img_info.get("hand_side", None)
        if hand_type == "right":
            hand_type = np.array([0, 1], dtype=np.float32)  # (left, right)
        elif hand_type == "left":
            hand_type = np.array([1, 0], dtype=np.float32)
        else:
            hand_type = np.array([0, 0], dtype=np.float32)  # 未知类型

        hand_type_valid = float(hand_type.sum() > 0)
        # **解析 image type , top or down**
        image_type = img_info.get("image_type", None)
        if image_type == "top":
            image_type = np.array([0, 1], dtype=np.float32) #(top, down)
        elif image_type == "down":
            image_type = np.array([1,0], dtype=np.float32)
        else:
            image_type = np.array([0,0], dtype=np.float32)


        # **计算相对深度**
        wrist_idx = 0  # 假设 wrist 关键点索引为 0
        if keypoints_3d.shape[0] > wrist_idx:
            wrist_depth = keypoints_3d[wrist_idx, 2]
        else:
            wrist_depth = 0.0

        # 将每个关键点的深度转换为相对于 wrist 的深度
        keypoints_3d[:, 2] = keypoints_3d[:, 2] - wrist_depth

        # 此时 wrist 的相对深度即为 0
        rel_root_depth = 0.0
        rel_root_valid = float(joints_3d_visible[wrist_idx] > 0)
        # print(f"wrist_depth:{wrist_depth}")
        return {
            "img_id": ann["image_id"],
            "img_path": img_path,
            "keypoints_2d": keypoints_2d,
            "keypoints_3d": keypoints_3d,
            "keypoints_visible": joints_3d_visible,
            "bbox": bbox_xyxy,
            "bbox_score": np.ones(1, dtype=np.float32),
            "hand_type": hand_type[np.newaxis, :],  # left , down
            "hand_type_valid": hand_type_valid,
            "rel_root_depth": np.array(wrist_depth),
            "rel_root_valid": np.array(rel_root_valid),
            "image_type": image_type, # top, down
        }

class LightHandDataset(Dataset):
    def __init__(self, config_path, mode="train", transform=None):
        assert os.path.exists(config_path), f"❌ 配置文件 {config_path} 不存在!"
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        assert mode in ["train", "val"], "❌ mode 必须是 'train' 或 'val'"

        self.data_root = self.config["dataset"][mode]["img_dir"]
        ann_file = self.config["dataset"][mode]["json_file"]
        self.transform = transform

        assert os.path.exists(ann_file), f"❌ 标注文件 {ann_file} 不存在!"
        assert os.path.exists(self.data_root), f"❌ 图片目录 {self.data_root} 不存在!"

        self.coco = COCO(ann_file)
        self.img_ids = list(self.coco.imgs.keys())
        assert len(self.img_ids) > 0, "❌ 数据集为空！请检查 JSON 文件！"

        self.mean = np.array(self.config["data_preprocessor"]["mean"]) / 255.0
        self.std = np.array(self.config["data_preprocessor"]["std"]) / 255.0
        self.normalize = self.config["transform"].get("normalize", False)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.data_root, img_info["file_name"])

        if not os.path.exists(img_path):
            print(f"⚠️ 图像 {img_path} 不存在")
            return None

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ 无法读取图像: {img_path}")
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        if len(anns) == 0:
            return None

        ann = anns[0]  # 默认使用第一个手部

        data_info = self.parse_data_info(ann, img_info, img_path)
        if data_info is None:
            return None

        results = {
            "img": img,
            "keypoints_2d": data_info["keypoints_2d"],
            "keypoints_3d": data_info["keypoints_3d"],
            "hand_typ": data_info["hand_type"],
            "hand_typ_valid": data_info["hand_type_valid"],
            "bbox": data_info["bbox"],
            "rel_root_depth": data_info["rel_root_depth"],
            "rel_root_valid": data_info["rel_root_valid"],
            "meta_info": {}
        }

        if self.transform:
            img, results = self.transform(img, results)
        results["img"] = img

        if "heatmaps" in results.get("heatmap_target", {}):
            results["meta_info"]["heatmaps"] = results["heatmap_target"]["heatmaps"]
        if "keypoint_weights" in results.get("heatmap_target", {}):
            results["meta_info"]["keypoint_weights"] = results["heatmap_target"]["keypoint_weights"]
        if "root_depth" in results.get("heatmap_target", {}):
            results["meta_info"]["root_depth"] = results["heatmap_target"]["root_depth"]
        if "root_depth_weight" in results.get("heatmap_target", {}):
            results["meta_info"]["root_depth_weight"] = results["heatmap_target"]["root_depth_weight"]
        if "type_weight" in results.get("heatmap_target", {}):
            results["meta_info"]["type_weight"] = results["heatmap_target"]["type_weight"]
        if 'hand_typ' in results:
            results['meta_info']['hand_typ'] = results['hand_typ']
        if 'hand_typ_valid' in results:
            results['meta_info']['hand_typ_valid'] = results.pop('hand_typ_valid')

        if self.normalize:
            results["img"] = (results["img"] / 255.0 - self.mean) / self.std

        return {
            "img": torch.tensor(results["img"], dtype=torch.float).permute(2, 0, 1),
            "keypoints_2d": torch.tensor(results["keypoints_2d"], dtype=torch.float),
            "keypoints_3d": torch.tensor(results["keypoints_3d"], dtype=torch.float),
            "hand_type": torch.tensor(results["hand_typ"], dtype=torch.float),
            "bbox": torch.tensor(results["bbox"], dtype=torch.float),
            "rel_root_depth": torch.tensor(results["rel_root_depth"], dtype=torch.float),
            "rel_root_valid": torch.tensor(results["rel_root_valid"], dtype=torch.float),
            "meta_info": results["meta_info"],
        }

    def parse_data_info(self, ann, img_info, img_path):
        # 解析 2D 关键点
        if "keypoints" not in ann:
            print(f"⚠️ COCO ann 中未包含 keypoints")
            return None
        keypoints_2d = np.array(ann["keypoints"], dtype=np.float32).reshape(-1, 3)

        # COCO 中一般没有 keypoints_3d，这里默认填充为 0
        keypoints_3d = np.zeros((keypoints_2d.shape[0], 3), dtype=np.float32)

        # bbox
        if "bbox" not in ann:
            print(f"⚠️ COCO ann 中未包含 bbox")
            return None
        bbox = np.array(ann["bbox"], dtype=np.float32)
        bbox_xyxy = np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]], dtype=np.float32)

        # hand type（COCO 一般没有 hand_side 字段）
        hand_type = np.array([0, 0], dtype=np.float32)
        hand_type_valid = 0.0

        # wrist 相对深度设为 0
        wrist_idx = 0
        rel_root_depth = np.array([0.0], dtype=np.float32)
        rel_root_valid = float(keypoints_2d[wrist_idx, 2] > 0)

        return {
            "img_id": ann["image_id"],
            "img_path": img_path,
            "keypoints_2d": keypoints_2d,
            "keypoints_3d": keypoints_3d,
            "bbox": bbox_xyxy,
            "hand_type": hand_type[np.newaxis, :],  # (1, 2)
            "hand_type_valid": hand_type_valid,
            "rel_root_depth": rel_root_depth,
            "rel_root_valid": rel_root_valid,
        }
