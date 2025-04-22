import sys
import os
import cv2
import numpy as np
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.registry import register_transform
from codecs_ import HandPoseCodec

# ------------------------
# Compose: 组合多个 transform
# ------------------------
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, results):
        for t in self.transforms:
            img, results = t(img, results)
        return img, results

# ------------------------
# LoadImage: 读取图像
# ------------------------
@register_transform
class LoadImage:
    def __init__(self, **kwargs):
        pass

    def __call__(self, img, results):
        # 这里假设 dataset 已经读取了图像，所以直接返回
        return img, results
# ------------------------
# Resize: 将图像调整到目标尺寸，同时更新关键点
# ------------------------
@register_transform
class Resize:
    def __init__(self, target_size):
        """
        target_size: [width, height]
        """
        self.target_size = tuple(target_size)

    def __call__(self, img, results):
        original_size = (img.shape[1], img.shape[0])
        img_resized = cv2.resize(img, self.target_size)
        scale_w = self.target_size[0] / original_size[0]
        scale_h = self.target_size[1] / original_size[1]
        # 更新 2D 关键点（前两维）
        results["keypoints_2d"][:, :2] *= np.array([scale_w, scale_h])
        # 更新 3D 关键点的前两维
        results["keypoints_3d"][:, :2] *= np.array([scale_w, scale_h])
        # 更新 bbox（假设格式为 [x1, y1, x2, y2]）
        bbox = results["bbox"].copy()
        bbox[0] *= scale_w
        bbox[1] *= scale_h
        bbox[2] *= scale_w
        bbox[3] *= scale_h
        results["bbox"] = bbox
        return img_resized, results

# ------------------------
# GetBBoxCenterScale: 根据 bbox 计算中心和尺度
# ------------------------

@register_transform
class GetBBoxCenterScale:
    def __init__(self, padding=1.25):
        self.padding = padding

    def __call__(self, img, results):
        bbox = results["bbox"]  # [x1, y1, x2, y2]
        center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        scale = np.array([bbox[2] - bbox[0], bbox[3] - bbox[1]], dtype=np.float32) * self.padding
        results["center"] = center
        results["scale"] = scale
        return img, results


# ------------------------
# TopdownAffine: 利用中心和尺度进行仿射变换
# ------------------------
@register_transform
class TopdownAffine:
    def __init__(self, input_size):
        self.input_size = tuple(input_size)

    def get_affine_transform(self, center, scale, rot, output_size, inv=0):
        src_w = scale[0]
        dst_w, dst_h = output_size
        rot_rad = np.pi * rot / 180
        # 构造源坐标
        src_dir = np.array([0, -src_w * 0.5], dtype=np.float32)
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        src_dir = np.array([src_dir[0]*cs - src_dir[1]*sn, src_dir[0]*sn + src_dir[1]*cs], dtype=np.float32)
        dst_dir = np.array([0, -dst_w * 0.5], dtype=np.float32)
        src = np.zeros((3,2), dtype=np.float32)
        dst = np.zeros((3,2), dtype=np.float32)
        src[0,:] = center
        src[1,:] = center + src_dir
        src[2,:] = [center[0] - src_dir[1], center[1] + src_dir[0]]
        dst[0,:] = [dst_w*0.5, dst_h*0.5]
        dst[1,:] = dst[0,:] + dst_dir
        dst[2,:] = [dst[0,0] - dst_dir[1], dst[0,1] + dst_dir[0]]
        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        return trans

    def __call__(self, img, results):
        center = results["center"]
        scale = results["scale"]
        trans = self.get_affine_transform(center, scale, rot=0, output_size=self.input_size)
        img_trans = cv2.warpAffine(img, trans, self.input_size)
        # 更新 2D 关键点
        pts = np.concatenate([results["keypoints_2d"][:, :2], np.ones((results["keypoints_2d"].shape[0], 1), dtype=np.float32)], axis=1)
        pts_trans = np.dot(trans, pts.T).T
        results["keypoints_2d"][:, :2] = pts_trans
        results["keypoints_3d"][:, :2] = pts_trans  # 同步更新 3D 关键点前两维
        return img_trans, results

# ------------------------
# RandomFlip: 随机水平翻转
# ------------------------
@register_transform
class RandomFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, results):
        if random.random() < self.prob:
            img = cv2.flip(img, 1)
            width = img.shape[1]
            results["keypoints_2d"][:, 0] = width - results["keypoints_2d"][:, 0] - 1
            results["keypoints_3d"][:, 0] = width - results["keypoints_3d"][:, 0] - 1
            bbox = results["bbox"]
            bbox[0], bbox[2] = width - bbox[2], width - bbox[0]
            results["bbox"] = bbox
        return img, results
# ------------------------
# RandomRotation: 随机旋转
# ------------------------
@register_transform
class RandomRotation:
    def __init__(self, angle_range):
        self.angle_range = angle_range

    def __call__(self, img, results):
        angle = random.uniform(*self.angle_range)
        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_rot = cv2.warpAffine(img, M, (w, h))
        pts = np.concatenate([results["keypoints_2d"][:, :2], np.ones((results["keypoints_2d"].shape[0], 1), dtype=np.float32)], axis=1)
        pts_rot = np.dot(M, pts.T).T
        results["keypoints_2d"][:, :2] = pts_rot
        pts3d = np.concatenate([results["keypoints_3d"][:, :2], np.ones((results["keypoints_3d"].shape[0], 1), dtype=np.float32)], axis=1)
        pts3d_rot = np.dot(M, pts3d.T).T
        results["keypoints_3d"][:, :2] = pts3d_rot
        return img_rot, results
# ------------------------
# RandomScale: 随机缩放
# ------------------------

@register_transform
class RandomScale:
    def __init__(self, scale_range):
        self.scale_range = scale_range

    def __call__(self, img, results):
        scale = random.uniform(*self.scale_range)
        h, w = img.shape[:2]
        img_scaled = cv2.resize(img, (int(w * scale), int(h * scale)))
        results["keypoints_2d"][:, :2] *= scale
        results["keypoints_3d"][:, :2] *= scale
        bbox = np.array(results["bbox"])
        bbox *= scale
        results["bbox"] = bbox
        return img_scaled, results
# ------------------------
# GenerateTarget: 利用 encoder 生成目标热图
# ------------------------
@register_transform
class GenerateTarget:
    def __init__(self, name='HandPoseCodec',image_size=[256,256],
                 heatmap_size=[64,64,64],heatmap3d_depth_bound=400.0,
                root_depth_bound=300.0, sigma=2.0, max_bound=1.0):
        ####
        self.codec = HandPoseCodec(
                          image_size=image_size,
                          heatmap_size=heatmap_size,  #D , H, W
                          heatmap3d_depth_bound=heatmap3d_depth_bound,
                          root_depth_bound=root_depth_bound,
                          sigma=sigma,
                          max_bound=max_bound
                         )

    def __call__(self, img, results):
        # keypoints = results["keypoints_3d"]  # (K, 3)
        # print(f"in Generate Target:")
        keypoints_2d = results['keypoints_2d'] # (21,3)
        # print(f"keypoints_2d:{keypoints_2d.shape}")
        keypoints_visible = keypoints_2d[:, 2].astype(np.float32)
        keypoints_3d = np.expand_dims(results['keypoints_3d'],axis=0) #(1,21,3)
        # print(f"keypoints_3d:{keypoints_3d.shape}")
        keypoints_2d = np.expand_dims(keypoints_2d, axis=0) # (1,21,3)
        # print(f"keypoints_2d:{keypoints_2d.shape}")
        # 将真实深度值传入到2d点第三个维度
        keypoints_2d[:,:,2] = keypoints_3d[:,:,2] * 1000 # 换成毫米单位然后传入
        
        keypoints_2d = keypoints_2d  # x,y is < 256 , z is mm
        keypoints_v = np.expand_dims(keypoints_visible, axis=0)
        rel_root_depth = np.expand_dims(results["rel_root_depth"], axis=0)
        rel_root_valid = np.expand_dims(results["rel_root_valid"], axis=0)
        hand_type = results["hand_typ"]
        hand_type_valid = results["hand_typ_valid"]

        targets = self.codec.encode( #
            keypoints_2d,
            keypoints_v,
            rel_root_depth,
            rel_root_valid,
            hand_type,
            hand_type_valid
        )

        results["heatmap_target"] = targets
        # print(f"targets:{targets}")

        return img, results
# ------------------------
# PackPoseInputs: 打包输入数据
# ------------------------

@register_transform
class PackPoseInputs:
    def __init__(self, meta_keys=None):
        self.meta_keys = meta_keys if meta_keys is not None else []

    def __call__(self, img, results):
        data_sample = {"img": img}
        meta_info = {}
        for key in self.meta_keys:
            meta_info[key] = results.get(key, None)
        data_sample["meta_info"] = meta_info
        return data_sample
