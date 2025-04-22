import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2
import argparse
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from datasets import HandPoseDataset, LightHandDataset, HandPoseDatasetTw
from codecs_ import HandPoseCodec
from utils.build_pipeline import *

def parse_args():
    """解析命令行参数"""
    # 获取当前脚本的目录
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # 拼接 config.yaml 绝对路径
    # CONFIG_PATH = os.path.join(BASE_DIR, "config", "config_Interhand99k_model01.yaml")
    CONFIG_PATH = os.path.join(BASE_DIR, "config", "config_HandPose_TwoInput.yaml")
    parser = argparse.ArgumentParser(description="可视化手部数据集")
    # parser.add_argument("--config", type=str, default="../config/config.yaml", help="YAML配置文件路径")
    parser.add_argument("--config", type=str, default=CONFIG_PATH, help="YAML配置文件路径")
    parser.add_argument("--num_images", type=int, default=900, help="可视化的图片数量")
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
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # 拇指
        (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
        (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
        (0, 13), (13, 14), (14, 15), (15, 16),  # 无名指
        (0, 17), (17, 18), (18, 19), (19, 20),  # 小指
        (5, 9), (9, 13), (13, 17)
    ]
    for x, y in keypoints:
        cv2.circle(img, (int(x), int(y)), 5, (255, 0, 0), -1)
    for conn in HAND_CONNECTIONS:
        pt1, pt2 = keypoints[conn[0]], keypoints[conn[1]]
        # if pt1[2] > 0 and pt2[2] > 0:  # 仅当两个关键点均可见时才绘制连接线
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0]), int(pt2[1])
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
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
    img_overlay = cv2.addWeighted(img, 0.5, heatmap_color, 0.3, 0)

    return img_overlay
def visualize_heatmap_3d_all(heatmaps_3d, threshold_ratio=0.2, save=False, save_path="heatmap_3d_all.png"):
    """
    可视化所有关键点的 3D Heatmap，将各个关键点的热图堆叠在一起展示。

    参数:
        heatmaps_3d (numpy.ndarray): 3D 热图，形状为 (K, D, H, W)，其中 K 为关键点个数
        threshold_ratio (float): 用于筛选热图值的阈值比例 (热图最大值乘以该比例, 默认 0.2)
        save (bool): 是否保存图像 (默认 False)
        save_path (str): 保存图像的路径 (默认 "heatmap_3d_all.png")
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # 新版本 matplotlib 可不显式导入

    K, D, H, W = heatmaps_3d.shape

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 为每个关键点生成不同的颜色 (使用 hsv 颜色映射)
    colors = plt.cm.hsv(np.linspace(0, 1, K))

    # 遍历每个关键点通道
    for k in range(K):
        hm = heatmaps_3d[k]  # 取出第 k 个关键点的热图，形状为 (D, H, W)
        # 计算阈值，筛选热图中大于此阈值的点
        threshold = hm.max() * threshold_ratio
        d_idx, h_idx, w_idx = np.where(hm > threshold)
        if len(d_idx) == 0:
            continue
        # 根据热图值获得每个点的 intensity
        intensities = hm[d_idx, h_idx, w_idx]
        # 将 intensity 归一化后映射到 marker 大小 (例如 10~50)
        size = 10 + (intensities - intensities.min()) / (intensities.max() - intensities.min() + 1e-5) * 40

        ax.scatter(w_idx, h_idx, d_idx,
                   color=colors[k],
                   s=size,
                   alpha=0.6,
                   label=f"Joint {k}")

    ax.set_xlabel('Width (W)')
    ax.set_ylabel('Height (H)')
    ax.set_zlabel('Depth (D)')
    ax.view_init(elev=30, azim=45)
    plt.title('3D Heatmap Visualization for All Joints')

    # 如果关键点数较少，则显示全部图例，否则仅显示前几个
    if K <= 10:
        ax.legend()
    else:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:min(10, len(handles))], labels[:min(10, len(labels))],
                  title="Joints (first 10)", loc="upper right")

    if save:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 3D 热图已保存到: {save_path}")
    plt.show()
def visualize_heatmap_3d_cube(heatmaps_3d, num_slices=5, save=False, save_path="heatmap_3d_cube.png"):
    """
    可视化所有关键点的 3D 热图，以正方体切片的形式展示。

    参数:
        heatmaps_3d (numpy.ndarray): 3D 热图，形状为 (K, D, H, W)，K 为关键点数量
        num_slices (int): 每个方向上显示的切片数量 (默认 5)
        save (bool): 是否保存图像 (默认 False)
        save_path (str): 保存图像的路径 (默认 "heatmap_3d_cube.png")
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # 对新版 matplotlib 可省略
    from matplotlib import cm
    import numpy as np

    # 聚合所有关键点热图，采用各位置的最大值（也可以换成均值）
    volume = np.max(heatmaps_3d, axis=0)  # shape: (D, H, W)
    D, H, W = volume.shape

    # 将体数据归一化到 [0, 1]
    vol_min, vol_max = volume.min(), volume.max()
    if vol_max - vol_min > 0:
        volume_norm = (volume - vol_min) / (vol_max - vol_min)
    else:
        volume_norm = volume.copy()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_zlim(0, D)
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    ax.set_zlabel('Depth')

    alpha = 0.1  # 切片透明度

    # ---------------------------
    # 沿轴向（Axial）方向：XY平面上的切片 (固定 z 值)
    # ---------------------------
    z_indices = np.linspace(0, D - 1, num_slices, dtype=int)
    for z in z_indices:
        slice_img = volume_norm[z, :, :]  # shape: (H, W)
        # 构建 XY 网格
        x = np.linspace(0, W, W)
        y = np.linspace(0, H, H)
        X, Y = np.meshgrid(x, y)
        Z = np.full_like(X, z)
        # 将灰度切片映射为彩色 (RGBA)
        colors = cm.jet(slice_img)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=colors,
                        shade=False, alpha=alpha, linewidth=0, antialiased=False)

    # ---------------------------
    # 沿矢状面（Sagittal）方向：YZ平面上的切片 (固定 x 值)
    # ---------------------------
    x_indices = np.linspace(0, W - 1, num_slices, dtype=int)
    for x in x_indices:
        slice_img = volume_norm[:, :, x]  # shape: (D, H)
        # 构建 YZ 网格
        y = np.linspace(0, H, H)
        z = np.linspace(0, D, D)
        Y, Z = np.meshgrid(y, z)
        X = np.full_like(Y, x)
        colors = cm.jet(slice_img)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=colors,
                        shade=False, alpha=alpha, linewidth=0, antialiased=False)

    # ---------------------------
    # 沿冠状面（Coronal）方向：XZ平面上的切片 (固定 y 值)
    # ---------------------------
    y_indices = np.linspace(0, H - 1, num_slices, dtype=int)
    for y in y_indices:
        slice_img = volume_norm[:, y, :]  # shape: (D, W)
        # 构建 XZ 网格
        x = np.linspace(0, W, W)
        z = np.linspace(0, D, D)
        X, Z = np.meshgrid(x, z)
        Y = np.full_like(X, y)
        colors = cm.jet(slice_img)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=colors,
                        shade=False, alpha=alpha, linewidth=0, antialiased=False)

    ax.view_init(elev=30, azim=45)
    plt.title("3D Heatmap Cube with Slices")

    if save:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 3D Cube Heatmap 已保存到: {save_path}")
    plt.show()
def visualize_depth_slices(heatmaps_3d, num_slices=5, alpha=0.6, save=False, save_path="depth_slices.png"):
    """
    可视化 3D 热图在不同深度切片的效果，将每个选定的深度切片以 2D 图像显示，并增加透明度。

    参数:
        heatmaps_3d (numpy.ndarray): 3D 热图，形状为 (K, D, H, W)，K 为关键点数量
        num_slices (int): 选择展示的深度切片数量（默认 5）
        alpha (float): 透明度参数，取值范围 [0,1] (默认 0.6)
        save (bool): 是否保存生成的图像 (默认 False)
        save_path (str): 保存图像的路径 (默认 "depth_slices.png")
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm

    # 聚合所有关键点的热图，采用各位置的最大值（也可以换成均值）
    volume = np.max(heatmaps_3d, axis=0)  # shape: (D, H, W)
    D, H, W = volume.shape

    # 将体数据归一化到 [0, 1]，便于颜色映射
    vol_min, vol_max = volume.min(), volume.max()
    if vol_max - vol_min > 0:
        volume_norm = (volume - vol_min) / (vol_max - vol_min)
    else:
        volume_norm = volume.copy()

    # 选择沿深度方向均匀的切片索引
    depth_indices = np.linspace(0, D - 1, num_slices, dtype=int)

    # 设置图像排列：这里采用一行多个子图的布局
    fig, axes = plt.subplots(1, num_slices, figsize=(4 * num_slices, 4))
    if num_slices == 1:
        axes = [axes]

    for i, z in enumerate(depth_indices):
        # 取出深度为 z 的切片，形状 (H, W)
        slice_img = volume_norm[z, :, :]
        axes[i].imshow(slice_img, cmap='jet', alpha=alpha)
        axes[i].set_title(f"Depth Slice {z}")
        axes[i].axis('off')

    plt.suptitle("Depth Slices of 3D Heatmap", fontsize=16)
    plt.tight_layout()

    if save:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Depth slices visualization 已保存到: {save_path}")
    plt.show()
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

def visualize(dataset, codec, num_images, save, save_path, mean, std):
    """可视化数据集"""
    os.makedirs(save_path, exist_ok=True)
    index = 0  # 初始化索引

    while index < num_images:
        sample = dataset[index]
        if sample == None:
            index += 1
            continue
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

        print(f"true encoder input:")
        print(f"GT keypoints2d: {keypoints_2d}")
        ##########不用上面的encode生成热图， 增强方法中设置GenerateTarget以后就生成了热图############
        heatmaps_3d = sample["meta_info"]["heatmaps"]  # 提取编码后的 3D Heatmap

        # print(f"heatmaps_3d:{heatmaps_3d}")

        print(f"heatmaps_3d:{heatmaps_3d.shape}")
        keypoints_weights = sample["meta_info"]['keypoint_weights']

        ##################################
        # model predict keypoints test
        ##################################
        from models.heads.hand_pose_head import HandPoseHead
        import torch
        head = HandPoseHead(in_channels=512, num_joints=21, depth_size=64,heatmap_size=[64,64,64])
        keypoints, scores = head.get_heatmap_3d_maximum( torch.tensor(np.expand_dims(heatmaps_3d, axis=0)),
                                                         image_size=[256,256])
        # print(f"head get_heatmap_3d_maximum : {keypoints}")

        keypoints_soft = head.soft_argmax_2d(torch.tensor(np.expand_dims(heatmaps_3d, axis=0)))
        keypoints_soft = keypoints_soft * 4.0
        print(f"after soft_argmax_2d * 4.0:{keypoints_soft }")

        keypoints_soft_3d,_ = head.soft_argmax_3d(torch.tensor(np.expand_dims(heatmaps_3d, axis=0)))
        keypoints_soft_3d = keypoints_soft_3d * 4.0
        print(f"after soft_argmax_3d * 4.0:{keypoints_soft_3d}")
        ################################################

        decoded = codec.decode(heatmaps = heatmaps_3d, 
                               root_depth = np.expand_dims(sample["rel_root_depth"].numpy(), axis=0),
                               hand_type = np.expand_dims(sample["hand_type"].numpy(), axis=0))
        decode_keypoints, decode_scores, decode_rel_root_depth, decode_hand_type = decoded
        # print(f"after decoded:{ decode_keypoints} ,{decode_scores}, {decode_rel_root_depth}, {decode_hand_type}")

        # **可视化**
        # img_with_kpts = draw_keypoints(img.copy(), keypoints_soft[0]) # use soft_argmax_2d result
        # img_with_kpts = draw_keypoints(img.copy(), keypoints_soft_3d[0]) # use soft_argmax_3d result
        img_with_kpts = draw_keypoints(img.copy(), keypoints[0,:,:2])  # use heatmap_3d_maximum
        img_with_heatmap = visualize_heatmaps(img_with_kpts, heatmaps_3d)

        # **显示图像**
        cv2.imshow(f"Image {index + 1}/{num_images}", img_with_heatmap)

        # # 假设 heatmaps_3d 的 shape 为 (K, D, H, W)
        # visualize_heatmap_3d_all(heatmaps_3d, threshold_ratio=0.2, save=False, save_path="3d_heatmap_all.png")
        # # 将所有关键点的 3D 热图聚合后，以切片的方式展示
        # visualize_heatmap_3d_cube(heatmaps_3d, num_slices=5, save=False, save_path="3d_heatmap_cube.png")

        # 假设 heatmaps_3d 的 shape 为 (K, D, H, W)
        # visualize_depth_slices(heatmaps_3d, num_slices=10, alpha=0.6, save=False, save_path="depth_slices.png")

        #在所有深度上聚合（如求和或最大），然后显示每个关键点在 H × W 平面上的响应分布

        visualize_keypoint_maps(
            heatmaps_3d=heatmaps_3d,  # 取第一张样本 (K, D, H, W)
            num_joints_to_show=15,
            method="max",  # 或者 "sum"
            alpha=0.8
        )

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


def visualizeTw(dataset, codec, num_images, save, save_path, mean, std):
    """可视化数据集"""
    os.makedirs(save_path, exist_ok=True)
    index = 300  # 初始化索引

    while index < num_images:
        cv2.destroyAllWindows()
        sample = dataset[index]
        if sample == None:
            index += 1
            continue
        img = sample["img"].permute(1, 2, 0).numpy()  # **反归一化**
        img = unnormalize(img, mean, std)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # **转换为 BGR 以供 OpenCV 显示**

        imgA = sample["assis_img"].permute(1,2,0).numpy()
        imgA = unnormalize(imgA, mean, std)
        imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)

        keypoints_2d = sample["keypoints_2d"].numpy()
        # **从 keypoints_2d[:, 2] 取出可见性**
        keypoints_visible = keypoints_2d[:, 2].astype(np.float32)
        keypoints_3d = np.expand_dims(sample["keypoints_3d"].numpy(), axis=0)  # (1, 21, 3)
        keypoints_2d = np.expand_dims(keypoints_2d, axis=0)  # (1,21,3)
        # 将真实深度值传入到2d点第三个维度
        keypoints_2d[:, :, 2] = keypoints_3d[:, :, 2] * 1000  # 换成毫米单位然后传入

        print(f"true encoder input:")
        print(f"GT keypoints2d: {keypoints_2d}")
        ##########不用上面的encode生成热图， 增强方法中设置GenerateTarget以后就生成了热图############
        heatmaps_3d = sample["meta_info"]["heatmaps"]  # 提取编码后的 3D Heatmap

        print(f"heatmaps_3d:{heatmaps_3d}")

        print(f"heatmaps_3d:{heatmaps_3d.shape}")
        keypoints_weights = sample["meta_info"]['keypoint_weights']

        ##################################
        # model predict keypoints test
        ##################################
        from models.heads.hand_pose_head import HandPoseHead
        import torch
        head = HandPoseHead(in_channels=512, num_joints=21, depth_size=64, heatmap_size=[64, 64, 64])
        keypoints, scores = head.get_heatmap_3d_maximum(torch.tensor(np.expand_dims(heatmaps_3d, axis=0)),
                                                        image_size=[256, 256])
        # print(f"head get_heatmap_3d_maximum : {keypoints}")

        keypoints_soft = head.soft_argmax_2d(torch.tensor(np.expand_dims(heatmaps_3d, axis=0)))
        keypoints_soft = keypoints_soft[..., :2] * 4.0
        print(f"after soft_argmax_2d:{keypoints_soft }")

        keypoints_soft_3d, _ = head.soft_argmax_3d(torch.tensor(np.expand_dims(heatmaps_3d, axis=0)))
        print(f"after soft_argmax_3d:{keypoints_soft_3d}")
        keypoints_soft_3d = keypoints_soft_3d[..., :2] * 4.0
        print(f"after soft_argmax_3d by scaling:{keypoints_soft_3d}")
        ################################################

        decoded = codec.decode(heatmaps=heatmaps_3d,
                               root_depth=np.expand_dims(sample["rel_root_depth"].numpy(), axis=0),
                               hand_type=np.expand_dims(sample["hand_type"].numpy(), axis=0))
        decode_keypoints, decode_scores, decode_rel_root_depth, decode_hand_type = decoded
        print(f"after decoded:{ decode_keypoints} ,{decode_scores}, {decode_rel_root_depth}, {decode_hand_type}")

        # **可视化**
        img_with_kpts = draw_keypoints(img.copy(), keypoints_soft[0]) # use soft_argmax_2d result
        # img_with_kpts = draw_keypoints(img.copy(), keypoints_soft_3d[0]) # use soft_argmax_3d result
        # img_with_kpts = draw_keypoints(img.copy(), keypoints[0])  # use heatmap_3d_maximum
        img_with_heatmap = visualize_heatmaps(img_with_kpts, heatmaps_3d)

        # **显示图像**
        # resize 两张图像到相同大小（防止 shape 不一致导致拼接失败）
        h, w = img_with_heatmap.shape[:2]
        imgA_resized = cv2.resize(imgA, (w, h))

        # 拼接两个图像：横向
        combined_img = np.hstack((img_with_heatmap, imgA_resized))
        cv2.imshow(f"Combined View -{index + 1}/{num_images}", combined_img)
        # cv2.imshow(f"Image {index + 1}/{num_images}", img_with_heatmap)
        # cv2.imshow(f"Ass Image {index + 1}/{num_images}", imgA)

        # # 假设 heatmaps_3d 的 shape 为 (K, D, H, W)
        # visualize_heatmap_3d_all(heatmaps_3d, threshold_ratio=0.2, save=False, save_path="3d_heatmap_all.png")
        # # 将所有关键点的 3D 热图聚合后，以切片的方式展示
        # visualize_heatmap_3d_cube(heatmaps_3d, num_slices=5, save=False, save_path="3d_heatmap_cube.png")

        # 假设 heatmaps_3d 的 shape 为 (K, D, H, W) 按照 深度维度切片，把 所有关键点 K 的最大值融合成一个体数据 (D, H, W) 来可视化不同深度层
        # visualize_depth_slices(heatmaps_3d, num_slices=15, alpha=0.6, save=False, save_path="depth_slices.png")

        #在所有深度上聚合（如求和或最大），然后显示每个关键点在 H × W 平面上的响应分布
        visualize_keypoint_maps(
            heatmaps_3d=heatmaps_3d,  # 取第一张样本 (K, D, H, W)
            num_joints_to_show=15,
            method="max",  # 或者 "sum"
            alpha=0.8
        )

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

    transf = build_pipeline(config, pipeline_key = 'val_pipeline')

    # select dataset
    dataset_name = config.get("dataset_name",'Interhand99k')
    input_views = config.get("input_views", None)
    if dataset_name == 'HandPose':
        if input_views == 2:
            dataset = HandPoseDatasetTw(config_path=args.config , mode ="train", transform=transf)
        else:
            dataset = HandPoseDataset(config_path=args.config, mode="train", transform=transf)
    else:
        dataset = LightHandDataset(config_path=args.config, mode="train", transform=transf)

    #  **创建 HandPoseCodec**
    codec = HandPoseCodec(image_size=[256,256],
                          heatmap_size=[64,64,64],  #D , H, W
                          heatmap3d_depth_bound=300.0,
                          root_depth_bound=300.0,
                          sigma=1,
                          max_bound=1.0
                         )

    # visualize(dataset, codec, args.num_images, args.save, args.save_path, mean, std)
    #
    visualizeTw(dataset, codec, args.num_images, args.save, args.save_path, mean, std)