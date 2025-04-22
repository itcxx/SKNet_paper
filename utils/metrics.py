import numpy as np
import matplotlib.pyplot as plt

# # 批量将图像空间关键点与相对深度还原为相机坐标系下的 3D 坐标
# # keypoints_2d: (B, K, 2), rel_z: (B, K), root_depth: (B,), K: (3, 3)
# def backproject_batch(keypoints_2d, rel_z, root_depth, K):
#     fx, fy = K[0, 0], K[1, 1]
#     cx, cy = K[0, 2], K[1, 2]
#
#     z = root_depth[:, None] + rel_z  # 计算每个关键点的绝对深度 (B, K)
#     x = (keypoints_2d[:, :, 0] - cx) * z / fx
#     y = (keypoints_2d[:, :, 1] - cy) * z / fy
#
#     return np.stack([x, y, z], axis=2)  # 返回 (B, K, 3)
#
# # 在整个验证集上评估 MPJPE、3D PCK、AUC 和 2D PCK 等指标
# def evaluate_handpose_dataset(all_preds_2d, all_preds_z, all_gts_2d, all_gts_z, all_root_depths, K, all_masks=None,
#                                pck_threshold=20.0, auc_max_thresh=50.0, auc_resolution=100, pck_2d_thresh=10.0):
#     all_pred_3d, all_gt_3d, all_pred_2d, all_gt_2d, all_mask = [], [], [], [], []
#
#     for i in range(len(all_preds_2d)):
#         # 将预测和 GT 的 (x,y,z) 坐标恢复为相机坐标系下的三维点
#         pred_3d = backproject_batch(all_preds_2d[i], all_preds_z[i], all_root_depths[i], K)
#         gt_3d = backproject_batch(all_gts_2d[i], all_gts_z[i], all_root_depths[i], K)
#
#         all_pred_3d.append(pred_3d)
#         all_gt_3d.append(gt_3d)
#         all_pred_2d.append(all_preds_2d[i])
#         all_gt_2d.append(all_gts_2d[i])
#
#         if all_masks is not None:
#             all_mask.append(all_masks[i])
#         else:
#             all_mask.append(np.ones(pred_3d.shape[:2], dtype=bool))
#
#     # 合并所有 batch
#     pred_3d = np.concatenate(all_pred_3d, axis=0)
#     gt_3d = np.concatenate(all_gt_3d, axis=0)
#     pred_2d = np.concatenate(all_pred_2d, axis=0)
#     gt_2d = np.concatenate(all_gt_2d, axis=0)
#     mask = np.concatenate(all_mask, axis=0)
#
#     # 3D 误差与指标计算
#     dist_3d = np.linalg.norm(pred_3d - gt_3d, axis=2)[mask]
#     mpjpe = dist_3d.mean()
#     pck = (dist_3d < pck_threshold).mean()
#
#     thresholds = np.linspace(0, auc_max_thresh, auc_resolution)
#     pcks = [(dist_3d < t).mean() for t in thresholds]
#     auc = np.trapz(pcks, thresholds) / auc_max_thresh
#
#     # 2D 投影误差
#     dist_2d = np.linalg.norm(pred_2d - gt_2d, axis=2)[mask]
#     pck_2d = (dist_2d < pck_2d_thresh).mean()
#
#     return {
#         'MPJPE': mpjpe,
#         f'3D PCK@{pck_threshold}mm': pck,
#         '3D AUC': auc,
#         f'2D PCK@{pck_2d_thresh}px': pck_2d,
#         '3D PCK Curve': (thresholds, pcks)
#     }
#
# # 绘制 PCK 曲线图
# def plot_pck_curve(thresholds, pcks, label="3D PCK"):
#     plt.plot(thresholds, pcks, label=label)
#     plt.xlabel("Threshold (mm)")
#     plt.ylabel("PCK")
#     plt.title("3D PCK Curve")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
#
# # 将单个样本从 2D + 相对深度恢复到相机坐标系下的 3D 点
# def backproject_to_camera_coords(keypoints_2d, rel_depth, root_depth, K):
#     fx, fy = K[0, 0], K[1, 1]
#     cx, cy = K[0, 2], K[1, 2]
#     z = root_depth + rel_depth
#     x = (keypoints_2d[:, 0] - cx) * z / fx
#     y = (keypoints_2d[:, 1] - cy) * z / fy
#     return np.stack([x, y, z], axis=1)
#
# # 单样本 MPJPE 计算
# def compute_mpjpe(pred_3d, gt_3d, mask=None):
#     if mask is not None:
#         pred_3d, gt_3d = pred_3d[mask], gt_3d[mask]
#     return np.mean(np.linalg.norm(pred_3d - gt_3d, axis=1))
#
# # 单样本 PCK
#
# def compute_3d_pck(pred_3d, gt_3d, threshold=20.0, mask=None):
#     if mask is not None:
#         pred_3d, gt_3d = pred_3d[mask], gt_3d[mask]
#     dist = np.linalg.norm(pred_3d - gt_3d, axis=1)
#     return np.mean(dist < threshold)
#
# # 单样本 AUC
#
# def compute_auc(pred_3d, gt_3d, max_threshold=50.0, mask=None, resolution=100):
#     if mask is not None:
#         pred_3d, gt_3d = pred_3d[mask], gt_3d[mask]
#     dist = np.linalg.norm(pred_3d - gt_3d, axis=1)
#     thresholds = np.linspace(0, max_threshold, resolution)
#     pcks = [np.mean(dist < t) for t in thresholds]
#     auc = np.trapz(pcks, thresholds) / max_threshold
#     return auc, thresholds, pcks
#
# # 单样本 2D PCK
#
# def compute_2d_pck(pred_2d, gt_2d, threshold=10.0, mask=None):
#     if mask is not None:
#         pred_2d, gt_2d = pred_2d[mask], gt_2d[mask]
#     dist = np.linalg.norm(pred_2d - gt_2d, axis=1)
#     return np.mean(dist < threshold)
#
# # ==========================================================================================
# # ✅ 无相机内参时的评估（结构相对指标））这是针对没有真实的世界坐标，只是对图片2D坐标加相对深度的评价方法
# # ==========================================================================================
# def evaluate_relative_pose(pred_2d, pred_z, gt_2d, gt_z, mask=None, threshold_mm=20.0, threshold_px=10.0):
#     """
#     输入:
#     - pred_2d, gt_2d: (B, K, 2)
#     - pred_z, gt_z:   (B, K)
#     - mask: (B, K) 可选可见性
#     输出:
#     - 相对结构 MPJPE、PCK@20mm、2D PCK@10px
#     """
#     if mask is None:
#         mask = np.ones(pred_2d.shape[:2], dtype=bool)
#
#     # 相对坐标对齐（相对于根节点）
#     pred_3d = np.concatenate([pred_2d, pred_z[:, :, None]], axis=-1)
#     gt_3d = np.concatenate([gt_2d, gt_z[:, :, None]], axis=-1)
#     pred_3d = pred_3d - pred_3d[:, [0], :]  # 对齐根节点
#     gt_3d = gt_3d - gt_3d[:, [0], :]
#
#     # 误差计算
#     dist_3d = np.linalg.norm(pred_3d - gt_3d, axis=2)[mask]
#     rel_mpjpe = dist_3d.mean()
#     rel_pck = (dist_3d < threshold_mm).mean()
#
#     dist_2d = np.linalg.norm(pred_2d - gt_2d, axis=2)[mask]
#     pck_2d = (dist_2d < threshold_px).mean()
#
#     return {
#         f"Relative Position Error (mixed unit)": rel_mpjpe,
#         f"Rel. PCK@{threshold_mm}mm": rel_pck,
#         f"2D PCK@{threshold_px}px": pck_2d,
#     }
#
# # ==============================
# # ✅ 伪造数据用于测试评估流程（有无内参通用）
# # ==============================
# def generate_mock_handpose_data(batch_size=4, num_joints=21, with_intrinsics=True):
#     # 图像大小假设为 256x256
#     if with_intrinsics:
#         K = np.array([
#             [1000, 0, 128],
#             [0, 1000, 128],
#             [0, 0, 1]
#         ], dtype=np.float32)
#     else:
#         K = None
#
#     # GT 图像坐标 (B, K, 2)，在图像中心附近
#     gt_2d = np.random.uniform(64, 192, size=(batch_size, num_joints, 2))
#     gt_z = np.random.uniform(-20, 20, size=(batch_size, num_joints))  # 相对深度
#     root_depth = np.random.uniform(500, 600, size=(batch_size,))
#
#     # 预测为 GT 添加噪声
#     pred_2d = gt_2d + np.random.normal(0, 2, size=gt_2d.shape)  # 2D 偏差
#     pred_z = gt_z + np.random.normal(0, 1, size=gt_z.shape)     # Z 偏差
#
#     # 可见性 mask（全为可见）
#     visible = np.ones((batch_size, num_joints), dtype=bool)
#
#     return pred_2d, pred_z, gt_2d, gt_z, root_depth, K, visible
#
# # ✅ 示例调用方式：
# pred_2d, pred_z, gt_2d, gt_z, _, _, mask = generate_mock_handpose_data(with_intrinsics=False)
# print(f"pred_2d shape: {pred_2d.shape} , pred_z shape: {pred_z.shape}")
# results = evaluate_relative_pose(pred_2d, pred_z, gt_2d, gt_z, mask)
# print(results)




import numpy as np
import matplotlib.pyplot as plt

# ==============================
# ✅ 自动选择评估方法（有无相机内参）
# ==============================
def evaluate_pose_dataset(pred_2d, pred_z, gt_2d, gt_z, root_depth=None, K=None, mask=None,
                          pck_threshold=20.0, auc_max_thresh=50.0, auc_resolution=100, pck_2d_thresh=10.0):
    if K is not None and root_depth is not None:
        return evaluate_with_intrinsics(
            [pred_2d], [pred_z], [gt_2d], [gt_z], [root_depth], K, [mask] if mask is not None else None,
            pck_threshold, auc_max_thresh, auc_resolution, pck_2d_thresh
        )
    else:
        return evaluate_relative_pose(
            pred_2d, pred_z, gt_2d, gt_z, mask,
            threshold_mm=pck_threshold, threshold_px=pck_2d_thresh
        )
# ==============================
# ✅ 有相机内参时的评估
# ==============================
def backproject_batch(keypoints_2d, rel_z, root_depth, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = root_depth[:, None] + rel_z
    x = (keypoints_2d[:, :, 0] - cx) * z / fx
    y = (keypoints_2d[:, :, 1] - cy) * z / fy
    return np.stack([x, y, z], axis=2)

def evaluate_with_intrinsics(all_preds_2d, all_preds_z, all_gts_2d, all_gts_z, all_root_depths, K, all_masks=None,
                              pck_threshold=20.0, auc_max_thresh=50.0, auc_resolution=100, pck_2d_thresh=10.0):
    all_pred_3d, all_gt_3d, all_pred_2d, all_gt_2d, all_mask = [], [], [], [], []
    for i in range(len(all_preds_2d)):
        pred_3d = backproject_batch(all_preds_2d[i], all_preds_z[i], all_root_depths[i], K)
        gt_3d = backproject_batch(all_gts_2d[i], all_gts_z[i], all_root_depths[i], K)
        all_pred_3d.append(pred_3d)
        all_gt_3d.append(gt_3d)
        all_pred_2d.append(all_preds_2d[i])
        all_gt_2d.append(all_gts_2d[i])
        all_mask.append(all_masks[i] if all_masks else np.ones(pred_3d.shape[:2], dtype=bool))

    pred_3d = np.concatenate(all_pred_3d, axis=0)
    gt_3d = np.concatenate(all_gt_3d, axis=0)
    pred_2d = np.concatenate(all_pred_2d, axis=0)
    gt_2d = np.concatenate(all_gt_2d, axis=0)
    mask = np.concatenate(all_mask, axis=0)

    dist_3d = np.linalg.norm(pred_3d - gt_3d, axis=2)[mask]
    mpjpe = dist_3d.mean()
    pck = (dist_3d < pck_threshold).mean()
    thresholds = np.linspace(0, auc_max_thresh, auc_resolution)
    pcks = [(dist_3d < t).mean() for t in thresholds]
    auc = np.trapz(pcks, thresholds) / auc_max_thresh

    dist_2d = np.linalg.norm(pred_2d - gt_2d, axis=2)[mask]
    pck_2d = (dist_2d < pck_2d_thresh).mean()

    return {
        'MPJPE (mm)': mpjpe,
        f'3D PCK@{pck_threshold}mm': pck,
        '3D AUC': auc,
        f'2D PCK@{pck_2d_thresh}px': pck_2d,
        '3D PCK Curve': (thresholds, pcks)
    }

# ==============================
# ✅ 无内参时的结构相似性评估
# ==============================
def evaluate_relative_pose(pred_2d, pred_z, gt_2d, gt_z, mask=None, threshold_mm=20.0, threshold_px=10.0):
    if mask is None:
        mask = np.ones(pred_2d.shape[:2], dtype=bool)

    pred_3d = np.concatenate([pred_2d, pred_z[:, :, None]], axis=-1)
    gt_3d = np.concatenate([gt_2d, gt_z[:, :, None]], axis=-1)
    pred_3d = pred_3d - pred_3d[:, [0], :]
    gt_3d = gt_3d - gt_3d[:, [0], :]

    dist_3d = np.linalg.norm(pred_3d - gt_3d, axis=2)[mask]
    rel_mpjpe = dist_3d.mean()
    rel_pck = (dist_3d < threshold_mm).mean()

    dist_2d = np.linalg.norm(pred_2d - gt_2d, axis=2)[mask]
    pck_2d = (dist_2d < threshold_px).mean()

    return {
        f"Relative Pos. Error (px+mm mix)": rel_mpjpe,
        f"Rel. PCK@{threshold_mm}mm": rel_pck,
        f"2D PCK@{threshold_px}px": pck_2d,
    }

# ==============================
# ✅ 可视化 PCK 曲线
# ==============================
def plot_pck_curve(thresholds, pcks, label="3D PCK"):
    plt.plot(thresholds, pcks, label=label)
    plt.xlabel("Threshold (mm)")
    plt.ylabel("PCK")
    plt.title("3D PCK Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# =====================================
# ✅ 伪造数据用于测试评估流程（有无内参通用）
# =====================================
# def generate_mock_handpose_data(batch_size=4, num_joints=21, with_intrinsics=True):
#     if with_intrinsics:
#         K = np.array([[1000, 0, 128], [0, 1000, 128], [0, 0, 1]], dtype=np.float32)
#     else:
#         K = None
#
#     gt_2d = np.random.uniform(64, 192, size=(batch_size, num_joints, 2))
#     gt_z = np.random.uniform(-20, 20, size=(batch_size, num_joints))
#     root_depth = np.random.uniform(500, 600, size=(batch_size,))
#
#     pred_2d = gt_2d + np.random.normal(0, 2, size=gt_2d.shape)
#     pred_z = gt_z + np.random.normal(0, 1, size=gt_z.shape)
#
#     visible = np.ones((batch_size, num_joints), dtype=bool)
#     return pred_2d, pred_z, gt_2d, gt_z, root_depth, K, visible
#

# ==============================
# ✅ 测试评估 + 可视化调用
# ==============================
# if __name__ == '__main__':
#     pred_2d, pred_z, gt_2d, gt_z, root_depth, K, mask = generate_mock_handpose_data(with_intrinsics=True)
#
#     results = evaluate_pose_dataset(pred_2d, pred_z, gt_2d, gt_z, root_depth, K, mask)
#     for k, v in results.items():
#         if isinstance(v, float):
#             print(f"{k}: {v:.2f}")
#
#     if '3D PCK Curve' in results:
#         thresholds, pcks = results['3D PCK Curve']
#         print(f"thresholds: {thresholds}, pcks: {pcks}")
#         plot_pck_curve(thresholds, pcks)
