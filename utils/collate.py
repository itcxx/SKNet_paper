import torch

def merge_nested(values):
    """
    递归合并列表中的元素。
    如果元素为 dict，则对每个键进行递归合并；否则尝试使用 torch.stack 合并。
    """
    # 如果列表中第一个元素为 dict，则对每个键递归处理
    if isinstance(values[0], dict):
        merged = {}
        for key in values[0]:
            sub_vals = [v[key] for v in values]
            merged[key] = merge_nested(sub_vals)
        return merged
    else:
        try:
            # 尝试将所有元素转为 tensor 并堆叠
            return torch.stack([v if isinstance(v, torch.Tensor) else torch.tensor(v) for v in values], dim=0)
        except Exception:
            # 如果堆叠失败，则直接返回列表
            return values


def custom_collate_fn(batch):
    """
    将一个 batch（列表，每个元素是样本字典）合并为一个字典。
    对于能堆叠的字段（img、keypoints_2d、keypoints_3d、hand_type、bbox 等）使用 torch.stack，
    对于 meta_info 字段采用递归合并。
    """
    # 过滤掉 None 样本
    batch = [sample for sample in batch if sample is not None]
    if len(batch) == 0:
        return None

    collated = {}
    # 对固定尺寸的字段直接 torch.stack
    collated["img"] = torch.stack([s["img"] for s in batch], dim=0)                   # (B, C, H, W)
    collated["keypoints_2d"] = torch.stack([s["keypoints_2d"] for s in batch], dim=0)     # (B, K, 3)
    collated["keypoints_3d"] = torch.stack([s["keypoints_3d"] for s in batch], dim=0)     # (B, K, 3)
    # collated["fusion_img"] = torch.stack([s["fusion_img"] for s in batch], dim=0)  # (B, K, 3)
    # collated["hand_type"] = torch.stack([s["hand_type"] for s in batch], dim=0)           # (B, 1, 2)
    # collated["hand_type_valid"] = torch.stack([s["hand_type_valid"] for s in batch], dim=0).view(-1, 1)
    collated["bbox"] = torch.stack([s["bbox"] for s in batch], dim=0)                     # (B, 4)
    collated["rel_root_depth"] = torch.stack([s["rel_root_depth"] for s in batch], dim=0).view(-1, 1)
    collated["rel_root_valid"] = torch.stack([s["rel_root_valid"] for s in batch], dim=0).view(-1, 1)
    # 判断 fusion_img 是否在每个样本中都存在
    if all("fusion_img" in s for s in batch):
        collated["fusion_img"] = torch.stack([s["fusion_img"] for s in batch], dim=0)     # (B, 6, H, W)


    # 对 meta_info 字段，递归合并
    meta_infos = [s["meta_info"] for s in batch]
    collated["meta_info"] = merge_nested(meta_infos)

    return collated