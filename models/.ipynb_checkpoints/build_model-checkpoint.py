import torch.nn as nn
from .backbones.resnet_skacot import ResNet_SKACOT
from .necks.necks import GlobalAveragePooling, FPN, SimpleConvNeck, DeformableConvNeck, HourglassNeck
from .heads.hand_pose_head import HandPoseHead


def build_backbone(backbone_name, resnet_depth,in_channels=3,channels=512,out_channels=64):
    """构建 Backbone"""
    if backbone_name == "ResNet_SKACOT":
        return ResNet_SKACOT(depth=resnet_depth, in_channels=in_channels,channel= channels, out_channels=out_channels)
    else:
        raise ValueError(f"❌ 未知的 Backbone: {backbone_name}")


def build_neck(neck_name, config):
    """构建 Neck 结构"""
    if neck_name == "GAP":
        return GlobalAveragePooling()
    elif neck_name == "FPN":
        return FPN(config["in_channels"], config["out_channels"], config["num_levels"])
    elif neck_name == "SimpleConv":
        return SimpleConvNeck(config["in_channels"], config["out_channels"])
    elif neck_name == "DeformableConv":
        return DeformableConvNeck(config["in_channels"], config["out_channels"])
    elif neck_name == "Hourglass":
        return HourglassNeck(config["num_stacks"], config["in_channels"], config["out_channels"])
    else:
        raise ValueError(f"❌ 未知的 Neck 结构: {neck_name}")
        
def build_head(head_name, config):
    """构建 Head 结构"""
    if head_name == "HandPoseHead":
        return HandPoseHead(
            in_channels=config["in_channels"],
            num_joints=config["num_joints"],
            depth_size=config["depth_size"]
        )
    else:
        raise ValueError(f"❌ 未知的 Head 结构: {head_name}")

class PoseEstimationModel(nn.Module):
    """完整的姿态估计模型，包含 Backbone + Neck + head"""

    def __init__(self, config):
        super(PoseEstimationModel, self).__init__()
        self.backbone = build_backbone(config["model"]['backbone_config']["backbone"],
                                       config["model"]['backbone_config']["resnet_depth"],
                                       config["model"]['backbone_config']["in_channels"],
                                       config["model"]['backbone_config']["channels"],
                                       config["model"]['backbone_config']["out_channels"],
                                      )
        self.neck = build_neck(config["model"]["neck"], config["model"]["neck_config"])
        # 构建 HandPoseHead
        self.head = build_head("HandPoseHead", config["model"]["head_config"])

    def forward(self, x):
        """前向传播"""
        x = self.backbone(x)  # 通过 Backbone 提取特征
        x = self.neck(x)  # 通过 Neck 进行特征融合
        x = self.head(x) # 通过head获取结果
        return x


def build_model(config):
    """创建完整模型"""
    model = PoseEstimationModel(config)
    return model

