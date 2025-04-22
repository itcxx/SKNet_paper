import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# **📌 CoT Attention (Contextual Transformer Networks)**
class CoTAttention(nn.Module):
    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=4, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.value_embed = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

        factor = 4
        self.attention_embed = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim // factor, 1, bias=False),
            nn.BatchNorm2d(2 * dim // factor),
            nn.ReLU(),
            nn.Conv2d(2 * dim // factor, kernel_size * kernel_size * dim, 1)
        )

    def forward(self, x, q):
        if isinstance(x, tuple):
            x = x[-1]
        bs, c, h, w = x.shape
        q = self.key_embed(q)
        k1 = self.key_embed(x)
        v = self.value_embed(x).view(bs, c, -1)

        y = torch.cat([k1, q], dim=1)
        att = self.attention_embed(y)
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1)

        att = F.softmax(att, dim=-1) * v
        k2 = att.view(bs, c, h, w)

        return k1 + k2


# **📌 SK Attention (Selective Kernel Networks)**
class SKAttention(nn.Module):
    def __init__(self, channel=512, kernels=(1, 3, 5, 7), reduction=8, group=1, L=32):
        super().__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2, groups=group),
                nn.BatchNorm2d(channel),
                nn.ReLU()
            ) for k in kernels
        ])
        
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([nn.Linear(self.d, channel) for _ in kernels])
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        B, C, H, W = x.size()
        conv_outs = [conv(x) for conv in self.convs]
        feats = torch.stack(conv_outs, 0)

        U = sum(conv_outs)
        S = U.mean(-1).mean(-1)
        Z = self.fc(S)

        weights = [fc(Z).view(B, C, 1, 1) for fc in self.fcs]
        scale_weight = torch.stack(weights, 0)
        scale_weight = self.softmax(scale_weight)

        V = (scale_weight * feats).sum(0)
        return V


# ** ResNet_SKACOT Backbone**
class ResNet_SKACOT(nn.Module):
    def __init__(self, depth=50,in_channels=3,
                 out_channels=21, channel=2048,
                 kernels=(1, 3, 5, 7), kernel_size=3):
        super().__init__()
        
        # 加载 torchvision 的 ResNet，并移除最后的分类层
        resnet = getattr(models, f"resnet{depth}")(pretrained=True)
        
        # **修改第一层卷积以适配不同的输入通道**
        if in_channels != 3:
            old_conv = resnet.conv1
            resnet.conv1 = nn.Conv2d(
                in_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # 去掉最后的 FC 层
        
        # 插入 SKAttention 和 CoTAttention
        self.sk_attention = SKAttention(channel=channel, kernels=kernels)
        self.cot_attention = CoTAttention(dim=channel, kernel_size=kernel_size)
        
        # 最终输出层
        self.final_conv = nn.Conv2d(channel, out_channels, kernel_size=1)

    # model -01
    # def forward(self, x):
    #     resnet_out = self.resnet(x)  # (B, 2048, H, W)
    #     sk_out = self.sk_attention(resnet_out)
    #     cot_out = self.cot_attention(resnet_out, sk_out)
    #     #
    #     fused_out = sk_out + cot_out
    #     # output = self.final_conv(resnet_out)
    #     output = self.final_conv(fused_out)
    #
    #     # output = resnet_out
    #     return output

    # model -02
    # 不使用SK, CoT
    # def forward(self, x):
    #     resnet_out = self.resnet(x)  # (B, 2048, H, W)
    #     output = self.final_conv(resnet_out)
    #     return output

    # model -03
    # 串联式融合
    # def forward(self, x):
    #     resnet_out = self.resnet(x)  # (B, 2048, H, W)
    #     sk_out = self.sk_attention(resnet_out)
    #     cot_out = self.cot_attention(sk_out, sk_out)  # 用 SK 输出作为 query 和 context
    #     fused_out = cot_out
    #     output = self.final_conv(fused_out)
    #     return output

    # model - 04
    # 残差加深 （CoT 作为残差调制 SK）
    # def forward(self, x):
    #     resnet_out = self.resnet(x)
    #     sk_out = self.sk_attention(resnet_out)
    #     cot_out = self.cot_attention(resnet_out, sk_out)
    #     fused_out = sk_out + cot_out + resnet_out  # 多一层残差增强
    #     output = self.final_conv(fused_out)
    #     return output

    # model - 05
    # 门控融合 （Gating)
    # def forward(self, x):
    #     resnet_out = self.resnet(x)
    #     sk_out = self.sk_attention(resnet_out)
    #     cot_out = self.cot_attention(resnet_out, sk_out)
    #
    #     # 学习一个通道注意力作为门控
    #     gate = torch.sigmoid(F.adaptive_avg_pool2d(sk_out + cot_out, 1))  # (B, C, 1, 1)
    #     fused_out = gate * sk_out + (1 - gate) * cot_out
    #
    #     output = self.final_conv(fused_out)
    #     return output

    # model - 06
    def forward(self,x):
        """
        When use model 06 , need to set in_channels=3
        and
        """
        if x.shape[1] == 6:
            x1 = x[:,:3]
            x2 = x[:,3:]
            x1_f = self.resnet(x1)
            x2_f = self.resnet(x2)
            sk_out_01 = self.sk_attention(x1_f)
            sk_out_02 = self.sk_attention(x2_f)
            cot_out = self.cot_attention(sk_out_01,sk_out_02)
            output = self.final_conv(cot_out)
            return output
        else:
            assert "The input shape is not true "





# # # ** 测试**
# if __name__ == "__main__":
#     model = ResNet_SKACOT(depth=50,in_channels=3, channel=2048, out_channels=64)
#     print(model)
#     x = torch.randn(1, 6, 224, 224)
#     output = model(x)
#     print("输出形状:", output.shape)  # (B, out_channels, H, W)
