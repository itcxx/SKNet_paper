import yaml
import torch
from models.build_model import build_model

# 读取配置文件
config_path = "./config/config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# 创建模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(config).to(device)

print(model)  # 打印模型结构

x = torch.randn(1, 6, 224, 224, dtype=torch.float).to('cuda')
output = model(x)
print("输出形状:", output)  # (B, out_channels, H, W)