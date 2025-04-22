import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets.transforms import Compose 
from utils.registry import TRANSFORMS

def build_pipeline(cfg, pipeline_key="train_pipeline"):
    pipeline_cfg = cfg.get(pipeline_key, [])
    print(f"pipeline_cfg:{pipeline_cfg}")
    transforms = []
    for transform_cfg in pipeline_cfg:
        t_type = transform_cfg.pop("type")
        print(f"t_type:{t_type}")
        if t_type not in TRANSFORMS:
            raise ValueError(f"Transform {t_type} not registered!")
        transforms.append(TRANSFORMS[t_type](**transform_cfg))
    return Compose(transforms)

# # 示例：使用配置文件构建 pipeline
# if __name__ == "__main__":
#     pipeline = build_pipeline("config/dataset.yaml", pipeline_key="train_pipeline")
#     print(pipeline)
