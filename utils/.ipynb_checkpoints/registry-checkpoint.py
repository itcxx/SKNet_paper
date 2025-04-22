# registry.py
# 一个简单的注册器
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

TRANSFORMS = {}

def register_transform(cls):
    TRANSFORMS[cls.__name__] = cls
    return cls
