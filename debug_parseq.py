"""Debug script to understand PARSeq output dimensions"""
import os
import sys
import torch
import numpy as np
from PIL import Image
from interfaces.base import TextSR
from easydict import EasyDict
import yaml

# 加载配置
config_path = os.path.join('config', 'super_resolution.yaml')
config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
config = EasyDict(config)

# 构建简单args
class Args:
    def __init__(self):
        self.resume = './ckpt/PEAN_final.pth'
        self.batch_size = 1
        self.srb = 1
        self.cuda = True
        self.cudnn_benchmark = False
        self.seed = 1234
        self.total_iter = 1
        self.total_epochs = 1
        
args = Args()

# 初始化
pean = TextSR(config, args)

# 加载PARSeq
print("Loading PARSeq...")
parseq = pean.PARSeq_init()
parseq.eval()
for p in parseq.parameters():
    p.requires_grad = False

# 加载一张测试图像
test_img_path = './demo_img/image1.jpeg'
if not os.path.exists(test_img_path):
    print(f"Test image not found: {test_img_path}")
    sys.exit(1)

# 加载并处理图像
img = Image.open(test_img_path)
print(f"Original image size: {img.size}")

# 使用pean的parse_parseq_data方法处理
pq_in = pean.parse_parseq_data(img)
print(f"PARSeq input type: {type(pq_in)}")
print(f"PARSeq input shape: {pq_in.shape if hasattr(pq_in, 'shape') else 'N/A'}")

# 推理
with torch.no_grad():
    output = parseq(pq_in, max_length=25)
    print(f"\nPARSeq output shape: {output.shape}")
    print(f"PARSeq output dtype: {output.dtype}")
    
    # 尝试softmax
    softmax_out = output.softmax(-1)
    print(f"After softmax shape: {softmax_out.shape}")
    print(f"After softmax dtype: {softmax_out.dtype}")
    print(f"After softmax values range: [{softmax_out.min().item():.4f}, {softmax_out.max().item():.4f}]")
    
    # 检查维度
    print(f"\nDimension analysis:")
    print(f"  Batch size: {softmax_out.shape[0] if len(softmax_out.shape) > 0 else 'scalar'}")
    print(f"  Seq length: {softmax_out.shape[1] if len(softmax_out.shape) > 1 else 'N/A'}")
    print(f"  Vocab size: {softmax_out.shape[2] if len(softmax_out.shape) > 2 else 'N/A'}")

print("\nDebug complete!")
