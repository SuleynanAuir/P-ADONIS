"""测试对原始尺寸图像的文本识别"""
import os
import sys
import yaml
import torch
import numpy as np
from PIL import Image
from easydict import EasyDict
from interfaces.super_resolution import TextSR
import string

# 导入 run_demo 中的函数
sys.path.insert(0, os.path.dirname(__file__))
from run_demo import _build_args

# 配置
config_path = 'config/super_resolution.yaml'
config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
config = EasyDict(config)

# 使用 run_demo 中的参数构建函数
args = _build_args(batch_size=1, srb=1)
args.resume = './ckpt/PEAN_final.pth'

print("=" * 100)
print("Testing Text Recognition on Original Images")
print("=" * 100)

# 初始化模型
print("\nInitializing models...")
pean = TextSR(config, args)

# 加载 ASTER
print("Loading ASTER...")
aster, aster_info = pean.Aster_init()
aster.eval()

# 加载 PARSeq  
print("Loading PARSeq...")
parseq = pean.PARSeq_init()
parseq.eval()

def recognize_aster(img_tensor):
    """使用 ASTER 识别文本"""
    with torch.no_grad():
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        # ASTER 期望 3 通道
        if img_tensor.shape[1] == 4:
            img_tensor = img_tensor[:, :3, :, :]
        
        aster_dict_result = aster(img_tensor)
        
        alphabet = string.digits + string.ascii_lowercase
        pred_str_lr = []
        for j in range(img_tensor.shape[0]):
            pred = []
            for i in aster_dict_result['output']['pred_rec'][j]:
                if i == aster_info['char2id']['EOS']:
                    break
                pred.append(alphabet[i])
            pred_str_lr.append(''.join(pred))
        
        return pred_str_lr[0] if pred_str_lr else ""

def recognize_parseq(img_tensor):
    """使用 PARSeq 识别文本"""
    with torch.no_grad():
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        # PARSeq 期望 3 通道
        if img_tensor.shape[1] == 4:
            img_tensor = img_tensor[:, :3, :, :]
        
        # 调整到 PARSeq 期望的尺寸 (32, 128)
        img_parseq = torch.nn.functional.interpolate(
            img_tensor, size=(32, 128), mode='bicubic', align_corners=False
        )
        
        # PARSeq 期望归一化
        img_parseq = pean.parse_parseq_data(img_parseq[0])
        
        # 识别
        logits = parseq(img_parseq, max_length=25)
        probs = logits.softmax(-1)
        
        # 解码
        pred_indices = probs.argmax(-1)[0]
        
        # 转换为字符
        charset = parseq.charset_test
        pred_str = []
        for idx in pred_indices:
            if idx < len(charset):
                char = charset[idx]
                if char == '[E]':  # EOS
                    break
                if char not in ['[B]', '[P]']:  # 忽略特殊标记
                    pred_str.append(char)
        
        return ''.join(pred_str)

# 测试图像
demo_dir = "./demo_img"
img_file = "image5.jpeg"
img_path = os.path.join(demo_dir, img_file)

print(f"\n{'='*100}")
print(f"Testing: {img_file}")
print('='*100)

# 加载原始图像
img = Image.open(img_path).convert('RGB')
img_np = np.array(img).astype(np.float32) / 255.0

print(f"\nOriginal image size: {img.size} (W×H)")

# 转换为 tensor
img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(pean.device)

print(f"Tensor shape: {img_tensor.shape}")

# 测试不同尺寸
test_sizes = [
    (img.size[1], img.size[0]),  # 原始尺寸
    (32, 128),   # 标准文本识别尺寸
    (64, 256),   # 2倍尺寸
    (16, 64),    # 当前使用的尺寸
]

for h, w in test_sizes:
    print(f"\n--- Testing at size {w}×{h} ---")
    
    # 调整尺寸
    img_resized = torch.nn.functional.interpolate(
        img_tensor, size=(h, w), mode='bicubic', align_corners=False
    )
    
    print(f"Resized shape: {img_resized.shape}")
    
    # ASTER 识别
    try:
        pred_aster = recognize_aster(img_resized)
        print(f"ASTER prediction: '{pred_aster}'")
    except Exception as e:
        print(f"ASTER failed: {e}")
    
    # PARSeq 识别
    try:
        pred_parseq = recognize_parseq(img_resized)
        print(f"PARSeq prediction: '{pred_parseq}'")
    except Exception as e:
        print(f"PARSeq failed: {e}")

print("\n" + "="*100)
