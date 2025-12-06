"""检查 demo_img 中图像的实际内容"""
import os
from PIL import Image
import numpy as np

demo_dir = "./demo_img"

print("=" * 80)
print("Checking demo images...")
print("=" * 80)

for img_file in sorted(os.listdir(demo_dir)):
    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(demo_dir, img_file)
        
        print(f"\n{img_file}:")
        
        try:
            # 打开图像
            img = Image.open(img_path)
            
            print(f"  - Mode: {img.mode}")
            print(f"  - Size: {img.size} (W×H)")
            print(f"  - Format: {img.format}")
            
            # 转换为 RGB
            img_rgb = img.convert('RGB')
            img_np = np.array(img_rgb)
            
            print(f"  - Array shape: {img_np.shape}")
            print(f"  - Value range: [{img_np.min()}, {img_np.max()}]")
            print(f"  - Mean pixel value: {img_np.mean():.2f}")
            
            # 检查是否是纯色或者有实际内容
            unique_colors = len(np.unique(img_np.reshape(-1, 3), axis=0))
            print(f"  - Unique colors: {unique_colors}")
            
            # 显示角落像素
            print(f"  - Top-left pixel (RGB): {img_np[0, 0]}")
            print(f"  - Center pixel (RGB): {img_np[img_np.shape[0]//2, img_np.shape[1]//2]}")
            
        except Exception as e:
            print(f"  ERROR: {e}")

print("\n" + "=" * 80)
