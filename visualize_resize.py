"""可视化图像调整到 16×64 后的效果"""
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

demo_dir = "./demo_img"
output_dir = "./demo_results/resize_check"
os.makedirs(output_dir, exist_ok=True)

target_h, target_w = 16, 64

print("=" * 80)
print("Visualizing image resizing to 16×64...")
print("=" * 80)

for img_file in sorted(os.listdir(demo_dir)):
    if 'image5' not in img_file.lower():
        continue
        
    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(demo_dir, img_file)
        
        print(f"\nProcessing {img_file}...")
        
        # 加载原始图像
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img).astype(np.float32) / 255.0
        
        print(f"  Original size: {img.size} (W×H)")
        
        # 转换为 torch 张量
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        
        # 调整到 16×64
        img_resized = F.interpolate(img_tensor, size=(target_h, target_w), 
                                    mode='bicubic', align_corners=False)
        
        # 转回 numpy 用于显示
        img_resized_np = img_resized.squeeze(0).permute(1, 2, 0).numpy()
        img_resized_np = np.clip(img_resized_np, 0, 1)
        
        print(f"  Resized shape: {img_resized_np.shape}")
        print(f"  Value range: [{img_resized_np.min():.3f}, {img_resized_np.max():.3f}]")
        
        # 创建对比图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始图像
        axes[0].imshow(img_np)
        axes[0].set_title(f'Original\n{img.size[0]}×{img.size[1]}', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # 调整后的图像（小尺寸）
        axes[1].imshow(img_resized_np)
        axes[1].set_title(f'Resized (actual size)\n{target_w}×{target_h}', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # 调整后的图像（放大显示）
        axes[2].imshow(img_resized_np, interpolation='nearest')
        axes[2].set_title(f'Resized (zoomed)\n{target_w}×{target_h}', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.suptitle(f'Resize Check: {img_file}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f'{os.path.splitext(img_file)[0]}_resize_check.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved to: {save_path}")
        
        # 保存实际 16×64 图像
        small_path = os.path.join(output_dir, f'{os.path.splitext(img_file)[0]}_16x64.png')
        img_resized_pil = Image.fromarray((img_resized_np * 255).astype(np.uint8))
        img_resized_pil.save(small_path)
        print(f"  Saved 16×64 image to: {small_path}")

print("\n" + "=" * 80)
print(f"Results saved to: {output_dir}")
print("=" * 80)
