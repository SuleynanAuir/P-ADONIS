"""
PEAN Demo Script
基于 demo_img 文件夹中的 JPEG 图片进行超分辨率评估
"""
import os
import csv
import yaml
import string
import argparse
from datetime import datetime
from easydict import EasyDict
import torch
import torchvision
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import time

from interfaces.super_resolution import TextSR
from utils.metrics import get_str_list


def _build_args(batch_size=1, rec='aster', mask=True, srb=1, testing=True):
    """构建 PEAN 所需的参数对象"""
    class Args:
        pass
    a = Args()
    a.test = testing
    a.pre_training = False
    a.test_data_dir = None
    a.batch_size = batch_size
    a.resume = None
    a.vis_dir = None
    a.rec = rec
    a.STN = False
    a.syn = False
    a.mixed = False
    a.mask = mask
    a.gradient = False
    a.hd_u = 32
    a.srb = srb
    a.demo = False
    a.demo_dir = ''
    a.prior_dim = 1024
    a.dec_num_heads = 16
    a.dec_mlp_ratio = 4
    a.dec_depth = 1
    a.max_gen_perms = 1
    a.rotate_train = 0.
    a.perm_forward = False
    a.perm_mirrored = False
    a.dropout = 0.1
    return a


def _ensure_dir(path):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)
    return path


def _to_3ch_float(x):
    """转换为 3 通道浮点张量 [0, 1]"""
    if x.dim() == 4:  # BCHW
        x = x[0]  # CHW
    if x.dim() == 3:
        if x.shape[0] == 1:  # 1HW -> 3HW
            x = x.repeat(3, 1, 1)
        elif x.shape[0] > 3:  # 4+HW -> 3HW
            x = x[:3, :, :]
    x = x.float()
    if x.max() > 1.0:
        x = x / 255.0
    return x.clamp(0, 1)


def _whiten_background_uint8(img_uint8, thresh=0.8, strength=0.6):
    """在可视化阶段对白色背景进行提亮，不影响文字结构。"""
    img = img_uint8.astype(np.float32) / 255.0
    gray = img.mean(axis=2)
    mask = (gray > thresh).astype(np.float32)[..., None]
    # 仅对背景区域进行提亮，文字区域保持原样
    img_whiten = img + strength * (1.0 - img) * mask
    img_whiten = np.clip(img_whiten, 0.0, 1.0)
    return (img_whiten * 255.0).round().astype(np.uint8)


def _calculate_optimal_tiling(orig_h, orig_w, tile_h=16, tile_w=64):
    """
    计算最优的分块策略
    
    Args:
        orig_h: 原始图像高度
        orig_w: 原始图像宽度
        tile_h: 单个块的高度（模型要求，默认16）
        tile_w: 单个块的宽度（模型要求，默认64）
    
    Returns:
        resized_h: 调整后的总高度（tile_h 的整数倍）
        resized_w: 调整后的总宽度（tile_w 的整数倍）
        n_tiles_h: 垂直方向的块数
        n_tiles_w: 水平方向的块数
    """
    # 计算需要多少个块来接近原始尺寸
    n_tiles_h = max(1, round(orig_h / tile_h))
    n_tiles_w = max(1, round(orig_w / tile_w))
    
    # 计算调整后的尺寸
    resized_h = n_tiles_h * tile_h
    resized_w = n_tiles_w * tile_w
    
    return resized_h, resized_w, n_tiles_h, n_tiles_w


def _split_into_tiles(img_tensor, tile_h=16, tile_w=64, overlap=0.25):
    """
    将图像分割成多个小块（带重叠）
    
    Args:
        img_tensor: 输入图像张量 (C, H, W) 或 (B, C, H, W)
        tile_h: 块高度
        tile_w: 块宽度
        overlap: 重叠比例 (0-1)，默认0.25表示25%重叠
    
    Returns:
        tiles: 分块列表，每个块的形状为 (1, C, tile_h, tile_w)
        positions: 每个块在原图中的位置 (h_start, h_end, w_start, w_end)
        n_tiles_h: 垂直方向的块数
        n_tiles_w: 水平方向的块数
    """
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)  # C,H,W -> 1,C,H,W
    
    B, C, H, W = img_tensor.shape
    
    # 计算步长（考虑重叠）
    stride_h = int(tile_h * (1 - overlap))
    stride_w = int(tile_w * (1 - overlap))
    
    # 确保至少步长为1
    stride_h = max(1, stride_h)
    stride_w = max(1, stride_w)
    
    tiles = []
    positions = []
    
    h_starts = list(range(0, H - tile_h + 1, stride_h))
    w_starts = list(range(0, W - tile_w + 1, stride_w))
    
    # 确保覆盖到边缘
    if h_starts[-1] + tile_h < H:
        h_starts.append(H - tile_h)
    if w_starts[-1] + tile_w < W:
        w_starts.append(W - tile_w)
    
    for h_start in h_starts:
        for w_start in w_starts:
            h_end = h_start + tile_h
            w_end = w_start + tile_w
            
            tile = img_tensor[:, :, h_start:h_end, w_start:w_end]
            tiles.append(tile)
            positions.append((h_start, h_end, w_start, w_end))
    
    n_tiles_h = len(h_starts)
    n_tiles_w = len(w_starts)
    
    return tiles, positions, n_tiles_h, n_tiles_w


def _create_blend_mask(tile_h, tile_w, overlap=0.25, device='cpu'):
    """
    创建平滑混合权重掩码（使用高斯平滑）
    
    Args:
        tile_h: 块高度
        tile_w: 块宽度
        overlap: 重叠比例
        device: 设备
    
    Returns:
        mask: 权重掩码 (1, 1, tile_h, tile_w)
    """
    # 计算重叠区域大小
    overlap_h = int(tile_h * overlap)
    overlap_w = int(tile_w * overlap)
    
    # 创建1D权重函数（使用更平滑的S曲线）
    def smooth_window(size, overlap_size):
        if overlap_size == 0:
            return torch.ones(size)
        
        weights = torch.ones(size)
        
        # 使用平滑的S曲线（sigmoid函数）而不是余弦
        if overlap_size > 0:
            # 左边缘渐变
            x = torch.linspace(-6, 6, overlap_size)
            fade_in = torch.sigmoid(x)
            weights[:overlap_size] = fade_in
            
            # 右边缘渐变
            fade_out = torch.sigmoid(-x)
            weights[-overlap_size:] = fade_out
        
        return weights
    
    # 创建2D权重
    weights_h = smooth_window(tile_h, overlap_h)
    weights_w = smooth_window(tile_w, overlap_w)
    
    # 外积得到2D掩码
    mask_2d = weights_h.unsqueeze(1) * weights_w.unsqueeze(0)
    mask = mask_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    return mask.to(device)


def _merge_tiles_smooth(tiles, positions, output_h, output_w, overlap=0.15):
    """
    将多个重叠的小块平滑合并成完整图像
    
    Args:
        tiles: 分块列表，每个块的形状为 (1, C, tile_h, tile_w)
        positions: 每个块在原图中的位置列表
        output_h: 输出图像高度
        output_w: 输出图像宽度
        overlap: 重叠比例
    
    Returns:
        merged: 合并后的图像张量 (1, C, H, W)
    """
    if not tiles:
        raise ValueError("No tiles to merge")
    
    device = tiles[0].device
    C = tiles[0].shape[1]
    tile_h = tiles[0].shape[2]
    tile_w = tiles[0].shape[3]
    
    # 创建输出张量和权重累积张量
    merged = torch.zeros(1, C, output_h, output_w, device=device)
    weight_sum = torch.zeros(1, 1, output_h, output_w, device=device)
    
    # 创建混合掩码
    blend_mask = _create_blend_mask(tile_h, tile_w, overlap, device)
    
    # 加权累加所有块
    for tile, (h_start, h_end, w_start, w_end) in zip(tiles, positions):
        # 应用权重掩码
        weighted_tile = tile * blend_mask
        
        # 累加到输出
        merged[:, :, h_start:h_end, w_start:w_end] += weighted_tile
        weight_sum[:, :, h_start:h_end, w_start:w_end] += blend_mask
    
    # 归一化（避免除零）
    weight_sum = torch.clamp(weight_sum, min=1e-8)
    merged = merged / weight_sum
    
    # 轻微高斯平滑以进一步减少接缝
    if overlap > 0:
        # 只对RGB通道应用平滑（如果有掩码通道则跳过）
        if C == 4:
            # 分离RGB和掩码
            rgb = merged[:, :3, :, :]
            mask = merged[:, 3:, :, :]
            # 对RGB应用轻微平滑
            rgb_smooth = F.avg_pool2d(
                F.pad(rgb, (1, 1, 1, 1), mode='replicate'),
                kernel_size=3, stride=1, padding=0
            )
            merged = torch.cat([rgb_smooth, mask], dim=1)
        else:
            # 全部通道平滑
            merged = F.avg_pool2d(
                F.pad(merged, (1, 1, 1, 1), mode='replicate'),
                kernel_size=3, stride=1, padding=0
            )
    
    return merged


def _sharpen_edges(img_tensor, strength=0.5):
    """
    对图像进行边缘锐化，使文字笔划更清晰
    
    Args:
        img_tensor: 输入图像张量 (B, C, H, W) 或 (C, H, W)
        strength: 锐化强度 (0-1)，默认0.5
    
    Returns:
        sharpened: 锐化后的图像张量
    """
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    
    device = img_tensor.device
    B, C, H, W = img_tensor.shape
    
    # Laplacian 锐化卷积核
    laplacian_kernel = torch.tensor([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=torch.float32, device=device)
    
    # 扩展为多通道
    laplacian_kernel = laplacian_kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
    laplacian_kernel = laplacian_kernel.repeat(C, 1, 1, 1)  # (C, 1, 3, 3)
    
    # 应用卷积提取边缘
    img_padded = F.pad(img_tensor, (1, 1, 1, 1), mode='replicate')
    edges = F.conv2d(img_padded, laplacian_kernel, groups=C)
    
    # 将边缘添加回原图以增强
    sharpened = img_tensor + strength * edges
    
    # 裁剪到有效范围
    sharpened = torch.clamp(sharpened, 0, 1)
    
    return sharpened


def _enhance_text_clarity(img_tensor, sharpen_strength=0.6, contrast_factor=1.2):
    """
    综合增强文字清晰度：锐化 + 对比度增强
    
    Args:
        img_tensor: 输入图像张量 (B, C, H, W) 或 (C, H, W)
        sharpen_strength: 锐化强度 (0-1)
        contrast_factor: 对比度增强因子 (>1增强, <1降低)
    
    Returns:
        enhanced: 增强后的图像张量
    """
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # 检查输入是否有效
    if torch.isnan(img_tensor).any() or torch.isinf(img_tensor).any():
        print("  WARNING: Invalid values in input tensor, skipping enhancement")
        return img_tensor.squeeze(0) if squeeze_output else img_tensor
    
    # 1. 边缘锐化
    enhanced = _sharpen_edges(img_tensor, strength=sharpen_strength)
    
    # 2. 对比度增强（使用自适应方法，避免过度增强导致全黑/全白）
    # 只对值域合理的图像应用对比度增强
    img_min = enhanced.min()
    img_max = enhanced.max()
    
    if img_max - img_min > 0.01:  # 确保有足够的动态范围
        # 计算每个通道的均值
        mean_val = enhanced.mean(dim=[2, 3], keepdim=True)
        
        # 以均值为中心进行对比度拉伸
        enhanced = (enhanced - mean_val) * contrast_factor + mean_val
        
        # 裁剪到有效范围
        enhanced = torch.clamp(enhanced, 0, 1)
    else:
        print("  WARNING: Low dynamic range, skipping contrast enhancement")
    
    if squeeze_output:
        enhanced = enhanced.squeeze(0)
    
    return enhanced


def _merge_tiles(tiles, n_tiles_h, n_tiles_w, tile_h, tile_w):
    """
    将多个小块合并成完整图像（旧版无重叠方法，保留用于兼容）
    
    Args:
        tiles: 分块列表，每个块的形状为 (1, C, tile_h, tile_w)
        n_tiles_h: 垂直方向的块数
        n_tiles_w: 水平方向的块数
        tile_h: 块高度
        tile_w: 块宽度
    
    Returns:
        merged: 合并后的图像张量 (1, C, H, W)
    """
    if not tiles:
        raise ValueError("No tiles to merge")
    
    C = tiles[0].shape[1]
    H = n_tiles_h * tile_h
    W = n_tiles_w * tile_w
    
    # 创建输出张量
    merged = torch.zeros(1, C, H, W, device=tiles[0].device)
    
    tile_idx = 0
    for i in range(n_tiles_h):
        for j in range(n_tiles_w):
            h_start = i * tile_h
            h_end = h_start + tile_h
            w_start = j * tile_w
            w_end = w_start + tile_w
            
            merged[:, :, h_start:h_end, w_start:w_end] = tiles[tile_idx]
            tile_idx += 1
    
    return merged


def _resize_chw(img_chw, target_h, target_w):
    """调整图像大小 (CHW 格式)"""
    img_1chw = img_chw.unsqueeze(0)
    resized = F.interpolate(img_1chw, size=(target_h, target_w), mode='bicubic', align_corners=False)
    return resized[0].clamp(0, 1)


def _metrics(sr, hr):
    """计算图像质量指标，自动处理不同大小的图像"""
    # 将 SR 调整到与 HR 相同的大小（用于公平的指标计算）
    if sr.shape != hr.shape:
        sr = F.interpolate(sr.unsqueeze(0), size=(hr.shape[1], hr.shape[2]), 
                          mode='bicubic', align_corners=False).squeeze(0)
    
    sr_np = sr.detach().cpu().numpy()
    hr_np = hr.detach().cpu().numpy()
    
    # L1, L2
    l1 = np.abs(sr_np - hr_np).mean()
    l2 = ((sr_np - hr_np) ** 2).mean() ** 0.5
    
    # PSNR
    mse = ((sr_np - hr_np) ** 2).mean()
    if mse < 1e-10:
        psnr = 100.0
    else:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    
    # SSIM (简化版本)
    from skimage.metrics import structural_similarity as ssim
    sr_gray = sr_np.mean(axis=0)
    hr_gray = hr_np.mean(axis=0)
    ssim_val = ssim(sr_gray, hr_gray, data_range=1.0)
    
    return {
        'psnr': psnr,
        'ssim': ssim_val,
        'l1': l1,
        'l2': l2
    }


def _save_comparison_grid_demo(lr, sr, img_name, pred_lr, pred_sr, save_path, elapsed_time_s=None):
    """
    保存专业的Demo对比图
    左侧：输入/输出水平排列
    右侧：量化分析可视化（PSNR/SSIM对比、局部放大、差异热图）
    
    Args:
        lr: 低分辨率图像张量 (3, H, W)
        sr: 超分辨率图像张量 (3, H, W)
        img_name: 图像文件名
        pred_lr: LR文本识别结果
        pred_sr: SR文本识别结果
        save_path: 保存路径
    """
    # 调整尺寸使两者一致
    if lr.shape != sr.shape:
        target_h, target_w = sr.shape[1], sr.shape[2]
        lr_resized = _resize_chw(lr, target_h, target_w)
    else:
        lr_resized = lr

    # 对齐尺寸，避免后续计算广播错误
    target_h = min(lr_resized.shape[1], sr.shape[1])
    target_w = min(lr_resized.shape[2], sr.shape[2])
    lr_resized = _resize_chw(lr_resized, target_h, target_w)
    sr_resized = _resize_chw(sr, target_h, target_w)
    
    # 转换为numpy数组
    lr_hwc = lr_resized.permute(1, 2, 0).detach().cpu().numpy()
    sr_hwc = sr_resized.permute(1, 2, 0).detach().cpu().numpy()

    # 可视化用的背景提亮版本（不影响后续指标计算）
    lr_disp_uint8 = _whiten_background_uint8((lr_hwc * 255).clip(0, 255).astype(np.uint8))
    sr_disp_uint8 = _whiten_background_uint8((sr_hwc * 255).clip(0, 255).astype(np.uint8))
    
    lr_hwc_uint8 = (lr_hwc * 255).clip(0, 255).astype(np.uint8)
    sr_hwc_uint8 = (sr_hwc * 255).clip(0, 255).astype(np.uint8)
    
    # 计算质量指标
    mse = ((lr_hwc - sr_hwc) ** 2).mean()
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 1e-10 else 100.0
    
    # 计算SSIM
    def simple_ssim(img1, img2):
        c1, c2 = (0.01)**2, (0.03)**2
        mu1, mu2 = img1.mean(), img2.mean()
        sigma1 = ((img1 - mu1)**2).mean()
        sigma2 = ((img2 - mu2)**2).mean()
        sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
        return ((2*mu1*mu2 + c1)*(2*sigma12 + c2)) / ((mu1**2 + mu2**2 + c1)*(sigma1 + sigma2 + c2))
    
    ssim_score = simple_ssim(lr_hwc, sr_hwc)
    
    # 计算更多质量指标
    psnr_normalized = min(psnr / 50.0, 1.0)
    
    # 计算MSE和MAE
    mse = ((lr_hwc - sr_hwc) ** 2).mean()
    mae = np.abs(lr_hwc - sr_hwc).mean()
    
    # 计算对比度改善度（简化版）
    lr_std = lr_hwc.std()
    sr_std = sr_hwc.std()
    contrast_improvement = min((sr_std / (lr_std + 1e-8)), 2.0) / 2.0  # 归一化到0-1
    
    # 计算清晰度指标（基于梯度）
    def calculate_sharpness(img):
        gray = img.mean(axis=2)
        grad_x = np.abs(np.diff(gray, axis=1)).mean()
        grad_y = np.abs(np.diff(gray, axis=0)).mean()
        return (grad_x + grad_y) / 2
    
    lr_sharpness = calculate_sharpness(lr_hwc)
    sr_sharpness = calculate_sharpness(sr_hwc)
    sharpness_improvement = min((sr_sharpness / (lr_sharpness + 1e-8)), 2.0) / 2.0  # 归一化
    
    # 计算差异图
    diff_map = np.abs(lr_hwc - sr_hwc).mean(axis=2)
    
    # 设置专业配色
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    fig = plt.figure(figsize=(20, 16), facecolor='white')
    
        # 创建网格布局: 9行×6列（8行内容 + 1行底部标题）
        # Row 0-7: 左侧4列（LR 2列 | SR 2列）+ 右侧2列（Zoom×8 + 雷达图 + Diff）
        # Row 8: 标题
    gs = fig.add_gridspec(9, 6, height_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 0.08], 
                          width_ratios=[1, 1, 1, 1, 0.85, 0.85],
                              hspace=0.25, wspace=0.08, top=0.99, bottom=0.04, left=0.01, right=0.99)
    
        # ==================== 左侧第1-2列：Input LR（占据行0-7）====================
    ax_lr = fig.add_subplot(gs[0:8, 0:2])
    ax_lr.imshow(lr_disp_uint8)
    ax_lr.set_title('INPUT - Low Resolution', fontsize=14, fontweight='bold',
                       color='#e74c3c', pad=12)
    ax_lr.axis('off')
    
        # ==================== 左侧第3-4列：Output SR（占据行0-7）====================
    ax_sr = fig.add_subplot(gs[0:8, 2:4])
    ax_sr.imshow(sr_disp_uint8)
    ax_sr.set_title('OUTPUT - Super Resolution (PEAN)', fontsize=14, fontweight='bold',
                       color='#27ae60', pad=12)
    ax_sr.axis('off')
    
    # ==================== 右侧第1列：8个Zoomed Detail示例（垂直排列）====================
    h, w = sr_disp_uint8.shape[:2]
    
    # 定义8个不同的裁剪区域：从上到下均匀分布
    crop_size_h, crop_size_w = min(h//5, 60), min(w//5, 140)
    
    zoom_positions = [
        ('Top-1', h//8, w//2),       # 第1区域
        ('Top-2', h*2//8, w//2),     # 第2区域
        ('Top-3', h*3//8, w//2),     # 第3区域
        ('Mid-1', h*4//8, w//2),     # 第4区域（中心）
        ('Mid-2', h*5//8, w//2),     # 第5区域
        ('Bot-1', h*6//8, w//2),     # 第6区域
        ('Bot-2', h*7//8, w//2),     # 第7区域
        ('Bot-3', h*7.5//8, w//2)    # 第8区域
    ]
    
    for idx, (pos_name, center_y, center_x) in enumerate(zoom_positions):
        y1, y2 = max(0, int(center_y - crop_size_h//2)), min(h, int(center_y + crop_size_h//2))
        x1, x2 = max(0, int(center_x - crop_size_w//2)), min(w, int(center_x + crop_size_w//2))
        
        lr_crop = lr_disp_uint8[y1:y2, x1:x2]
        sr_crop = sr_disp_uint8[y1:y2, x1:x2]
        
        # 组合放大图（左右排列更紧凑）
        separator = np.ones((lr_crop.shape[0], 2, 3), dtype=np.uint8) * 200
        zoom_combined = np.hstack([lr_crop, separator, sr_crop])
        
        # 8个zoom垂直排列在第5列的row 0-7
        ax_zoom = fig.add_subplot(gs[idx, 4])
        
        ax_zoom.imshow(zoom_combined)
        ax_zoom.set_title(f'Zoom: {pos_name}', fontsize=8, fontweight='bold',
                         color='#2c3e50', pad=4)
        ax_zoom.axis('off')
        
        # 添加标签（左右排列）
        w_crop = lr_crop.shape[1]
        ax_zoom.text(w_crop//2, 3, 'LR', fontsize=6, color='white', fontweight='bold',
                    ha='center',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='#e74c3c', alpha=0.9))
        ax_zoom.text(w_crop + 2 + sr_crop.shape[1]//2, 3, 'SR', fontsize=6,
                    color='white', fontweight='bold', ha='center',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='#27ae60', alpha=0.9))
    
    # ==================== 右上：质量指标雷达图（占据2行）====================
    ax_radar = fig.add_subplot(gs[0:2, 5], projection='polar')
    
    # 准备雷达图数据 - 使用4个质量指标
    categories = ['PSNR', 'SSIM', 'Contrast', 'Sharpness']
    values = [psnr_normalized, ssim_score, contrast_improvement, sharpness_improvement]
    
    # 为了闭合雷达图，需要重复第一个点
    values_closed = values + [values[0]]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles_closed = angles + [angles[0]]
    
    # 绘制雷达图
    ax_radar.plot(angles_closed, values_closed, 'o-', linewidth=2.5, 
                 color='#3498db', label='Quality Scores', markersize=8)
    ax_radar.fill(angles_closed, values_closed, alpha=0.3, color='#3498db')
    
    # 设置标签
    ax_radar.set_xticks(angles)
    ax_radar.set_xticklabels(categories, fontsize=10, fontweight='bold', color='#2c3e50')
    
    # 设置刻度
    ax_radar.set_ylim(0, 1)
    ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_radar.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], 
                            fontsize=8, color='#7f8c8d')
    ax_radar.grid(True, linestyle='--', linewidth=1, alpha=0.6, color='#bdc3c7')
    
    # 添加参考圆圈
    ax_radar.plot(angles_closed, [0.9]*len(angles_closed), '--', 
                 linewidth=1.5, color='#27ae60', alpha=0.7, label='Excellent (0.9)')
    ax_radar.plot(angles_closed, [0.7]*len(angles_closed), '--', 
                 linewidth=1.2, color='#f39c12', alpha=0.6, label='Good (0.7)')
    
    # 添加标题
    ax_radar.set_title('Quality Metrics Radar', fontsize=11,
                      fontweight='bold', color='#1a1a2e', pad=15,
                             bbox=dict(boxstyle='round,pad=0.4', facecolor='#ecf0f1', alpha=0.9))
    
    # 添加图例
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
                         fontsize=7, framealpha=0.95)
    
    # 在雷达图上标注具体数值
    for angle, value, category in zip(angles, values, categories):
        ax_radar.text(angle, value + 0.08, f'{value:.3f}', 
                     ha='center', va='bottom', fontsize=7, 
                     fontweight='bold', color='#2c3e50')
    
    # ==================== 右下：差异热图（占据6行）====================
    ax_diff = fig.add_subplot(gs[2:8, 5])
    im = ax_diff.imshow(diff_map, cmap='hot', aspect='auto', interpolation='bilinear')
    ax_diff.set_title('Difference Heatmap', fontsize=11, fontweight='bold',
                     color='#2c3e50', pad=8)
    ax_diff.axis('off')
    
    # 添加颜色条
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax_diff)
    cax = divider.append_axes("right", size="6%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Intensity', fontsize=9, fontweight='bold', color='#2c3e50')
    cbar.ax.tick_params(labelsize=8, colors='#34495e')
    
    # ==================== 底部标题 ====================
    ax_title = fig.add_subplot(gs[8, :])
    ax_title.axis('off')
    ax_title.text(0.5, 0.5, f'Demo: {img_name}',
                  ha='center', va='center', fontsize=20, fontweight='bold',
                  color='#1a1a2e',
                  bbox=dict(boxstyle='round,pad=1.0', facecolor='#eef2f7', alpha=0.95))

    # 顶部时间标记（如果有）
    if elapsed_time_s is not None:
        fig.text(0.98, 0.985, f'Time: {elapsed_time_s:.1f}s', ha='right', va='top',
                fontsize=10, fontweight='bold', color='#2c3e50')
    
    # ==================== 底部额外信息 ====================
    time_info = '' if elapsed_time_s is None else f'  |  Time: {elapsed_time_s:.1f}s'
    info_left = f'Resolution: {lr_resized.shape[2]}×{lr_resized.shape[1]} → {sr.shape[2]}×{sr.shape[1]} px  |  Scale Factor: {sr.shape[2]/lr_resized.shape[2]:.1f}x  |  MSE: {mse:.6f}  |  MAE: {mae:.4f}{time_info}'
    fig.text(0.02, 0.015, info_left,
            ha='left', va='bottom', fontsize=9, color='#2c3e50', fontweight='bold')
    
    fig.text(0.98, 0.015, 'ours Deep Learning Text Super-Resolution',
            ha='right', va='bottom', fontsize=9, color='#7f8c8d',
            style='italic', alpha=0.85)
    
    # 保存超高清图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none', pad_inches=0.08)
    plt.close()
    plt.style.use('default')

    # 追加随机采样可视化与专业量化可视化
    base_dir = os.path.dirname(save_path)
    base_name = os.path.splitext(os.path.basename(save_path))[0]
    random_save = os.path.join(base_dir, base_name + '_random_samples.png')
    metrics_save = os.path.join(base_dir, base_name + '_metrics_pro.png')
    _save_random_sampling(lr_disp_uint8, sr_disp_uint8, img_name, random_save)
    _save_metrics_pro(lr_hwc, sr_hwc, img_name, metrics_save)


def _save_random_sampling(lr_img_uint8, sr_img_uint8, img_name, save_path, num_patches=5, patch_h=80, patch_w=160):
    """保存随机区域采样对比图(LR vs SR)。"""
    h, w, _ = sr_img_uint8.shape
    rng = np.random.default_rng()
    fig = plt.figure(figsize=(14, 10), facecolor='white')
    gs = fig.add_gridspec(num_patches, 2, hspace=0.25, wspace=0.08)
    for i in range(num_patches):
        cy = int(rng.integers(low=patch_h//2, high=max(patch_h//2 + 1, h - patch_h//2)))
        cx = int(rng.integers(low=patch_w//2, high=max(patch_w//2 + 1, w - patch_w//2)))
        y1, y2 = max(0, cy - patch_h//2), min(h, cy + patch_h//2)
        x1, x2 = max(0, cx - patch_w//2), min(w, cx + patch_w//2)
        lr_crop = lr_img_uint8[y1:y2, x1:x2]
        sr_crop = sr_img_uint8[y1:y2, x1:x2]

        ax_lr = fig.add_subplot(gs[i, 0])
        ax_lr.imshow(lr_crop)
        ax_lr.set_title(f'LR Patch #{i+1} ({y1}:{y2}, {x1}:{x2})', fontsize=9, fontweight='bold', color='#e74c3c')
        ax_lr.axis('off')

        ax_sr = fig.add_subplot(gs[i, 1])
        ax_sr.imshow(sr_crop)
        ax_sr.set_title(f'SR Patch #{i+1}', fontsize=9, fontweight='bold', color='#27ae60')
        ax_sr.axis('off')

    fig.suptitle(f'Random Region Sampling - {img_name}', fontsize=16, fontweight='bold', color='#1a1a2e')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()


def _save_metrics_pro(lr_hwc, sr_hwc, img_name, save_path):
    """保存专业级量化可视化: 字体结构/清晰度/噪声等。"""
    # 转灰度
    lr_gray = lr_hwc.mean(axis=2)
    sr_gray = sr_hwc.mean(axis=2)

    # 梯度(结构与清晰度)
    grad_lr_x = np.diff(lr_gray, axis=1)
    grad_lr_y = np.diff(lr_gray, axis=0)
    grad_sr_x = np.diff(sr_gray, axis=1)
    grad_sr_y = np.diff(sr_gray, axis=0)

    # 对齐梯度尺寸，避免广播错误
    gh_lr = min(grad_lr_x.shape[0], grad_lr_y.shape[0])
    gw_lr = min(grad_lr_x.shape[1], grad_lr_y.shape[1])
    grad_lr = np.sqrt(grad_lr_x[:gh_lr, :gw_lr]**2 + grad_lr_y[:gh_lr, :gw_lr]**2)

    gh_sr = min(grad_sr_x.shape[0], grad_sr_y.shape[0])
    gw_sr = min(grad_sr_x.shape[1], grad_sr_y.shape[1])
    grad_sr = np.sqrt(grad_sr_x[:gh_sr, :gw_sr]**2 + grad_sr_y[:gh_sr, :gw_sr]**2)

    # 噪声估计: 与高斯平滑差值的标准差
    def _gaussian_smooth(img, sigma=1.0):
        try:
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(img, sigma=sigma)
        except Exception:
            try:
                import cv2
                ksize = max(3, int(2 * round(3 * sigma) + 1))
                return cv2.GaussianBlur(img, (ksize, ksize), sigma)
            except Exception:
                # 简单均值滤波兜底（纯 numpy）
                k = np.ones((3, 3), dtype=np.float32) / 9.0
                pad = np.pad(img, 1, mode='reflect')
                out = (
                    k[1,1]*pad[1:-1,1:-1] + k[0,0]*pad[:-2,:-2] + k[0,1]*pad[:-2,1:-1] +
                    k[0,2]*pad[:-2,2:] + k[1,0]*pad[1:-1,:-2] + k[1,2]*pad[1:-1,2:] +
                    k[2,0]*pad[2:,:-2] + k[2,1]*pad[2:,1:-1] + k[2,2]*pad[2:,2:]
                )
                return out

    def estimate_noise(img, sigma=1.0):
        smooth = _gaussian_smooth(img, sigma=sigma)
        residual = img - smooth
        return residual.std()

    noise_lr = estimate_noise(lr_gray)
    noise_sr = estimate_noise(sr_gray)

    # 字体结构密度(简单二值化前景占比)
    thresh_lr = lr_gray.mean()
    thresh_sr = sr_gray.mean()
    stroke_density_lr = (lr_gray <= thresh_lr).mean()
    stroke_density_sr = (sr_gray <= thresh_sr).mean()

    # 对比度与清晰度指标
    contrast_lr = lr_gray.std()
    contrast_sr = sr_gray.std()
    sharp_lr = grad_lr.mean()
    sharp_sr = grad_sr.mean()

    # 频谱能量(径向平均)
    def radial_power(img):
        f = np.fft.fftshift(np.fft.fft2(img))
        p = np.abs(f)**2
        cy, cx = np.array(p.shape)//2
        y, x = np.indices(p.shape)
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        r_int = r.astype(np.int32)
        max_r = r_int.max()
        radial_mean = [p[r_int == i].mean() for i in range(max_r + 1)]
        return radial_mean

    power_lr = radial_power(lr_gray)
    power_sr = radial_power(sr_gray)
    freq_axis = np.arange(len(power_lr))

    # 量化指标条形对比
    metrics = ['Contrast', 'Sharpness', 'StrokeDensity', 'Noise(↓)']
    lr_vals = [contrast_lr, sharp_lr, stroke_density_lr, noise_lr]
    sr_vals = [contrast_sr, sharp_sr, stroke_density_sr, noise_sr]

    fig = plt.figure(figsize=(16, 12), facecolor='white')
    gs = fig.add_gridspec(3, 2, height_ratios=[0.5, 1, 1], hspace=0.32, wspace=0.25)

    # 条形对比
    ax_bar = fig.add_subplot(gs[0, :])
    x = np.arange(len(metrics))
    width = 0.35
    ax_bar.bar(x - width/2, lr_vals, width, label='LR', color='#e74c3c', alpha=0.8)
    ax_bar.bar(x + width/2, sr_vals, width, label='SR', color='#27ae60', alpha=0.8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metrics, fontsize=10, fontweight='bold')
    ax_bar.set_ylabel('Value', fontsize=11, fontweight='bold')
    ax_bar.set_title('Structural & Noise Metrics (LR vs SR)', fontsize=13, fontweight='bold')
    ax_bar.legend()

    # 梯度直方图(结构清晰度)
    ax_hist = fig.add_subplot(gs[1, 0])
    ax_hist.hist(grad_lr.ravel(), bins=50, alpha=0.6, label='LR', color='#e67e22')
    ax_hist.hist(grad_sr.ravel(), bins=50, alpha=0.6, label='SR', color='#2980b9')
    ax_hist.set_title('Gradient Magnitude Distribution', fontsize=12, fontweight='bold')
    ax_hist.set_xlabel('Gradient')
    ax_hist.set_ylabel('Frequency')
    ax_hist.legend()

    # 频谱径向平均(纹理细节)
    ax_freq = fig.add_subplot(gs[1, 1])
    ax_freq.plot(freq_axis, power_lr, label='LR', color='#e74c3c', alpha=0.7)
    ax_freq.plot(freq_axis, power_sr, label='SR', color='#27ae60', alpha=0.8)
    ax_freq.set_xlim(0, min(len(freq_axis), 200))
    ax_freq.set_title('Radial Power Spectrum', fontsize=12, fontweight='bold')
    ax_freq.set_xlabel('Frequency radius')
    ax_freq.set_ylabel('Power')
    ax_freq.legend()

    # 单行剖面对比(字体边缘)
    ax_profile = fig.add_subplot(gs[2, 0])
    mid_row = lr_gray.shape[0] // 2
    ax_profile.plot(sr_gray[mid_row, :], label='SR', color='#27ae60')
    ax_profile.plot(lr_gray[mid_row, :], label='LR', color='#e74c3c', alpha=0.8)
    ax_profile.set_title('Center Row Intensity Profile', fontsize=12, fontweight='bold')
    ax_profile.set_xlabel('X')
    ax_profile.set_ylabel('Intensity')
    ax_profile.legend()

    # 噪声残差可视化
    res_lr = lr_gray - _gaussian_smooth(lr_gray, sigma=1.0)
    res_sr = sr_gray - _gaussian_smooth(sr_gray, sigma=1.0)
    vmax = max(np.abs(res_lr).max(), np.abs(res_sr).max())
    ax_noise_lr = fig.add_subplot(gs[2, 1])
    im = ax_noise_lr.imshow(np.hstack([res_lr, res_sr]), cmap='bwr', vmin=-vmax, vmax=vmax)
    ax_noise_lr.set_title('Noise Residual (LR | SR)', fontsize=12, fontweight='bold')
    ax_noise_lr.axis('off')
    cbar = plt.colorbar(im, ax=ax_noise_lr, fraction=0.046, pad=0.04)
    cbar.set_label('Residual Intensity', fontsize=9)

    fig.suptitle(f'Academic Metrics Visualization - {img_name}', fontsize=16, fontweight='bold', color='#1a1a2e')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()


def _save_comparison_grid(lr, sr, hr, img_name, save_path):
    """保存 LR | SR | HR 对比图"""
    # 统一到 HR 分辨率
    hr_h, hr_w = hr.shape[1], hr.shape[2]
    lr_resized = _resize_chw(lr, hr_h, hr_w)
    sr_resized = _resize_chw(sr, hr_h, hr_w)
    
    # 转换为 numpy 数组
    lr_hwc = lr_resized.permute(1, 2, 0).detach().cpu().numpy()
    sr_hwc = sr_resized.permute(1, 2, 0).detach().cpu().numpy()
    hr_hwc = hr.permute(1, 2, 0).detach().cpu().numpy()
    
    lr_hwc = (lr_hwc * 255).clip(0, 255).astype(np.uint8)
    sr_hwc = (sr_hwc * 255).clip(0, 255).astype(np.uint8)
    hr_hwc = (hr_hwc * 255).clip(0, 255).astype(np.uint8)
    
    # 设置美观的配色方案
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 7))
    
    # 创建网格布局
    gs = fig.add_gridspec(2, 3, height_ratios=[0.1, 1], hspace=0.2, wspace=0.15)
    
    # 顶部标题区域
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    
    # 主标题
    title_text = f"Demo: {img_name}"
    ax_title.text(0.5, 0.5, title_text, ha='center', va='center', 
                  fontsize=18, fontweight='bold', color='#2c3e50',
                  bbox=dict(boxstyle='round,pad=0.8', facecolor='white', 
                           edgecolor='#3498db', linewidth=2))
    
    # 图像子图
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[1, 2])
    
    # LR (Low Resolution Input)
    ax1.imshow(lr_hwc)
    ax1.set_title('Low Resolution (Input)', fontsize=14, fontweight='bold', 
                  color='#34495e', pad=15)
    ax1.axis('off')
    # 添加边框
    for spine in ax1.spines.values():
        spine.set_edgecolor('#95a5a6')
        spine.set_linewidth(2)
    
    # SR (ours Super-Resolution)
    ax2.imshow(sr_hwc)
    title_color = '#3498db'
    ax2.set_title('Super-Resolution (ours)', fontsize=14, fontweight='bold', 
                  color=title_color, pad=15)
    ax2.axis('off')
    # 添加高亮边框
    for spine in ax2.spines.values():
        spine.set_edgecolor(title_color)
        spine.set_linewidth(3)
    
    # HR (Ground Truth - Original Image)
    ax3.imshow(hr_hwc)
    ax3.set_title('Original (Reference)', fontsize=14, fontweight='bold', 
                  color='#27ae60', pad=15)
    ax3.axis('off')
    # 添加边框
    for spine in ax3.spines.values():
        spine.set_edgecolor('#27ae60')
        spine.set_linewidth(3)
    
    # 添加水印/信息
    fig.text(0.99, 0.01, 'ours Demo', ha='right', va='bottom',
            fontsize=9, color='#95a5a6', style='italic')
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    plt.style.use('default')  # 恢复默认样式


def _save_heatmap(sr, hr, save_path, img_name, metrics=None):
    """保存美化的误差热图"""
    # 将 SR 调整到与 HR 相同的尺寸
    if sr.shape != hr.shape:
        sr = F.interpolate(sr.unsqueeze(0), size=(hr.shape[1], hr.shape[2]), 
                          mode='bicubic', align_corners=False).squeeze(0)
    
    # 计算误差
    diff = torch.abs(sr - hr).mean(dim=0).detach().cpu().numpy()
    
    # 创建图形
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.15, 1], width_ratios=[1, 1], 
                          hspace=0.3, wspace=0.3)
    
    # 标题区域
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    ax_title.text(0.5, 0.5, f'Error Analysis: {img_name}', ha='center', va='center',
                  fontsize=16, fontweight='bold', color='#e74c3c',
                  bbox=dict(boxstyle='round,pad=0.8', facecolor='white',
                           edgecolor='#e74c3c', linewidth=2))
    
    # SR 图像
    ax_sr = fig.add_subplot(gs[1, 0])
    sr_hwc = sr.permute(1, 2, 0).detach().cpu().numpy()
    sr_hwc = (sr_hwc * 255).clip(0, 255).astype(np.uint8)
    ax_sr.imshow(sr_hwc)
    ax_sr.set_title('Super-Resolution Output', fontsize=12, fontweight='bold', color='#3498db')
    ax_sr.axis('off')
    
    # 误差热图
    ax_heat = fig.add_subplot(gs[1, 1])
    im = ax_heat.imshow(diff, cmap='hot', vmin=0, vmax=0.3)
    ax_heat.set_title('Pixel-wise Error Map', fontsize=12, fontweight='bold', 
                     color='#e74c3c')
    ax_heat.axis('off')
    
    # 添加色条
    cbar = plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label('Absolute Error', fontsize=10, fontweight='bold')
    cbar.ax.tick_params(labelsize=9)
    
    # 添加统计信息
    if metrics:
        stats_text = (f"Quality Metrics:\n"
                     f"PSNR: {metrics['psnr']:.2f} dB\n"
                     f"SSIM: {metrics['ssim']:.4f}\n"
                     f"Mean Error (L1): {metrics['l1']:.4f}")
        fig.text(0.5, 0.02, stats_text, ha='center', va='bottom',
                fontsize=10, fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='#ecf0f1',
                         edgecolor='#34495e', linewidth=1.5))
    
    # 添加水印
    fig.text(0.99, 0.01, 'ours Demo', ha='right', va='bottom',
            fontsize=8, color='#95a5a6', style='italic')
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()


def _generate_adaptive_mask(img_pil):
    """
    生成自适应二值化mask(与eval数据处理一致)
    
    Args:
        img_pil: PIL Image对象(RGB或灰度)
    
    Returns:
        mask_tensor: mask张量 (1, H, W),值域[0, 1]
    """
    # 转换为灰度图
    img_gray = img_pil.convert('L')
    
    # 计算自适应阈值(使用均值)
    img_array = np.array(img_gray)
    threshold = img_array.mean()
    
    # 二值化:大于阈值为0(背景),小于等于阈值为1(前景/文字)
    # 这与dataset.py中的逻辑一致
    mask_array = np.where(img_array > threshold, 0, 255).astype(np.uint8)
    
    # 转换为张量
    mask_pil = Image.fromarray(mask_array)
    mask_tensor = transforms.ToTensor()(mask_pil)  # (1, H, W), [0, 1]
    
    return mask_tensor


def _load_jpeg_image_tiled(image_path, device, tile_h=16, tile_w=64, overlap=0.15, add_mask=True):
    """
    从 JPEG 文件加载低分辨率图像(Demo场景:输入=LR，无ground truth)
    将原图调整为 tile_h*M × tile_w*N 的尺寸，然后分成多个重叠的 tile_h×tile_w 小块
    
    Demo场景逻辑:
    - 输入图片 = LR (低分辨率，需要超分辨)
    - 没有 ground truth HR
    - 模型输出 SR (超分辨率结果)
    
    Args:
        image_path: JPEG 图像的路径
        device: 目标设备 (CPU or CUDA)
        tile_h: 单个块的高度（默认 16）
        tile_w: 单个块的宽度（默认 64）
        overlap: 重叠比例（默认0.15表示15%重叠，减少以降低条纹）
        add_mask: 是否添加掩码通道（ours 需要 4 通道输入）
    
    Returns:
        img_lr_tiles: 低分辨率分块列表 (模型输入)
        lr_positions: LR块的位置信息
        img_name: 图像文件名
        resized_h: 调整后的总高度
        resized_w: 调整后的总宽度
        original_pil: 原始PIL图像(用于可视化对比)
    """
    # 打开图像 - 这是LR(低分辨率输入)
    img_lr = Image.open(image_path).convert('RGB')
    img_name = os.path.basename(image_path)
    orig_w, orig_h = img_lr.size
    
    print(f'  - Original LR size: {orig_w}×{orig_h}')
    
    # 计算最优分块策略
    resized_h, resized_w, _, _ = _calculate_optimal_tiling(
        orig_h, orig_w, tile_h, tile_w
    )
    
    # 使用PIL的BICUBIC插值resize到分块尺寸(与eval一致)
    img_lr_resized = img_lr.resize((resized_w, resized_h), Image.BICUBIC)
    
    # 转换为张量(使用transforms.ToTensor自动归一化到[0,1])
    to_tensor = transforms.ToTensor()
    img_lr_tensor = to_tensor(img_lr_resized).unsqueeze(0)  # (1, 3, H, W)
    
    # 分块（带重叠）
    lr_tiles, lr_positions, n_tiles_h, n_tiles_w = _split_into_tiles(
        img_lr_tensor[0], tile_h, tile_w, overlap
    )
    
    print(f'  - Resized to: {resized_w}×{resized_h}')
    print(f'  - Tiles with {int(overlap*100)}% overlap: {len(lr_tiles)} tiles')
    
    # 添加自适应mask通道(与eval一致)
    if add_mask:
        # 生成自适应mask
        mask_lr = _generate_adaptive_mask(img_lr_resized)  # (1, H, W)
        
        # 将整图mask分块
        mask_lr_tiles, _, _, _ = _split_into_tiles(mask_lr, tile_h, tile_w, overlap)
        
        lr_tiles_masked = []
        
        for tile_lr, mask_lr_t in zip(lr_tiles, mask_lr_tiles):
            # 拼接RGB + mask
            tile_lr_masked = torch.cat([tile_lr, mask_lr_t], dim=1)
            lr_tiles_masked.append(tile_lr_masked.to(device))
        
        lr_tiles = lr_tiles_masked
    else:
        lr_tiles = [t.to(device) for t in lr_tiles]
    
    return lr_tiles, lr_positions, img_name, resized_h, resized_w, img_lr


def _load_jpeg_image(image_path, device, target_h=16, target_w=64, add_mask=True):
    """
    从 JPEG 文件加载图像，并将其调整为目标尺寸（旧版单块处理）
    同时生成低分辨率版本用作输入
    
    注意：ours 模型架构要求
    - 训练配置：height=32, width=128, scale_factor=2
    - TPS 输出尺寸：tps_outputsize = [height/scale_factor, width/scale_factor] = [16, 64]
    - 因此必须使用 16×64 作为输入尺寸
    
    Args:
        image_path: JPEG 图像的路径
        device: 目标设备 (CPU or CUDA)
        target_h: 目标高度 (默认 16 = 32/2)
        target_w: 目标宽度 (默认 64 = 128/2)
        add_mask: 是否添加掩码通道（ours 需要 4 通道输入）
    
    Returns:
        img_lr: 低分辨率张量 (1, 4, 16, 64)
        img_hr: 高分辨率张量 (1, 4, 16, 64)
        img_name: 图像文件名
    """
    # 打开图像
    img = Image.open(image_path).convert('RGB')
    img_name = os.path.basename(image_path)
    
    # 转换为 numpy 数组
    img_np = np.array(img).astype(np.float32) / 255.0  # 归一化到 [0, 1]
    
    # 转换为 torch 张量 HWC -> CHW
    img_hwc = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # BCHW
    
    # 调整大小到目标尺寸 (作为 HR 参考)
    # target_h=16, target_w=64 对应 ours 模型中 tps_outputsize
    img_hr = F.interpolate(img_hwc, size=(target_h, target_w), 
                          mode='bicubic', align_corners=False)
    
    # 生成低分辨率版本 (双重下采样: 先下采样再上采样)
    scale_factor = 4  # 4倍超分
    lr_h = target_h // scale_factor
    lr_w = target_w // scale_factor
    
    # 先下采样到低分辨率
    img_lr_small = F.interpolate(img_hwc, size=(lr_h, lr_w), 
                                mode='bicubic', align_corners=False)
    
    # 再上采样回目标尺寸 (用于模型输入)
    img_lr = F.interpolate(img_lr_small, size=(target_h, target_w), 
                          mode='bicubic', align_corners=False)
    
    # 添加掩码通道（ours 模型期望 4 通道输入）
    if add_mask:
        # 使用全 1 掩码表示有效区域
        mask_lr = torch.ones_like(img_lr[:, :1, :, :])  # (1, 1, H, W)
        mask_hr = torch.ones_like(img_hr[:, :1, :, :])
        img_lr = torch.cat([img_lr, mask_lr], dim=1)  # (1, 4, H, W)
        img_hr = torch.cat([img_hr, mask_hr], dim=1)
    
    # 移到设备
    img_lr = img_lr.to(device)
    img_hr = img_hr.to(device)
    
    return img_lr, img_hr, img_name


def _split_pdf_to_images(pdf_path, output_dir, dpi=200):
    """将PDF分割成单独的图像文件。
    
    Args:
        pdf_path: PDF文件路径
        output_dir: 输出图像目录
        dpi: 渲染DPI (default: 200)
    
    Returns:
        image_paths: 生成的图像文件路径列表
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("ERROR: PyMuPDF (fitz) not installed. Install with: pip install PyMuPDF")
        return []
    
    os.makedirs(output_dir, exist_ok=True)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    print(f"  - Opening PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    image_paths = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        # 渲染页面为图像
        mat = fitz.Matrix(dpi/72, dpi/72)  # 缩放矩阵
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # 保存为PNG
        img_filename = f"{pdf_name}_page{page_num+1:03d}.png"
        img_path = os.path.join(output_dir, img_filename)
        pix.save(img_path)
        image_paths.append(img_path)
        print(f"    - Saved page {page_num+1}/{len(doc)}: {img_filename}")
    
    doc.close()
    print(f"  - Total {len(image_paths)} pages extracted from PDF")
    return image_paths


def _predict_aster(aster, aster_info, img):
    """使用 ASTER 识别文本"""
    with torch.no_grad():
        # img 应该是 CHW 格式
        if img.dim() == 3:
            img = img.unsqueeze(0)  # BCHW
        
        # ASTER 需要 3 通道
        if img.shape[1] == 1:
            img = img.repeat(1, 3, 1, 1)
        elif img.shape[1] > 3:
            img = img[:, :3, :, :]
        
        # 预测
        input_dict = {}
        input_dict['images'] = img
        rec_targets = torch.IntTensor(1, aster_info.max_len).fill_(1)
        input_dict['rec_targets'] = rec_targets
        input_dict['rec_lengths'] = [aster_info.max_len]
        
        output_dict = aster(input_dict)
        pred_rec = output_dict['output']['pred_rec']
        
        # 使用 get_str_list 解码
        pred_str, _ = get_str_list(pred_rec, rec_targets, dataset=aster_info)
        
        return pred_str[0] if pred_str and len(pred_str) > 0 else ''


def main():
    parser = argparse.ArgumentParser(description='PEAN Demo - Process Images or PDFs')
    parser.add_argument('--PEAN_ckpt', type=str, default='./ckpt/PEAN_final.pth',
                       help='Path to PEAN checkpoint')
    parser.add_argument('--tpem_ckpt', type=str, default='./ckpt/TPEM_final.pth',
                       help='Path to TPEM checkpoint (optional)')
    parser.add_argument('--input_type', type=str, required=True, choices=['img', 'pdf'],
                       help='Input type: "img" for images or "pdf" for PDF files')
    parser.add_argument('--demo_dir', type=str, default='./demo_img',
                       help='Directory containing demo files (images or PDFs based on input_type)')
    parser.add_argument('--out_dir', type=str, default='./demo_results',
                       help='Output directory for results')
    parser.add_argument('--pdf_dpi', type=int, default=200,
                       help='DPI for PDF rendering (default: 200)')
    parser.add_argument('--srb', type=int, default=1,
                       help='Number of SRB blocks (should match checkpoint)')
    parser.add_argument('--target_h', type=int, default=16,
                       help='Target image height (tps_outputsize = height/scale_factor = 32/2 = 16)')
    parser.add_argument('--target_w', type=int, default=64,
                       help='Target image width (tps_outputsize = width/scale_factor = 128/2 = 64)')
    parser.add_argument('--enhance_edges', action='store_true', default=True,
                       help='Enable edge enhancement for clearer text strokes (default: True)')
    parser.add_argument('--sharpen_strength', type=float, default=0.5,
                       help='Edge sharpening strength (0-1, default: 0.5)')
    parser.add_argument('--contrast_factor', type=float, default=1.15,
                       help='Contrast enhancement factor (>1 enhance, default: 1.15)')
    args = parser.parse_args()

    print('=' * 100)
    print(f'PEAN Demo - Processing {args.input_type.upper()} Files from Demo Folder')
    print('=' * 100)
    print(f'Input Type: {args.input_type.upper()}')
    print(f'PEAN Checkpoint: {args.PEAN_ckpt}')
    if args.tpem_ckpt and os.path.exists(args.tpem_ckpt):
        print(f'TPEM Checkpoint: {args.tpem_ckpt}')
    print(f'Demo Directory: {args.demo_dir}')
    if args.input_type == 'pdf':
        print(f'PDF Rendering DPI: {args.pdf_dpi}')
    print(f'Target Size (tps_outputsize): {args.target_h}x{args.target_w}')
    print(f'Edge Enhancement: {"Enabled" if args.enhance_edges else "Disabled"}')
    if args.enhance_edges:
        print(f'  - Sharpen Strength: {args.sharpen_strength}')
        print(f'  - Contrast Factor: {args.contrast_factor}')
    print('=' * 100)

    # 检查 demo 目录是否存在
    if not os.path.isdir(args.demo_dir):
        print(f'ERROR: Demo directory "{args.demo_dir}" not found!')
        return

    # 准备临时目录用于PDF转换
    temp_img_dir = None
    
    if args.input_type == 'pdf':
        # 处理PDF文件
        pdf_files = [f for f in os.listdir(args.demo_dir) 
                     if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f'ERROR: No PDF files found in {args.demo_dir}')
            return
        
        pdf_files = sorted(pdf_files)
        print(f'\nFound {len(pdf_files)} PDF file(s) to process:')
        for f in pdf_files:
            print(f'  - {f}')
        
        # 创建临时目录存储PDF转换的图像
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_img_dir = _ensure_dir(os.path.join(args.out_dir, f'temp_pdf_images_{ts}'))
        
        # 转换所有PDF为图像
        print('\nConverting PDFs to images...')
        all_image_paths = []
        for pdf_file in pdf_files:
            pdf_path = os.path.join(args.demo_dir, pdf_file)
            img_paths = _split_pdf_to_images(pdf_path, temp_img_dir, dpi=args.pdf_dpi)
            all_image_paths.extend(img_paths)
        
        # 使用转换后的图像路径
        jpeg_files = [os.path.basename(p) for p in all_image_paths]
        # 更新demo_dir指向临时目录
        original_demo_dir = args.demo_dir
        args.demo_dir = temp_img_dir
        
    else:
        # 处理图像文件
        jpeg_files = [f for f in os.listdir(args.demo_dir) 
                      if f.lower().endswith(('.jpeg', '.jpg', '.png', '.webp'))]
        
        if not jpeg_files:
            print(f'ERROR: No images found in {args.demo_dir}')
            return
        
        jpeg_files = sorted(jpeg_files)
        print(f'\nFound {len(jpeg_files)} image(s) to process:')
        for f in jpeg_files:
            print(f'  - {f}')

    # 加载配置
    config_path = os.path.join('config', 'super_resolution.yaml')
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    config = EasyDict(config)

    # 构建参数
    args_pean = _build_args(batch_size=1, srb=args.srb)
    args_pean.resume = args.PEAN_ckpt

    # 实例化 PEAN
    print('\nInitializing PEAN...')
    pean = TextSR(config, args_pean)
    setattr(pean, '_comparison_name', 'PEAN')

    # 加载 PEAN 模型
    print('Loading PEAN model...')
    pean_model = pean.generator_init(resume_this=args.PEAN_ckpt)['model']
    pean_model.eval()
    print('PEAN model loaded successfully')

    # 加载 PARSeq
    print('Loading PARSeq recognizer...')
    parseq = pean.PARSeq_init()
    parseq.eval()
    for p in parseq.parameters():
        p.requires_grad = False
    print('PARSeq loaded')

    # 加载 ASTER
    print('Loading ASTER recognizer...')
    aster, aster_info = pean.Aster_init()
    aster.eval()
    for p in aster.parameters():
        p.requires_grad = False
    print('ASTER loaded')

    # 准备输出目录(Demo: 不需要heatmaps)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_root = _ensure_dir(os.path.join(args.out_dir, f'demo_{ts}'))
    img_dir = _ensure_dir(os.path.join(out_root, 'comparison'))
    hr_dir = _ensure_dir(os.path.join(out_root, 'demo_single_HR'))  # 新增：单独保存HR图像的目录
    csv_path = os.path.join(out_root, 'demo_results.csv')

    print(f'\nOutput directory: {out_root}')
    print(f'HR images will be saved to: {hr_dir}')

    # 创建 CSV 文件 (Demo: 只记录识别结果)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['index', 'image_name', 'pred_lr', 'pred_sr'])

    # 评估循环
    print('\n' + '=' * 100)
    print('Processing demo images...')
    print('=' * 100)
    
    all_predictions_lr = []
    all_predictions_sr = []
    
    # 获取字符映射 (alphabet to digit)
    alphabet = ':'.join(string.digits + string.ascii_lowercase + '$')
    a2d = {ch: i for i, ch in enumerate(alphabet.split(':'))}
    
    for idx, img_file in enumerate(tqdm(jpeg_files, desc='Images', unit='img')):
        start_time = time.time()
        print(f'\n[{idx+1}/{len(jpeg_files)}] Processing: {img_file}')
        
        img_path = os.path.join(args.demo_dir, img_file)
        
        try:
            # 加载 JPEG 图像(Demo: 输入=LR, 无ground truth)
            lr_tiles, lr_positions, img_name, resized_h, resized_w, original_lr_pil = \
                _load_jpeg_image_tiled(img_path, pean.device, args.target_h, args.target_w)
            
            print(f'  - Total tiles: {len(lr_tiles)}')

            # 处理每个块
            sr_tiles = []
            sr_positions = []
            
            for tile_idx, lr_tile in enumerate(tqdm(lr_tiles, desc='  Tiles', leave=False, unit='tile')):
                # 生成虚拟标签
                weighted_mask = torch.tensor([0]).long()
                text_len = torch.tensor([1]).long()

                try:
                    # 构建 PARSeq 先验 (LR)
                    pq_in_lr = pean.parse_parseq_data(lr_tile[0, :3, :, :])
                    prob_str_lr = parseq(pq_in_lr, max_length=25).softmax(-1)
                    
                    # Demo场景:没有HR,使用LR的先验作为HR先验的近似
                    prob_str_hr = prob_str_lr  # 使用LR先验代替HR先验
                    
                    # 使用 TPEM 扩散先验
                    try:
                        if tile_idx == 0:  # 只在第一个块时初始化
                            pean.diffusion = pean.init_diffusion_model()
                            if args.tpem_ckpt and os.path.exists(args.tpem_ckpt):
                                pean.diffusion.load_network()
                        
                        predicted_length = torch.ones(prob_str_lr.shape[0]) * prob_str_lr.shape[1]
                        
                        data_diff = {
                            "HR": prob_str_hr,  # Demo: 使用LR先验
                            "SR": prob_str_lr,
                            "weighted_mask": weighted_mask,
                            "predicted_length": predicted_length,
                            "text_len": text_len
                        }
                        pean.diffusion.feed_data(data_diff)
                        _, label_vec_final = pean.diffusion.process()
                        label_vec_final = label_vec_final.to(pean.device)
                    except Exception as e:
                        # 直接使用 PARSeq 输出
                        label_vec_final = prob_str_lr
                        
                except Exception as e:
                    print(f'  - Tile {tile_idx+1} feature extraction failed: {str(e)}')
                    raise

                # PEAN 超分辨率
                with torch.no_grad():
                    tile_sr, _ = pean_model(lr_tile, label_vec_final)
                
                sr_tiles.append(tile_sr)
                
                # SR输出的位置需要根据缩放因子调整
                sr_scale = tile_sr.shape[2] // lr_tile.shape[2]
                h_start, h_end, w_start, w_end = lr_positions[tile_idx]
                sr_pos = (h_start * sr_scale, h_end * sr_scale, 
                         w_start * sr_scale, w_end * sr_scale)
                sr_positions.append(sr_pos)
            
            print(f'  - All {len(sr_tiles)} tiles processed, merging with smooth blending...')
            
            # 使用平滑合并（带重叠和加权融合）
            sr_scale = sr_tiles[0].shape[2] // lr_tiles[0].shape[2]
            
            sr_merged = _merge_tiles_smooth(
                sr_tiles, sr_positions, 
                resized_h * sr_scale, resized_w * sr_scale,
                overlap=0.15
            )
            lr_merged = _merge_tiles_smooth(
                lr_tiles, lr_positions,
                resized_h, resized_w,
                overlap=0.15
            )
            
            print(f'  - Merged shapes: LR={lr_merged.shape}, SR={sr_merged.shape}')
            
            # 调试：检查合并后的值域
            sr_min, sr_max, sr_mean = sr_merged.min().item(), sr_merged.max().item(), sr_merged.mean().item()
            print(f'  - SR merged range: [{sr_min:.4f}, {sr_max:.4f}], mean: {sr_mean:.4f}')
            
            # 如果SR超出范围，进行裁剪
            if sr_max > 1.0 or sr_min < 0.0:
                print(f'  - WARNING: SR values out of range, clipping...')
                sr_merged = torch.clamp(sr_merged, 0, 1)
            
            # 对SR图像应用边缘增强，使文字笔划更清晰
            if args.enhance_edges:
                # 对于高亮度图像（mean > 0.7），降低对比度增强以避免过曝
                if sr_mean > 0.7:
                    contrast_factor = min(args.contrast_factor, 1.05)
                    print(f'  - High brightness image detected, reducing contrast factor to {contrast_factor:.2f}')
                else:
                    contrast_factor = args.contrast_factor
                
                print(f'  - Enhancing text clarity (sharpen={args.sharpen_strength}, contrast={contrast_factor})...')
                sr_merged_enhanced = _enhance_text_clarity(
                    sr_merged, 
                    sharpen_strength=args.sharpen_strength,
                    contrast_factor=contrast_factor
                )
                print(f'  - Enhanced range: [{sr_merged_enhanced.min().item():.4f}, {sr_merged_enhanced.max().item():.4f}]')
            else:
                sr_merged_enhanced = sr_merged

            # 转换为 3 通道
            lr_vis = _to_3ch_float(lr_merged[:, :3, :, :])
            sr_vis = _to_3ch_float(sr_merged_enhanced[:, :3, :, :])  # 使用增强后的SR

            # 识别预测文本 (LR 和 SR)
            print('  - Recognizing text on merged image...')
            pred_lr = _predict_aster(aster, aster_info, lr_merged)
            pred_sr = _predict_aster(aster, aster_info, sr_merged_enhanced)  # 使用增强后的SR
            
            print(f'    - LR Prediction: {pred_lr}')
            print(f'    - SR Prediction: {pred_sr}')

            # Demo场景:没有ground truth,所以不计算PSNR/SSIM
            # 只保存可视化对比
            all_predictions_lr.append(pred_lr)
            all_predictions_sr.append(pred_sr)

            # 保存对比图 (LR vs SR, 无HR)
            grid_path = os.path.join(img_dir, f'{idx:02d}_{os.path.splitext(img_name)[0]}_comparison.png')
            elapsed_time = time.time() - start_time
            _save_comparison_grid_demo(lr_vis, sr_vis, img_name, pred_lr, pred_sr, grid_path, elapsed_time_s=elapsed_time)
            print(f'  - Saved comparison to {os.path.basename(grid_path)}')
            
            # 额外保存单独的 HR (SR) 图像到 demo_single_HR 文件夹
            hr_path = os.path.join(hr_dir, f'{idx:02d}_{os.path.splitext(img_name)[0]}_HR.png')
            sr_pil = transforms.ToPILImage()(sr_vis.cpu())
            sr_pil.save(hr_path)
            print(f'  - Saved HR image to {os.path.basename(hr_path)}')
            
            print(f'  - Elapsed time: {elapsed_time:.1f}s')
            
            # 记录到 CSV (Demo: 只记录文本识别结果)
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow([idx, img_name, pred_lr, pred_sr])
            
        except Exception as e:
            print(f'  ERROR processing {img_file}: {str(e)}')
            import traceback
            traceback.print_exc()
            continue
    
    # 最终统计
    if len(jpeg_files) > 0:
        print('\n' + '=' * 100)
        print('Demo Processing Complete!')
        print('=' * 100)
        print(f'Total Images: {len(jpeg_files)}')
        print(f'\nResults saved to: {out_root}')
        print(f'  - Comparison Images: {img_dir}')
        print(f'  - HR Images (SR): {hr_dir}')
        print(f'  - Results CSV: {csv_path}')
        print('=' * 100)


if __name__ == '__main__':
    main()
