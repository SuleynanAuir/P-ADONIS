"""
PEAN Single Model Evaluation Script
使用作者提供的预训练模型评估 PEAN 性能
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from interfaces.super_resolution import TextSR
from utils.metrics import get_str_list


def _build_args(batch_size=8, rec='aster', mask=True, srb=1, testing=True):
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


def _resize_chw(img_chw, target_h, target_w):
    """调整图像大小 (CHW 格式)"""
    img_1chw = img_chw.unsqueeze(0)
    resized = F.interpolate(img_1chw, size=(target_h, target_w), mode='bicubic', align_corners=False)
    return resized[0].clamp(0, 1)


def _metrics(sr, hr):
    """计算图像质量指标"""
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


def _save_single_image(img_chw, save_path):
    """保存单张图像"""
    img_hwc = img_chw.permute(1, 2, 0).detach().cpu().numpy()
    img_hwc = (img_hwc * 255).clip(0, 255).astype(np.uint8)
    plt.imsave(save_path, img_hwc)


def _save_comparison_grid(lr, sr, hr, label, pred, save_path, metrics=None):
    """保存 LR | SR | HR 对比图，每张子图带标签和量化指标"""
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
    gs = fig.add_gridspec(2, 3, height_ratios=[0.15, 1], hspace=0.3, wspace=0.15)
    
    # 顶部标题区域
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    
    # 判断预测是否正确
    is_correct = (label.lower() == pred.lower())
    correctness_color = '#2ecc71' if is_correct else '#e74c3c'
    correctness_text = '✓ Correct' if is_correct else '✗ Incorrect'
    
    # 主标题
    title_text = f"Ground Truth: \"{label}\"  |  Prediction: \"{pred}\"  |  {correctness_text}"
    ax_title.text(0.5, 0.5, title_text, ha='center', va='center', 
                  fontsize=18, fontweight='bold', color=correctness_color,
                  bbox=dict(boxstyle='round,pad=0.8', facecolor='white', 
                           edgecolor=correctness_color, linewidth=2))
    
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
    
    # SR (PEAN Super-Resolution)
    ax2.imshow(sr_hwc)
    title_color = '#3498db'
    ax2.set_title('Super-Resolution (PEAN)', fontsize=14, fontweight='bold', 
                  color=title_color, pad=15)
    ax2.axis('off')
    # 添加高亮边框
    for spine in ax2.spines.values():
        spine.set_edgecolor(title_color)
        spine.set_linewidth(3)
    
    # 在SR图下方添加量化指标
    if metrics:
        metrics_text = (f"PSNR: {metrics['psnr']:.2f} dB\n"
                       f"SSIM: {metrics['ssim']:.4f}\n"
                       f"L1: {metrics['l1']:.4f}")
        ax2.text(0.5, -0.12, metrics_text, ha='center', va='top',
                transform=ax2.transAxes, fontsize=11,
                bbox=dict(boxstyle='round,pad=0.6', facecolor='#ecf0f1', 
                         edgecolor=title_color, linewidth=2),
                fontfamily='monospace', fontweight='bold')
    
    # HR (Ground Truth)
    ax3.imshow(hr_hwc)
    ax3.set_title('High Resolution (Ground Truth)', fontsize=14, fontweight='bold', 
                  color='#27ae60', pad=15)
    ax3.axis('off')
    # 添加边框
    for spine in ax3.spines.values():
        spine.set_edgecolor('#27ae60')
        spine.set_linewidth(3)
    
    # 添加水印/信息
    fig.text(0.99, 0.01, 'PEAN Evaluation', ha='right', va='bottom',
            fontsize=9, color='#95a5a6', style='italic')
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    plt.style.use('default')  # 恢复默认样式


def _save_heatmap(sr, hr, save_path, title='Error Map', metrics=None):
    """保存美化的误差热图"""
    # 计算误差
    diff = torch.abs(sr - hr).mean(dim=0).detach().cpu().numpy()
    
    # 创建图形
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.15, 1], width_ratios=[1, 1], 
                          hspace=0.3, wspace=0.3)
    
    # 标题区域
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    ax_title.text(0.5, 0.5, title, ha='center', va='center',
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
    for spine in ax_sr.spines.values():
        spine.set_edgecolor('#3498db')
        spine.set_linewidth(2)
    
    # 误差热图
    ax_heat = fig.add_subplot(gs[1, 1])
    im = ax_heat.imshow(diff, cmap='hot', vmin=0, vmax=0.3)
    ax_heat.set_title('Pixel-wise Error Map', fontsize=12, fontweight='bold', color='#e74c3c')
    ax_heat.axis('off')
    for spine in ax_heat.spines.values():
        spine.set_edgecolor('#e74c3c')
        spine.set_linewidth(2)
    
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
    fig.text(0.99, 0.01, 'PEAN Evaluation', ha='right', va='bottom',
            fontsize=8, color='#95a5a6', style='italic')
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()


def _save_combined_heatmap(samples_data, save_path):
    """将多个样本的热图组合：左边是误差热图，右边是误差统计分析
    samples_data: list of dict, 每个dict包含 {'sr', 'hr', 'label', 'pred', 'metrics'}
    """
    n_samples = len(samples_data)
    if n_samples == 0:
        return
    
    # 创建图形：左边热图，右边统计
    fig = plt.figure(figsize=(24, 3 * n_samples + 2))
    
    # 使用 GridSpec 创建左右两部分布局
    gs_main = fig.add_gridspec(1, 2, width_ratios=[2.5, 1], wspace=0.15,
                               left=0.02, right=0.98, top=0.94, bottom=0.04)
    
    # 左侧：热图网格
    gs_left = gs_main[0, 0].subgridspec(n_samples + 1, 2, 
                                        height_ratios=[0.08] + [1] * n_samples,
                                        hspace=0.12, wspace=0.08)
    
    # 右侧：误差统计分析
    gs_right = gs_main[0, 1].subgridspec(4, 1, height_ratios=[0.15, 1, 1, 1],
                                         hspace=0.25)
    
    # ============ 左侧：误差热图 ============
    # 标题
    ax_title = fig.add_subplot(gs_left[0, :])
    ax_title.axis('off')
    avg_psnr = np.mean([s['metrics']['psnr'] for s in samples_data])
    avg_ssim = np.mean([s['metrics']['ssim'] for s in samples_data])
    avg_l1 = np.mean([s['metrics']['l1'] for s in samples_data])
    
    title_text = f"Error Analysis: {n_samples} Samples | Avg Error (L1): {avg_l1:.4f}"
    ax_title.text(0.5, 0.3, title_text, ha='center', va='center',
                  fontsize=15, fontweight='bold', color='#e74c3c',
                  bbox=dict(boxstyle='round,pad=0.6', facecolor='#fff5f5',
                           edgecolor='#e74c3c', linewidth=2.5))
    
    # 为每个样本创建热图
    all_errors = []
    for idx, sample in enumerate(samples_data):
        row = idx + 1
        
        # 计算误差
        diff = torch.abs(sample['sr'] - sample['hr']).mean(dim=0).detach().cpu().numpy()
        all_errors.append(diff)
        
        # SR图像
        sr_hwc = sample['sr'].permute(1, 2, 0).detach().cpu().numpy()
        sr_hwc = (sr_hwc * 255).clip(0, 255).astype(np.uint8)
        
        is_correct = sample['label'].lower() == sample['pred'].lower()
        mark = '✓' if is_correct else '✗'
        
        # SR
        ax_sr = fig.add_subplot(gs_left[row, 0])
        ax_sr.imshow(sr_hwc)
        ax_sr.set_title(f"#{idx+1} SR {mark}", fontsize=9, fontweight='bold', 
                       color='#3498db', pad=5)
        ax_sr.axis('off')
        
        # 热图
        ax_heat = fig.add_subplot(gs_left[row, 1])
        im = ax_heat.imshow(diff, cmap='hot', vmin=0, vmax=0.3)
        ax_heat.set_title(f"Error Map", fontsize=9, fontweight='bold', 
                         color='#e74c3c', pad=5)
        ax_heat.axis('off')
        
        # 在热图下方添加指标
        m = sample['metrics']
        error_text = f"L1: {m['l1']:.4f} | L2: {m['l2']:.4f}"
        ax_heat.text(0.5, -0.06, error_text, ha='center', va='top',
                    transform=ax_heat.transAxes, fontsize=7.5,
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                             edgecolor='#e74c3c', linewidth=1.2),
                    fontfamily='monospace', fontweight='bold', color='#c0392b')
    
    # 在最后一个热图旁边添加色条
    cbar = plt.colorbar(im, ax=fig.add_subplot(gs_left[n_samples, 1]), 
                       fraction=0.046, pad=0.04, orientation='horizontal')
    cbar.set_label('Absolute Error', fontsize=9, fontweight='bold')
    cbar.ax.tick_params(labelsize=8)
    
    # ============ 右侧：误差统计分析 ============
    # 标题
    ax_stats_title = fig.add_subplot(gs_right[0])
    ax_stats_title.axis('off')
    ax_stats_title.text(0.5, 0.5, 'Error Statistics', ha='center', va='center',
                       fontsize=14, fontweight='bold', color='#c0392b',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff5f5',
                                edgecolor='#e74c3c', linewidth=2))
    
    # 提取误差数据
    l1_values = [s['metrics']['l1'] for s in samples_data]
    l2_values = [s['metrics']['l2'] for s in samples_data]
    psnr_values = [s['metrics']['psnr'] for s in samples_data]
    sample_labels = [f"#{i+1}" for i in range(n_samples)]
    colors = ['#2ecc71' if samples_data[i]['label'].lower() == samples_data[i]['pred'].lower() 
              else '#e74c3c' for i in range(n_samples)]
    
    # 1. L1 误差柱状图
    ax_l1 = fig.add_subplot(gs_right[1])
    bars_l1 = ax_l1.bar(sample_labels, l1_values, color=colors, alpha=0.8, 
                        edgecolor='#c0392b', linewidth=1.5)
    ax_l1.set_ylabel('L1 Error', fontsize=11, fontweight='bold')
    ax_l1.set_title(f'Mean Absolute Error (Avg: {avg_l1:.4f})', 
                   fontsize=11, fontweight='bold', color='#c0392b', pad=10)
    ax_l1.grid(axis='y', alpha=0.3, linestyle='--')
    ax_l1.set_ylim(0, max(l1_values) * 1.2)
    
    for bar, val in zip(bars_l1, l1_values):
        ax_l1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(l1_values)*0.02,
                  f'{val:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 2. L2 误差柱状图
    ax_l2 = fig.add_subplot(gs_right[2])
    bars_l2 = ax_l2.bar(sample_labels, l2_values, color=colors, alpha=0.8,
                        edgecolor='#c0392b', linewidth=1.5)
    ax_l2.set_ylabel('L2 Error', fontsize=11, fontweight='bold')
    avg_l2 = np.mean(l2_values)
    ax_l2.set_title(f'Mean Squared Error (Avg: {avg_l2:.4f})', 
                   fontsize=11, fontweight='bold', color='#c0392b', pad=10)
    ax_l2.grid(axis='y', alpha=0.3, linestyle='--')
    ax_l2.set_ylim(0, max(l2_values) * 1.2)
    
    for bar, val in zip(bars_l2, l2_values):
        ax_l2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(l2_values)*0.02,
                  f'{val:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 3. 误差分布统计表格
    ax_table = fig.add_subplot(gs_right[3])
    ax_table.axis('off')
    
    # 计算误差统计
    table_data = [['Sample', 'L1 Error', 'L2 Error', 'PSNR', 'Status']]
    for i, s in enumerate(samples_data):
        result_mark = '✓' if s['label'].lower() == s['pred'].lower() else '✗'
        table_data.append([
            f"#{i+1}",
            f"{s['metrics']['l1']:.4f}",
            f"{s['metrics']['l2']:.4f}",
            f"{s['metrics']['psnr']:.2f}",
            result_mark
        ])
    
    # 添加统计行
    table_data.append([
        'Avg',
        f"{avg_l1:.4f}",
        f"{avg_l2:.4f}",
        f"{avg_psnr:.2f}",
        ''
    ])
    table_data.append([
        'Min',
        f"{min(l1_values):.4f}",
        f"{min(l2_values):.4f}",
        f"{min(psnr_values):.2f}",
        ''
    ])
    table_data.append([
        'Max',
        f"{max(l1_values):.4f}",
        f"{max(l2_values):.4f}",
        f"{max(psnr_values):.2f}",
        ''
    ])
    
    table = ax_table.table(cellText=table_data, cellLoc='center', loc='center',
                          bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    
    # 设置表格样式
    for i in range(len(table_data)):
        for j in range(5):
            cell = table[(i, j)]
            if i == 0:  # 表头
                cell.set_facecolor('#c0392b')
                cell.set_text_props(weight='bold', color='white')
            elif i >= len(table_data) - 3:  # 统计行
                cell.set_facecolor('#ecf0f1')
                cell.set_text_props(weight='bold')
            else:  # 数据行
                if samples_data[i-1]['label'].lower() == samples_data[i-1]['pred'].lower():
                    cell.set_facecolor('#e8f8f5')
                else:
                    cell.set_facecolor('#fadbd8')
            cell.set_edgecolor('#95a5a6')
            cell.set_linewidth(1.5)
    
    # 水印
    fig.text(0.99, 0.01, 'PEAN Error Analysis', ha='right', va='bottom',
            fontsize=8, color='#95a5a6', style='italic', alpha=0.7)
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.close()


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


def _save_combined_grid(samples_data, save_path):
    """将多个样本组合成一张图：左边是图像对比，右边是量化指标可视化
    samples_data: list of dict, 每个dict包含 {'lr', 'sr', 'hr', 'label', 'pred', 'metrics'}
    """
    n_samples = len(samples_data)
    if n_samples == 0:
        return
    
    # 创建图形：左边图像，右边指标
    fig = plt.figure(figsize=(24, 3 * n_samples + 2))
    
    # 使用 GridSpec 创建左右两部分布局
    gs_main = fig.add_gridspec(1, 2, width_ratios=[2.5, 1], wspace=0.15,
                               left=0.02, right=0.98, top=0.94, bottom=0.04)
    
    # 左侧：图像网格
    gs_left = gs_main[0, 0].subgridspec(n_samples + 1, 3, 
                                        height_ratios=[0.08] + [1] * n_samples,
                                        hspace=0.12, wspace=0.06)
    
    # 右侧：指标可视化
    gs_right = gs_main[0, 1].subgridspec(4, 1, height_ratios=[0.15, 1, 1, 1],
                                         hspace=0.25)
    
    # ============ 左侧：图像对比 ============
    # 标题
    ax_title = fig.add_subplot(gs_left[0, :])
    ax_title.axis('off')
    correct_count = sum(1 for s in samples_data if s['label'].lower() == s['pred'].lower())
    avg_psnr = np.mean([s['metrics']['psnr'] for s in samples_data])
    avg_ssim = np.mean([s['metrics']['ssim'] for s in samples_data])
    
    title_text = f"PEAN Evaluation: {n_samples} Samples | Accuracy: {correct_count}/{n_samples} ({correct_count/n_samples*100:.1f}%)"
    ax_title.text(0.5, 0.3, title_text, ha='center', va='center',
                  fontsize=15, fontweight='bold', color='#2c3e50',
                  bbox=dict(boxstyle='round,pad=0.6', facecolor='#ecf0f1',
                           edgecolor='#3498db', linewidth=2.5))
    
    # 为每个样本创建一行
    for idx, sample in enumerate(samples_data):
        row = idx + 1
        
        # 统一分辨率
        hr_h, hr_w = sample['hr'].shape[1], sample['hr'].shape[2]
        lr_resized = _resize_chw(sample['lr'], hr_h, hr_w)
        sr_resized = _resize_chw(sample['sr'], hr_h, hr_w)
        
        # 转换为 numpy
        lr_hwc = (lr_resized.permute(1, 2, 0).detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        sr_hwc = (sr_resized.permute(1, 2, 0).detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        hr_hwc = (sample['hr'].permute(1, 2, 0).detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        
        is_correct = sample['label'].lower() == sample['pred'].lower()
        mark = '✓' if is_correct else '✗'
        
        # LR
        ax_lr = fig.add_subplot(gs_left[row, 0])
        ax_lr.imshow(lr_hwc)
        ax_lr.set_title(f"#{idx+1} LR", fontsize=9, fontweight='bold', color='#7f8c8d', pad=5)
        ax_lr.axis('off')
        
        # SR
        ax_sr = fig.add_subplot(gs_left[row, 1])
        ax_sr.imshow(sr_hwc)
        ax_sr.set_title(f"SR {mark}", fontsize=9, fontweight='bold', color='#3498db', pad=5)
        ax_sr.axis('off')
        
        # HR
        ax_hr = fig.add_subplot(gs_left[row, 2])
        ax_hr.imshow(hr_hwc)
        ax_hr.set_title(f"HR", fontsize=9, fontweight='bold', color='#27ae60', pad=5)
        ax_hr.axis('off')
        
        # 在图像下方添加文本信息
        text_color = '#2ecc71' if is_correct else '#e74c3c'
        text_info = f"GT: \"{sample['label']}\" | Pred: \"{sample['pred']}\""
        ax_hr.text(0.5, -0.06, text_info, ha='center', va='top',
                  transform=ax_hr.transAxes, fontsize=7.5,
                  bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                           edgecolor=text_color, linewidth=1.2),
                  fontweight='bold', color=text_color)
    
    # ============ 右侧：量化指标可视化 ============
    # 标题
    ax_metrics_title = fig.add_subplot(gs_right[0])
    ax_metrics_title.axis('off')
    ax_metrics_title.text(0.5, 0.5, 'Quality Metrics', ha='center', va='center',
                         fontsize=14, fontweight='bold', color='#34495e',
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa',
                                  edgecolor='#95a5a6', linewidth=2))
    
    # 提取指标数据
    psnr_values = [s['metrics']['psnr'] for s in samples_data]
    ssim_values = [s['metrics']['ssim'] for s in samples_data]
    l1_values = [s['metrics']['l1'] for s in samples_data]
    sample_labels = [f"#{i+1}" for i in range(n_samples)]
    colors = ['#2ecc71' if samples_data[i]['label'].lower() == samples_data[i]['pred'].lower() 
              else '#e74c3c' for i in range(n_samples)]
    
    # 1. PSNR 柱状图
    ax_psnr = fig.add_subplot(gs_right[1])
    bars_psnr = ax_psnr.bar(sample_labels, psnr_values, color=colors, alpha=0.8, edgecolor='#2c3e50', linewidth=1.5)
    ax_psnr.set_ylabel('PSNR (dB)', fontsize=11, fontweight='bold')
    ax_psnr.set_title(f'Peak Signal-to-Noise Ratio (Avg: {avg_psnr:.2f} dB)', 
                     fontsize=11, fontweight='bold', color='#2c3e50', pad=10)
    ax_psnr.grid(axis='y', alpha=0.3, linestyle='--')
    ax_psnr.set_ylim(0, max(psnr_values) * 1.15)
    
    # 在柱子上显示数值
    for i, (bar, val) in enumerate(zip(bars_psnr, psnr_values)):
        ax_psnr.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. SSIM 柱状图
    ax_ssim = fig.add_subplot(gs_right[2])
    bars_ssim = ax_ssim.bar(sample_labels, ssim_values, color=colors, alpha=0.8, edgecolor='#2c3e50', linewidth=1.5)
    ax_ssim.set_ylabel('SSIM', fontsize=11, fontweight='bold')
    ax_ssim.set_title(f'Structural Similarity Index (Avg: {avg_ssim:.4f})', 
                     fontsize=11, fontweight='bold', color='#2c3e50', pad=10)
    ax_ssim.grid(axis='y', alpha=0.3, linestyle='--')
    ax_ssim.set_ylim(0, 1.1)
    
    # 在柱子上显示数值
    for i, (bar, val) in enumerate(zip(bars_ssim, ssim_values)):
        ax_ssim.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 3. 综合指标对比表格
    ax_table = fig.add_subplot(gs_right[3])
    ax_table.axis('off')
    
    # 创建表格数据
    table_data = [['Sample', 'PSNR↑', 'SSIM↑', 'L1↓', 'Result']]
    for i, s in enumerate(samples_data):
        result_mark = '✓' if s['label'].lower() == s['pred'].lower() else '✗'
        table_data.append([
            f"#{i+1}",
            f"{s['metrics']['psnr']:.2f}",
            f"{s['metrics']['ssim']:.4f}",
            f"{s['metrics']['l1']:.4f}",
            result_mark
        ])
    
    # 添加平均值行
    avg_l1 = np.mean(l1_values)
    table_data.append([
        'Avg',
        f"{avg_psnr:.2f}",
        f"{avg_ssim:.4f}",
        f"{avg_l1:.4f}",
        f"{correct_count}/{n_samples}"
    ])
    
    table = ax_table.table(cellText=table_data, cellLoc='center', loc='center',
                          bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    
    # 设置表格样式
    for i in range(len(table_data)):
        for j in range(5):
            cell = table[(i, j)]
            if i == 0:  # 表头
                cell.set_facecolor('#34495e')
                cell.set_text_props(weight='bold', color='white')
            elif i == len(table_data) - 1:  # 平均值行
                cell.set_facecolor('#ecf0f1')
                cell.set_text_props(weight='bold')
            else:  # 数据行
                if samples_data[i-1]['label'].lower() == samples_data[i-1]['pred'].lower():
                    cell.set_facecolor('#e8f8f5')
                else:
                    cell.set_facecolor('#fadbd8')
            cell.set_edgecolor('#95a5a6')
            cell.set_linewidth(1.5)
    
    # 水印
    fig.text(0.99, 0.01, 'PEAN Error Analysis', ha='right', va='bottom',
            fontsize=8, color='#95a5a6', style='italic', alpha=0.7)
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='PEAN Single Model Evaluation')
    parser.add_argument('--pean_ckpt', type=str, default='./ckpt/PEAN_final.pth',
                       help='Path to PEAN checkpoint')
    parser.add_argument('--tpem_ckpt', type=str, default='./ckpt/TPEM_final.pth',
                       help='Path to TPEM checkpoint (optional)')
    parser.add_argument('--subset', type=str, default='easy', choices=['easy', 'medium', 'hard'],
                       help='Test subset to evaluate on')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for evaluation (recommend 1 for visualization)')
    parser.add_argument('--max_samples', type=int, default=50,
                       help='Maximum number of samples to evaluate')
    parser.add_argument('--random_sample', type=int, default=0,
                       help='Number of random samples to evaluate (0=use max_samples, >0=random sample count)')
    parser.add_argument('--samples_per_grid', type=int, default=5,
                       help='Number of samples to combine in one grid image')
    parser.add_argument('--num_grids', type=int, default=1,
                       help='Number of combined grid images to generate')
    parser.add_argument('--out_dir', type=str, default='./eval_results',
                       help='Output directory for results')
    parser.add_argument('--srb', type=int, default=1,
                       help='Number of SRB blocks (should match checkpoint)')
    args = parser.parse_args()

    print('=' * 100)
    print('PEAN Single Model Evaluation')
    print('=' * 100)
    print(f'PEAN Checkpoint: {args.pean_ckpt}')
    if args.tpem_ckpt and os.path.exists(args.tpem_ckpt):
        print(f'TPEM Checkpoint: {args.tpem_ckpt}')
    print(f'Test Subset: {args.subset}')
    print(f'Max Samples: {args.max_samples}')
    print('=' * 100)

    # 加载配置
    config_path = os.path.join('config', 'super_resolution.yaml')
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    config = EasyDict(config)

    # 构建参数
    args_pean = _build_args(batch_size=args.batch_size, srb=args.srb)
    args_pean.resume = args.pean_ckpt

    # 实例化 PEAN
    print('\nInitializing PEAN...')
    pean = TextSR(config, args_pean)
    setattr(pean, '_comparison_name', 'PEAN')

    # 选择测试子集
    subset_map = {}
    for p in config.TRAIN.VAL.val_data_dir:
        key = p.replace('\\', '/').split('/')[-1]
        subset_map[key] = p
    
    if args.subset not in subset_map:
        raise RuntimeError(f"Subset '{args.subset}' not found. Available: {list(subset_map.keys())}")
    
    subset_path = subset_map[args.subset]
    print(f'Loading test data from: {subset_path}')

    # 构建测试数据加载器
    _, test_loader = pean.get_test_data(subset_path)
    print(f'Test loader created: {len(test_loader)} batches')

    # 加载 PEAN 模型
    print('\nLoading PEAN model...')
    pean_model = pean.generator_init(resume_this=args.pean_ckpt)['model']
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

    # 准备输出目录
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_root = _ensure_dir(os.path.join(args.out_dir, f'pean_eval_{args.subset}_{ts}'))
    img_dir = _ensure_dir(os.path.join(out_root, 'images'))
    heat_dir = _ensure_dir(os.path.join(out_root, 'heatmaps'))
    combined_dir = _ensure_dir(os.path.join(out_root, 'combined'))
    csv_path = os.path.join(out_root, 'metrics.csv')

    print(f'Output directory: {out_root}')
    print(f'Samples per combined grid: {args.samples_per_grid}')
    print(f'Number of combined grids: {args.num_grids}')
    
    # 处理随机采样
    if args.random_sample > 0:
        print(f'Random sample mode: {args.random_sample} samples')
        sample_count = args.random_sample
    else:
        print(f'Standard mode: up to {args.max_samples} samples')
        sample_count = args.max_samples
    
    # 计算总样本数
    total_samples_needed = args.samples_per_grid * args.num_grids
    sample_count = max(sample_count, total_samples_needed)

    # 创建 CSV 文件
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['index', 'label', 'pred_pean', 'psnr', 'ssim', 'l1', 'l2', 'correct'])

    # 评估循环
    print('\n' + '=' * 100)
    print('Starting evaluation...')
    print('=' * 100)
    
    collected = 0
    total_psnr = 0
    total_ssim = 0
    total_correct = 0
    
    # 用于存储组合图的样本数据
    current_grid_samples = []
    current_heatmap_samples = []
    grid_count = 0
    
    # 获取字符映射 (alphabet to digit)
    alphabet = ':'.join(string.digits + string.ascii_lowercase + '$')
    a2d = {ch: i for i, ch in enumerate(alphabet.split(':'))}
    
    for bidx, data in enumerate(test_loader):
        if collected >= sample_count:
            break
        
        images_hr, images_lr, label_strs, label_vecs = data
        images_lr = images_lr.to(pean.device)
        images_hr = images_hr.to(pean.device)

        for i in range(images_lr.shape[0]):
            if collected >= sample_count:
                break
            
            img_lr = images_lr[i:i+1]
            img_hr = images_hr[i:i+1]
            label = label_strs[i]

            # 构建 weighted_mask (字符ID序列)
            label_lower = label.lower()
            label_list = [a2d[ch] for ch in label_lower if ch in a2d]
            if len(label_list) == 0:
                weighted_mask = torch.tensor([0]).long()  # 空白标签
            else:
                weighted_mask = torch.tensor(label_list).long()
            
            # 计算 text_len
            text_len = torch.tensor([len(weighted_mask)]).long()

            # 构建 PARSeq 先验 (LR)
            pq_in_lr = pean.parse_parseq_data(img_lr[0, :3, :, :])
            prob_str_lr = parseq(pq_in_lr, max_length=25).softmax(-1)
            
            # 构建 PARSeq 先验 (HR)
            pq_in_hr = pean.parse_parseq_data(img_hr[0, :3, :, :])
            prob_str_hr = parseq(pq_in_hr, max_length=25).softmax(-1)
            
            # TPEM 扩散先验
            pean.diffusion = pean.init_diffusion_model()
            if args.tpem_ckpt and os.path.exists(args.tpem_ckpt):
                pean.diffusion.load_network()  # 会从配置加载 TPEM
            
            # TPEM 需要 HR, SR, weighted_mask, predicted_length, text_len
            predicted_length = torch.ones(prob_str_lr.shape[0]) * prob_str_lr.shape[1]
            
            data_diff = {
                "HR": prob_str_hr, 
                "SR": prob_str_lr,
                "weighted_mask": weighted_mask,
                "predicted_length": predicted_length,
                "text_len": text_len
            }
            pean.diffusion.feed_data(data_diff)
            _, label_vec_final = pean.diffusion.process()
            label_vec_final = label_vec_final.to(pean.device)

            # PEAN 超分辨率
            with torch.no_grad():
                img_sr, _ = pean_model(img_lr, label_vec_final)

            # 转换为 3 通道
            lr_vis = _to_3ch_float(img_lr)
            sr_vis = _to_3ch_float(img_sr)
            hr_vis = _to_3ch_float(img_hr)

            # 识别预测文本
            pred = _predict_aster(aster, aster_info, img_sr)
            correct = 1 if pred.lower() == label.lower() else 0

            # 计算指标
            metrics = _metrics(sr_vis, hr_vis)
            
            total_psnr += metrics['psnr']
            total_ssim += metrics['ssim']
            total_correct += correct

            # 保存对比图
            grid_path = os.path.join(img_dir, f'{collected:04d}_comparison.png')
            _save_comparison_grid(lr_vis, sr_vis, hr_vis, label, pred, grid_path, metrics)

            # 保存误差热图
            heat_path = os.path.join(heat_dir, f'{collected:04d}_error.png')
            _save_heatmap(sr_vis, hr_vis, heat_path, title=f'Error: {label} → {pred}', metrics=metrics)
            
            # 收集组合图数据
            current_grid_samples.append({
                'lr': lr_vis.clone(),
                'sr': sr_vis.clone(),
                'hr': hr_vis.clone(),
                'label': label,
                'pred': pred,
                'metrics': metrics.copy()
            })
            
            # 收集热图数据
            current_heatmap_samples.append({
                'sr': sr_vis.clone(),
                'hr': hr_vis.clone(),
                'label': label,
                'pred': pred,
                'metrics': metrics.copy()
            })
            
            # 当收集够一组样本时，生成组合图
            if len(current_grid_samples) == args.samples_per_grid:
                grid_count += 1
                
                # 生成对比组合图
                combined_path = os.path.join(combined_dir, f'grid_{grid_count:02d}.png')
                _save_combined_grid(current_grid_samples, combined_path)
                print(f'  → Saved combined grid {grid_count} to {combined_path}')
                
                # 生成热图组合图
                heatmap_combined_path = os.path.join(combined_dir, f'heatmap_{grid_count:02d}.png')
                _save_combined_heatmap(current_heatmap_samples, heatmap_combined_path)
                print(f'  → Saved combined heatmap {grid_count} to {heatmap_combined_path}')
                
                current_grid_samples = []
                current_heatmap_samples = []
                
                # 如果已生成足够的组合图，停止
                if grid_count >= args.num_grids:
                    break

            # 记录到 CSV
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow([collected, label, pred, 
                           f'{metrics["psnr"]:.2f}', f'{metrics["ssim"]:.4f}',
                           f'{metrics["l1"]:.6f}', f'{metrics["l2"]:.6f}', correct])

            collected += 1
            
            if (collected) % 5 == 0:
                print(f'Processed {collected}/{args.max_samples} samples | '
                      f'Avg PSNR: {total_psnr/collected:.2f} | '
                      f'Avg SSIM: {total_ssim/collected:.4f} | '
                      f'Accuracy: {total_correct/collected*100:.1f}%')
        
        # 如果已生成足够的组合图，退出外层循环
        if grid_count >= args.num_grids:
            break
    
    # 处理剩余样本（如果有）
    if len(current_grid_samples) > 0 and grid_count < args.num_grids:
        grid_count += 1
        combined_path = os.path.join(combined_dir, f'grid_{grid_count:02d}.png')
        _save_combined_grid(current_grid_samples, combined_path)
        print(f'  → Saved combined grid {grid_count} (partial) to {combined_path}')
        
        heatmap_combined_path = os.path.join(combined_dir, f'heatmap_{grid_count:02d}.png')
        _save_combined_heatmap(current_heatmap_samples, heatmap_combined_path)
        print(f'  → Saved combined heatmap {grid_count} (partial) to {heatmap_combined_path}')

    # 最终统计
    print('\n' + '=' * 100)
    print('Evaluation Complete!')
    print('=' * 100)
    print(f'Total Samples: {collected}')
    print(f'Average PSNR: {total_psnr/collected:.2f} dB')
    print(f'Average SSIM: {total_ssim/collected:.4f}')
    print(f'Accuracy: {total_correct}/{collected} ({total_correct/collected*100:.1f}%)')
    print(f'Combined Grids Generated: {grid_count}')
    print(f'\nResults saved to: {out_root}')
    print(f'  - Individual Images: {img_dir}')
    print(f'  - Error Heatmaps: {heat_dir}')
    print(f'  - Combined Grids: {combined_dir}')
    print(f'  - Metrics CSV: {csv_path}')
    print('=' * 100)


if __name__ == '__main__':
    main()
