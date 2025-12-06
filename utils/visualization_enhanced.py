"""
Enhanced Visualization Tools for PEAN Baseline
===============================================
Academic-grade visualizations for deep learning research:
1. Training dynamics (loss, learning rate, gradients)
2. Feature map visualizations (encoder, decoder, attention)
3. Diffusion process visualization
4. Prediction quality analysis (correct/incorrect examples)
5. Metric correlation heatmaps
6. Attention mechanism visualization

Author: Enhanced PEAN
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import os
from matplotlib import cm
from datetime import datetime

# Set publication-quality defaults
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Define a nice color palette manually (similar to seaborn's husl)
COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#e67e22']


def plot_training_dynamics(metrics_history, save_path='training_dynamics.png', figsize=(20, 12)):
    """
    Plot comprehensive training dynamics including loss, metrics, and learning rate
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    iterations = metrics_history.get('iterations', [])
    if not iterations:
        print("No training data to plot")
        return
    
    # 1. Training Loss
    ax1 = fig.add_subplot(gs[0, :2])
    if 'loss' in metrics_history and metrics_history['loss']:
        ax1.plot(iterations, metrics_history['loss'], 'b-', linewidth=2, alpha=0.7, label='Training Loss')
        # Add moving average
        if len(metrics_history['loss']) > 10:
            window = min(50, len(metrics_history['loss']) // 10)
            ma = np.convolve(metrics_history['loss'], np.ones(window)/window, mode='valid')
            ma_iters = iterations[window-1:]
            ax1.plot(ma_iters, ma, 'r-', linewidth=2, label=f'MA({window})')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Learning Rate Schedule
    ax2 = fig.add_subplot(gs[0, 2])
    if 'learning_rate' in metrics_history and metrics_history['learning_rate']:
        ax2.plot(iterations, metrics_history['learning_rate'], 'g-', linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('LR Schedule')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
    
    # 3. PSNR Curves
    ax3 = fig.add_subplot(gs[1, 0])
    for dataset in ['easy', 'medium', 'hard']:
        key = f'psnr_{dataset}'
        if key in metrics_history and metrics_history[key]:
            val_iters = iterations[:len(metrics_history[key])]
            ax3.plot(val_iters, metrics_history[key], marker='o', linewidth=2, 
                    label=dataset.capitalize(), markersize=4)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('PSNR (dB)')
    ax3.set_title('PSNR on Validation Sets')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. SSIM Curves
    ax4 = fig.add_subplot(gs[1, 1])
    for dataset in ['easy', 'medium', 'hard']:
        key = f'ssim_{dataset}'
        if key in metrics_history and metrics_history[key]:
            val_iters = iterations[:len(metrics_history[key])]
            ax4.plot(val_iters, metrics_history[key], marker='s', linewidth=2,
                    label=dataset.capitalize(), markersize=4)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('SSIM')
    ax4.set_title('SSIM on Validation Sets')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. ASTER Accuracy
    ax5 = fig.add_subplot(gs[1, 2])
    for dataset in ['easy', 'medium', 'hard']:
        key = f'acc_aster_{dataset}'
        if key in metrics_history and metrics_history[key]:
            val_iters = iterations[:len(metrics_history[key])]
            acc_percent = [a * 100 for a in metrics_history[key]]
            ax5.plot(val_iters, acc_percent, marker='^', linewidth=2,
                    label=dataset.capitalize(), markersize=4)
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('Accuracy (%)')
    ax5.set_title('ASTER Recognition Accuracy')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. CER (Character Error Rate)
    ax6 = fig.add_subplot(gs[2, 0])
    for dataset in ['easy', 'medium', 'hard']:
        key = f'cer_{dataset}'
        if key in metrics_history and metrics_history[key]:
            val_iters = iterations[:len(metrics_history[key])]
            cer_percent = [c * 100 for c in metrics_history[key]]
            ax6.plot(val_iters, cer_percent, marker='d', linewidth=2,
                    label=dataset.capitalize(), markersize=4)
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('CER (%)')
    ax6.set_title('Character Error Rate')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Combined Metrics (PSNR vs Accuracy scatter)
    ax7 = fig.add_subplot(gs[2, 1])
    colors = {'easy': 'green', 'medium': 'orange', 'hard': 'red'}
    for dataset in ['easy', 'medium', 'hard']:
        psnr_key = f'psnr_{dataset}'
        acc_key = f'acc_aster_{dataset}'
        if psnr_key in metrics_history and acc_key in metrics_history:
            if metrics_history[psnr_key] and metrics_history[acc_key]:
                psnr = metrics_history[psnr_key]
                acc = [a * 100 for a in metrics_history[acc_key]]
                ax7.scatter(psnr, acc, c=colors[dataset], label=dataset.capitalize(),
                           s=50, alpha=0.6)
    ax7.set_xlabel('PSNR (dB)')
    ax7.set_ylabel('Accuracy (%)')
    ax7.set_title('PSNR vs Accuracy Correlation')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Training Summary Stats
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    summary_text = "Training Summary\n" + "="*30 + "\n"
    if iterations:
        summary_text += f"Total Iterations: {iterations[-1]}\n"
    if 'loss' in metrics_history and metrics_history['loss']:
        summary_text += f"Final Loss: {metrics_history['loss'][-1]:.4f}\n"
        summary_text += f"Min Loss: {min(metrics_history['loss']):.4f}\n"
    if 'acc_aster_easy' in metrics_history and metrics_history['acc_aster_easy']:
        summary_text += f"\nBest Accuracy:\n"
        for dataset in ['easy', 'medium', 'hard']:
            key = f'acc_aster_{dataset}'
            if key in metrics_history and metrics_history[key]:
                best_acc = max(metrics_history[key]) * 100
                summary_text += f"  {dataset.capitalize()}: {best_acc:.2f}%\n"
    
    ax8.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.3))
    
    plt.suptitle('PEAN Training Dynamics Dashboard', fontsize=18, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training dynamics saved to {save_path}")


def plot_feature_maps(feature_dict, save_path='feature_maps.png', max_channels=16):
    """
    Visualize feature maps from different layers
    feature_dict: {'layer_name': tensor of shape (B, C, H, W)}
    """
    num_layers = len(feature_dict)
    if num_layers == 0:
        return
    
    fig, axes = plt.subplots(num_layers, max_channels, 
                            figsize=(max_channels*2, num_layers*2))
    if num_layers == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (layer_name, features) in enumerate(feature_dict.items()):
        # features: (B, C, H, W) -> take first batch item
        if isinstance(features, torch.Tensor):
            features = features[0].detach().cpu().numpy()
        
        num_channels = min(features.shape[0], max_channels)
        
        for ch in range(num_channels):
            ax = axes[idx, ch] if num_layers > 1 else axes[ch]
            feature_map = features[ch]
            
            # Normalize for visualization
            fmin, fmax = feature_map.min(), feature_map.max()
            if fmax - fmin > 1e-6:
                feature_map = (feature_map - fmin) / (fmax - fmin)
            
            im = ax.imshow(feature_map, cmap='viridis', aspect='auto')
            ax.axis('off')
            if ch == 0:
                ax.set_title(f'{layer_name}\nCh {ch}', fontsize=8)
            else:
                ax.set_title(f'Ch {ch}', fontsize=8)
        
        # Hide unused subplots
        for ch in range(num_channels, max_channels):
            ax = axes[idx, ch] if num_layers > 1 else axes[ch]
            ax.axis('off')
    
    plt.suptitle('Feature Map Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Feature maps saved to {save_path}")


def plot_attention_maps(attention_weights, images_lr, images_sr, labels, preds,
                       save_path='attention_maps.png', num_samples=4):
    """
    Visualize attention weights from recognition module
    attention_weights: list of attention tensors (seq_len, batch, spatial)
    """
    num_samples = min(num_samples, len(labels))
    fig = plt.figure(figsize=(20, num_samples * 4))
    gs = gridspec.GridSpec(num_samples, 5, figure=fig, hspace=0.3, wspace=0.3)
    
    for i in range(num_samples):
        # 1. LR Image
        ax_lr = fig.add_subplot(gs[i, 0])
        lr_img = images_lr[i].permute(1, 2, 0).cpu().numpy()
        lr_img = np.clip(lr_img, 0, 1)
        ax_lr.imshow(lr_img)
        ax_lr.set_title(f'LR Image\n"{labels[i]}"', fontsize=10)
        ax_lr.axis('off')
        
        # 2. SR Image
        ax_sr = fig.add_subplot(gs[i, 1])
        sr_img = images_sr[i].permute(1, 2, 0).cpu().numpy()
        sr_img = np.clip(sr_img, 0, 1)
        ax_sr.imshow(sr_img)
        ax_sr.set_title(f'SR Image\n"{preds[i]}"', fontsize=10)
        ax_sr.axis('off')
        
        # 3-5. Attention at different timesteps
        if attention_weights and len(attention_weights) > i:
            attn = attention_weights[i]  # (seq_len, spatial)
            if isinstance(attn, torch.Tensor):
                attn = attn.detach().cpu().numpy()
            
            seq_len = min(3, attn.shape[0])
            for t in range(seq_len):
                ax_att = fig.add_subplot(gs[i, 2 + t])
                attn_map = attn[t].reshape(-1)  # Flatten spatial dims
                
                # Reshape for visualization (assume 1D attention over width)
                h, w = 8, len(attn_map) // 8
                if len(attn_map) >= h * w:
                    attn_map = attn_map[:h*w].reshape(h, w)
                else:
                    attn_map = np.pad(attn_map, (0, h*w - len(attn_map))).reshape(h, w)
                
                im = ax_att.imshow(attn_map, cmap='hot', aspect='auto')
                ax_att.set_title(f'Attention t={t}', fontsize=10)
                ax_att.axis('off')
                plt.colorbar(im, ax=ax_att, fraction=0.046)
    
    plt.suptitle('Attention Mechanism Visualization', fontsize=16, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Attention maps saved to {save_path}")


def plot_super_resolution_comparison(images_lr, images_sr, images_hr, 
                                    labels, preds, psnr_values, ssim_values,
                                    save_path='sr_comparison.png', num_samples=8):
    """
    Side-by-side comparison of LR, SR, HR with metrics
    """
    num_samples = min(num_samples, len(labels))
    fig = plt.figure(figsize=(15, num_samples * 2.5))
    gs = gridspec.GridSpec(num_samples, 3, figure=fig, hspace=0.4, wspace=0.2)
    
    for i in range(num_samples):
        # LR
        ax_lr = fig.add_subplot(gs[i, 0])
        lr_img = images_lr[i][:3].permute(1, 2, 0).cpu().numpy()
        lr_img = np.clip(lr_img, 0, 1)
        ax_lr.imshow(lr_img)
        ax_lr.set_title(f'LR\n"{labels[i]}"', fontsize=9)
        ax_lr.axis('off')
        
        # SR
        ax_sr = fig.add_subplot(gs[i, 1])
        sr_img = images_sr[i][:3].permute(1, 2, 0).cpu().numpy()
        sr_img = np.clip(sr_img, 0, 1)
        ax_sr.imshow(sr_img)
        match = '✓' if preds[i] == labels[i] else '✗'
        ax_sr.set_title(f'SR {match}\n"{preds[i]}"', fontsize=9,
                       color='green' if match == '✓' else 'red')
        ax_sr.axis('off')
        
        # HR
        ax_hr = fig.add_subplot(gs[i, 2])
        hr_img = images_hr[i][:3].permute(1, 2, 0).cpu().numpy()
        hr_img = np.clip(hr_img, 0, 1)
        ax_hr.imshow(hr_img)
        ax_hr.set_title(f'HR (GT)\nPSNR: {psnr_values[i]:.2f}dB\nSSIM: {ssim_values[i]:.3f}', 
                       fontsize=9)
        ax_hr.axis('off')
    
    plt.suptitle('Super-Resolution Quality Comparison', fontsize=16, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"SR comparison saved to {save_path}")


def plot_loss_landscape(loss_history, save_path='loss_landscape.png'):
    """
    3D visualization of loss landscape (smoothed)
    """
    if len(loss_history) < 100:
        return
    
    # Reshape loss into 2D grid for landscape
    n = int(np.sqrt(len(loss_history)))
    losses = np.array(loss_history[:n*n]).reshape(n, n)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y)
    
    surf = ax.plot_surface(X, Y, losses, cmap='viridis', alpha=0.8, 
                          edgecolor='none', antialiased=True)
    
    ax.set_xlabel('Iteration Dim 1')
    ax.set_ylabel('Iteration Dim 2')
    ax.set_zlabel('Loss')
    ax.set_title('Loss Landscape Visualization', fontsize=14, fontweight='bold')
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss landscape saved to {save_path}")


def plot_gradient_flow(named_parameters, save_path='gradient_flow.png'):
    """
    Visualize gradient flow through layers
    """
    ave_grads = []
    max_grads = []
    layers = []
    
    for n, p in named_parameters:
        if p.requires_grad and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu().item())
            max_grads.append(p.grad.abs().max().cpu().item())
    
    if not layers:
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(layers))
    ax.bar(x, max_grads, alpha=0.5, lw=1, color='c', label='Max Gradient')
    ax.bar(x, ave_grads, alpha=0.7, lw=1, color='b', label='Mean Gradient')
    
    ax.set_xlabel('Layers')
    ax.set_ylabel('Gradient Magnitude')
    ax.set_title('Gradient Flow Through Network', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Gradient flow saved to {save_path}")


def plot_confusion_matrix(true_chars, pred_chars, save_path='confusion_matrix.png'):
    """
    Character-level confusion matrix
    """
    from collections import defaultdict
    
    # Build confusion matrix
    char_set = sorted(set(true_chars + pred_chars))
    if len(char_set) > 50:  # Limit for visualization
        char_set = char_set[:50]
    
    char_to_idx = {c: i for i, c in enumerate(char_set)}
    matrix = np.zeros((len(char_set), len(char_set)))
    
    for t, p in zip(true_chars, pred_chars):
        if t in char_to_idx and p in char_to_idx:
            matrix[char_to_idx[t], char_to_idx[p]] += 1
    
    # Normalize
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix = np.divide(matrix, row_sums, where=row_sums > 0)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(matrix, cmap='Blues', aspect='auto')
    
    ax.set_xticks(np.arange(len(char_set)))
    ax.set_yticks(np.arange(len(char_set)))
    ax.set_xticklabels(char_set, fontsize=8)
    ax.set_yticklabels(char_set, fontsize=8)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    ax.set_xlabel('Predicted Character')
    ax.set_ylabel('True Character')
    ax.set_title('Character Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def create_visualization_dashboard(metrics_history, feature_maps=None, 
                                  attention_maps=None, save_dir='./vis_dashboard'):
    """
    Create a complete visualization dashboard
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    subdir = os.path.join(save_dir, f'dashboard_{timestamp}')
    os.makedirs(subdir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Creating Visualization Dashboard: {subdir}")
    print(f"{'='*60}")
    
    # 1. Training Dynamics
    plot_training_dynamics(metrics_history, 
                          os.path.join(subdir, '01_training_dynamics.png'))
    
    # 2. Feature Maps (if provided)
    if feature_maps:
        plot_feature_maps(feature_maps, 
                         os.path.join(subdir, '02_feature_maps.png'))
    
    # 3. Loss Landscape
    if 'loss' in metrics_history and len(metrics_history['loss']) > 100:
        plot_loss_landscape(metrics_history['loss'],
                          os.path.join(subdir, '03_loss_landscape.png'))
    
    print(f"{'='*60}")
    print(f"Dashboard created successfully!")
    print(f"{'='*60}\n")
    
    return subdir
