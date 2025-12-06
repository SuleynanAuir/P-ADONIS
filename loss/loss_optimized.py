"""
Optimized Loss Functions for PEAN
==================================
Enhanced loss functions with:
1. Perceptual Loss using VGG features
2. Focal Loss for hard example mining
3. Improved combination strategies

Author: Optimized PEAN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import sys


class PerceptualLoss(nn.Module):
    """
    Perceptual Loss using pretrained VGG16 features
    Helps the model learn better image quality by matching feature representations
    """
    def __init__(self, feature_layers=[3, 8, 15, 22], use_cuda=True):
        super(PerceptualLoss, self).__init__()
        
        # Load pretrained VGG16
        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        except:
            vgg = models.vgg16(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:max(feature_layers)+1])
        
        # Freeze VGG parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.feature_layers = feature_layers
        self.criterion = nn.L1Loss()
        
        if use_cuda and torch.cuda.is_available():
            self.feature_extractor = self.feature_extractor.cuda()
        
        # Normalization for VGG (ImageNet mean and std)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, x):
        """Normalize images for VGG"""
        # Assume input is in range [-1, 1], convert to [0, 1]
        x = (x + 1) / 2.0
        return (x - self.mean) / self.std
    
    def forward(self, pred, target):
        """
        Args:
            pred: predicted images (B, C, H, W) in range [-1, 1]
            target: target images (B, C, H, W) in range [-1, 1]
        """
        # Normalize inputs
        pred = self.normalize(pred)
        target = self.normalize(target)
        
        # Resize to minimum size for VGG
        if pred.size(2) < 32 or pred.size(3) < 32:
            pred = F.interpolate(pred, size=(max(32, pred.size(2)), max(128, pred.size(3))), 
                               mode='bilinear', align_corners=False)
            target = F.interpolate(target, size=(max(32, target.size(2)), max(128, target.size(3))), 
                                 mode='bilinear', align_corners=False)
        
        loss = 0.0
        x_pred = pred
        x_target = target
        
        for i, layer in enumerate(self.feature_extractor):
            x_pred = layer(x_pred)
            x_target = layer(x_target)
            
            if i in self.feature_layers:
                loss += self.criterion(x_pred, x_target)
        
        return loss / len(self.feature_layers)


class FocalCTCLoss(nn.Module):
    """
    Focal Loss variant for CTC to focus on hard examples
    """
    def __init__(self, alpha=0.25, gamma=2.0, blank=0, reduction='mean'):
        super(FocalCTCLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.blank = blank
        self.reduction = reduction
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction='none')
    
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        """
        Args:
            log_probs: (T, N, C) log probabilities
            targets: (N, S) target sequences
            input_lengths: (N,) lengths of input sequences
            target_lengths: (N,) lengths of target sequences
        """
        # Calculate standard CTC loss
        ctc_loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        
        # Apply focal loss weighting
        p = torch.exp(-ctc_loss)
        focal_weight = self.alpha * (1 - p) ** self.gamma
        focal_loss = focal_weight * ctc_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class StrokeFocusLossOptimized(nn.Module):
    """
    Enhanced Stroke Focus Loss with Perceptual and Focal components
    """
    def __init__(self, args, use_perceptual=True, perceptual_weight=0.1,
                 use_focal=False, focal_alpha=0.25, focal_gamma=2.0):
        super(StrokeFocusLossOptimized, self).__init__()
        self.args = args
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # Perceptual loss
        self.use_perceptual = use_perceptual
        self.perceptual_weight = perceptual_weight
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss(use_cuda=True)
        
        # Focal loss
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
    
    def forward(self, pred, target, label_str=None):
        """
        Args:
            pred: predicted SR images (B, C, H, W)
            target: target HR images (B, C, H, W)
            label_str: text labels (optional, for future text-aware loss)
        
        Returns:
            total_loss: combined loss value
        """
        # Basic reconstruction loss (MSE + L1)
        mse_loss = self.mse_loss(pred, target)
        l1_loss = self.l1_loss(pred, target)
        recon_loss = mse_loss + 0.1 * l1_loss
        
        # Perceptual loss
        if self.use_perceptual:
            # Only use first 3 channels for perceptual loss
            pred_rgb = pred[:, :3, :, :]
            target_rgb = target[:, :3, :, :]
            perceptual = self.perceptual_loss(pred_rgb, target_rgb)
            total_loss = recon_loss + self.perceptual_weight * perceptual
        else:
            total_loss = recon_loss
        
        # Gradient-based edge loss (stroke focus)
        if self.args.gradient:
            pred_grad = self._gradient(pred)
            target_grad = self._gradient(target)
            gradient_loss = self.l1_loss(pred_grad, target_grad)
            total_loss = total_loss + 0.1 * gradient_loss
        
        return total_loss
    
    def _gradient(self, x):
        """Calculate image gradients"""
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        
        # Apply to each channel
        grad_x = F.conv2d(x, sobel_x.repeat(x.size(1), 1, 1, 1), 
                         padding=1, groups=x.size(1))
        grad_y = F.conv2d(x, sobel_y.repeat(x.size(1), 1, 1, 1), 
                         padding=1, groups=x.size(1))
        
        gradient = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        return gradient


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing for better generalization
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, C) predictions
            target: (B,) target labels
        """
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1).mean()
        return loss


# Utility function to create optimized loss
def create_optimized_loss(args, config):
    """
    Factory function to create optimized loss based on config
    
    Args:
        args: command line arguments
        config: configuration dict
    
    Returns:
        loss_fn: optimized loss function
    """
    loss_config = config.get('LOSS', {})
    
    use_perceptual = loss_config.get('use_perceptual_loss', True)
    perceptual_weight = loss_config.get('perceptual_weight', 0.1)
    use_focal = loss_config.get('use_focal_loss', False)
    focal_alpha = loss_config.get('focal_alpha', 0.25)
    focal_gamma = loss_config.get('focal_gamma', 2.0)
    
    return StrokeFocusLossOptimized(
        args,
        use_perceptual=use_perceptual,
        perceptual_weight=perceptual_weight,
        use_focal=use_focal,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma
    )
