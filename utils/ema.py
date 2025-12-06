"""
Exponential Moving Average (EMA) for model weights
===================================================
Maintains a moving average of model parameters for more stable inference

Author: Optimized PEAN
"""

import torch
import torch.nn as nn
from copy import deepcopy


class ModelEMA:
    """
    Exponential Moving Average of model weights
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    """
    
    def __init__(self, model, decay=0.9999, device=None):
        """
        Args:
            model: PyTorch model
            decay: decay rate for EMA
            device: device to store EMA model
        """
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.device = device
        
        # Disable gradient for EMA model
        for param in self.ema.parameters():
            param.requires_grad = False
        
        if device is not None:
            self.ema.to(device)
    
    def update(self, model):
        """Update EMA parameters"""
        with torch.no_grad():
            # Update parameters
            for ema_param, model_param in zip(self.ema.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)
            
            # Update buffers (like running_mean in BatchNorm)
            for ema_buffer, model_buffer in zip(self.ema.buffers(), model.buffers()):
                ema_buffer.copy_(model_buffer)
    
    def state_dict(self):
        """Return EMA model state dict"""
        return self.ema.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load EMA model state dict"""
        self.ema.load_state_dict(state_dict)
    
    def __call__(self, *args, **kwargs):
        """Forward pass using EMA model"""
        return self.ema(*args, **kwargs)
