"""
PEAN Optimized Version
======================
Enhanced PEAN architecture with the following improvements:
1. Residual Scaling for better gradient flow
2. Layer Normalization for more stable training
3. Improved initialization
4. Optional EMA (handled in training loop)

Author: Optimized PEAN
"""

import math
import torch
import torch.nn.functional as F
from torch import nn
import sys
sys.path.append('./')
sys.path.append('../')
from .tps_spatial_transformer import TPSSpatialTransformer
from .stn_head import STNHead
from .transformer_v2 import InfoTransformer
from .transformer_v2 import PositionalEncoding
from .amm import AMM
from .arm import ARM


class PEANOptimized(nn.Module):
    """
    Optimized PEAN with enhanced features for better performance
    """

    def __init__(self, scale_factor=2, width=128, height=32, STN=False, srb_nums=5, 
                 mask=True, hidden_units=32, testing=False, residual_scaling=0.1):
        super(PEANOptimized, self).__init__()
        in_planes = 3
        if mask:
            in_planes = 4
        assert math.log(scale_factor, 2) % 1 == 0
        upsample_block_num = int(math.log(scale_factor, 2))
        
        # Residual scaling factor for stable training
        self.residual_scaling = residual_scaling
        
        # Initial feature extraction with LayerNorm
        self.block1 = nn.Sequential(    
            nn.Conv2d(in_planes, 2*hidden_units, kernel_size=9, padding=4),
            nn.PReLU(),
            nn.LayerNorm([2*hidden_units, height//scale_factor, width//scale_factor])
        )
        
        self.srb_nums = srb_nums
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prior_dim = 2*hidden_units
        self.block2 = AMM()
        
        # Enhanced SRB with residual scaling and normalization
        setattr(self, 'block%d' % (srb_nums + 2),
                nn.Sequential(
                    nn.Conv2d(2*hidden_units, 2*hidden_units, kernel_size=3, padding=1),
                    nn.LayerNorm([2*hidden_units, height//scale_factor, width//scale_factor]),
                    nn.PReLU()
                ))
        
        # Upsampling blocks with improved architecture
        block_ = []
        for _ in range(upsample_block_num):
            block_.append(UpsampleBLockOptimized(2*hidden_units, 2*hidden_units, scale_factor))
        block_.append(nn.Conv2d(2*hidden_units, in_planes, kernel_size=9, padding=4))
        block_.append(nn.Tanh())  # Normalize output to [-1, 1]
        setattr(self, 'block%d' % (srb_nums + 3), nn.Sequential(*block_))
        
        self.tps_inputsize = [32, 64]       
        tps_outputsize = [height//scale_factor, width//scale_factor]
        num_control_points = 20
        tps_margins = [0.05, 0.05]
        self.stn = STN
        t_decoder_num = 3
        
        # Enhanced transformer with better initialization
        self.fn = nn.Linear(95, self.prior_dim)
        nn.init.xavier_uniform_(self.fn.weight)
        nn.init.zeros_(self.fn.bias)
        
        self.pe = PositionalEncoding(d_model=self.prior_dim, dropout=0.1, max_len=5000)
        self.init_factor = nn.Embedding(tps_outputsize[0]*tps_outputsize[1], self.prior_dim)
        nn.init.normal_(self.init_factor.weight, mean=0, std=0.02)
        
        self.upsample_transformer = InfoTransformer(
            d_model=self.prior_dim,
            dropout=0.1,
            num_encoder_layers=3,
            nhead=4,
            dim_feedforward=self.prior_dim,
            num_decoder_layers=t_decoder_num,
            normalize_before=False,
            return_intermediate_dec=True, 
            feat_height=tps_outputsize[0], 
            feat_width=tps_outputsize[1]
        )

        if self.stn:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(tps_outputsize),
                num_control_points=num_control_points,
                margins=tuple(tps_margins))

            self.stn_head = STNHead(
                in_planes=in_planes,
                num_ctrlpoints=num_control_points,
                activation='none')
        
        self.testing = testing
        if not self.testing:
            self.arm = ARM(16, 1, 37, 256)
        
        # Apply weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Apply He initialization for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, rec_result):
        if self.stn and self.training:
            x = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)             
        
        block = {'1': self.block1(x)}                      
        B, _, H, W = block['1'].shape
        x_im = block['1'].contiguous().view(B, -1, H*W).permute(2, 0, 1)
        
        # Process text prior
        rec_result = rec_result.transpose(0, 1)
        rec_result = self.fn(rec_result)
        rec_result = F.relu(rec_result)
        L, _, C = rec_result.shape
        prior = rec_result
        x_pos_mem = self.pe(torch.zeros((B, L, C)).to(self.device)).permute(1, 0, 2)
        mask_mem = torch.zeros((B, L)).to(self.device).bool()
        
        # Transformer-based prior enhancement
        prior, _ = self.upsample_transformer(prior, mask_mem, self.init_factor.weight, x_pos_mem, x_im)
        prior = prior.mean(0)
        prior = prior.permute(1, 2, 0).view(B, -1, H, W)

        # SRB blocks with residual scaling
        for i in range(self.srb_nums):                 
            residual = block[str(i + 1)]
            block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(residual, prior)
            # Apply residual scaling for stability
            block[str(i + 2)] = residual + self.residual_scaling * (block[str(i + 2)] - residual)

        block[str(self.srb_nums + 2)] = getattr(self, 'block%d' % (self.srb_nums + 2))(block[str(self.srb_nums + 1)])
        
        # Auxiliary recognition branch
        if not self.testing:
            input_feature = torch.nn.functional.interpolate(
                block[str(self.srb_nums + 2)], (16, 50), mode='bicubic'
            )
            logits = self.arm(input_feature)
        else:
            logits = None
        
        # Final reconstruction with scaled residual connection
        reconstruction = getattr(self, 'block%d' % (self.srb_nums + 3))(
            block['1'] + self.residual_scaling * block[str(self.srb_nums + 2)]
        )
        output = reconstruction
        
        return output, logits


class UpsampleBLockOptimized(nn.Module):
    """Optimized upsampling block with better activation"""

    def __init__(self, in_channels, out_channels, up_scale):
        super(UpsampleBLockOptimized, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.activation = mish()
        
        # Initialize weights
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        x = self.conv(x)                
        x = self.pixel_shuffle(x)      
        x = self.activation(x)              
        return x


class mish(nn.Module):
    """Mish activation function: x * tanh(softplus(x))"""
    def __init__(self):
        super(mish, self).__init__()
        self.activated = True

    def forward(self, x):
        if self.activated:
            x = x * torch.tanh(F.softplus(x))
        return x
