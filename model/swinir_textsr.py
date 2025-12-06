"""
SwinIR backbone tailored for Text Image Super-Resolution (TextSR).
This implementation is self-contained and does not touch existing project bases.
"""
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * random_tensor


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition feature map (B, H, W, C) into windows."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Reconstruct feature map from windows."""
    B = int(windows.shape[0] // (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window based multi-head self-attention with relative position bias."""

    def __init__(self, dim: int, window_size: int, num_heads: int, qkv_bias: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

        relative_bias_shape = (2 * window_size - 1) * (2 * window_size - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(relative_bias_shape, num_heads))

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(
            self.window_size * self.window_size,
            self.window_size * self.window_size,
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            n_windows = mask.shape[0]
            attn = attn.view(B_ // n_windows, n_windows, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class ResidualSwinTransformerBlock(nn.Module):
    """Residual Swin Transformer block with optional shifted window."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        assert 0 <= shift_size < window_size, "shift_size must be in [0, window_size)"

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, qkv_bias=qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self._attn_mask_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def _generate_attn_mask(self, H: int, W: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if self.shift_size == 0:
            return None
        key = (H, W)
        if key not in self._attn_mask_cache:
            img_mask = torch.zeros((1, H, W, 1), device=device)
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
            self._attn_mask_cache[key] = attn_mask
        return self._attn_mask_cache[key].to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        shortcut = x

        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        attn_mask = self._generate_attn_mask(H, W, x.device, x.dtype)
        x_windows = window_partition(shifted_x, self.window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows.view(attn_windows.shape[0], -1, C), self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        shortcut = shortcut.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x


class Upsample(nn.Sequential):
    """Pixel-shuffle based upsampler."""

    def __init__(self, scale: int, num_feat: int) -> None:
        modules = []
        if scale in (2, 3):
            modules.extend([
                nn.Conv2d(num_feat, num_feat * scale * scale, 3, 1, 1),
                nn.PixelShuffle(scale),
            ])
        elif scale == 4:
            modules.extend([
                nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
                nn.PixelShuffle(2),
            ])
        elif scale == 1:
            modules.append(nn.Identity())
        else:
            raise ValueError(f"Unsupported upscale factor: {scale}")
        super().__init__(*modules)


class SwinIRTextSR(nn.Module):
    """Complete SwinIR backbone for Text SR."""

    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 64,
        depths: Tuple[int, ...] = (6, 6, 6, 6),
        num_heads: Tuple[int, ...] = (6, 6, 6, 6),
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.1,
        upscale: int = 2,
    ) -> None:
        super().__init__()
        assert len(depths) == len(num_heads), "depths and num_heads must have the same length"

        self.window_size = window_size
        self.upscale = upscale
        self.embed_dim = embed_dim

        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_last = nn.Conv2d(embed_dim, in_chans, 3, 1, 1)

        dpr = torch.linspace(0.0, drop_path_rate, sum(depths)).tolist()
        block_id = 0
        stages = []
        for stage_idx, num_blocks in enumerate(depths):
            blocks = []
            for i in range(num_blocks):
                shift = 0 if (i % 2 == 0) else window_size // 2
                blocks.append(
                    ResidualSwinTransformerBlock(
                        dim=embed_dim,
                        num_heads=num_heads[stage_idx],
                        window_size=window_size,
                        shift_size=shift,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop_path=dpr[block_id],
                    )
                )
                block_id += 1
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)
        self.norm = nn.LayerNorm(embed_dim)
        self.upsample = Upsample(upscale, embed_dim)

    def forward(self, x: torch.Tensor, rec_result: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass compatible with PEAN interface.
        
        Args:
            x: Input low-resolution images (B, C, H, W)
            rec_result: Text recognition prior (optional, not used in SwinIR)
        
        Returns:
            Tuple of (super-resolved images, None) to match PEAN's return signature
        """
        fea = self.conv_first(x)
        H_ori, W_ori = fea.shape[2:]

        pad_r = (self.window_size - W_ori % self.window_size) % self.window_size
        pad_b = (self.window_size - H_ori % self.window_size) % self.window_size
        if pad_r != 0 or pad_b != 0:
            fea = F.pad(fea, (0, pad_r, 0, pad_b), mode="reflect")
        residual = fea

        for stage in self.stages:
            fea = stage(fea)

        B, C, H, W = fea.shape
        fea = fea.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        fea = self.norm(fea)
        fea = fea.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        fea = self.conv_after_body(fea) + residual
        fea = self.upsample(fea)
        out = self.conv_last(fea)

        if pad_r != 0 or pad_b != 0:
            h = H_ori * self.upscale
            w = W_ori * self.upscale
            out = out[:, :, :h, :w]

        # Return (SR_image, None) to match PEAN's return signature (SR_image, logits)
        return out, None


if __name__ == "__main__":
    model = SwinIRTextSR(upscale=2)
    dummy = torch.randn(1, 3, 32, 128)
    with torch.no_grad():
        output, _ = model(dummy)
    print("Input shape:", dummy.shape)
    print("Output shape:", output.shape)
    params = sum(math.prod(p.shape) for p in model.parameters()) / 1e6
    print(f"Model parameters: {params:.2f}M")
