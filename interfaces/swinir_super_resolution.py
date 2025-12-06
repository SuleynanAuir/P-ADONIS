"""SwinIR-based Text Super-Resolution interface.

This module mirrors the existing training pipeline while instantiating the
SwinIRTextSR generator. It lives in a separate file so the original
implementation stays untouched.
"""
from typing import Any, Dict, Tuple

import torch

from interfaces.super_resolution import TextSR
from loss import stroke_focus_loss
from model.swinir_textsr import SwinIRTextSR


class SwinIRTextSRInterface(TextSR):
    """Drop-in replacement interface that swaps the generator for SwinIR."""

    def __init__(self, config, args):
        # Override directories before calling parent __init__
        # This ensures SwinIR uses separate output folders
        if not hasattr(config.TRAIN, 'ckpt_dir_override'):
            config.TRAIN.ckpt_dir = './ckpt_swinir'
        if not hasattr(config.TRAIN.VAL, 'vis_dir_override'):
            config.TRAIN.VAL.vis_dir = './vis_swinir'
        
        super().__init__(config, args)
        self._swin_resume = getattr(args, "swin_resume", None)
        # Set comparison name for standardized checkpoint naming
        self._comparison_name = 'SWINIR'
        
        print(f"SwinIR output directories:")
        print(f"  - Checkpoints: {self.config.TRAIN.ckpt_dir}")
        print(f"  - Visualizations: {self.vis_dir}")

    def _parse_swinir_config(self) -> Dict[str, Any]:
        cfg_train = self.config.TRAIN
        swin_cfg = getattr(cfg_train, "SWINIR", None)

        def _get(attr: str, default: Any) -> Any:
            return getattr(swin_cfg, attr, default) if swin_cfg is not None else default

        embed_dim = _get("embed_dim", 60)  # Changed from 64 to 60 (divisible by 6)
        depths = tuple(_get("depths", (6, 6, 6, 6)))
        num_heads = tuple(_get("num_heads", (6, 6, 6, 6)))
        window_size = _get("window_size", 8)
        mlp_ratio = _get("mlp_ratio", 2.0)
        qkv_bias = _get("qkv_bias", True)
        drop_path_rate = _get("drop_path_rate", 0.1)
        upscale = _get("upscale", self.scale_factor)
        # Use 4 channels if mask is enabled (RGB + mask), otherwise 3
        in_chans = _get("in_chans", 4 if self.args.mask else 3)

        return {
            "in_chans": in_chans,
            "embed_dim": embed_dim,
            "depths": depths,
            "num_heads": num_heads,
            "window_size": window_size,
            "mlp_ratio": mlp_ratio,
            "qkv_bias": qkv_bias,
            "drop_path_rate": drop_path_rate,
            "upscale": upscale,
        }

    def generator_init(self, resume_this: str = "") -> Dict[str, Any]:
        cfg = self.config.TRAIN
        swin_kwargs = self._parse_swinir_config()

        generator = SwinIRTextSR(**swin_kwargs)
        image_crit = stroke_focus_loss.StrokeFocusLoss(self.args)

        generator = generator.to(self.device)
        image_crit = image_crit.to(self.device)

        if cfg.ngpu > 1:
            generator = torch.nn.DataParallel(generator, device_ids=range(cfg.ngpu))
            image_crit = torch.nn.DataParallel(image_crit, device_ids=range(cfg.ngpu))

        resume_path = self._swin_resume or resume_this or self.resume
        if resume_path:
            print(f"Loading SwinIRTextSR weights from {resume_path}")
            checkpoint = torch.load(resume_path, map_location=self.device)
            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    checkpoint = checkpoint["state_dict"]
                elif "model" in checkpoint:
                    checkpoint = checkpoint["model"]
            if isinstance(generator, torch.nn.DataParallel):
                generator.load_state_dict(checkpoint)
            else:
                if isinstance(checkpoint, dict) and any(k.startswith("module.") for k in checkpoint.keys()):
                    checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
                generator.load_state_dict(checkpoint)

        return {"model": generator, "crit": image_crit}

    # The remaining training pipeline (diffusion, optimizer, evaluation, etc.)
    # is inherited from TextSR without modification.
