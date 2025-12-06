import os
import csv
import yaml
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
from interfaces.swinir_super_resolution import SwinIRTextSRInterface


def _build_args(batch_size=8, rec='aster', mask=True, srb=1, testing=True):
    # Minimal args object compatible with TextSR
    class Args:
        pass
    a = Args()
    # Set testing=True to disable ARM head in PEAN for broader ckpt compatibility
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
    # Align SRB count with typical released checkpoints (e.g., --srb=1)
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
    # special for swinir interface
    a.swin_resume = None
    return a


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def _to_3ch_float(img: torch.Tensor) -> torch.Tensor:
    """Ensure CHW 3-channel float tensor in [0,1] on CPU."""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu()
    # keep first 3 channels (RGB)
    img = img[:3]
    # some models may output beyond [0,1]; clamp for visualization
    img = img.float().clamp(0.0, 1.0)
    return img


def _resize_chw(img: torch.Tensor, size_hw: tuple) -> torch.Tensor:
    """Resize CHW tensor to (H,W) using bicubic."""
    h, w = size_hw
    if img.shape[-2] == h and img.shape[-1] == w:
        return img
    img_b = img.unsqueeze(0)
    img_r = F.interpolate(img_b, size=(h, w), mode='bicubic', align_corners=False)
    return img_r.squeeze(0)


def _save_grid(lr, sr_pean, sr_swin, hr, save_path):
    # Normalize and resize all to HR resolution for a consistent grid
    hr_t = _to_3ch_float(hr)
    H, W = hr_t.shape[-2], hr_t.shape[-1]
    lr_t = _resize_chw(_to_3ch_float(lr), (H, W))
    pean_t = _resize_chw(_to_3ch_float(sr_pean), (H, W))
    swin_t = _resize_chw(_to_3ch_float(sr_swin), (H, W))
    tensors = [lr_t, pean_t, swin_t, hr_t]
    grid = torchvision.utils.make_grid(tensors, nrow=4, padding=4)
    torchvision.utils.save_image(grid, save_path)


def _save_diff_maps(sr_pean, sr_swin, hr, save_dir, stem):
    A = _to_3ch_float(sr_pean)
    B = _to_3ch_float(sr_swin)
    H = _to_3ch_float(hr)
    # absolute differences
    err_pean = (A - H).abs().mean(0)  # HW
    err_swin = (B - H).abs().mean(0)
    diff_ab = (A - B).abs().mean(0)

    def _plot_heat(mat, title, path):
        plt.figure(figsize=(4,3), dpi=150)
        plt.imshow(mat.numpy(), cmap='magma')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    _plot_heat(err_pean, 'AbsErr: PEAN vs HR (mean|RGB)', os.path.join(save_dir, f'{stem}_err_pean_hr.png'))
    _plot_heat(err_swin, 'AbsErr: SwinIR vs HR (mean|RGB)', os.path.join(save_dir, f'{stem}_err_swinir_hr.png'))
    _plot_heat(diff_ab, 'AbsDiff: PEAN vs SwinIR (mean|RGB)', os.path.join(save_dir, f'{stem}_diff_pean_swinir.png'))


def _metrics(sr, hr):
    # Basic metrics: L1, L2; PSNR/SSIM approximations via skimage if available
    s = _to_3ch_float(sr)
    h = _to_3ch_float(hr)
    l1 = (s - h).abs().mean().item()
    l2 = torch.sqrt(((s - h) ** 2).mean()).item()
    # PSNR
    mse = ((s - h) ** 2).mean().item()
    if mse == 0:
        psnr = 99.0
    else:
        psnr = 10.0 * np.log10(1.0 / mse)
    # SSIM (simple channel-average using skimage if present)
    try:
        from skimage.metrics import structural_similarity as ssim
        s_np = s.permute(1,2,0).numpy()
        h_np = h.permute(1,2,0).numpy()
        ssim_val = ssim(h_np, s_np, channel_axis=2, data_range=1.0)
    except Exception:
        ssim_val = float('nan')
    return l1, l2, psnr, ssim_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pean_ckpt', type=str, required=True)
    parser.add_argument('--swinir_ckpt', type=str, required=True)
    parser.add_argument('--subset', type=str, default='easy', choices=['easy','medium','hard'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_samples', type=int, default=12)
    parser.add_argument('--out_dir', type=str, default='./ckpt_comparison')
    args = parser.parse_args()

    config_path = os.path.join('config', 'super_resolution.yaml')
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    config = EasyDict(config)

    # prepare args for both models
    args_pean = _build_args(batch_size=args.batch_size)
    args_pean.resume = args.pean_ckpt
    args_swin = _build_args(batch_size=args.batch_size)
    args_swin.resume = args.swinir_ckpt

    # Instantiate interfaces
    pean = TextSR(config, args_pean)
    setattr(pean, '_comparison_name', 'PEAN')
    swin = SwinIRTextSRInterface(config, args_swin)
    setattr(swin, '_comparison_name', 'SwinIR')

    # Pick subset path from config
    subset_map = {}
    for p in config.TRAIN.VAL.val_data_dir:
        key = p.replace('\\','/').split('/')[-1]
        subset_map[key] = p
    if args.subset not in subset_map:
        raise RuntimeError(f"Subset '{args.subset}' not found in config.TRAIN.VAL.val_data_dir. Available: {list(subset_map)}")
    subset_path = subset_map[args.subset]

    # Build deterministic test loader (shuffle=False in base)
    _, test_loader = pean.get_test_data(subset_path)

    # Load generators
    pean_model = pean.generator_init(resume_this=args.pean_ckpt)['model']
    swin_model = swin.generator_init(resume_this=args.swinir_ckpt)['model']
    pean_model.eval(); swin_model.eval()

    # Load PARSeq (shared)
    parseq = pean.PARSeq_init()
    for p in parseq.parameters():
        p.requires_grad = False
    parseq.eval()

    # Prepare output dir
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_root = _ensure_dir(os.path.join(args.out_dir, f'eval_{args.subset}_{ts}'))
    img_dir = _ensure_dir(os.path.join(out_root, 'images'))
    heat_dir = _ensure_dir(os.path.join(out_root, 'heatmaps'))
    csv_path = os.path.join(out_root, 'metrics.csv')

    # Metrics CSV header
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['index','label','pred_pean','pred_swinir',
                    'psnr_pean','ssim_pean','psnr_swinir','ssim_swinir',
                    'l1_pean','l2_pean','l1_swinir','l2_swinir',
                    'psnr_delta(pean-swinir)','ssim_delta(pean-swinir)'])

    # Evaluation loop (limited samples)
    collected = 0
    aster, aster_info = pean.Aster_init()
    aster.eval()
    for p in aster.parameters():
        p.requires_grad = False

    # Collect tiles for a single big gallery image
    gallery_tiles = []  # flattened as [LR, PEAN, SwinIR, HR, LR, PEAN, ...]
    gallery_rows_meta = []  # [(label, pred_pean, pred_swinir), ...]

    for bidx, data in enumerate(test_loader):
        if collected >= args.max_samples:
            break
        images_hr, images_lr, label_strs, label_vecs = data
        images_lr = images_lr.to(pean.device)
        images_hr = images_hr.to(pean.device)

        # Build prior for LR via PARSeq
        prob_str_lr = []
        for i in range(images_lr.shape[0]):
            pq_in = pean.parse_parseq_data(images_lr[i, :3, :, :])
            pq_out = parseq(pq_in)
            prob = pq_out.softmax(-1)
            prob_str_lr.append(prob)
        prob_str_lr = torch.cat(prob_str_lr, dim=0)
        predicted_length = torch.ones(prob_str_lr.shape[0]) * prob_str_lr.shape[1]

        # Diffusion prior
        data_diff = {"SR": prob_str_lr}
        pean.diffusion = pean.init_diffusion_model()
        swin.diffusion = swin.init_diffusion_model()
        # Per-sample diffusion (as in eval)
        label_vecs_final = None
        for j in range(images_lr.shape[0]):
            data_diff_j = {"SR": prob_str_lr[j, :, :].unsqueeze(0)}
            pean.diffusion.feed_data(data_diff_j)
            pean.diffusion.test()
            visuals = pean.diffusion.get_current_visuals()
            prior = visuals['SR']
            if label_vecs_final is None:
                label_vecs_final = prior
            else:
                label_vecs_final = torch.cat([label_vecs_final, prior], dim=0)
        label_vecs_final = label_vecs_final.to(pean.device)

        # Forward both models
        with torch.no_grad():
            sr_pean, _ = pean_model(images_lr, label_vecs_final)
            sr_swin, _ = swin_model(images_lr, label_vecs_final)

        # Aster predictions for SR (qualitative/recognition diff)
        aster_dict_lr = pean.parse_aster_data(images_lr[:, :3, :, :])
        aster_dict_sr_pean = pean.parse_aster_data(sr_pean[:, :3, :, :])
        aster_dict_sr_swin = pean.parse_aster_data(sr_swin[:, :3, :, :])
        pred_rec_lr = aster(aster_dict_lr)['output']['pred_rec']
        pred_rec_pean = aster(aster_dict_sr_pean)['output']['pred_rec']
        pred_rec_swin = aster(aster_dict_sr_swin)['output']['pred_rec']
        from utils.metrics import get_str_list
        pred_str_lr, _ = get_str_list(pred_rec_lr, aster_dict_lr['rec_targets'], dataset=aster_info)
        pred_str_pean, _ = get_str_list(pred_rec_pean, aster_dict_sr_pean['rec_targets'], dataset=aster_info)
        pred_str_swin, _ = get_str_list(pred_rec_swin, aster_dict_sr_swin['rec_targets'], dataset=aster_info)

        batch_n = images_lr.shape[0]
        for i in range(batch_n):
            if collected >= args.max_samples:
                break
            label = str(label_strs[i])
            stem = f'idx{collected:04d}'
            # Save grid image
            grid_path = os.path.join(img_dir, f'{stem}_grid.png')
            _save_grid(images_lr[i], sr_pean[i], sr_swin[i], images_hr[i], grid_path)
            # Save heatmaps
            _save_diff_maps(sr_pean[i], sr_swin[i], images_hr[i], heat_dir, stem)
            # Metrics
            l1_p, l2_p, psnr_p, ssim_p = _metrics(sr_pean[i], images_hr[i])
            l1_s, l2_s, psnr_s, ssim_s = _metrics(sr_swin[i], images_hr[i])
            with open(csv_path, 'a', newline='') as f:
                w = csv.writer(f)
                w.writerow([
                    collected, label,
                    pred_str_pean[i], pred_str_swin[i],
                    f'{psnr_p:.4f}', f'{ssim_p:.6f}', f'{psnr_s:.4f}', f'{ssim_s:.6f}',
                    f'{l1_p:.6f}', f'{l2_p:.6f}', f'{l1_s:.6f}', f'{l2_s:.6f}',
                    f'{(psnr_p-psnr_s):.4f}', f'{(ssim_p-ssim_s):.6f}',
                ])

            # Add to gallery tiles (resize all to HR size)
            hr_t = _to_3ch_float(images_hr[i])
            H, W = hr_t.shape[-2], hr_t.shape[-1]
            lr_t = _resize_chw(_to_3ch_float(images_lr[i]), (H, W))
            pean_t = _resize_chw(_to_3ch_float(sr_pean[i]), (H, W))
            swin_t = _resize_chw(_to_3ch_float(sr_swin[i]), (H, W))
            gallery_tiles.extend([lr_t, pean_t, swin_t, hr_t])
            gallery_rows_meta.append((label, pred_str_pean[i], pred_str_swin[i]))
            collected += 1
        if collected >= args.max_samples:
            break

    # Aggregate summary
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        def _fmean(col):
            try:
                return float(pd.to_numeric(df[col], errors='coerce').mean())
            except Exception:
                return float('nan')
        summary = {
            'psnr_pean_mean': _fmean('psnr_pean'),
            'psnr_swinir_mean': _fmean('psnr_swinir'),
            'ssim_pean_mean': _fmean('ssim_pean'),
            'ssim_swinir_mean': _fmean('ssim_swinir'),
            'psnr_delta_mean': _fmean('psnr_delta(pean-swinir)'),
            'ssim_delta_mean': _fmean('ssim_delta(pean-swinir)')
        }
        with open(os.path.join(out_root, 'summary.txt'), 'w') as f:
            for k,v in summary.items():
                f.write(f"{k}: {v}\n")
    except Exception:
        pass

    # Build a single gallery image (rows=samples, cols=[LR, PEAN, SwinIR, HR])
    try:
        if len(gallery_tiles) > 0:
            gallery = torchvision.utils.make_grid(gallery_tiles, nrow=4, padding=4)
            torchvision.utils.save_image(gallery, os.path.join(out_root, 'gallery.png'))

            # Labeled gallery using matplotlib (optional, adds column headers)
            gal_np = gallery.permute(1, 2, 0).numpy()
            fig_h = max(4, int(2 + 2 * (len(gallery_rows_meta))))
            plt.figure(figsize=(10, fig_h), dpi=150)
            plt.imshow(gal_np)
            plt.axis('off')
            # Column headers
            cols = ['LR', 'PEAN', 'SwinIR', 'HR']
            Himg, Wimg = gal_np.shape[0], gal_np.shape[1]
            col_w = Wimg / 4.0
            for ci, name in enumerate(cols):
                x = (ci + 0.5) * col_w
                plt.text(x, 12, name, color='white', fontsize=12, ha='center', va='top',
                         bbox=dict(facecolor='black', alpha=0.6, pad=2))
            plt.tight_layout()
            plt.savefig(os.path.join(out_root, 'gallery_labeled.png'))
            plt.close()
    except Exception:
        pass

    print(f"\n✓ Saved paired comparison to: {out_root}")
    print(f"  - Grids: {img_dir}")
    print(f"  - Heatmaps: {heat_dir}")
    print(f"  - Metrics CSV: {csv_path}")
    print(f"  - Gallery: {os.path.join(out_root, 'gallery.png')}")


if __name__ == '__main__':
    main()
