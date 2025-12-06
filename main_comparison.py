"""
Dual training script: Train PEAN and SwinIR simultaneously and compare their performance.
This script runs both methods in parallel and generates comparison visualizations.
"""
import argparse
import csv
import os
import warnings
import threading
import time
import shutil
from datetime import datetime
import copy

import yaml
import matplotlib.pyplot as plt
from easydict import EasyDict

from interfaces.super_resolution import TextSR
from interfaces.swinir_super_resolution import SwinIRTextSRInterface
from utils.util import set_seed

warnings.filterwarnings("ignore")

# Matplotlib configuration for comparison plots
plt.rcParams['font.size'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.titlesize'] = 12


class DualTrainer:
    """Manages parallel training of PEAN and SwinIR with real-time comparison."""
    
    def __init__(self, config, args):
        self.config = config
        self.args = args
        self.comparison_dir = './ckpt_comparison'
        os.makedirs(self.comparison_dir, exist_ok=True)
        self.model_names = ("PEAN", "SwinIR")
        self.comparison_index = 0
        self.comparison_history_path = os.path.join(self.comparison_dir, 'comparison_history.txt')
        self.last_pean_rows = 0
        self.last_swinir_rows = 0
        if not os.path.exists(self.comparison_history_path):
            with open(self.comparison_history_path, 'w', encoding='utf-8') as history_file:
                history_file.write("PEAN vs SwinIR Comparison History\n")
                history_file.write("=" * 80 + "\n\n")
        
        # Training metrics storage
        self.pean_metrics = {'iters': [], 'loss': [], 'psnr': [], 'ssim': [], 'accuracy': []}
        self.swinir_metrics = {'iters': [], 'loss': [], 'psnr': [], 'ssim': [], 'accuracy': []}
        
        # Locks for thread-safe metric updates
        self.pean_lock = threading.Lock()
        self.swinir_lock = threading.Lock()
        
        print("=" * 100)
        print("DUAL TRAINING MODE: PEAN vs SwinIR")
        print("=" * 100)
        print(f"Output directories:")
        print(f"  PEAN:       ./ckpt/       ./vis/")
        print(f"  SwinIR:     ./ckpt_swinir/ ./vis_swinir/")
        print(f"  Comparison: {self.comparison_dir}/")
        print(f"Comparing models: {self.model_names[0]} vs {self.model_names[1]}")
        print("=" * 100)
    
    def train_pean(self):
        """Train PEAN baseline in a separate thread."""
        try:
            print("\n[PEAN] Initializing...")
            # Work on isolated copies so each model owns its directories
            pean_config = copy.deepcopy(self.config)
            pean_args = copy.deepcopy(self.args)

            # Force training from scratch (disable resume)
            pean_args.resume = None
            pean_args.vis_dir = './vis'
            pean_config.TRAIN.ckpt_dir = './ckpt'
            pean_config.TRAIN.VAL.vis_dir = './vis'

            mission = TextSR(pean_config, pean_args)
            # Inject model name before training starts
            mission._comparison_name = "PEAN"
            
            # Override train method to capture metrics
            original_train = mission.train
            
            def wrapped_train():
                # Inject metric capture
                mission._comparison_metrics = self.pean_metrics
                mission._comparison_lock = self.pean_lock
                original_train()
            
            wrapped_train()
            
        except Exception as e:
            print(f"[PEAN] Error: {e}")
            import traceback
            traceback.print_exc()
    
    def train_swinir(self):
        """Train SwinIR in a separate thread."""
        try:
            print("\n[SwinIR] Initializing...")
            swinir_config = copy.deepcopy(self.config)
            swinir_args = copy.deepcopy(self.args)

            # Force training from scratch (disable resume)
            swinir_args.resume = None
            swinir_args.swin_resume = None
            swinir_args.vis_dir = './vis_swinir'
            swinir_config.TRAIN.ckpt_dir = './ckpt_swinir'
            swinir_config.TRAIN.VAL.vis_dir = './vis_swinir'

            mission = SwinIRTextSRInterface(swinir_config, swinir_args)
            # Inject model name before training starts
            mission._comparison_name = "SwinIR"
            
            # Override train method to capture metrics
            original_train = mission.train
            
            def wrapped_train():
                # Inject metric capture
                mission._comparison_metrics = self.swinir_metrics
                mission._comparison_lock = self.swinir_lock
                original_train()
            
            wrapped_train()
            
        except Exception as e:
            print(f"[SwinIR] Error: {e}")
            import traceback
            traceback.print_exc()
    
    def monitor_and_compare(self):
        """Monitor training progress and generate comparison plots."""
        print("\n[Monitor] Starting comparison monitor...")
        
        while True:
            time.sleep(30)  # Check every 30 seconds
            
            with self.pean_lock:
                pean_has_data = len(self.pean_metrics['iters']) > 0
            with self.swinir_lock:
                swinir_has_data = len(self.swinir_metrics['iters']) > 0
            
            if pean_has_data or swinir_has_data:
                try:
                    self.generate_comparison_plots()
                except Exception as e:
                    print(f"[Monitor] Failed to generate comparison: {e}")
    
    def generate_comparison_plots(self):
        """Generate comprehensive comparison visualizations."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Read logs from CSV files
        pean_log = self._read_log('./ckpt/log.csv')
        swinir_log = self._read_log('./ckpt_swinir/log.csv')
        pean_train_log = self._read_training_metrics('./ckpt/train_metrics.csv')
        swinir_train_log = self._read_training_metrics('./ckpt_swinir/train_metrics.csv')
        
        if not pean_log and not swinir_log:
            return

        # Handle potential log resets (e.g., restarted training)
        if len(pean_log) < self.last_pean_rows:
            self.last_pean_rows = 0
        if len(swinir_log) < self.last_swinir_rows:
            self.last_swinir_rows = 0

        # Skip if no new log rows were added
        if len(pean_log) <= self.last_pean_rows and len(swinir_log) <= self.last_swinir_rows:
            return

        summary_text, summary_stats = self._generate_summary_text(pean_log, swinir_log)

        self.comparison_index += 1
        comparison_id = self.comparison_index
        comparison_name = f'comparison_{comparison_id:04d}_{timestamp}'
        output_dir = os.path.join(self.comparison_dir, comparison_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"[Monitor] Comparison #{comparison_id}: {self.model_names[0]} vs {self.model_names[1]}")
        self._print_dataset_summary(summary_stats)

        # Generate multiple comparison plots
        plot_jobs = [
            (self._plot_accuracy_comparison, (pean_log, swinir_log, output_dir), 'accuracy comparison'),
            (self._plot_psnr_ssim_comparison, (pean_log, swinir_log, output_dir), 'PSNR/SSIM comparison'),
            (self._plot_comprehensive_dashboard, (pean_log, swinir_log, output_dir, summary_text), 'dashboard'),
            (self._plot_training_dynamics, (pean_train_log, swinir_train_log, output_dir), 'training dynamics'),
        ]

        for plot_func, args, description in plot_jobs:
            try:
                plot_func(*args)
            except Exception as exc:
                print(f"[Monitor] Warning: failed to render {description} plot ({exc})")

        # Persist comparison history
        self._write_comparison_history(
            comparison_id=comparison_id,
            comparison_name=comparison_name,
            timestamp=timestamp,
            summary_text=summary_text,
            summary_stats=summary_stats,
            output_dir=output_dir,
        )

        # Persist textual summary and log snapshots
        self._write_summary_files(output_dir, summary_text, summary_stats)

        # Maintain "latest" shortcut
        latest_dir = os.path.join(self.comparison_dir, 'latest')
        if os.path.exists(latest_dir):
            shutil.rmtree(latest_dir)
        shutil.copytree(output_dir, latest_dir)

        # Update counters
        self.last_pean_rows = len(pean_log)
        self.last_swinir_rows = len(swinir_log)

        print(f"[Monitor] Saved comparison #{comparison_id} to: {output_dir}")
    
    def _read_log(self, log_path):
        """Read training log CSV file."""
        if not os.path.exists(log_path):
            return []

        rows = []
        try:
            with open(log_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    dataset = row.get('dataset', '')
                    if not dataset:
                        # Skip rows without dataset information (e.g., best_sum markers)
                        continue

                    try:
                        epoch = int(float(row.get('epoch', 0) or 0))
                    except ValueError:
                        epoch = 0

                    try:
                        accuracy = float(row.get('accuracy', 0.0) or 0.0)
                    except ValueError:
                        accuracy = 0.0
                    try:
                        psnr = float(row.get('psnr_avg', 0.0) or 0.0)
                    except ValueError:
                        psnr = 0.0
                    try:
                        ssim = float(row.get('ssim_avg', 0.0) or 0.0)
                    except ValueError:
                        ssim = 0.0

                    rows.append({
                        'epoch': epoch,
                        'dataset': dataset,
                        'accuracy': accuracy,
                        'psnr_avg': psnr,
                        'ssim_avg': ssim,
                        'best': row.get('best', ''),
                        'best_sum': row.get('best_sum', ''),
                    })
        except Exception as e:
            print(f"[Monitor] Failed to read {log_path}: {e}")
            return []

        return rows

    def _read_training_metrics(self, log_path, max_points=2000):
        """Read per-iteration training metrics and downsample for plotting."""
        if not os.path.exists(log_path):
            return []

        rows = []
        try:
            with open(log_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        iteration = int(float(row.get('iteration', 0) or 0))
                    except ValueError:
                        iteration = 0
                    try:
                        loss = float(row.get('loss', 0.0) or 0.0)
                    except ValueError:
                        loss = 0.0
                    try:
                        lr = float(row.get('learning_rate', 0.0) or 0.0)
                    except ValueError:
                        lr = 0.0
                    try:
                        grad_norm = float(row.get('gradient_norm', 0.0) or 0.0)
                    except ValueError:
                        grad_norm = 0.0
                    rows.append({
                        'iteration': iteration,
                        'loss': loss,
                        'learning_rate': lr,
                        'gradient_norm': grad_norm,
                    })
        except Exception as exc:
            print(f"[Monitor] Failed to read training metrics {log_path}: {exc}")
            return []

        if len(rows) > max_points:
            step = max(1, len(rows) // max_points)
            rows = rows[::step]
        return rows
    
    @staticmethod
    def _get_dataset_entries(log_rows, dataset_keyword):
        """Return log entries matching the dataset keyword (case-insensitive)."""
        if not log_rows:
            return []
        keyword = dataset_keyword.lower()
        matches = []
        for row in log_rows:
            dataset_name = str(row.get('dataset', '')).lower()
            if keyword in dataset_name:
                matches.append(row)
        return matches

    def _plot_accuracy_comparison(self, pean_log, swinir_log, output_dir):
        """Plot accuracy comparison across datasets."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        datasets = ['easy', 'medium', 'hard']
        colors = {'PEAN': '#2E86AB', 'SwinIR': '#A23B72'}
        
        for idx, dataset in enumerate(datasets):
            ax = axes[idx]
            
            subset = self._get_dataset_entries(pean_log, dataset)
            if subset:
                accuracy_values = [row['accuracy'] * 100 for row in subset]
                ax.plot(range(len(accuracy_values)), accuracy_values,
                        'o-', label='PEAN', color=colors['PEAN'], linewidth=2, markersize=6)

            subset = self._get_dataset_entries(swinir_log, dataset)
            if subset:
                accuracy_values = [row['accuracy'] * 100 for row in subset]
                ax.plot(range(len(accuracy_values)), accuracy_values,
                        's-', label='SwinIR', color=colors['SwinIR'], linewidth=2, markersize=6)
            
            ax.set_xlabel('Validation Step', fontweight='bold')
            ax.set_ylabel('Accuracy (%)', fontweight='bold')
            ax.set_title(f'{dataset.capitalize()} Dataset', fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        fig.suptitle('Accuracy Comparison: PEAN vs SwinIR', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '01_accuracy_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_psnr_ssim_comparison(self, pean_log, swinir_log, output_dir):
        """Plot PSNR and SSIM comparison."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        datasets = ['easy', 'medium', 'hard']
        colors = {'PEAN': '#2E86AB', 'SwinIR': '#A23B72'}
        
        for idx, dataset in enumerate(datasets):
            # PSNR plot
            ax_psnr = axes[0, idx]
            subset = self._get_dataset_entries(pean_log, dataset)
            if subset:
                psnr_values = [row['psnr_avg'] for row in subset]
                ax_psnr.plot(range(len(psnr_values)), psnr_values,
                             'o-', label='PEAN', color=colors['PEAN'], linewidth=2, markersize=6)

            subset = self._get_dataset_entries(swinir_log, dataset)
            if subset:
                psnr_values = [row['psnr_avg'] for row in subset]
                ax_psnr.plot(range(len(psnr_values)), psnr_values,
                             's-', label='SwinIR', color=colors['SwinIR'], linewidth=2, markersize=6)
            
            ax_psnr.set_ylabel('PSNR (dB)', fontweight='bold')
            ax_psnr.set_title(f'{dataset.capitalize()} - PSNR', fontweight='bold')
            ax_psnr.legend(loc='best')
            ax_psnr.grid(True, alpha=0.3)
            
            # SSIM plot
            ax_ssim = axes[1, idx]
            subset = self._get_dataset_entries(pean_log, dataset)
            if subset:
                ssim_values = [row['ssim_avg'] for row in subset]
                ax_ssim.plot(range(len(ssim_values)), ssim_values,
                             'o-', label='PEAN', color=colors['PEAN'], linewidth=2, markersize=6)

            subset = self._get_dataset_entries(swinir_log, dataset)
            if subset:
                ssim_values = [row['ssim_avg'] for row in subset]
                ax_ssim.plot(range(len(ssim_values)), ssim_values,
                             's-', label='SwinIR', color=colors['SwinIR'], linewidth=2, markersize=6)
            
            ax_ssim.set_xlabel('Validation Step', fontweight='bold')
            ax_ssim.set_ylabel('SSIM', fontweight='bold')
            ax_ssim.set_title(f'{dataset.capitalize()} - SSIM', fontweight='bold')
            ax_ssim.legend(loc='best')
            ax_ssim.grid(True, alpha=0.3)
        
        fig.suptitle('Image Quality Comparison: PSNR & SSIM', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '02_psnr_ssim_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_comprehensive_dashboard(self, pean_log, swinir_log, output_dir, summary_text):
        """Generate comprehensive comparison dashboard."""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        colors = {'PEAN': '#2E86AB', 'SwinIR': '#A23B72'}
        datasets = ['easy', 'medium', 'hard']
        
        # Row 1: Accuracy trends
        for idx, dataset in enumerate(datasets):
            ax = fig.add_subplot(gs[0, idx])
            
            subset = self._get_dataset_entries(pean_log, dataset)
            if subset:
                accuracy_values = [row['accuracy'] * 100 for row in subset]
                ax.plot(range(len(accuracy_values)), accuracy_values,
                        'o-', label='PEAN', color=colors['PEAN'], linewidth=2, markersize=5)

            subset = self._get_dataset_entries(swinir_log, dataset)
            if subset:
                accuracy_values = [row['accuracy'] * 100 for row in subset]
                ax.plot(range(len(accuracy_values)), accuracy_values,
                        's-', label='SwinIR', color=colors['SwinIR'], linewidth=2, markersize=5)
            
            ax.set_title(f'{dataset.capitalize()} Accuracy', fontweight='bold')
            ax.set_ylabel('Accuracy (%)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Row 2: PSNR comparison
        for idx, dataset in enumerate(datasets):
            ax = fig.add_subplot(gs[1, idx])
            
            subset = self._get_dataset_entries(pean_log, dataset)
            if subset:
                psnr_values = [row['psnr_avg'] for row in subset]
                ax.plot(range(len(psnr_values)), psnr_values,
                        'o-', label='PEAN', color=colors['PEAN'], linewidth=2, markersize=5)

            subset = self._get_dataset_entries(swinir_log, dataset)
            if subset:
                psnr_values = [row['psnr_avg'] for row in subset]
                ax.plot(range(len(psnr_values)), psnr_values,
                        's-', label='SwinIR', color=colors['SwinIR'], linewidth=2, markersize=5)
            
            ax.set_title(f'{dataset.capitalize()} PSNR', fontweight='bold')
            ax.set_ylabel('PSNR (dB)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Row 3: Summary statistics
        ax_summary = fig.add_subplot(gs[2, :])
        ax_summary.axis('off')

        try:
            ax_summary.text(
                0.5,
                0.5,
                summary_text,
                ha='center',
                va='center',
                fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
                wrap=True,
            )
        except Exception as exc:
            print(f"[Monitor] Warning: failed to render summary panel ({exc}). Summary saved to summary.txt.")
            ax_summary.text(
                0.5,
                0.5,
                "Summary text saved to summary.txt",
                ha='center',
                va='center',
                fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
            )
        
        fig.suptitle('Comprehensive Training Comparison Dashboard', fontsize=18, fontweight='bold')
        plt.savefig(os.path.join(output_dir, '03_comprehensive_dashboard.png'), dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_training_dynamics(self, pean_train, swinir_train, output_dir):
        """Plot per-iteration training metrics for both models."""
        if not pean_train and not swinir_train:
            return

        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        colors = {'PEAN': '#2E86AB', 'SwinIR': '#A23B72'}
        series = [
            ('loss', 'Training Loss', 'Loss (log scale)', True),
            ('learning_rate', 'Learning Rate Schedule', 'Learning Rate', False),
            ('gradient_norm', 'Gradient Norm', '||grad||₂', False),
        ]

        def _plot_series(ax, data, label, color, key, log_scale):
            if not data:
                return
            x = [row['iteration'] for row in data]
            y = [row[key] for row in data]
            if not any(y):
                return
            ax.plot(x, y, label=label, color=color, linewidth=1.6)
            if log_scale:
                ax.set_yscale('log')

        for ax, (key, title, ylabel, log_scale) in zip(axes, series):
            _plot_series(ax, pean_train, 'PEAN', colors['PEAN'], key, log_scale)
            _plot_series(ax, swinir_train, 'SwinIR', colors['SwinIR'], key, log_scale)
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Iteration', fontweight='bold')
            ax.set_ylabel(ylabel, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')

        fig.suptitle('Per-Iteration Training Dynamics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '04_training_dynamics.png'), dpi=180, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def _summarize_entries(entries):
        """Return per-model summary metrics for a dataset."""
        if not entries:
            return {
                'count': 0,
                'latest': {'epoch': None, 'accuracy': None, 'psnr': None, 'ssim': None},
                'best': {
                    'accuracy': {'value': None, 'epoch': None},
                    'psnr': {'value': None, 'epoch': None},
                    'ssim': {'value': None, 'epoch': None},
                },
            }

        latest = entries[-1]
        best_accuracy_entry = max(entries, key=lambda item: item['accuracy'])
        best_psnr_entry = max(entries, key=lambda item: item['psnr_avg'])
        best_ssim_entry = max(entries, key=lambda item: item['ssim_avg'])

        return {
            'count': len(entries),
            'latest': {
                'epoch': latest.get('epoch'),
                'accuracy': latest.get('accuracy', 0.0) * 100.0,
                'psnr': latest.get('psnr_avg', 0.0),
                'ssim': latest.get('ssim_avg', 0.0),
            },
            'best': {
                'accuracy': {
                    'value': best_accuracy_entry.get('accuracy', 0.0) * 100.0,
                    'epoch': best_accuracy_entry.get('epoch'),
                },
                'psnr': {
                    'value': best_psnr_entry.get('psnr_avg', 0.0),
                    'epoch': best_psnr_entry.get('epoch'),
                },
                'ssim': {
                    'value': best_ssim_entry.get('ssim_avg', 0.0),
                    'epoch': best_ssim_entry.get('epoch'),
                },
            },
        }

    def _generate_summary_text(self, pean_log, swinir_log):
        """Generate summary statistics text and structured metrics."""
        datasets = ['easy', 'medium', 'hard']
        summary_lines = ["PERFORMANCE SUMMARY", "=" * 80, ""]
        model_stats = {
            dataset: {name: self._summarize_entries(self._get_dataset_entries(log, dataset))
                      for name, log in zip(self.model_names, (pean_log, swinir_log))}
            for dataset in datasets
        }

        for dataset in datasets:
            summary_lines.append(f"{dataset.upper()} Dataset:")
            summary_lines.append("-" * 80)

            for model_name in self.model_names:
                stats = model_stats[dataset][model_name]
                summary_lines.extend(self._format_summary_lines(model_name, stats))

            summary_lines.append("")

        summary_text = "\n".join(summary_lines).strip()
        return summary_text, model_stats

    def _format_summary_lines(self, model_name, stats):
        """Return formatted lines describing latest and best metrics for a model."""
        if stats['count'] == 0:
            return [f"  {model_name}:   (no data)"]

        latest = stats['latest']
        best = stats['best']

        latest_line = (
            f"  {model_name}:   Latest Acc {self._format_accuracy(latest['accuracy'], latest['epoch'])}  "
            f"PSNR {self._format_psnr(latest['psnr'], latest['epoch'])}  "
            f"SSIM {self._format_ssim(latest['ssim'], latest['epoch'])}"
        )

        best_line = (
            f"              Best   Acc {self._format_accuracy(best['accuracy']['value'], best['accuracy']['epoch'])}  | "
            f"PSNR {self._format_psnr(best['psnr']['value'], best['psnr']['epoch'])}  | "
            f"SSIM {self._format_ssim(best['ssim']['value'], best['ssim']['epoch'])}  | "
            f"Samples {stats['count']}"
        )

        return [latest_line, best_line]

    def _print_dataset_summary(self, summary_stats):
        """Print comparison summary for each dataset."""
        datasets = ['easy', 'medium', 'hard']
        print("  Dataset comparison summary:")
        for dataset in datasets:
            for idx, model_name in enumerate(self.model_names):
                stats = summary_stats[dataset][model_name]
                prefix = f"    {dataset.capitalize():<6} | " if idx == 0 else "             | "

                if stats['count'] == 0:
                    print(f"{prefix}{model_name:<6} (no data yet)")
                    continue

                latest = stats['latest']
                best = stats['best']
                print(
                    f"{prefix}{model_name:<6} Latest Acc {self._format_accuracy(latest['accuracy'], latest['epoch'])}  "
                    f"PSNR {self._format_psnr(latest['psnr'], latest['epoch'])}  "
                    f"SSIM {self._format_ssim(latest['ssim'], latest['epoch'])}"
                )
                print(
                    f"             | {model_name:<6} Best   Acc {self._format_accuracy(best['accuracy']['value'], best['accuracy']['epoch'])}  "
                    f"PSNR {self._format_psnr(best['psnr']['value'], best['psnr']['epoch'])}  "
                    f"SSIM {self._format_ssim(best['ssim']['value'], best['ssim']['epoch'])}  "
                    f"Samples {stats['count']}"
                )
        print("  " + "-" * 70)

    def _write_comparison_history(self, comparison_id, comparison_name, timestamp, summary_text, summary_stats, output_dir):
        """Append comparison details to the history log."""
        pean_name, swinir_name = self.model_names
        with open(self.comparison_history_path, 'a', encoding='utf-8') as history_file:
            history_file.write(f"[{timestamp}] Comparison #{comparison_id} ({comparison_name})\n")
            history_file.write(f"Models: {pean_name} vs {swinir_name}\n")
            history_file.write(f"Output: {os.path.abspath(output_dir)}\n")
            for dataset in ['easy', 'medium', 'hard']:
                history_file.write(f"  {dataset.capitalize()} Dataset\n")
                for model_name in self.model_names:
                    stats = summary_stats[dataset][model_name]
                    if stats['count'] == 0:
                        history_file.write(f"    {model_name:<6} (no data yet)\n")
                        continue

                    latest = stats['latest']
                    best = stats['best']
                    history_file.write(
                        f"    {model_name:<6} Latest Acc {self._format_accuracy(latest['accuracy'], latest['epoch'])}  "
                        f"PSNR {self._format_psnr(latest['psnr'], latest['epoch'])}  "
                        f"SSIM {self._format_ssim(latest['ssim'], latest['epoch'])}\n"
                    )
                    history_file.write(
                        f"              Best   Acc {self._format_accuracy(best['accuracy']['value'], best['accuracy']['epoch'])}  "
                        f"PSNR {self._format_psnr(best['psnr']['value'], best['psnr']['epoch'])}  "
                        f"SSIM {self._format_ssim(best['ssim']['value'], best['ssim']['epoch'])}  "
                        f"Samples {stats['count']}\n"
                    )
            history_file.write(summary_text + "\n")
            history_file.write("-" * 80 + "\n\n")

    def _write_summary_files(self, output_dir, summary_text, summary_stats):
        """Persist textual and tabular summaries for the comparison snapshot."""
        summary_txt_path = os.path.join(output_dir, 'summary.txt')
        with open(summary_txt_path, 'w', encoding='utf-8') as summary_file:
            summary_file.write(summary_text + "\n")

        summary_csv_path = os.path.join(output_dir, 'summary_metrics.csv')
        with open(summary_csv_path, 'w', encoding='utf-8', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                'dataset',
                'model',
                'metric_type',
                'accuracy_percent',
                'accuracy_epoch',
                'psnr',
                'psnr_epoch',
                'ssim',
                'ssim_epoch',
                'observations',
            ])

            for dataset in ['easy', 'medium', 'hard']:
                for model_name in self.model_names:
                    stats = summary_stats[dataset][model_name]
                    if stats['count'] == 0:
                        writer.writerow([dataset, model_name, 'none', '', '', '', '', '', '', 0])
                        continue

                    latest = stats['latest']
                    best = stats['best']

                    writer.writerow([
                        dataset,
                        model_name,
                        'latest',
                        f"{latest['accuracy']:.4f}",
                        latest['epoch'] if latest['epoch'] is not None else '',
                        f"{latest['psnr']:.4f}",
                        latest['epoch'] if latest['epoch'] is not None else '',
                        f"{latest['ssim']:.6f}",
                        latest['epoch'] if latest['epoch'] is not None else '',
                        stats['count'],
                    ])

                    writer.writerow([
                        dataset,
                        model_name,
                        'best',
                        f"{best['accuracy']['value']:.4f}",
                        best['accuracy']['epoch'] if best['accuracy']['epoch'] is not None else '',
                        f"{best['psnr']['value']:.4f}",
                        best['psnr']['epoch'] if best['psnr']['epoch'] is not None else '',
                        f"{best['ssim']['value']:.6f}",
                        best['ssim']['epoch'] if best['ssim']['epoch'] is not None else '',
                        stats['count'],
                    ])

        # Copy current log files for traceability
        log_snapshots = [
            ('pean_log_snapshot.csv', './ckpt/log.csv'),
            ('swinir_log_snapshot.csv', './ckpt_swinir/log.csv'),
            ('pean_train_metrics_snapshot.csv', './ckpt/train_metrics.csv'),
            ('swinir_train_metrics_snapshot.csv', './ckpt_swinir/train_metrics.csv'),
        ]
        for target_name, source_path in log_snapshots:
            if os.path.exists(source_path):
                try:
                    shutil.copy(source_path, os.path.join(output_dir, target_name))
                except Exception as exc:
                    print(f"[Monitor] Warning: failed to copy {source_path} ({exc})")

    @staticmethod
    def _format_accuracy(value, epoch=None):
        if value is None:
            return "--"
        base = f"{value:.2f}%"
        if epoch is not None:
            return f"{base} (ep{epoch})"
        return base

    @staticmethod
    def _format_psnr(value, epoch=None):
        if value is None:
            return "--"
        base = f"{value:.2f}dB"
        if epoch is not None:
            return f"{base} (ep{epoch})"
        return base

    @staticmethod
    def _format_ssim(value, epoch=None):
        if value is None:
            return "--"
        base = f"{value:.4f}"
        if epoch is not None:
            return f"{base} (ep{epoch})"
        return base
    
    def run(self):
        """Start dual training with monitoring."""
        # Create training threads
        pean_thread = threading.Thread(target=self.train_pean, name="PEAN-Training")
        swinir_thread = threading.Thread(target=self.train_swinir, name="SwinIR-Training")
        monitor_thread = threading.Thread(target=self.monitor_and_compare, name="Monitor", daemon=True)
        
        # Start all threads
        print("\n[Main] Starting PEAN training thread...")
        pean_thread.start()
        
        time.sleep(5)  # Stagger starts to avoid resource conflicts
        
        print("[Main] Starting SwinIR training thread...")
        swinir_thread.start()
        
        print("[Main] Starting comparison monitor thread...")
        monitor_thread.start()
        
        # Wait for training to complete
        print("\n[Main] Both models are training in parallel...")
        print("[Main] Press Ctrl+C to stop\n")
        
        try:
            pean_thread.join()
            swinir_thread.join()
        except KeyboardInterrupt:
            print("\n[Main] Training interrupted by user")
        
        # Generate final comparison
        print("\n[Main] Generating final comparison...")
        self.generate_comparison_plots()
        
        print("\n[Main] Dual training completed!")
        print(f"[Main] Comparison results saved to: {self.comparison_dir}/")


def main(config, args):
    set_seed(config.TRAIN.manualSeed)
    
    if args.test:
        print("Test mode not supported in comparison script")
        print("Use main.py or main_swinir.py for testing")
        return
    
    # Initialize dual trainer
    trainer = DualTrainer(config, args)
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual training: PEAN vs SwinIR with real-time comparison")
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--pre_training', action='store_true', default=False)
    parser.add_argument('--test_data_dir', type=str, default='/root/dataset/TextZoom/test/medium', help='')
    parser.add_argument('--batch_size', type=int, default=None, help='')
    parser.add_argument('--resume', type=str, default=None, help='Base model resume path')
    parser.add_argument('--swin_resume', type=str, default=None, help='Resume path for SwinIR generator')
    parser.add_argument('--vis_dir', type=str, default=None, help='')
    parser.add_argument('--rec', default='aster', choices=['aster', 'moran', 'crnn'])
    parser.add_argument('--STN', action='store_true', default=False, help='')
    parser.add_argument('--syn', action='store_true', default=False, help='use synthetic LR')
    parser.add_argument('--mixed', action='store_true', default=False, help='mix synthetic with real LR')
    parser.add_argument('--mask', action='store_true', default=False, help='')
    parser.add_argument('--gradient', action='store_true', default=False, help='')
    parser.add_argument('--hd_u', type=int, default=32, help='')
    parser.add_argument('--srb', type=int, default=5, help='')
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--demo_dir', type=str, default='')
    parser.add_argument('--prior_dim', type=int, default=1024, help='')
    parser.add_argument('--dec_num_heads', type=int, default=16, help='')
    parser.add_argument('--dec_mlp_ratio', type=int, default=4, help='')
    parser.add_argument('--dec_depth', type=int, default=1, help='')
    parser.add_argument('--max_gen_perms', type=int, default=1, help='')
    parser.add_argument('--rotate_train', type=float, default=0., help='')
    parser.add_argument('--perm_forward', action='store_true', default=False, help='')
    parser.add_argument('--perm_mirrored', action='store_true', default=False, help='')
    parser.add_argument('--dropout', type=float, default=0.1, help='')
    args = parser.parse_args()

    config_path = os.path.join('config', 'super_resolution.yaml')
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    config = EasyDict(config)
    main(config, args)
