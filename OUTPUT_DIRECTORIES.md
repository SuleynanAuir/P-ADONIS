# 输出目录说明

## 目录结构

训练输出已完全分离，避免混淆：

```
/
CWordClearer├── ckpt/                           # PEAN baseline 输出目录
│   ├── log.csv                     # 训练日志
│   ├── *_gen.pth                   # 模型检查点
│   ├── PEAN_final.pth              # 最终模型
│   ├── visualizations/             # 原始可视化
│   │   ├── iter_*_epoch_*/         # 每次验证的可视化
│   │   └── latest/                 # 最新可视化快捷方式
│   └── visualizations_enhanced/    # 增强可视化
│       └── dashboard_*/            # 每次验证的增强仪表板
│
├── ckpt_swinir/                    # SwinIR 输出目录
│   ├── log.csv                     # SwinIR训练日志
│   ├── *_gen.pth                   # SwinIR模型检查点
│   ├── visualizations/             # SwinIR原始可视化
│   └── visualizations_enhanced/    # SwinIR增强可视化
│
├── ckpt_comparison/                # ★ 对比可视化目录 ★
│   ├── comparison_*/               # 每次对比生成的结果
│   │   ├── 01_accuracy_comparison.png      # 准确率对比
│   │   ├── 02_psnr_ssim_comparison.png     # PSNR/SSIM对比
│   │   └── 03_comprehensive_dashboard.png  # 综合仪表板
│   └── latest/                     # 最新对比结果
│
├── vis/                            # PEAN baseline 验证可视化
│   ├── 0/                          # Easy数据集
│   ├── 1/                          # Medium数据集
│   └── 2/                          # Hard数据集
│
└── vis_swinir/                     # SwinIR 验证可视化
    ├── 0/                          # Easy数据集
    ├── 1/                          # Medium数据集
    └── 2/                          # Hard数据集
```

## 训练命令

### PEAN Baseline (单独训练)
```bash
python main.py --batch_size=8 --mask --rec="aster" --srb=1
```
**输出位置：**
- 检查点: `./ckpt/`
- 可视化: `./vis/`
- 日志: `./ckpt/log.csv`

### SwinIR (单独训练)
```bash
python main_swinir.py --batch_size=8 --mask --rec="aster" --srb=1
```
**输出位置：**
- 检查点: `./ckpt_swinir/`
- 可视化: `./vis_swinir/`
- 日志: `./ckpt_swinir/log.csv`

### 双模型对比训练 (推荐)
```bash
# PowerShell
.\run_comparison.ps1

# 或直接运行
python main_comparison.py --batch_size=8 --mask --rec="aster" --srb=1
```
**输出位置：**
- PEAN: `./ckpt/` + `./vis/`
- SwinIR: `./ckpt_swinir/` + `./vis_swinir/`
- 对比可视化: `./ckpt_comparison/`

**特点：**
- 同时训练两个模型，节省时间
- 每30秒自动生成对比图表
- 实时监控性能差异
- 包含准确率、PSNR、SSIM全方位对比

## 监控训练进度

### PEAN Baseline
```bash
# 实时查看日志
Get-Content .\ckpt\log.csv -Wait

# 查看最新可视化
explorer .\ckpt\visualizations_enhanced\

# 检查最新检查点
Get-ChildItem .\ckpt\*_gen.pth | Sort-Object LastWriteTime -Descending | Select-Object -First 5
```

### SwinIR
```bash
# 实时查看日志
Get-Content .\ckpt_swinir\log.csv -Wait

# 查看最新可视化
explorer .\ckpt_swinir\visualizations_enhanced\

# 检查最新检查点
Get-ChildItem .\ckpt_swinir\*_gen.pth | Sort-Object LastWriteTime -Descending | Select-Object -First 5
```

## 模型对比

训练完成后，可以对比两种方法的性能：

```bash
# 对比训练日志
python -c "
import pandas as pd
baseline = pd.read_csv('ckpt/log.csv')
swinir = pd.read_csv('ckpt_swinir/log.csv')
print('PEAN Baseline最佳准确率:')
print(baseline.groupby('dataset')['accuracy'].max())
print('\nSwinIR最佳准确率:')
print(swinir.groupby('dataset')['accuracy'].max())
"
```

## 清理旧输出

如果需要重新开始训练：

```bash
# 清理PEAN baseline
Remove-Item -Recurse -Force .\ckpt\*_gen.pth
Remove-Item -Recurse -Force .\ckpt\visualizations\*
Remove-Item -Recurse -Force .\vis\*

# 清理SwinIR
Remove-Item -Recurse -Force .\ckpt_swinir\*_gen.pth
Remove-Item -Recurse -Force .\ckpt_swinir\visualizations\*
Remove-Item -Recurse -Force .\vis_swinir\*
```

## 文件说明

### 检查点文件命名
- `I{iter}_E{epoch}_aster_gen.pth` - ASTER识别器最佳模型
- `I{iter}_E{epoch}_crnn_gen.pth` - CRNN识别器最佳模型
- `I{iter}_E{epoch}_moran_gen.pth` - MORAN识别器最佳模型
- `I{iter}_E{epoch}_sum_*_gen.pth` - 综合最佳模型
- `PEAN_final.pth` / `SwinIR_final.pth` - 最终保存的模型

### 日志文件格式
CSV列: `epoch`, `dataset`, `accuracy`, `psnr_avg`, `ssim_avg`, `best`, `best_sum`

### 可视化文件
- `training_curves.png` - 训练曲线
- `metrics_table.png` - 指标表格
- `metrics_scatter.png` - 指标散点图
- `diffusion_process.png` - 扩散过程可视化
- `prediction_examples.png` - 预测示例

### 增强可视化文件
- `01_training_dynamics.png` - 8合1训练动态仪表板
- `02_feature_maps.png` - 特征图可视化
- `03_loss_landscape.png` - 损失地形图
- `04_gradient_flow.png` - 梯度流可视化
- `05_sr_comparison_*.png` - 超分辨率对比图

### 对比可视化文件 (main_comparison.py)
- `01_accuracy_comparison.png` - 三个数据集(Easy/Medium/Hard)准确率对比曲线
- `02_psnr_ssim_comparison.png` - PSNR和SSIM六宫格对比图
- `03_comprehensive_dashboard.png` - 综合仪表板（准确率+PSNR+统计摘要）

**说明：**
- 每30秒自动更新一次对比图表
- 蓝色线条/圆圈 = PEAN baseline
- 紫色线条/方块 = SwinIR
- 包含最佳性能统计摘要表格
