# 环境配置文件更新说明

**更新日期:** 2025年11月20日  
**项目名称:** CWordClearer  
**基于环境:** pean (Python 3.8.20, PyTorch 1.10.1+cu113, CUDA 11.3)

---

## 📋 更新内容

### 1. 文件列表

已更新以下环境配置文件:

| 文件 | 用途 | 包数量 |
|------|------|--------|
| `requirements.txt` | pip 安装 | 140+ 包 |
| `environment.yml` | conda 环境创建 | 140+ 包 |

### 2. 主要改进

#### ✨ 结构化组织
- **分类清晰**: 按功能模块分组 (深度学习、计算机视觉、科学计算等)
- **注释详细**: 每个部分都有明确说明
- **易于维护**: 核心依赖和自动依赖分开

#### 📦 包分类

**核心框架 (8个包)**
```
PyTorch 1.10.1+cu113
torchvision 0.11.2+cu113
pytorch-lightning 1.5.10
torchmetrics 1.5.2
```

**计算机视觉 (6个包)**
```
opencv-python 4.12.0.88
pillow 10.4.0
scikit-image 0.19.3
imageio 2.35.1
imgaug 0.4.0
```

**TPAN-E & SwinIR 专用 (7个包)**
```
einops 0.8.1          # 张量操作
timm 0.6.5            # 预训练模型库
lpips 0.1.4           # 感知损失
pytorch-msssim 1.0.0  # 结构相似度
pyiqa 0.1.7           # 图像质量评估
thop 0.1.1            # 参数量计算
```

**实验追踪 (3个包)**
```
tensorboard 2.14.0
wandb 0.14.0
```

**文本处理 (4个包)**
```
nltk 3.6.7
ftfy 6.2.3
regex 2024.11.6
editdistance 0.6.0
```

**配置管理 (4个包)**
```
omegaconf 2.1.1
easydict 1.9
pyyaml 6.0.3
lmdb 1.3.0
```

#### 🔧 优化细节

1. **PyTorch CUDA 支持**
   - 使用 `-f https://download.pytorch.org/whl/cu113/torch_stable.html` 确保安装 CUDA 版本
   - conda 环境中同时指定 `cudatoolkit=11.3`

2. **版本固定**
   - 所有包都固定了精确版本号
   - 避免依赖冲突和不兼容问题

3. **自动依赖**
   - 100+ 个自动安装的依赖包已列出
   - 便于完整环境复现

---

## 🚀 使用方法

### 方法一: 使用 environment.yml (推荐)

**优点:** 一键创建完整环境，包含 conda 和 pip 依赖

```bash
# 1. 创建新环境
conda env create -f environment.yml

# 2. 激活环境
conda activate cwordclearer

# 3. 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 4. (可选) 删除旧环境
conda env remove -n pean
```

**预期输出:**
```
PyTorch: 1.10.1+cu113, CUDA: True
```

### 方法二: 使用 requirements.txt

**优点:** 适用于已有 Python 3.8 环境的情况

```bash
# 1. 创建 conda 环境 (如果还没有)
conda create -n cwordclearer python=3.8
conda activate cwordclearer

# 2. 安装所有依赖
pip install -r requirements.txt

# 3. 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 方法三: 更新现有 pean 环境

**如果想保留现有环境并更新名称:**

```bash
# 1. 克隆现有环境
conda create --name cwordclearer --clone pean

# 2. 激活新环境
conda activate cwordclearer

# 3. 验证
python -c "import torch; print(torch.__version__)"

# 4. (可选) 删除旧环境
conda env remove -n pean
```

---

## 📊 环境信息

### 系统要求

| 组件 | 版本要求 |
|------|----------|
| **操作系统** | Windows 10/11, Linux |
| **Python** | 3.8.20 |
| **CUDA** | 11.3 |
| **NVIDIA Driver** | ≥ 465.89 |
| **GPU Memory** | ≥ 6GB (建议 8GB+) |
| **磁盘空间** | ~8GB |

### 核心版本

```yaml
Python:        3.8.20
PyTorch:       1.10.1+cu113
torchvision:   0.11.2+cu113
CUDA:          11.3
NumPy:         1.24.4
OpenCV:        4.12.0.88
```

### 包统计

- **总包数:** 140+
- **核心依赖:** ~20 个
- **自动依赖:** ~120 个
- **环境大小:** ~8GB

---

## 🔍 关键依赖说明

### 深度学习核心

```python
torch==1.10.1+cu113          # PyTorch 主框架
torchvision==0.11.2+cu113    # 计算机视觉工具
pytorch-lightning==1.5.10    # 高级训练框架
```

### TPAN-E 模型专用

```python
einops==0.8.1                # 张量重排 (用于注意力机制)
timm==0.6.5                  # 预训练视觉模型
lpips==0.1.4                 # 感知损失 (TPAN-E 损失函数)
pytorch-msssim==1.0.0        # MS-SSIM 损失
pyiqa==0.1.7                 # 图像质量评估指标
```

### SwinIR 模型专用

```python
einops==0.8.1                # Swin Transformer 窗口注意力
timm==0.6.5                  # Swin 架构实现
```

### 文本识别器

```python
nltk==3.6.7                  # 自然语言处理 (ASTER, CRNN)
ftfy==6.2.3                  # 文本修复
editdistance==0.6.0          # 编辑距离计算 (准确率评估)
```

### 实验管理

```python
tensorboard==2.14.0          # 训练可视化
wandb==0.14.0                # 实验追踪 (云端)
omegaconf==2.1.1             # 配置文件管理
```

---

## ⚠️ 常见问题

### 1. CUDA 不可用

**问题:** `torch.cuda.is_available()` 返回 `False`

**解决方案:**
```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 重新安装 PyTorch with CUDA
pip uninstall torch torchvision
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

### 2. SSL 证书错误

**问题:** `SSLError` 在下载包时

**解决方案:**
```bash
# 临时使用 HTTP (不推荐生产环境)
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

# 或更新 certifi
pip install --upgrade certifi
```

### 3. 包冲突

**问题:** 版本冲突或依赖解析失败

**解决方案:**
```bash
# 清理缓存
pip cache purge
conda clean --all

# 重新创建环境
conda env remove -n cwordclearer
conda env create -f environment.yml
```

### 4. 内存不足

**问题:** 安装过程中内存溢出

**解决方案:**
```bash
# 分步安装核心包
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html
    
# 然后安装其他包
pip install -r requirements.txt
```

### 5. Windows 特定问题

**问题:** Visual C++ 构建工具缺失

**解决方案:**
- 下载安装 [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- 或使用预编译轮子 (已在 requirements.txt 中指定)

---

## 📝 环境验证

### 完整验证脚本

创建 `verify_env.py`:

```python
import sys
import torch
import torchvision
import cv2
import numpy as np
import PIL
import matplotlib
import wandb
import omegaconf

print("=" * 70)
print("CWordClearer Environment Verification")
print("=" * 70)

# Python
print(f"Python: {sys.version}")

# PyTorch
print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# Core packages
print(f"\ntorchvision: {torchvision.__version__}")
print(f"OpenCV: {cv2.__version__}")
print(f"NumPy: {np.__version__}")
print(f"Pillow: {PIL.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")

# Model-specific
try:
    import einops
    import timm
    import lpips
    print(f"\neinops: {einops.__version__}")
    print(f"timm: {timm.__version__}")
    print(f"lpips: {lpips.__version__}")
except ImportError as e:
    print(f"\nWarning: {e}")

# Experiment tracking
print(f"\nwandb: {wandb.__version__}")
print(f"omegaconf: {omegaconf.__version__}")

# Simple GPU test
if torch.cuda.is_available():
    print("\n" + "=" * 70)
    print("GPU Test")
    print("=" * 70)
    x = torch.rand(1000, 1000).cuda()
    y = torch.rand(1000, 1000).cuda()
    z = x @ y
    print(f"Matrix multiplication test: {'PASSED' if z.shape == (1000, 1000) else 'FAILED'}")

print("\n" + "=" * 70)
print("Verification Complete!")
print("=" * 70)
```

运行验证:

```bash
conda activate cwordclearer
python verify_env.py
```

**预期输出示例:**

```
======================================================================
CWordClearer Environment Verification
======================================================================
Python: 3.8.20 (default, ...)

PyTorch: 1.10.1+cu113
CUDA Available: True
CUDA Version: 11.3
GPU Count: 1
GPU Name: NVIDIA GeForce RTX 3080

torchvision: 0.11.2+cu113
OpenCV: 4.12.0.88
NumPy: 1.24.4
Pillow: 10.4.0
Matplotlib: 3.3.4

einops: 0.8.1
timm: 0.6.5
lpips: 0.1.4

wandb: 0.14.0
omegaconf: 2.1.1

======================================================================
GPU Test
======================================================================
Matrix multiplication test: PASSED

======================================================================
Verification Complete!
======================================================================
```

---

## 🔄 环境导出

### 导出当前环境 (未来更新用)

如果您对环境进行了修改，可以重新导出:

```bash
# 导出 conda 环境
conda env export > environment_new.yml

# 导出 pip 包
pip list --format=freeze > requirements_new.txt

# 更精简的导出 (仅用户安装的包)
pip freeze > requirements_frozen.txt
```

---

## 📚 相关文档

- **README.md** - 项目主文档
- **ENVIRONMENT_SETUP.md** - 详细环境配置指南
- **TRAINING_GUIDE.md** - 训练指南
- **CHECKPOINT_NAMING.md** - 检查点命名规范

---

## 🎯 总结

### 更新要点

1. ✅ **完整性**: 包含所有 140+ 个包及精确版本
2. ✅ **组织性**: 按功能分类，注释清晰
3. ✅ **可复现**: 固定版本号确保环境一致
4. ✅ **易用性**: 提供多种安装方法
5. ✅ **项目化**: 环境名称改为 `cwordclearer`

### 下一步

```bash
# 1. 创建新环境
conda env create -f environment.yml
conda activate cwordclearer

# 2. 验证环境
python verify_env.py

# 3. 开始训练
python main.py --batch_size=8 --mask --rec="aster" --srb=1
```

---

**维护者:** Minghao Lee (t330034027@mail.uic.edu.cn)  
**最后更新:** 2025年11月20日
