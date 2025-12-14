# 双臂深度图训练指南

## 快速开始

### 1. 数据准备

```bash
# 转换HDF5为Memmap格式（推荐）
python scripts/convert_hdf5_to_mmap.py \
  --input-dir data/dual_arm_demos \
  --output-dir data/dual_arm_demos_mmap

# 验证数据
python test_dual_arm_dataset.py
```

### 2. 启动训练

```bash
# 第一次训练
CUDA_VISIBLE_DEVICES=1 python train.py \
  --config-dir=diffusion_policy/config \
  --config-name=train_diffusion_dual_arm_depth_mmap_workspace.yaml
```

预期输出：
```
Training epoch 0: 100%|████| 475/475 [10:32<00:00, loss=0.0435]
Training epoch 1: 100%|████| 475/475 [10:28<00:00, loss=0.0298]
```

---

## 继续训练（关键！）

**❌ 错误做法**（会重新开始）：
```bash
python train.py --config-name=train_diffusion_dual_arm_depth_mmap_workspace.yaml
# 这样会创建新的时间戳目录，loss会重新变高！
```

**✅ 正确做法**（继续从checkpoint）：
```bash
# 1. 找到你的run目录
ls data/outputs/2025.12.12/
# → 16.00.12_train_diffusion_dual_arm_depth_mmap_dual_arm_depth_mmap

# 2. 在原目录继续训练
CUDA_VISIBLE_DEVICES=1 python train.py \
  --config-dir=diffusion_policy/config \
  --config-name=train_diffusion_dual_arm_depth_mmap_workspace.yaml \
  hydra.run.dir=data/outputs/2025.12.12/16.00.12_train_diffusion_dual_arm_depth_mmap_dual_arm_depth_mmap \
  hydra.output_subdir=null
```

这样会：
- ✅ 自动加载 `latest.ckpt`
- ✅ 从 epoch 101 继续（不是 epoch 0）
- ✅ WandB日志连续

---

## 数据配置（已优化）

当前配置已针对RTX 6000 Ada优化：

```yaml
dataloader:
  batch_size: 128
  num_workers: 3              # 加速数据加载
  persistent_workers: False  # ⭐ 防止内存累积（关键）
  prefetch_factor: 2         # ⭐ 适度预加载（关键）
  pin_memory: True
```

**为什么这样配置**：
- `num_workers: 3` → CPU并行加载数据
- `persistent_workers: False` → worker用完立即释放（防止12-15epoch死机）
- `prefetch_factor: 2` → 足够预加载，不会堆积数据

**如果出现问题**：
- 显存爆满 → 降低 `batch_size` (128→96)
- 系统内存爆炸 → 降低 `prefetch_factor` (2→1)
- 数据加载慢 → 增加 `batch_size` (128→160)

---

## 训练时长

根据RTX 6000 Ada实际测试：

| 进度 | 耗时 | 累计 | loss变化 |
|------|------|------|---------|
| 0-50 epochs | 1.5h | 1.5h | 0.40 → 0.10 |
| 50-100 epochs | 1.5h | 3h | 0.10 → 0.004 |
| 100-500 epochs | 15h | ~18h | 微调和收敛 |

**建议**：
- 先训100 epochs（3小时）检查loss趋势
- loss应该快速下降（不是持续很高）
- 如果loss持续高于0.1，检查数据或配置

---

## 模型保存与加载

### Checkpoint自动保存

```
data/outputs/2025.12.12/16.00.12_.../checkpoints/
├── latest.ckpt                    # 自动加载用这个
├── epoch=0050-train_loss=0.006.ckpt
├── epoch=0100-train_loss=0.004.ckpt  ← 选这个做机器人测试
└── ...
```

### 加载最佳模型

```python
import torch
from diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace import TrainDiffusionUnetHybridWorkspace
from hydra import compose, initialize

with initialize(config_path="diffusion_policy/config", version_base=None):
    cfg = compose(config_name="train_diffusion_dual_arm_depth_mmap_workspace")

workspace = TrainDiffusionUnetHybridWorkspace(cfg)

# 加载checkpoint
ckpt_path = "data/outputs/2025.12.12/.../epoch=0100-train_loss=0.004.ckpt"
checkpoint = torch.load(ckpt_path)
workspace.load_state_dict(checkpoint['state_dict'])

# 使用EMA模型预测（推荐）
policy = workspace.ema_model
policy.eval()

with torch.no_grad():
    obs_dict = {
        'left_hand_depth': torch.zeros(1, 2, 3, 240, 320).cuda(),
        'right_hand_depth': torch.zeros(1, 2, 3, 240, 320).cuda(),
        'head_depth': torch.zeros(1, 2, 3, 240, 320).cuda(),
    }
    action = policy.predict_action(obs_dict)
```

---

## 后台运行（推荐）

```bash
# 启动tmux session
tmux new-session -d -s train "cd /home/zzx/diffusion_policy && \
  CUDA_VISIBLE_DEVICES=1 python train.py \
  --config-dir=diffusion_policy/config \
  --config-name=train_diffusion_dual_arm_depth_mmap_workspace.yaml"

# 查看日志
tmux attach -t train

# 查看GPU（另一个终端）
watch -n 2 nvidia-smi
```

---

## 关键配置文件

| 文件 | 说明 |
|------|------|
| [train_diffusion_dual_arm_depth_mmap_workspace.yaml](diffusion_policy/config/train_diffusion_dual_arm_depth_mmap_workspace.yaml) | 训练配置（核心） |
| [dual_arm_depth_mmap.yaml](diffusion_policy/config/task/dual_arm_depth_mmap.yaml) | 任务配置 |
| [dual_arm_depth_mmap_dataset.py](diffusion_policy/dataset/dual_arm_depth_mmap_dataset.py) | Memmap数据集 |

---

## 故障排查

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| loss重新变高 | 没从checkpoint恢复 | 用`hydra.run.dir`指定原目录 |
| 12-15 epoch死机 | `persistent_workers: True` | 改为`False` |
| 显存爆满 | batch_size太大 | 改为96 |
| 系统内存爆满 | prefetch_factor太高 | 改为1 |
| 数据加载慢 | num_workers或batch太小 | 增加num_workers或batch_size |

---

## 为什么用Memmap

| 格式 | 优点 | 缺点 |
|------|------|------|
| HDF5全加载 | 速度快 | 大数据集OOM |
| HDF5懒加载 | 通用 | CPU瓶颈，加载慢 |
| **Memmap** | **零拷贝，高效** | 需要预转换 |

105个HDF5文件（94GB）转为26.59GB memmap，训练速度1.3秒/批次。

---

## 实用命令速查

```bash
# 查看最新checkpoint修改时间
stat data/outputs/2025.12.12/*/checkpoints/latest.ckpt | grep Modify

# 查找所有checkpoint
find data/outputs -name "*.ckpt" -type f | sort

# 查看当前GPU状态
nvidia-smi -l 1

# 杀死训练
pkill -f "train.py"

# 查看WandB日志
ls data/outputs/2025.12.12/*/wandb/
```

---

## 检查清单

- [ ] 数据转换完成：`convert_hdf5_to_mmap.py`
- [ ] 数据验证通过：`test_dual_arm_dataset.py`
- [ ] GPU可用：`nvidia-smi`
- [ ] 磁盘空间：≥50GB（数据+checkpoints）
- [ ] WandB登录：`wandb login`

---

**🎉 准备好了？开始训练！**

```bash
CUDA_VISIBLE_DEVICES=1 python train.py \
  --config-dir=diffusion_policy/config \
  --config-name=train_diffusion_dual_arm_depth_mmap_workspace.yaml
```

预计 18-20 小时完成 500 epochs。
