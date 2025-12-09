# 双臂深度图训练 - 项目就绪总结

## ✅ 所有文件已准备完成

### 核心文件

| 文件 | 状态 | 说明 |
|------|------|------|
| [dual_arm_depth_dataset.py](diffusion_policy/dataset/dual_arm_depth_dataset.py) | ✅ | 数据集类 |
| [dual_arm_depth.yaml](diffusion_policy/config/task/dual_arm_depth.yaml) | ✅ | 任务配置 |
| [train_diffusion_dual_arm_depth_workspace.yaml](diffusion_policy/config/train_diffusion_dual_arm_depth_workspace.yaml) | ✅ | 训练配置（双GPU） |
| [test_dual_arm_dataset.py](test_dual_arm_dataset.py) | ✅ | 测试脚本 |

### 文档文件

| 文件 | 内容 |
|------|------|
| [DUAL_ARM_TRAINING_GUIDE.md](DUAL_ARM_TRAINING_GUIDE.md) | 完整训练指南 |
| [MULTI_GPU_SETUP.md](MULTI_GPU_SETUP.md) | 双GPU配置说明 |
| [TRAINING_PROCESS.md](TRAINING_PROCESS.md) | 训练过程详解（本文档） |

---

## 📊 训练时会有什么

### 自动记录的指标

#### 1. **每个训练步** (每个batch)
- `train_loss`: 训练损失
- `lr`: 学习率
- `global_step`: 全局步数

#### 2. **验证阶段** (每1个epoch)
- `val_loss`: 验证集损失
- 用于监控过拟合

#### 3. **动作采样** (每5个epoch)
- `train_action_mse_error`: 预测动作MSE
- 检查模型预测质量

#### 4. **Checkpoint保存** (每50个epoch)
- 保存最新模型
- 保存Top-5最佳模型

### ⚠️ 没有环境评估 (Rollout)

**原因**：你提供的是离线数据集，没有仿真环境

**影响**：无影响！对于离线学习，validation loss + action MSE已经足够

**最终评估**：训练完成后在真实机器人上测试

---

## 🚀 如何开始训练

### 步骤1: 测试数据加载

```bash
python test_dual_arm_dataset.py
```

**预期输出**：
```
================================================================================
Testing DualArmDepthDataset
================================================================================

Loading dataset from: data/dual_arm_demos
Found 100 HDF5 files
Loading episodes: 100%|████████████| 100/100 [00:15<00:00]

✓ Dataset loaded successfully!
  Number of episodes: 100
  Total steps: 60000
  Dataset length (sequences): 59984

✓ Validation split:
  Train sequences: 58784
  Val sequences: 1200

✓ Sample structure:
  Keys: ['obs', 'action']

✓ Observation shapes:
  left_hand_depth     : (2, 3, 240, 320)    dtype=torch.float32
  right_hand_depth    : (2, 3, 240, 320)    dtype=torch.float32
  head_depth          : (2, 3, 240, 320)    dtype=torch.float32

✓ Action shape:
  action: (16, 14) dtype=torch.float32

✓ No invalid values (inf/NaN已处理)
================================================================================
✓ All tests passed!
================================================================================
```

### 步骤2: 开始训练（双GPU）

```bash
python train.py --config-name=train_diffusion_dual_arm_depth_workspace
```

**预期输出**：
```
Using DataParallel with GPUs: [0, 1]
Found 100 HDF5 files
Loading episodes: 100%|████████████| 100/100
Loaded 100 episodes, 60000 total steps

wandb: Run URL: https://wandb.ai/your-username/diffusion_policy_dual_arm/runs/xxxxx

Training epoch 0: 100%|████████| 469/469 [02:15<00:00, loss=0.234]
Validation epoch 0: 100%|████████| 10/10 [00:05<00:00]

Epoch 0: train_loss=0.234, val_loss=0.245

Training epoch 1: ...
```

### 步骤3: 监控训练

打开WandB链接，实时查看：
- Loss曲线（train vs val）
- 学习率变化
- 动作MSE
- GPU使用率

---

## 📈 训练监控指标

### 健康的训练曲线

```
Loss
  │
0.5│╲
   │ ╲           训练loss快速下降
0.3│  ╲___
   │      ╲___   验证loss跟随下降
0.1│          ╲______  两者差距小（没有过拟合）
   │                 ╲_____
0.0│                       ╲____
   └────────────────────────────→ Epoch
   0   500   1000  1500   2000
```

### 异常情况

#### 过拟合
```
Loss
  │
0.5│╲
   │ ╲           训练loss持续下降
0.3│  ╲___
   │      ╲___
0.1│          ╲  验证loss开始上升 ⚠️
   │           ╱
0.2│          ╱  差距变大 ⚠️
   │         ╱
   └────────────────→ Epoch
```

**解决方案**：
- 增加数据增强
- 减小模型容量
- 早停（使用验证loss最低的checkpoint）

#### 欠拟合
```
Loss
  │
0.5│╲
   │ ╲____
   │      ╲____  两者都没怎么下降 ⚠️
0.3│           ╲____
   │                ╲____
0.2│                     ╲____
   └────────────────────────→ Epoch
```

**解决方案**：
- 增加训练时间
- 增大模型容量
- 调整学习率

---

## 🎯 训练目标

### 收敛标准

建议在以下条件满足时停止训练：

1. ✅ `val_loss < 0.02` 且稳定
2. ✅ `train_action_mse_error < 0.01`
3. ✅ `train_loss`和`val_loss`差距 < 20%
4. ✅ 连续50个epoch无明显改善

### 预期训练时长

根据你的数据规模（100 episodes × 600 frames）：

| GPU配置 | 预计时长（3000 epochs） | 建议 |
|---------|------------------------|------|
| RTX 3090 × 2 | 30-40小时 | 先训练500 epochs |
| RTX 4090 × 2 | 20-30小时 | 先训练500 epochs |
| A100 × 2 | 15-25小时 | 先训练1000 epochs |

**实用建议**：
- 先训练500 epochs（~5-8小时）
- 检查val_loss是否收敛
- 如果收敛，可以提前停止
- 如果还在下降，继续训练

---

## 💾 Checkpoint管理

### 保存策略

训练会自动保存：

1. **最新checkpoint**: `data/outputs/*/checkpoints/latest.ckpt`
   - 每50个epoch更新
   - 用于断点续训

2. **Top-5最佳checkpoint**:
   - `epoch=0100-train_loss=0.123.ckpt`
   - `epoch=0200-train_loss=0.098.ckpt`
   - `epoch=0350-train_loss=0.067.ckpt`
   - `epoch=0500-train_loss=0.045.ckpt`
   - `epoch=0750-train_loss=0.023.ckpt` ← **最佳！**

### 断点续训

如果训练中断，重新运行命令会自动从`latest.ckpt`恢复：

```bash
python train.py --config-name=train_diffusion_dual_arm_depth_workspace

# 输出：
# Resuming from checkpoint .../checkpoints/latest.ckpt
# Resuming from epoch 123
```

### 加载最佳模型

```python
from hydra import compose, initialize
from diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace import TrainDiffusionUnetHybridWorkspace

# 加载配置
with initialize(config_path="diffusion_policy/config"):
    cfg = compose(config_name="train_diffusion_dual_arm_depth_workspace")

# 创建workspace
workspace = TrainDiffusionUnetHybridWorkspace(cfg)

# 加载最佳checkpoint
workspace.load_checkpoint('data/outputs/.../checkpoints/epoch=0750-train_loss=0.023.ckpt')

# 使用EMA模型（通常效果更好）
policy = workspace.ema_model
policy.eval()

# 预测
with torch.no_grad():
    action = policy.predict_action(obs_dict)
```

---

## 🔧 常见调整

### 如果显存不足 (OOM)

编辑配置文件，降低batch_size：

```yaml
dataloader:
  batch_size: 64   # 从128降到64
  num_workers: 8

val_dataloader:
  batch_size: 64
  num_workers: 8
```

或使用梯度累积：
```yaml
training:
  gradient_accumulate_every: 2  # 累积2个batch
dataloader:
  batch_size: 64  # 实际等效batch_size = 64 × 2 = 128
```

### 如果训练太慢

1. **检查数据加载**：
```yaml
dataloader:
  num_workers: 16  # 增加到16
```

2. **减少验证频率**：
```yaml
training:
  val_every: 5      # 从1改到5
  sample_every: 10  # 从5改到10
```

### 如果想加快收敛

```yaml
optimizer:
  lr: 2.0e-4  # 从1.0e-4提高到2.0e-4

training:
  lr_warmup_steps: 200  # 从500降到200
```

⚠️ 注意：过高的学习率可能导致不稳定

---

## 📝 训练日志位置

所有训练输出保存在：
```
data/outputs/2025.12.09/14.30.15_train_diffusion_dual_arm_depth_dual_arm_depth/
├── checkpoints/
│   ├── latest.ckpt
│   ├── epoch=0050-train_loss=0.123.ckpt
│   ├── epoch=0100-train_loss=0.098.ckpt
│   └── ...
├── logs.json.txt           # JSON格式日志
└── wandb/                  # WandB日志
```

---

## ✅ 准备清单

在开始训练前，确认：

- [x] 数据集放在 `data/dual_arm_demos/`
- [x] HDF5文件格式正确（包含actions和observations）
- [x] 运行`test_dual_arm_dataset.py`通过
- [x] 有两个GPU可用
- [x] 足够的磁盘空间（checkpoint会占用~5-10GB）
- [x] WandB账号登录（或设置`mode: offline`）

---

## 🎉 你已经准备好了！

所有代码和配置都已优化完成。现在只需要：

1. 准备数据
2. 运行测试脚本
3. 开始训练
4. 监控WandB
5. 加载最佳checkpoint
6. 部署到机器人测试

祝训练顺利！🚀
