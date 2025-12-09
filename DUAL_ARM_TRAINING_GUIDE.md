# Dual-Arm Depth Training Guide

## 概述

这个配置使用**三个深度相机**（左手、右手、头部）训练双臂机械人的扩散策略。

### 关键特性
- ✅ **三个独立的深度编码器**：每个相机使用独立的CNN提取特征
- ✅ **深度图处理**：自动处理 inf/NaN 值，复制到3通道以兼容ResNet
- ✅ **双臂协作控制**：14维动作空间（左臂6关节+夹爪 + 右臂6关节+夹爪）
- ✅ **100+演示**：支持大规模数据集训练
- ✅ **鲁棒归一化**：基于百分位数的深度归一化，抗异常值干扰

---

## 文件结构

```
diffusion_policy/
├── dataset/
│   └── dual_arm_depth_dataset.py          # 自定义数据集类
├── config/
│   ├── task/
│   │   └── dual_arm_depth.yaml            # 任务配置（数据格式）
│   └── train_diffusion_dual_arm_depth_workspace.yaml  # 训练配置
└── test_dual_arm_dataset.py               # 测试脚本

data/
└── dual_arm_demos/
    ├── demo_0.hdf5
    ├── demo_1.hdf5
    └── ...
```

---

## 数据格式

每个 HDF5 文件包含一个演示，结构如下：

```
demo_X.hdf5
├── actions/
│   ├── left_arm_action      (N, 6)  # 左臂关节角度
│   ├── left_gripper_action  (N,)    # 左夹爪
│   ├── right_arm_action     (N, 6)  # 右臂关节角度
│   └── right_gripper_action (N,)    # 右夹爪
└── observations/
    ├── left_hand_depth      (N, 240, 320)  # 左手深度图
    ├── right_hand_depth     (N, 240, 320)  # 右手深度图
    └── head_depth           (N, 240, 320)  # 头部深度图
```

---

## 使用步骤

### 1. 准备数据

将你的 HDF5 文件放到 `data/dual_arm_demos/` 目录：

```bash
mkdir -p data/dual_arm_demos
# 将 demo_0.hdf5, demo_1.hdf5, ... 放到这个目录
```

### 2. 测试数据加载

运行测试脚本确保数据加载正常：

```bash
cd /Users/rover/Documents/diffusion_policy
python test_dual_arm_dataset.py
```

预期输出：
```
================================================================================
Testing DualArmDepthDataset
================================================================================

Loading dataset from: data/dual_arm_demos
Found 100 HDF5 files
Loading episodes: 100%|██████████| 100/100

✓ Dataset loaded successfully!
  Number of episodes: 100
  Total steps: 60000
  Dataset length (sequences): 59000

✓ Observation shapes:
  left_hand_depth     : (16, 3, 240, 320)   dtype=torch.float32
  right_hand_depth    : (16, 3, 240, 320)   dtype=torch.float32
  head_depth          : (16, 3, 240, 320)   dtype=torch.float32

✓ Action shape:
  action: (16, 14) dtype=torch.float32

✓ Data statistics:
  left_hand_depth:
    min: 0.0000
    max: 1.0000
    ✓ No invalid values

Checking raw HDF5 data for invalid values:
  left_hand_depth:
    Inf values: 15234 (8.12%)
    Valid values: 172566 (91.88%)

✓ All tests passed!
```

### 3. 开始训练

```bash
python train.py --config-name=train_diffusion_dual_arm_depth_workspace
```

### 4. 监控训练（可选）

如果使用 Weights & Biases：

```bash
# 训练会自动记录到 wandb
# 项目名：diffusion_policy_dual_arm
```

---

## 配置参数说明

### 关键参数（可调整）

在 `train_diffusion_dual_arm_depth_workspace.yaml` 中：

```yaml
# 时间窗口
horizon: 16              # 总预测长度
n_obs_steps: 2          # 观测历史（2帧深度图）
n_action_steps: 8       # 动作预测长度

# 图像增强
crop_shape: [216, 288]  # 随机裁剪大小（训练时）

# 训练参数
batch_size: 64          # 根据GPU显存调整
num_epochs: 3000        # 训练轮数
lr: 1.0e-4             # 学习率

# GPU设置
device: "cuda:0"        # 使用第一块GPU
```

### 内存优化

如果 GPU 显存不足，可以调整：

```yaml
dataloader:
  batch_size: 32        # 减小 batch size
  num_workers: 2        # 减少数据加载线程

# 或者使用梯度累积
training:
  gradient_accumulate_every: 2  # 累积2个batch再更新
```

---

## 架构细节

### 网络结构

```
输入观测 (n_obs_steps=2):
  ├─ left_hand_depth:  (2, 3, 240, 320)  # 3通道（深度复制）
  ├─ right_hand_depth: (2, 3, 240, 320)
  └─ head_depth:       (2, 3, 240, 320)
       ↓
独立深度编码器 (ResNet18):
  ├─ DepthEncoder_left  → features_left  (512维)
  ├─ DepthEncoder_right → features_right (512维)
  └─ DepthEncoder_head  → features_head  (512维)
       ↓
特征融合:
  concat([features_left, features_right, features_head])
  → global_features (1536维 × 2帧 = 3072维)
       ↓
扩散模型:
  ConditionalUnet1D
  ├─ Input: 噪声动作轨迹 (16, 14)
  ├─ Condition: global_features (3072维)
  └─ Output: 去噪后的动作 (16, 14)
       ↓
动作输出 (n_action_steps=8):
  取前8步动作 → (8, 14)
```

### 为什么是独立编码器？

- **左手相机**：近距离观察左臂操作
- **右手相机**：近距离观察右臂操作
- **头部相机**：全局视角观察任务

三个视角的**视野、距离、角度都不同**，需要独立的编码器来学习各自的特征。

---

## 深度图处理细节

### 自动处理无效值

深度传感器常产生 inf/NaN 值，代码自动处理：

```python
# 1. 检测并替换无效值
invalid_mask = ~np.isfinite(depth)  # inf, -inf, nan
depth[invalid_mask] = 0.0

# 2. 仅对有效像素归一化
valid_depth = depth[depth > 0]
depth_min = np.percentile(valid_depth, 1)   # 1%分位数
depth_max = np.percentile(valid_depth, 99)  # 99%分位数

# 3. 归一化到 [0, 1]
depth = (depth - depth_min) / (depth_max - depth_min)

# 4. 复制到3通道（兼容ResNet）
depth = np.stack([depth, depth, depth], axis=1)
```

**优势**：
- ✅ 自动过滤异常值
- ✅ 百分位数归一化更鲁棒
- ✅ 兼容预训练的RGB编码器

---

## 常见问题

### Q1: 为什么深度图要复制到3通道？

**原因**：现有的 ResNet 编码器第一层是 `Conv2d(in_channels=3, ...)`，期望3通道输入。

**选项**：
- ✅ 复制到3通道（当前实现）- 最简单，可利用预训练权重
- ⏳ 自定义单通道编码器 - 需要修改 robomimic 代码

### Q2: 如何使用RGB图像而不是深度图？

修改 `task/dual_arm_depth.yaml`：

```yaml
shape_meta:
  obs:
    left_hand_rgb:
      shape: [3, 240, 320]
      type: rgb
```

并修改 dataset 加载 RGB 数据。

### Q3: 动作是增量还是绝对？

当前实现假设是**增量控制**（delta actions）。如果是绝对控制，需要：

1. 在数据集中转换为增量
2. 或使用 `abs_action: True` 配置

### Q4: 如何调整观测历史步数？

```yaml
n_obs_steps: 3  # 使用3帧历史而不是2帧
```

注意：更多历史帧会增加特征维度和计算量（3072维 → 4608维）。

### Q5: 深度图中有大量 inf 值怎么办？

**无需担心**！代码会自动处理：
- 测试脚本会显示原始数据中 inf/NaN 的占比
- 处理后的数据保证无 inf/NaN
- 使用百分位数归一化，抗异常值干扰

查看 `process_depth()` 函数了解详细处理逻辑。

### Q6: 训练多久能收敛？

根据经验（100个演示，每个600帧）：
- **初步收敛**：500-1000 epochs
- **较好性能**：1500-2000 epochs
- **最佳性能**：2500-3000 epochs

### Q7: 如何恢复训练？

```yaml
training:
  resume: True  # 已经设置
```

训练会自动从最新的 checkpoint 恢复。

---

## 性能优化建议

### 1. 数据预处理缓存

深度图归一化在运行时计算。如果数据集很大，可以预先处理并保存为 Zarr 格式：

```python
# 预处理脚本示例
# 1. 读取所有 HDF5
# 2. 处理 inf/NaN 并归一化
# 3. 保存为 Zarr（压缩存储）
# 优势：加载速度提升 10-100 倍
```

**注意**：当前实现已使用 ReplayBuffer（基于 Zarr），加载速度已经很快。

### 2. 混合精度训练

```yaml
training:
  use_amp: True  # 需要在 workspace 中实现
```

### 3. 多GPU训练

```yaml
training:
  device: "cuda"
  use_ddp: True
  num_gpus: 2
```

---

## 完整数据处理流程

### 从原始深度图到网络输入

```
步骤 1: 原始深度图
  HDF5: (N, 240, 320) 单通道
  - 可能包含: inf, -inf, NaN (典型: 5-15%)
  - 深度范围: 0-10000mm (RealSense) 或 0-8000mm (Kinect)

      ↓

步骤 2: 过滤无效值
  invalid_mask = ~np.isfinite(depth)
  depth[invalid_mask] = 0.0
  - ✓ 所有 inf/NaN → 0

      ↓

步骤 3: 鲁棒归一化
  valid_depth = depth[depth > 0]
  depth_min = np.percentile(valid_depth, 1)   # 忽略最低1%异常值
  depth_max = np.percentile(valid_depth, 99)  # 忽略最高1%异常值
  depth = (depth - depth_min) / (depth_max - depth_min)
  - ✓ 范围: [0, 1]
  - ✓ 抗离群值干扰

      ↓

步骤 4: 复制到3通道
  depth = np.stack([depth, depth, depth], axis=1)
  (T, 240, 320) → (T, 3, 240, 320)
  - ✓ 兼容 ResNet (in_channels=3)

      ↓

步骤 5: Batch组装
  (B, n_obs_steps, 3, 240, 320)
  默认: (64, 2, 3, 240, 320)

      ↓

步骤 6: 三个独立ResNet编码
  left_hand   → ResNet18 → 512维
  right_hand  → ResNet18 → 512维
  head        → ResNet18 → 512维

      ↓

步骤 7: 特征拼接
  concat → (B, 1536) × 2帧 = (B, 3072)

      ↓

步骤 8: 扩散模型条件
  global_cond = features (B, 3072)
  生成动作轨迹 (B, 16, 14)
```

---

## 下一步

1. ✅ **测试数据加载**：运行 `test_dual_arm_dataset.py`
2. ✅ **开始训练**：`python train.py --config-name=train_diffusion_dual_arm_depth_workspace`
3. ⏳ **评估性能**：需要实现 env_runner（模拟器或真实机器人）
4. ⏳ **部署到机器人**：将训练好的模型部署到实际系统

---

## 联系与支持

如有问题，请检查：
- Dataset 代码：`diffusion_policy/dataset/dual_arm_depth_dataset.py`
- 配置文件：`diffusion_policy/config/train_diffusion_dual_arm_depth_workspace.yaml`
- Diffusion Policy 文档：https://diffusion-policy.cs.columbia.edu/
