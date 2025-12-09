# 双GPU训练配置说明

## 已完成的配置

### 1. 配置文件修改

**文件**: `diffusion_policy/config/train_diffusion_dual_arm_depth_workspace.yaml`

已添加以下配置：

```yaml
training:
  device: "cuda:0"
  multi_gpu: True      # 启用多GPU训练
  gpu_ids: [0, 1]      # 使用GPU 0和1
  ...

dataloader:
  batch_size: 128      # 每个GPU 64个样本
  num_workers: 8       # 增加数据加载线程
  ...

val_dataloader:
  batch_size: 128
  num_workers: 8
  ...
```

### 2. 代码修改

**文件**: `diffusion_policy/workspace/train_diffusion_unet_hybrid_workspace.py`

在第138-147行添加了DataParallel支持：

```python
# Multi-GPU support with DataParallel
if cfg.training.get('multi_gpu', False) and torch.cuda.device_count() > 1:
    gpu_ids = cfg.training.get('gpu_ids', None)
    if gpu_ids is not None:
        print(f"Using DataParallel with GPUs: {gpu_ids}")
        self.model = torch.nn.DataParallel(self.model, device_ids=gpu_ids)
    else:
        print(f"Using DataParallel with all available GPUs: {torch.cuda.device_count()}")
        self.model = torch.nn.DataParallel(self.model)
```

---

## 使用方法

### 开始训练（使用双GPU）

```bash
python train.py --config-name=train_diffusion_dual_arm_depth_workspace
```

训练开始时会看到类似输出：
```
Using DataParallel with GPUs: [0, 1]
```

### 监控GPU使用

在另一个终端运行：
```bash
watch -n 1 nvidia-smi
```

应该能看到两个GPU都在使用，显存占用相近。

---

## 性能优化建议

### 1. Batch Size调整

根据GPU显存调整batch_size：
- **16GB显存**: `batch_size: 128` (每个GPU 64)
- **24GB显存**: `batch_size: 192-256` (每个GPU 96-128)
- **12GB显存**: `batch_size: 64-96` (每个GPU 32-48)

如果遇到OOM (Out of Memory)错误，降低batch_size。

### 2. Num Workers调整

`num_workers`应该设置为：
- 单GPU: `4-8`
- 双GPU: `8-16` (当前配置: 8)

太多workers会占用过多CPU和内存。

### 3. Gradient Accumulation

如果显存不足，可以使用梯度累积：
```yaml
training:
  gradient_accumulate_every: 2  # 累积2个batch再更新
```

并相应减小batch_size。

---

## 性能对比

### 预期加速比

- **单GPU**: 1x基准速度
- **双GPU (DataParallel)**: 1.7-1.8x (不是完美的2x，因为通信开销)

### 训练时间估算

假设100个episode，每个600帧，3000 epochs：

- **单GPU**: ~48-72小时（取决于GPU型号）
- **双GPU**: ~27-40小时

---

## 故障排除

### 1. 如果只看到一个GPU在工作

检查配置：
```yaml
training:
  multi_gpu: True
  gpu_ids: [0, 1]
```

### 2. 如果遇到CUDA out of memory

降低batch_size：
```yaml
dataloader:
  batch_size: 64  # 从128降到64
```

### 3. 如果想临时使用单GPU训练

修改配置：
```yaml
training:
  multi_gpu: False  # 禁用多GPU
  device: "cuda:0"  # 只使用GPU 0
```

或者使用命令行覆盖：
```bash
python train.py --config-name=train_diffusion_dual_arm_depth_workspace \
    training.multi_gpu=False training.device=cuda:0
```

### 4. 选择特定GPU

如果有多个GPU但只想用其中某几个：
```yaml
training:
  gpu_ids: [1, 2]  # 只使用GPU 1和2
```

或使用环境变量：
```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py --config-name=train_diffusion_dual_arm_depth_workspace
```

---

## 技术细节

### DataParallel vs DistributedDataParallel

当前使用**DataParallel**：
- ✅ 优点：配置简单，单进程
- ❌ 缺点：有通信瓶颈，加速比不是线性的

如果需要更好的性能，可以升级到**DistributedDataParallel** (DDP)：
- ✅ 优点：接近线性加速，更高效
- ❌ 缺点：需要多进程，配置更复杂

### Checkpoint兼容性

- 双GPU训练保存的checkpoint可以在单GPU上加载（需要设置`multi_gpu=False`）
- 单GPU训练保存的checkpoint可以在双GPU上加载
- PyTorch会自动处理`module.`前缀的差异

---

## 验证多GPU是否工作

在训练脚本中添加以下代码验证：

```python
# 在训练开始后
print(f"Model device: {next(self.model.parameters()).device}")
if hasattr(self.model, 'module'):
    print("Model is wrapped in DataParallel")
    print(f"Using devices: {self.model.device_ids}")
```

---

## 推荐配置总结

**推荐使用当前配置（已设置）**：

```yaml
training:
  multi_gpu: True
  gpu_ids: [0, 1]

dataloader:
  batch_size: 128
  num_workers: 8
```

这个配置在两个GPU上应该能获得约1.7-1.8x的训练加速。
