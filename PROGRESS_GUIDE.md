# 训练进度监控指南 📊

## 你将看到的进度输出

### 1. 数据加载进度

```
Loading data from /kaggle/working/processed/product_daily_panel.parquet...
Train split: 487 days, 2016-10-03 00:00:00 to 2018-02-01 00:00:00
Creating sequences for 12,543 products...
Processing products: 100%|██████████| 12543/12543 [00:45<00:00, 278.51it/s]
Created 880,743 sequences for train split
```

### 2. 训练进度（每个 Epoch）

```
============================================================
EPOCH 1/3
============================================================
  Step 100/1376 (7.3%) | D_loss: 0.0294 | G_loss: -2.6907
  Step 200/1376 (14.5%) | D_loss: 0.0735 | G_loss: -2.0942
  Step 300/1376 (21.8%) | D_loss: 0.0320 | G_loss: -2.2840
  ...
  Step 1300/1376 (94.5%) | D_loss: 0.0145 | G_loss: -2.1234

  Running validation...
  ✓ Validation MAE: 0.0166

────────────────────────────────────────────────────────────
Epoch 1 completed in 18.5 minutes
  Final D_loss: 0.0234
  Final G_loss: -2.1567
  ETA for completion: 37.0 minutes (0.6 hours)
```

### 3. Checkpoint 保存通知

```
Epoch 1: 100%|██████████| 1376/1376 [18:30<00:00, 1.24it/s]
Saving checkpoint to /kaggle/working/checkpoints/gan-epoch01-gloss-2.157.ckpt
```

### 4. 完成信息

```
============================================================
✅ TRAINING COMPLETED!
============================================================
Total epochs: 3
Best checkpoint saved

📁 Saved 4 checkpoint(s):
  ✓ gan-epoch01-gloss-2.157.ckpt (27.6 MB)
  ✓ gan-epoch02-gloss-2.089.ckpt (27.6 MB)
  ✓ gan-epoch03-gloss-1.945.ckpt (27.6 MB)
  ✓ last.ckpt (27.6 MB)

🏆 Best checkpoint: gan-epoch03-gloss-1.945.ckpt
```

---

## 实时监控方法

### 方法 1: 直接看输出

训练时你会看到：
- ✅ 每 100 steps 显示一次进度
- ✅ 当前 D_loss 和 G_loss
- ✅ 每个 epoch 完成时的总结
- ✅ 剩余时间估算（ETA）

### 方法 2: TensorBoard（推荐）

在新的 Cell 中运行：

```python
%load_ext tensorboard
%tensorboard --logdir /kaggle/working/logs
```

你可以看到：
- 📈 Loss 曲线（实时更新）
- 📊 学习率变化
- 🎯 验证指标
- 📉 梯度惩罚值

### 方法 3: 检查文件

训练过程中，在新的 Cell 运行：

```python
# 查看最新的 checkpoint
!ls -lt /kaggle/working/checkpoints/ | head -5

# 查看日志目录
!ls -lh /kaggle/working/logs/cgan_sales/
```

### 方法 4: 监控 GPU

```python
# 查看 GPU 使用情况
!nvidia-smi

# 实时监控（每 5 秒刷新）
!watch -n 5 nvidia-smi
```

---

## 进度显示频率

| 事件 | 频率 | 说明 |
|------|------|------|
| Step 进度 | 每 100 steps | 显示当前 loss |
| Epoch 总结 | 每个 epoch 结束 | 完整统计信息 |
| Validation | 每个 epoch 结束 | VAL MAE 指标 |
| Checkpoint 保存 | 每个 epoch | 自动保存 |
| ETA 更新 | 每个 epoch | 剩余时间估算 |

---

## 理解输出指标

### D_loss (Discriminator Loss)
- **含义**: 判别器的损失
- **期望**: 在 0 附近小范围波动
- **正常范围**: -0.5 到 0.5
- **异常**: 如果 > 1 或 < -1，可能训练不稳定

### G_loss (Generator Loss)
- **含义**: 生成器的损失
- **期望**: 逐渐下降（变得更负）
- **正常范围**: -3 到 -1
- **趋势**: 应该有下降趋势，但可能波动

### VAL MAE (Validation Mean Absolute Error)
- **含义**: 验证集上的平均绝对误差
- **期望**: 越小越好
- **正常范围**: 0.01 到 0.05（归一化数据）
- **趋势**: 应该逐渐下降

### ETA (Estimated Time to Arrival)
- **含义**: 预计剩余时间
- **基于**: 已完成 epoch 的平均时间
- **准确性**: 第一个 epoch 后会更准确

---

## 常见进度模式

### ✅ 正常训练

```
Epoch 1: D_loss: 0.05 → G_loss: -2.5 → VAL MAE: 0.025
Epoch 2: D_loss: 0.03 → G_loss: -2.2 → VAL MAE: 0.022
Epoch 3: D_loss: 0.02 → G_loss: -2.0 → VAL MAE: 0.020
```
**特征**: 损失稳定，验证 MAE 逐渐下降

### ⚠️ 训练不稳定

```
Epoch 1: D_loss: 0.05 → G_loss: -2.5 → VAL MAE: 0.025
Epoch 2: D_loss: 1.50 → G_loss: -8.3 → VAL MAE: 0.055
Epoch 3: D_loss: 2.20 → G_loss: -12.5 → VAL MAE: 0.080
```
**特征**: 损失爆炸，验证 MAE 上升

**解决**: 降低学习率或增加梯度惩罚

### ⚠️ 模式崩溃

```
Epoch 1: D_loss: 0.05 → G_loss: -2.5 → VAL MAE: 0.025
Epoch 2: D_loss: 0.01 → G_loss: -0.5 → VAL MAE: 0.040
Epoch 3: D_loss: 0.00 → G_loss: -0.1 → VAL MAE: 0.050
```
**特征**: G_loss 趋近于 0，验证性能下降

**解决**: 增加 n_critic 或调整学习率比例

---

## 训练中断后恢复

如果训练中断，可以从最新的 checkpoint 恢复：

```python
# 找到最新的 checkpoint
!ls -lt /kaggle/working/checkpoints/ | head -2

# 从 checkpoint 继续训练
!python kaggle_train.py --resume /kaggle/working/checkpoints/last.ckpt
```

---

## 性能优化提示

### 如果训练太慢

```python
# 减少数据量
!python kaggle_train.py --data-fraction 0.5

# 减少 batch size（如果内存不足）
!python kaggle_train.py --batch-size 32

# 减少 epochs
!python kaggle_train.py --epochs 10
```

### 如果想加速测试

```python
# 最快速测试（5分钟）
!python kaggle_train.py --quick --data-fraction 0.05 --epochs 1
```

---

## 进度输出示例（完整）

```
===========================================================
CONFIGURATION - QUICK TEST
===========================================================
  Epochs: 3
  Batch size: 64
  Data fraction: 10%
  Estimated time: ~45 minutes
===========================================================

===========================================================
LOADING DATA
===========================================================
ℹ️  Kaggle environment detected: setting num_workers=0 (was 4)
Loading data from /kaggle/working/processed/product_daily_panel.parquet...
Train split: 487 days, 2016-10-03 00:00:00 to 2018-02-01 00:00:00
Creating sequences for 12,543 products...
Processing products: 100%|██████████| 12543/12543 [00:45<00:00]
Created 880,743 sequences for train split
...

===========================================================
INITIALIZING MODEL
===========================================================

Model parameters:
  Generator: 3,802,241
  Discriminator: 3,086,209
  Total: 6,888,450
  Model size: ~27.6 MB

===========================================================
INITIALIZING TRAINER
===========================================================

Trainer configuration:
  Accelerator: CUDAAccelerator
  Devices: 1
  Max epochs: 3
  Train batches: 10%

===========================================================
STARTING TRAINING
===========================================================

============================================================
EPOCH 1/3
============================================================
  Step 100/138 (72.5%) | D_loss: 0.0294 | G_loss: -2.6907

  Running validation...
  ✓ Validation MAE: 0.0166

────────────────────────────────────────────────────────────
Epoch 1 completed in 2.5 minutes
  Final D_loss: 0.0234
  Final G_loss: -2.1567
  ETA for completion: 5.0 minutes (0.1 hours)

Saving checkpoint: gan-epoch01-gloss-2.157.ckpt

============================================================
EPOCH 2/3
============================================================
  Step 100/138 (72.5%) | D_loss: 0.0189 | G_loss: -2.3456

  Running validation...
  ✓ Validation MAE: 0.0158

────────────────────────────────────────────────────────────
Epoch 2 completed in 2.3 minutes
  Final D_loss: 0.0198
  Final G_loss: -2.0891
  ETA for completion: 2.3 minutes (0.0 hours)

Saving checkpoint: gan-epoch02-gloss-2.089.ckpt

============================================================
EPOCH 3/3
============================================================
  Step 100/138 (72.5%) | D_loss: 0.0145 | G_loss: -2.1234

  Running validation...
  ✓ Validation MAE: 0.0152

────────────────────────────────────────────────────────────
Epoch 3 completed in 2.2 minutes
  Final D_loss: 0.0156
  Final G_loss: -1.9445
  ETA for completion: 0.0 minutes (0.0 hours)

Saving checkpoint: gan-epoch03-gloss-1.945.ckpt

============================================================
✅ TRAINING COMPLETED!
============================================================
Total epochs: 3
Best checkpoint saved

📁 Saved 4 checkpoint(s):
  ✓ gan-epoch01-gloss-2.157.ckpt (27.6 MB)
  ✓ gan-epoch02-gloss-2.089.ckpt (27.6 MB)
  ✓ gan-epoch03-gloss-1.945.ckpt (27.6 MB)
  ✓ last.ckpt (27.6 MB)

🏆 Best checkpoint: gan-epoch03-gloss-1.945.ckpt

📊 View training logs:
  TensorBoard: %tensorboard --logdir /kaggle/working/logs
  Checkpoint dir: /kaggle/working/checkpoints

============================================================
Next steps:
  1. View TensorBoard logs to check training curves
  2. Generate synthetic samples: !python scripts/generate_samples.py
  3. Train baseline models: !python scripts/train_baseline.py
============================================================
```

---

**现在你可以清楚地看到训练的每一步进度了！** 📊✨
