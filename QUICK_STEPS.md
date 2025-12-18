# 快速完成所有步骤 🚀

## 🎯 两种方式

### 方式 A: 一键自动完成（推荐）

```python
# 自动完成步骤 2-5（约 1-2 小时）
!python scripts/run_full_evaluation.py

# 或快速测试版（约 30 分钟）
!python scripts/run_full_evaluation.py --quick
```

### 方式 B: 手动逐步运行

如果你想看到每一步的详细过程，可以按顺序运行下面的步骤：

---

## ✅ 步骤 1: 查看训练曲线（已完成）

```python
!python scripts/quick_view_results.py
```

---

## 📊 步骤 2: 生成合成样本

```python
# 自动找到最新 checkpoint 并生成合成样本（每个真实样本生成5个）
!python scripts/generate_samples.py --num_samples_per_real 5
```

**预计时间**: 5-10分钟
**输出**: `/kaggle/working/output/synthetic_samples.pt`

> 注: 脚本会自动找到最新的 checkpoint，无需手动指定

---

## 🏃 步骤 3: 训练 Baseline 模型（仅真实数据）

```python
!python scripts/train_baseline.py
```

**预计时间**: 10-15分钟
**说明**: 在真实数据上训练 LSTM，作为对比基准

---

## 🏆 步骤 4: 训练 Augmented 模型（真实 + 合成数据）

```python
!python scripts/train_baseline.py --augmented
```

**预计时间**: 15-20分钟
**说明**: 在真实+合成数据上训练 LSTM

---

## 📈 步骤 5: 对比结果

```python
# 查看两个模型的性能对比
import pandas as pd
from pathlib import Path

results_dir = Path('/kaggle/working/output')

# 读取结果
baseline_results = pd.read_csv(results_dir / 'real_only_results.csv')
augmented_results = pd.read_csv(results_dir / 'augmented_results.csv')

print("="*60)
print("模型性能对比 (Model Performance Comparison)")
print("="*60)

print("\n【Baseline - 仅真实数据】")
print(baseline_results.to_string(index=False))

print("\n【Augmented - 真实+合成数据】")
print(augmented_results.to_string(index=False))

# 计算改进
test_loss_baseline = baseline_results['test_loss'].values[0]
test_loss_augmented = augmented_results['test_loss'].values[0]

improvement = (test_loss_baseline - test_loss_augmented) / test_loss_baseline * 100

print("\n" + "="*60)
print(f"📊 性能变化: {improvement:+.2f}%")
if improvement > 0:
    print(f"✅ Augmented 模型更好！Loss 降低了 {improvement:.2f}%")
else:
    print(f"⚠️  Baseline 模型表现更好")
print("="*60)
```

---

## 📦 步骤 6: 打包下载所有结果

```python
# 创建结果摘要
import shutil
from pathlib import Path

output_dir = Path('/kaggle/working/final_results')
output_dir.mkdir(exist_ok=True)

print("打包结果...")

# 复制重要文件
files_to_copy = [
    ('/kaggle/working/checkpoints', 'checkpoints'),
    ('/kaggle/working/output', 'visualizations'),
    ('/kaggle/working/results', 'metrics'),
    ('/kaggle/working/data/synthetic', 'synthetic_data'),
]

for src, dst_name in files_to_copy:
    src_path = Path(src)
    if src_path.exists():
        dst_path = output_dir / dst_name
        if src_path.is_dir():
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dst_path)
        print(f"✓ Copied {dst_name}")

print("\n" + "="*60)
print("✅ 所有结果已打包到: /kaggle/working/final_results/")
print("="*60)
print("\n在 Kaggle 右侧点击 'Output' → 'Download All'")
```

---

## ⏱️ 预计总时间

### 方式 A: 一键自动（推荐）
- **完整版**: 1-2 小时（50 epochs）
- **快速版** (`--quick`): 30-45 分钟（10 epochs）

### 方式 B: 手动逐步
- 步骤 2: 5-10分钟
- 步骤 3: 10-15分钟
- 步骤 4: 15-20分钟
- 步骤 5: 1分钟
- 步骤 6: 1分钟

**总计**: 约 30-45 分钟

---

## 如果遇到错误

### 找不到 checkpoint
```python
!ls -lh /kaggle/working/checkpoints/
```

### 找不到合成数据
```python
!ls -lh /kaggle/working/data/synthetic/
```

### 内存不足
在步骤 2 中减少生成数量：
```python
!python scripts/generate_samples.py \
    --checkpoint {latest_ckpt} \
    --num_samples_per_real 3  # 改成 3
```
