# Kaggle 快速上手指南 🚀

## 第一步：准备GitHub仓库

在你的本地终端运行：

```bash
# 确保所有文件都已提交
git add .
git commit -m "Add Kaggle support"
git push origin main
```

---

## 第二步：在Kaggle上创建Notebook

### 2.1 创建新Notebook

1. 访问 https://www.kaggle.com/code
2. 点击右上角的 **"New Notebook"** 按钮
3. 等待Notebook加载完成

### 2.2 配置GPU

1. 点击右侧边栏的 **"Settings"** (齿轮图标)
2. 找到 **"Accelerator"** 选项
3. 选择 **"GPU P100"** (或 GPU T4 x2)
4. 点击 **"Save"**

### 2.3 添加Olist数据集

1. 点击右侧边栏的 **"Add Input"** 按钮
2. 点击 **"Datasets"** 标签
3. 在搜索框输入：**"Brazilian E-Commerce"**
4. 找到 **"Brazilian E-Commerce Public Dataset by Olist"** (by olistbr)
5. 点击数据集右侧的 **"Add"** 按钮
6. 等待数据集加载完成（右侧会显示绿色对勾）

---

## 第三步：在Notebook中设置代码

### 方法A：使用GitHub集成（推荐）

#### 连接GitHub（首次使用需要）

1. 点击右上角头像旁的设置图标
2. 选择 **"Settings"**
3. 找到 **"GitHub"** 部分
4. 点击 **"Link to GitHub"**
5. 授权Kaggle访问你的GitHub

#### 导入仓库

1. 在Notebook中点击 **"File"** → **"Import Notebook"**
2. 选择 **"GitHub"** 标签
3. 输入你的仓库URL：`https://github.com/YOUR_USERNAME/dda4220`
4. 选择导入

### 方法B：手动克隆（简单快速）

在Kaggle Notebook的第一个cell中输入并运行：

```python
# Cell 1: 克隆仓库
!git clone https://github.com/YOUR_USERNAME/dda4220.git
%cd dda4220
!ls -la
```

**注意：** 把 `YOUR_USERNAME` 替换成你的GitHub用户名！

---

## 第四步：快速测试运行

### Cell 2: 安装依赖

```python
# 安装所需的Python包
!pip install -q -r requirements.txt

# 验证安装
import torch
import pytorch_lightning as pl
print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ Lightning: {pl.__version__}")
print(f"✓ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### Cell 3: 快速测试（推荐先运行这个）

```python
# 快速测试模式 - 约10-15分钟完成
# 使用10%数据，训练3个epochs
!python kaggle_quickstart.py
```

**这个命令会：**
- ✅ 自动检测环境和GPU
- ✅ 预处理Olist数据
- ✅ 训练GAN模型（3 epochs, 10%数据）
- ✅ 生成少量合成样本
- ⏱️ 预计耗时：10-15分钟（P100 GPU）

---

## 第五步：查看测试结果

### Cell 4: 检查输出文件

```python
import os

# 检查生成的文件
print("Generated files:")
print("\n📁 Checkpoints:")
for f in os.listdir('/kaggle/working/checkpoints'):
    size = os.path.getsize(f'/kaggle/working/checkpoints/{f}') / 1e6
    print(f"  - {f} ({size:.1f} MB)")

print("\n📁 Logs:")
for f in os.listdir('/kaggle/working/logs'):
    print(f"  - {f}")
```

### Cell 5: 查看训练曲线

```python
# 加载TensorBoard查看训练过程
%load_ext tensorboard
%tensorboard --logdir /kaggle/working/logs
```

---

## 第六步：运行完整训练（测试成功后）

如果快速测试成功，运行完整版本：

### Cell 6: 完整训练

```python
# 完整训练 - 约2-3小时
!python scripts/run_kaggle.py --max-epochs 50
```

或者使用中等规模训练（更快）：

```python
# 中等训练 - 约1小时
!python scripts/run_kaggle.py --max-epochs 20
```

---

## 运行模式对比

| 模式 | 命令 | 数据量 | Epochs | 耗时 (P100) | 适用场景 |
|------|------|--------|--------|-------------|----------|
| 🔬 快速测试 | `kaggle_quickstart.py` | 10% | 3 | ~15分钟 | 验证代码可运行 |
| 🏃 快速训练 | `--quick` | 20% | 5 | ~30分钟 | 快速原型 |
| 🎯 中等训练 | `--max-epochs 20` | 100% | 20 | ~1小时 | 平衡质量和时间 |
| 🏆 完整训练 | `--max-epochs 50` | 100% | 50 | ~2-3小时 | 最佳结果 |

---

## 常见问题解决

### ❌ 问题1: "Dataset not found"

**原因：** 没有添加Olist数据集

**解决：**
1. 点击右侧 "Add Input" → "Datasets"
2. 搜索 "Brazilian E-Commerce Public Dataset by Olist"
3. 点击 "Add"

### ❌ 问题2: "CUDA out of memory"

**原因：** GPU内存不足

**解决：**
```python
# 使用更小的batch size
!python scripts/run_kaggle.py --quick
```

### ❌ 问题3: "No module named 'src'"

**原因：** 不在正确的目录

**解决：**
```python
%cd /kaggle/working/dda4220
!python kaggle_quickstart.py
```

### ❌ 问题4: 训练中断

**原因：** Kaggle有12小时运行限制

**解决：**
- 使用 `--max-epochs 30` 限制训练时间
- 或者分步运行：先预处理，再训练

---

## 下载结果

训练完成后，下载你的模型：

1. 点击右侧 **"Output"** 标签
2. 等待输出文件生成完成
3. 点击 **"Download"** 下载所有文件

主要文件：
- `checkpoints/` - 训练好的模型权重
- `logs/` - TensorBoard日志
- `synthetic_samples.parquet` - 生成的合成数据

---

## 完整代码示例（复制粘贴使用）

```python
# ========== Cell 1: Setup ==========
!git clone https://github.com/YOUR_USERNAME/dda4220.git
%cd dda4220

# ========== Cell 2: Install ==========
!pip install -q -r requirements.txt
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")

# ========== Cell 3: Quick Test ==========
!python kaggle_quickstart.py

# ========== Cell 4: Check Results ==========
!ls -lh /kaggle/working/checkpoints/

# ========== Cell 5: View Logs ==========
%load_ext tensorboard
%tensorboard --logdir /kaggle/working/logs

# ========== Cell 6: Full Training (optional) ==========
# !python scripts/run_kaggle.py --max-epochs 30
```

---

## 预期输出

快速测试成功后，你会看到：

```
==============================================================
✅ QUICK TEST COMPLETED SUCCESSFULLY!
==============================================================

Next steps:
  1. Check /kaggle/working/checkpoints/ for model weights
  2. Check /kaggle/working/logs/ for training logs
  3. Run full training: !python scripts/run_kaggle.py

==============================================================
```

---

## 性能参考（P100 GPU）

- **数据预处理**: ~5分钟
- **GAN训练 (3 epochs, 10% data)**: ~8分钟
- **样本生成**: ~2分钟
- **总计**: ~15分钟

完整训练：
- **50 epochs, 100% data**: ~2-3小时
- **20 epochs, 100% data**: ~1小时

---

## 需要帮助？

如果遇到问题：

1. 📖 查看 `README.md` 的 "Troubleshooting" 部分
2. 🔍 检查 Kaggle Notebook 的错误信息
3. 💬 在GitHub Issues提问

---

**祝训练顺利！🎉**
