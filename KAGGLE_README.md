# Kaggle 快速上手指南 🚀

## 一分钟开始训练

### 1. 在 Kaggle 创建 Notebook

1. 访问 https://www.kaggle.com/code
2. 点击 "New Notebook"
3. 设置 GPU: Settings → Accelerator → **GPU P100**
4. 添加数据: Add Input → 搜索 "Brazilian E-Commerce Public Dataset by Olist" → Add

### 2. 复制粘贴运行

在 Notebook 中新建 Cell，复制粘贴：

```python
# Cell 1: 克隆代码
!git clone https://github.com/YOUR_USERNAME/dda4220.git
%cd dda4220

# Cell 2: 安装依赖
!pip install -q -r requirements.txt

# Cell 3: 预处理数据（首次运行，约5分钟）
!python scripts/preprocess_data.py

# Cell 4: 快速测试（30分钟）
!python kaggle_train.py --quick
```

**注意**: 把 `YOUR_USERNAME` 改成你的 GitHub 用户名！

---

## 训练模式

### 🔬 快速测试（推荐先运行）

```python
!python kaggle_train.py --quick
```

- **时间**: ~30分钟
- **数据**: 10%
- **Epochs**: 3
- **用途**: 验证代码可以运行

### 🏃 中等训练

```python
!python kaggle_train.py --epochs 10 --data-fraction 0.5
```

- **时间**: ~2-3小时
- **数据**: 50%
- **Epochs**: 10
- **用途**: 获得decent结果

### 🏆 完整训练

```python
!python kaggle_train.py
```

- **时间**: ~12小时
- **数据**: 100%
- **Epochs**: 50
- **用途**: 最佳结果

---

## 查看训练结果

### 检查 Checkpoint

```python
!ls -lh /kaggle/working/checkpoints/
```

### 查看 TensorBoard

```python
%load_ext tensorboard
%tensorboard --logdir /kaggle/working/logs
```

### 下载模型

训练完成后：
1. 点击右侧 "Output" 标签
2. 等待文件生成
3. 点击 "Download All"

---

## 常见问题

### ❌ "Dataset not found"

**解决**: 确保添加了 Olist 数据集
- Add Input → Datasets → 搜索 "Brazilian E-Commerce"

### ❌ "Out of memory"

**解决**: 使用更小的batch size
```python
!python kaggle_train.py --quick --batch-size 32
```

### ❌ "No checkpoints saved"

**原因**: 代码已优化，现在会自动保存
- 每个epoch保存一次
- 检查 `/kaggle/working/checkpoints/`

### ❌ 训练卡住

**如果卡在 "Loading data"**:
- 正常，第一次加载需要几分钟
- 会显示进度条

**如果超过10分钟卡住**:
- 停止运行（Stop按钮）
- 重新运行 Cell

---

## 性能优化已完成 ✅

这个仓库已经优化了以下问题：
- ✅ 自动检测 Kaggle 环境
- ✅ 自动设置 `num_workers=0`（避免多进程卡住）
- ✅ 修复了递归错误（sample_and_visualize）
- ✅ 自动跳过交互式输入（Kaggle环境）
- ✅ 添加了进度条（数据加载）
- ✅ 强制保存 checkpoint（每个epoch）

**你不需要手动修改任何代码！**

---

## 完整工作流

```python
# ===== 一次性设置 =====
!git clone https://github.com/YOUR_USERNAME/dda4220.git
%cd dda4220
!pip install -q -r requirements.txt
!python scripts/preprocess_data.py

# ===== 训练 GAN =====
!python kaggle_train.py --quick  # 先测试
# !python kaggle_train.py  # 再完整训练

# ===== 查看结果 =====
%load_ext tensorboard
%tensorboard --logdir /kaggle/working/logs

# ===== 生成合成数据 =====
!python scripts/generate_samples.py \
    --checkpoint /kaggle/working/checkpoints/gan-epoch02-gloss-2.XXX.ckpt \
    --num_samples_per_real 5

# ===== 训练 Baseline =====
!python scripts/train_baseline.py  # Real only
!python scripts/train_baseline.py --augmented  # Real + Synthetic

# ===== 下载结果 =====
# 点击右侧 Output → Download All
```

---

## 预计时间线（P100 GPU）

| 步骤 | 时间 | 说明 |
|------|------|------|
| 预处理数据 | 5分钟 | 只需运行一次 |
| 快速测试 | 30分钟 | 验证代码可运行 |
| 完整训练 | 12小时 | 50 epochs，最佳结果 |
| 生成样本 | 10分钟 | 生成合成数据 |
| 训练Baseline | 1小时 | Real + Augmented |
| **总计** | **~14小时** | 从零到完整结果 |

---

## 代码已优化列表

✅ **src/models/gan/wgan_gp.py**
- 移除了导致递归的 print 语句
- 使用 TensorBoard logging 代替

✅ **src/data/datamodule.py**
- 自动检测 Kaggle 环境
- 自动设置 `num_workers=0`

✅ **src/data/dataset.py**
- 添加了进度条（tqdm）
- 显示序列创建进度

✅ **scripts/run_kaggle.py**
- 修复了 `input()` 阻塞问题
- Kaggle 环境自动跳过交互

✅ **kaggle_train.py** (新文件)
- 统一的训练入口
- 自动checkpoint保存
- 清晰的进度输出

---

## 需要帮助？

- 📖 查看 `README.md` 获取详细文档
- 🐛 遇到问题？在 GitHub 提 Issue
- 💬 或联系：122040057@link.cuhk.edu.cn

---

**祝训练顺利！🎉**
