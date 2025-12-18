# Kaggle Notebook Template

Copy and paste the following cells into your Kaggle Notebook to run the complete pipeline.

---

## Cell 1: Setup Environment

```python
# Install dependencies
!pip install -q -r requirements.txt

# Verify installation
import torch
import pytorch_lightning as pl
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ PyTorch Lightning version: {pl.__version__}")
print(f"✓ GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU name: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

---

## Cell 2: Check Data Availability

```python
import os
from pathlib import Path

# Check if Olist dataset is available
input_dir = Path('/kaggle/input')
print("Available input datasets:")
for d in input_dir.iterdir():
    if d.is_dir():
        print(f"  - {d.name}")
        # List some files in the dataset
        files = list(d.glob('*.csv'))[:3]
        for f in files:
            print(f"    └─ {f.name}")
```

---

## Cell 3: Run Complete Pipeline (Full Training)

```python
# Run the complete pipeline
# This will:
# 1. Preprocess Olist data
# 2. Train conditional GAN
# 3. Generate synthetic samples
# 4. Train and compare baseline models

!python scripts/run_kaggle.py
```

---

## Cell 3 (Alternative): Quick Test Run

```python
# For a quick test run (5 epochs, 20% of data)
# Good for testing the pipeline before full training

!python scripts/run_kaggle.py --quick
```

---

## Cell 3 (Alternative): Custom Configuration

```python
# Customize the training
# Options:
#   --max-epochs N     : Set number of training epochs
#   --skip-preprocess  : Skip data preprocessing (if already done)
#   --skip-baseline    : Skip baseline model training
#   --quick            : Quick mode for testing

# Example: Train for 30 epochs only
!python scripts/run_kaggle.py --max-epochs 30

# Example: Skip preprocessing if data is already processed
!python scripts/run_kaggle.py --skip-preprocess

# Example: Train GAN only, skip baselines
!python scripts/run_kaggle.py --skip-baseline
```

---

## Cell 4: View Training Results

```python
# Load TensorBoard to view training curves
%load_ext tensorboard
%tensorboard --logdir /kaggle/working/logs
```

---

## Cell 5: Inspect Generated Checkpoints

```python
import os

# List all checkpoints
checkpoint_dir = '/kaggle/working/checkpoints'
if os.path.exists(checkpoint_dir):
    print("Available checkpoints:")
    for f in os.listdir(checkpoint_dir):
        filepath = os.path.join(checkpoint_dir, f)
        size_mb = os.path.getsize(filepath) / 1e6
        print(f"  - {f} ({size_mb:.2f} MB)")
else:
    print("No checkpoints found yet. Training may still be in progress.")
```

---

## Cell 6: Load and Inspect Synthetic Data

```python
import pandas as pd

# Load synthetic samples
synthetic_path = '/kaggle/working/synthetic_samples.parquet'

if os.path.exists(synthetic_path):
    df_synthetic = pd.read_parquet(synthetic_path)
    print(f"✓ Loaded {len(df_synthetic)} synthetic samples")
    print(f"\nDataset info:")
    print(df_synthetic.info())
    print(f"\nFirst few samples:")
    print(df_synthetic.head())
else:
    print("Synthetic samples not generated yet.")
```

---

## Cell 7: Visualize Sample Predictions

```python
import matplotlib.pyplot as plt
import numpy as np

# Visualize some synthetic vs real samples
if os.path.exists(synthetic_path):
    df_synthetic = pd.read_parquet(synthetic_path)

    # Plot first 5 samples
    fig, axes = plt.subplots(5, 1, figsize=(12, 10))

    for i, ax in enumerate(axes):
        if i < len(df_synthetic):
            sales = df_synthetic.iloc[i]['synthetic_sales']
            ax.plot(sales, marker='o', label=f'Sample {i+1}')
            ax.set_ylabel('Sales')
            ax.set_xlabel('Day')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
else:
    print("No synthetic data available to visualize.")
```

---

## Cell 8: Compare Model Performance

```python
# After training completes, view the final results
import json

results_path = '/kaggle/working/results.json'
if os.path.exists(results_path):
    with open(results_path) as f:
        results = json.load(f)

    print("="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)

    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
else:
    print("Results not available yet. Training may still be in progress.")
```

---

## Tips for Kaggle Usage

### 1. Enable GPU
- Go to Settings (right sidebar) → Accelerator → Select "GPU T4 x2" or "GPU P100"
- This will significantly speed up training

### 2. Monitor Progress
- The training will print progress updates
- You can scroll up to see intermediate results
- Checkpoints are saved every few epochs

### 3. Save Your Work
- Kaggle automatically saves notebook outputs
- Download checkpoints from the Output tab after training
- Download logs for offline TensorBoard viewing

### 4. Session Management
- Kaggle notebooks have a 12-hour runtime limit
- Use `--max-epochs 30` if you need faster training
- Checkpoints are saved regularly, so you can resume if needed

### 5. Resource Usage
- Check GPU memory: `nvidia-smi` in a code cell
- If out of memory, use `--quick` mode

---

## Expected Timeline

- **Setup and data preprocessing**: 5-10 minutes
- **GAN training (50 epochs)**: 1.5-2 hours
- **Sample generation**: 10-15 minutes
- **Baseline training**: 30-45 minutes
- **Total**: ~3-4 hours for complete pipeline

With `--quick` mode: ~15-20 minutes total
