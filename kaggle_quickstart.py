"""
Kaggle Quick Start Script - Fast testing version

This script runs a minimal pipeline for quick testing:
- Uses only 10% of data
- Trains for 3 epochs
- Skips baseline comparison (optional)

Perfect for verifying everything works before full training!

Usage in Kaggle Notebook:
    !python kaggle_quickstart.py
"""

import sys
import os
from pathlib import Path

print("="*60)
print("KAGGLE QUICK START - FAST TESTING MODE")
print("="*60)

# Check if running on Kaggle
if not os.path.exists('/kaggle'):
    print("\n⚠️  Warning: Not running on Kaggle!")
    print("This script is optimized for Kaggle environment.")
    response = input("Continue anyway? [y/N]: ")
    if response.lower() != 'y':
        sys.exit(0)

# Print environment info
print("\n📊 Environment Information:")
print("-" * 60)

import torch
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✓ GPU memory: {mem_gb:.1f} GB")
    print(f"✓ CUDA version: {torch.version.cuda}")
else:
    print("⚠️  No GPU detected! Training will be slow.")

# Check for Olist dataset
print("\n📁 Checking datasets:")
print("-" * 60)

input_dir = Path('/kaggle/input')
if input_dir.exists():
    datasets = [d.name for d in input_dir.iterdir() if d.is_dir()]
    print(f"Available datasets: {len(datasets)}")
    for ds in datasets:
        print(f"  - {ds}")

    # Check for Olist data
    has_olist = any('olist' in ds.lower() or 'brazilian' in ds.lower() for ds in datasets)
    if has_olist:
        print("\n✓ Olist dataset found!")
    else:
        print("\n⚠️  Olist dataset not found!")
        print("Please add 'Brazilian E-Commerce Public Dataset by Olist' to inputs.")
        sys.exit(1)
else:
    print("⚠️  /kaggle/input not found!")

print("\n" + "="*60)
print("STARTING QUICK TEST PIPELINE")
print("="*60)
print("\nThis will:")
print("  1. Preprocess Olist data")
print("  2. Train GAN for 3 epochs (10% data)")
print("  3. Generate small batch of synthetic samples")
print("  4. Show results")
print("\nEstimated time: ~10-15 minutes on P100 GPU")
print("="*60)

# Run the main script with quick settings
print("\n🚀 Launching training...\n")

import subprocess

cmd = [
    'python', 'scripts/run_kaggle.py',
    '--quick',
    '--skip-baseline',  # Skip baseline for fastest test
]

try:
    result = subprocess.run(cmd, check=True, text=True)

    print("\n" + "="*60)
    print("✅ QUICK TEST COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Check /kaggle/working/checkpoints/ for model weights")
    print("  2. Check /kaggle/working/logs/ for training logs")
    print("  3. Run full training: !python scripts/run_kaggle.py")
    print("\n" + "="*60)

except subprocess.CalledProcessError as e:
    print("\n" + "="*60)
    print("❌ ERROR OCCURRED")
    print("="*60)
    print(f"Error: {e}")
    print("\nTroubleshooting:")
    print("  1. Make sure all dependencies are installed")
    print("  2. Check that Olist dataset is added as input")
    print("  3. Verify GPU is enabled in notebook settings")
    sys.exit(1)
except KeyboardInterrupt:
    print("\n\n⚠️  Interrupted by user")
    sys.exit(0)
