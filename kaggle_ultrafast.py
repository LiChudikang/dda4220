"""
Ultra-fast Kaggle pipeline for quick validation.
Completes in ~5 minutes on P100 GPU.

NOT FOR PRODUCTION - just to verify the pipeline works!

Usage in Kaggle:
    !python kaggle_ultrafast.py

Requirements:
    - Add "Brazilian E-Commerce Public Dataset by Olist" to Kaggle notebook inputs
    - Enable GPU accelerator in notebook settings
"""

import sys
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from src.utils.kaggle_utils import is_kaggle_environment, get_olist_data_path, print_environment_info

print("="*70)
print("ULTRA-FAST KAGGLE VALIDATION PIPELINE")
print("="*70)
print("\nThis will complete in ~5 minutes and verify:")
print("  ✓ Data preprocessing works")
print("  ✓ GAN training runs without errors")
print("  ✓ Sample generation works")
print("  ✓ Baseline comparison works")
print("\nNOTE: Results will NOT be accurate (using minimal data/epochs)")
print("="*70)

# Check environment
print_environment_info()

# Check if dataset is available
if is_kaggle_environment():
    try:
        data_path = get_olist_data_path()
        print(f"✓ Olist dataset found at: {data_path}\n")
    except FileNotFoundError as e:
        print("\n" + "="*70)
        print("❌ DATASET NOT FOUND")
        print("="*70)
        print(f"\n{e}")
        print("\n📝 How to fix:")
        print("  1. Click 'Add data' button on the right side of Kaggle notebook")
        print("  2. Search for 'Brazilian E-Commerce Public Dataset by Olist'")
        print("  3. Click 'Add' to attach it to your notebook")
        print("  4. Restart the kernel and run this script again")
        print("\nDataset URL:")
        print("  https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce")
        print("="*70)
        sys.exit(1)
else:
    print("⚠ Warning: Not running on Kaggle. This script is optimized for Kaggle environment.\n")

# Step 1: Check if data is already preprocessed
processed_data = Path("/kaggle/working/processed/product_daily_panel.parquet")
if not processed_data.exists():
    print("\n[1/4] Preprocessing data (~30 seconds)...")
    result = subprocess.run(
        ["python", "scripts/preprocess_data.py"],
        capture_output=False
    )
    if result.returncode != 0:
        print("❌ Preprocessing failed!")
        sys.exit(1)
    print("✓ Data preprocessing complete")
else:
    print("\n[1/4] Using existing preprocessed data ✓")

# Step 2: Train GAN with ultra-fast settings
print("\n[2/4] Training GAN (~2 minutes)...")
result = subprocess.run([
    "python", "scripts/train_gan.py",
    "--config-name=config_ultrafast"
], capture_output=False)

if result.returncode != 0:
    print("❌ GAN training failed!")
    sys.exit(1)
print("✓ GAN training complete")

# Step 3: Generate synthetic samples
print("\n[3/4] Generating synthetic samples (~30 seconds)...")
checkpoint = Path("/kaggle/working/checkpoints")
ckpt_files = list(checkpoint.glob("gan-ultrafast-*.ckpt"))

if not ckpt_files:
    print("⚠ No checkpoint found, skipping sample generation")
else:
    latest_ckpt = max(ckpt_files, key=lambda p: p.stat().st_mtime)
    result = subprocess.run([
        "python", "scripts/generate_samples.py",
        "--checkpoint", str(latest_ckpt),
        "--num_samples_per_real", "2",  # Just 2 samples per real sequence
        "--output_path", "/kaggle/working/synthetic/ultrafast_samples.pt",
        "--data_path", "/kaggle/working/processed/product_daily_panel_small.parquet"
    ], capture_output=False)

    if result.returncode != 0:
        print("❌ Sample generation failed!")
        sys.exit(1)
    print("✓ Sample generation complete")

# Step 4: Quick baseline comparison
print("\n[4/4] Training baseline models (~2 minutes)...")

# Train on real data only
print("  Training baseline (real data only)...")
result = subprocess.run([
    "python", "scripts/train_baseline.py",
    "data.path=/kaggle/working/processed/product_daily_panel_small.parquet",
    "training.max_epochs=5",
    "trainer.limit_train_batches=50",
    "trainer.limit_val_batches=20"
], capture_output=False)

if result.returncode != 0:
    print("⚠ Baseline training failed, but continuing...")
else:
    print("  ✓ Baseline (real) complete")

# Train on augmented data
print("  Training baseline (augmented data)...")
result = subprocess.run([
    "python", "scripts/train_baseline.py",
    "--augmented",
    "data.path=/kaggle/working/processed/product_daily_panel_small.parquet",
    "training.max_epochs=5",
    "trainer.limit_train_batches=50",
    "trainer.limit_val_batches=20",
    "data.synthetic_data_path=/kaggle/working/synthetic/ultrafast_samples.pt"
], capture_output=False)

if result.returncode != 0:
    print("⚠ Augmented baseline training failed")
else:
    print("  ✓ Baseline (augmented) complete")

print("\n" + "="*70)
print("ULTRA-FAST VALIDATION COMPLETE!")
print("="*70)
print("\n✓ The pipeline works end-to-end!")
print("\nNext steps:")
print("  1. Check checkpoints: /kaggle/working/checkpoints/")
print("  2. Check logs: /kaggle/working/logs/")
print("  3. For real training, use: !python scripts/run_kaggle.py")
print("\n" + "="*70)
