"""
Optimized Kaggle training script for cGAN sales forecasting.

This script is designed to run smoothly on Kaggle with:
- Automatic environment detection
- Proper checkpoint saving
- No recursion errors
- Progress bars for data loading

Usage in Kaggle Notebook:
    !python kaggle_train.py
    !python kaggle_train.py --quick       # Fast test (3 epochs, 10% data)
    !python kaggle_train.py --epochs 5    # Custom epochs
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from src.data.datamodule import SalesDataModule
from src.models.gan.wgan_gp import WGANGP
from src.utils.kaggle_utils import (
    is_kaggle_environment,
    print_environment_info,
    setup_kaggle_directories,
    get_processed_data_path,
    get_kaggle_paths
)
from src.utils.progress_callback import KaggleProgressCallback


def parse_args():
    parser = argparse.ArgumentParser(description='Train cGAN on Kaggle')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test mode (3 epochs, 10% data, ~30 min)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (default: 50 for full, 3 for quick)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (default: 128 for full, 64 for quick)')
    parser.add_argument('--data-fraction', type=float, default=None,
                        help='Fraction of data to use (0.0-1.0, default: 1.0 for full, 0.1 for quick)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Print environment info
    print_environment_info()

    # Setup directories if on Kaggle
    if is_kaggle_environment():
        setup_kaggle_directories()

    # Configuration
    if args.quick:
        config = {
            'epochs': args.epochs or 3,
            'batch_size': args.batch_size or 64,
            'data_fraction': args.data_fraction or 0.1,
            'mode': 'QUICK TEST'
        }
    else:
        config = {
            'epochs': args.epochs or 50,
            'batch_size': args.batch_size or 128,
            'data_fraction': args.data_fraction or 1.0,
            'mode': 'FULL TRAINING'
        }

    print("\n" + "="*60)
    print(f"CONFIGURATION - {config['mode']}")
    print("="*60)
    print(f"  Epochs: {config['epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Data fraction: {config['data_fraction']*100:.0f}%")
    print(f"  Estimated time: ~{config['epochs'] * 15 * config['data_fraction']:.0f} minutes")
    print("="*60)

    # Set seed
    pl.seed_everything(42)

    # Get paths
    paths = get_kaggle_paths()
    data_path = get_processed_data_path()

    if not data_path.exists():
        print(f"\n❌ Error: Data not found at {data_path}")
        print("Please run data preprocessing first:")
        print("  !python scripts/preprocess_data.py")
        sys.exit(1)

    # Initialize data module
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    datamodule = SalesDataModule(
        data_path=str(data_path),
        batch_size=config['batch_size'],
        num_workers=4,  # Will auto-adjust to 0 on Kaggle
        history_window=30,
        forecast_horizon=7,
        train_ratio=0.7,
        val_ratio=0.15,
    )

    # Initialize model
    print("\n" + "="*60)
    print("INITIALIZING MODEL")
    print("="*60)
    model = WGANGP(
        noise_dim=128,
        condition_dim=512,
        hidden_dim=256,
        output_len=7,
        lambda_gp=10.0,
        n_critic=5,
        lr_g=0.0001,
        lr_d=0.0004
    )

    # Print model info
    total_params_g = sum(p.numel() for p in model.generator.parameters())
    total_params_d = sum(p.numel() for p in model.discriminator.parameters())
    print(f"\nModel parameters:")
    print(f"  Generator: {total_params_g:,}")
    print(f"  Discriminator: {total_params_d:,}")
    print(f"  Total: {total_params_g + total_params_d:,}")
    print(f"  Model size: ~{(total_params_g + total_params_d) * 4 / 1e6:.1f} MB")

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(paths['checkpoints']),
        filename='gan-epoch{epoch:02d}-gloss{g_loss:.3f}',
        save_top_k=3,
        monitor='g_loss',
        mode='min',
        save_last=True,
        every_n_epochs=1,
        verbose=True,
        auto_insert_metric_name=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    progress_callback = KaggleProgressCallback()  # For better progress visibility

    # Logger
    logger = TensorBoardLogger(
        save_dir=str(paths['logs']),
        name='cgan_sales'
    )

    # Initialize trainer
    print("\n" + "="*60)
    print("INITIALIZING TRAINER")
    print("="*60)

    trainer = pl.Trainer(
        max_epochs=config['epochs'],
        accelerator='auto',
        devices=1,
        callbacks=[checkpoint_callback, lr_monitor, progress_callback],
        logger=logger,
        log_every_n_steps=50,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False,  # For better performance
        limit_train_batches=config['data_fraction'],
        limit_val_batches=min(config['data_fraction'] * 2, 1.0),
    )

    print(f"\nTrainer configuration:")
    print(f"  Accelerator: {trainer.accelerator.__class__.__name__}")
    print(f"  Devices: {trainer.num_devices}")
    print(f"  Max epochs: {config['epochs']}")
    print(f"  Train batches: {config['data_fraction']*100:.0f}%")

    # Train
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    print()
    sys.stdout.flush()

    try:
        trainer.fit(model, datamodule)

        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE!")
        print("="*60)

        # Show checkpoint info
        import os
        ckpt_dir = paths['checkpoints']
        if ckpt_dir.exists():
            checkpoints = sorted([f for f in ckpt_dir.glob('*.ckpt')],
                                key=lambda x: x.stat().st_mtime)

            print(f"\n📁 Saved {len(checkpoints)} checkpoint(s):")
            for ckpt in checkpoints[-5:]:  # Show last 5
                size_mb = ckpt.stat().st_size / 1e6
                print(f"  ✓ {ckpt.name} ({size_mb:.1f} MB)")

            if checkpoint_callback.best_model_path:
                print(f"\n🏆 Best checkpoint: {Path(checkpoint_callback.best_model_path).name}")

        print(f"\n📊 View training logs:")
        print(f"  TensorBoard: %tensorboard --logdir {paths['logs']}")
        print(f"  Checkpoint dir: {paths['checkpoints']}")

        print("\n" + "="*60)
        print("Next steps:")
        print("  1. View TensorBoard logs to check training curves")
        print("  2. Generate synthetic samples: !python scripts/generate_samples.py")
        print("  3. Train baseline models: !python scripts/train_baseline.py")
        print("="*60)

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Checkpoints saved up to this point are available.")
        sys.exit(0)

    except Exception as e:
        print("\n" + "="*60)
        print("❌ ERROR OCCURRED")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
