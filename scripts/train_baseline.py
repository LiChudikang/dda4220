"""
Training script for LSTM baseline forecaster.

Usage:
    # Train on real data only
    python scripts/train_baseline.py

    # Train on augmented data (real + synthetic)
    python scripts/train_baseline.py --augmented --synthetic_path data/synthetic/gan_samples.pt
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from src.data.datamodule import SalesDataModule
from src.models.baselines.lstm_forecaster import LSTMForecaster


def train_baseline(use_augmentation: bool = False,
                   synthetic_data_path: str = None,
                   max_epochs: int = 100,
                   batch_size: int = 128):
    """
    Train LSTM baseline forecaster.

    Args:
        use_augmentation: Whether to use synthetic data augmentation
        synthetic_data_path: Path to synthetic samples
        max_epochs: Maximum training epochs
        batch_size: Batch size
    """

    data_type = "augmented" if use_augmentation else "real_only"

    print("="*60)
    print(f"TRAINING LSTM BASELINE ({data_type.upper()})")
    print("="*60)

    # Set seed
    pl.seed_everything(42, workers=True)

    # Initialize data module
    datamodule = SalesDataModule(
        data_path="data/processed/product_daily_panel.parquet",
        batch_size=batch_size,
        num_workers=4,
        use_augmentation=use_augmentation,
        synthetic_data_path=synthetic_data_path,
        synthetic_ratio=1.0
    )

    # Initialize model
    model = LSTMForecaster(
        input_dim=30,
        hidden_dim=128,
        output_len=7,
        num_layers=2,
        lr=1e-3
    )

    # Callbacks
    checkpoint_dir = f"checkpoints/lstm_{data_type}"
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        dirpath=checkpoint_dir,
        filename='lstm-{epoch:02d}-{val_loss:.4f}',
        verbose=True
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        verbose=True
    )

    # Logger
    logger = TensorBoardLogger(
        save_dir="logs",
        name=f"lstm_{data_type}"
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='auto',
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
        deterministic=True
    )

    print(f"\nTraining configuration:")
    print(f"  Data type: {data_type}")
    print(f"  Max epochs: {max_epochs}")
    print(f"  Batch size: {batch_size}")
    if use_augmentation:
        print(f"  Synthetic data: {synthetic_data_path}")

    # Train
    trainer.fit(model, datamodule)

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"✓ Best checkpoint: {checkpoint_callback.best_model_path}")

    # Test
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)

    test_results = trainer.test(model, datamodule)

    print("\nTest results:")
    for key, value in test_results[0].items():
        print(f"  {key}: {value:.4f}")

    return test_results


def main():
    parser = argparse.ArgumentParser(description='Train LSTM baseline')
    parser.add_argument('--augmented', action='store_true',
                       help='Use augmented dataset with synthetic samples')
    parser.add_argument('--synthetic_path', type=str,
                       default='data/synthetic/gan_samples.pt',
                       help='Path to synthetic samples')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')

    args = parser.parse_args()

    train_baseline(
        use_augmentation=args.augmented,
        synthetic_data_path=args.synthetic_path if args.augmented else None,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size
    )

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
