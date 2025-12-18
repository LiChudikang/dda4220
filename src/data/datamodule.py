"""
PyTorch Lightning DataModule for sales forecasting.

Handles data loading, train/val/test splits, and batching.
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .dataset import SalesDataset, AugmentedSalesDataset


class SalesDataModule(pl.LightningDataModule):
    """Lightning DataModule for sales forecasting."""

    def __init__(self,
                 data_path: str,
                 batch_size: int = 128,
                 num_workers: int = 4,
                 history_window: int = 30,
                 forecast_horizon: int = 7,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 use_augmentation: bool = False,
                 synthetic_data_path: str = None,
                 synthetic_ratio: float = 1.0):
        """
        Args:
            data_path: Path to processed parquet file
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
            history_window: Historical window size
            forecast_horizon: Forecast horizon
            train_ratio: Training data ratio
            val_ratio: Validation data ratio
            use_augmentation: Whether to use synthetic augmentation
            synthetic_data_path: Path to synthetic data
            synthetic_ratio: Ratio of synthetic to real data
        """
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size

        # Auto-adjust num_workers for Kaggle environment
        from ..utils.kaggle_utils import is_kaggle_environment
        if is_kaggle_environment() and num_workers > 0:
            self.num_workers = 0
            print(f"ℹ️  Kaggle environment detected: setting num_workers=0 (was {num_workers})")
        else:
            self.num_workers = num_workers

        self.history_window = history_window
        self.forecast_horizon = forecast_horizon
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.use_augmentation = use_augmentation
        self.synthetic_data_path = synthetic_data_path
        self.synthetic_ratio = synthetic_ratio

    def setup(self, stage: str = None):
        """Setup datasets for each split."""

        # Create base datasets
        self.train_dataset_base = SalesDataset(
            data_path=self.data_path,
            history_window=self.history_window,
            forecast_horizon=self.forecast_horizon,
            split='train',
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio
        )

        self.val_dataset = SalesDataset(
            data_path=self.data_path,
            history_window=self.history_window,
            forecast_horizon=self.forecast_horizon,
            split='val',
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio
        )

        self.test_dataset = SalesDataset(
            data_path=self.data_path,
            history_window=self.history_window,
            forecast_horizon=self.forecast_horizon,
            split='test',
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio
        )

        # Use augmented dataset if specified
        if self.use_augmentation and self.synthetic_data_path is not None:
            print("\nUsing augmented dataset with synthetic samples")
            self.train_dataset = AugmentedSalesDataset(
                real_dataset=self.train_dataset_base,
                synthetic_data_path=self.synthetic_data_path,
                synthetic_ratio=self.synthetic_ratio
            )
        else:
            self.train_dataset = self.train_dataset_base

    def train_dataloader(self):
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def test_dataloader(self):
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )


if __name__ == "__main__":
    # Test datamodule
    datamodule = SalesDataModule(
        data_path="data/processed/product_daily_panel.parquet",
        batch_size=32,
        num_workers=0
    )

    datamodule.setup()

    # Test train loader
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    print("\nBatch structure:")
    for key, value in batch.items():
        print(f"  {key}: {value.shape}")

    print(f"\nNumber of batches:")
    print(f"  Train: {len(train_loader)}")
    print(f"  Val: {len(datamodule.val_dataloader())}")
    print(f"  Test: {len(datamodule.test_dataloader())}")
