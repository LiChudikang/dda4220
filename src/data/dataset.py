"""
PyTorch Dataset for sales forecasting with GAN.

Creates sequences of (sales_history, conditions, target_sales) for training.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Dict, Tuple


class SalesDataset(Dataset):
    """
    Dataset for conditional sales forecasting.

    Returns:
        - sales_history: Past 30 days of sales (tensor of shape (30,))
        - temporal_features: Day-of-week one-hot + is_weekend (tensor of shape (8,))
        - review_features: avg_rating_norm + review_count_norm (tensor of shape (2,))
        - target_sales: Next 7 days of sales (tensor of shape (7,))
    """

    def __init__(self,
                 data_path: str,
                 history_window: int = 30,
                 forecast_horizon: int = 7,
                 split: str = 'train',
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15):
        """
        Args:
            data_path: Path to processed parquet file
            history_window: Number of historical days to use (default: 30)
            forecast_horizon: Number of days to forecast (default: 7)
            split: One of 'train', 'val', 'test'
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
        """
        self.history_window = history_window
        self.forecast_horizon = forecast_horizon
        self.split = split

        # Load data
        print(f"Loading data from {data_path}...")
        df = pd.read_parquet(data_path)

        # Sort by product and date
        df = df.sort_values(['product_id', 'date']).reset_index(drop=True)

        # Split data chronologically
        df = self._split_data(df, train_ratio, val_ratio, split)

        # Create sequences
        self.sequences = self._create_sequences(df)

        print(f"Created {len(self.sequences)} sequences for {split} split")

    def _split_data(self, df: pd.DataFrame, train_ratio: float,
                   val_ratio: float, split: str) -> pd.DataFrame:
        """Split data chronologically."""
        # Get unique dates
        unique_dates = sorted(df['date'].unique())
        n_dates = len(unique_dates)

        # Calculate split indices
        train_end_idx = int(n_dates * train_ratio)
        val_end_idx = int(n_dates * (train_ratio + val_ratio))

        if split == 'train':
            split_dates = unique_dates[:train_end_idx]
        elif split == 'val':
            split_dates = unique_dates[train_end_idx:val_end_idx]
        elif split == 'test':
            split_dates = unique_dates[val_end_idx:]
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")

        df_split = df[df['date'].isin(split_dates)].copy()
        print(f"{split.capitalize()} split: {len(split_dates)} days, "
              f"{df_split['date'].min()} to {df_split['date'].max()}")

        return df_split

    def _create_sequences(self, df: pd.DataFrame) -> list:
        """Create sequences for each product."""
        from tqdm import tqdm

        sequences = []
        unique_products = df['product_id'].unique()
        print(f"Creating sequences for {len(unique_products)} products...")

        for product_id in tqdm(unique_products, desc="Processing products", leave=False):
            product_data = df[df['product_id'] == product_id].sort_values('date')

            # Need at least history_window + forecast_horizon days
            if len(product_data) < self.history_window + self.forecast_horizon:
                continue

            # Create sliding window sequences
            for i in range(len(product_data) - self.history_window - self.forecast_horizon + 1):
                history_data = product_data.iloc[i:i+self.history_window]
                target_data = product_data.iloc[i+self.history_window:i+self.history_window+self.forecast_horizon]

                # Extract features
                sales_history = history_data['daily_sales_norm'].values
                target_sales = target_data['daily_sales_norm'].values

                # Temporal features (from first day of forecast horizon)
                first_target_day = target_data.iloc[0]
                temporal_features = self._create_temporal_features(first_target_day)

                # Review features (aggregated over history window)
                review_features = self._create_review_features(history_data)

                sequences.append({
                    'sales_history': sales_history,
                    'temporal_features': temporal_features,
                    'review_features': review_features,
                    'target_sales': target_sales,
                    'product_id': product_id,
                    'date': first_target_day['date']
                })

        return sequences

    def _create_temporal_features(self, row: pd.Series) -> np.ndarray:
        """Create temporal feature vector."""
        # One-hot encode day of week (7 dimensions)
        day_of_week_onehot = np.zeros(7)
        day_of_week_onehot[int(row['day_of_week'])] = 1.0

        # Is weekend binary feature
        is_weekend = float(row['is_weekend'])

        # Concatenate
        temporal = np.concatenate([day_of_week_onehot, [is_weekend]])
        return temporal.astype(np.float32)

    def _create_review_features(self, history_data: pd.DataFrame) -> np.ndarray:
        """Aggregate review features over history window."""
        # Take mean of normalized ratings and review counts
        avg_rating_norm = history_data['avg_rating_norm'].mean()
        review_count_norm = history_data['review_count_norm'].mean()

        review_features = np.array([avg_rating_norm, review_count_norm], dtype=np.float32)
        return review_features

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single sequence."""
        seq = self.sequences[idx]

        return {
            'sales_history': torch.FloatTensor(seq['sales_history']),
            'temporal_features': torch.FloatTensor(seq['temporal_features']),
            'review_features': torch.FloatTensor(seq['review_features']),
            'target_sales': torch.FloatTensor(seq['target_sales'])
        }


class AugmentedSalesDataset(Dataset):
    """
    Dataset that combines real and synthetic (GAN-generated) sales sequences.

    Used for Strategy A: augmentation-based training.
    """

    def __init__(self,
                 real_dataset: SalesDataset,
                 synthetic_data_path: str = None,
                 synthetic_ratio: float = 1.0):
        """
        Args:
            real_dataset: Original SalesDataset
            synthetic_data_path: Path to saved synthetic samples (from GAN)
            synthetic_ratio: Ratio of synthetic to real samples (e.g., 1.0 means 1:1)
        """
        self.real_dataset = real_dataset
        self.real_samples = []
        self.synthetic_samples = []

        # Collect real samples
        for i in range(len(real_dataset)):
            self.real_samples.append(real_dataset[i])

        # Load synthetic samples if available
        if synthetic_data_path is not None:
            print(f"Loading synthetic data from {synthetic_data_path}...")
            synthetic_data = torch.load(synthetic_data_path)

            # Sample synthetic data according to ratio
            n_synthetic = int(len(self.real_samples) * synthetic_ratio)
            if len(synthetic_data) > n_synthetic:
                import random
                synthetic_data = random.sample(synthetic_data, n_synthetic)

            self.synthetic_samples = synthetic_data
            print(f"Loaded {len(self.synthetic_samples)} synthetic samples")

        # Combine
        self.all_samples = self.real_samples + self.synthetic_samples
        print(f"Augmented dataset: {len(self.real_samples)} real + "
              f"{len(self.synthetic_samples)} synthetic = {len(self.all_samples)} total")

    def __len__(self) -> int:
        return len(self.all_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.all_samples[idx]


if __name__ == "__main__":
    # Test dataset
    dataset = SalesDataset(
        data_path="data/processed/product_daily_panel.parquet",
        split='train'
    )

    print(f"\nDataset size: {len(dataset)}")

    # Get a sample
    sample = dataset[0]
    print("\nSample structure:")
    for key, value in sample.items():
        print(f"  {key}: {value.shape}")

    print("\nSample values:")
    print(f"  Sales history (first 5 days): {sample['sales_history'][:5]}")
    print(f"  Temporal features: {sample['temporal_features']}")
    print(f"  Review features: {sample['review_features']}")
    print(f"  Target sales: {sample['target_sales']}")
