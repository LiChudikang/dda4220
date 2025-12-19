"""
Data preprocessing module for Olist Brazilian E-Commerce dataset.

Joins orders, items, reviews, and products tables into a product-day panel format.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class OlistPreprocessor:
    """Preprocesses Olist dataset into product-day panel."""

    def __init__(self, raw_data_path: str, processed_data_path: str):
        self.raw_path = Path(raw_data_path)
        self.processed_path = Path(processed_data_path)
        self.processed_path.mkdir(parents=True, exist_ok=True)

    def load_raw_data(self) -> Tuple[pd.DataFrame, ...]:
        """Load all raw Olist CSV files."""
        print("Loading raw Olist data...")

        orders = pd.read_csv(self.raw_path / 'olist_orders_dataset.csv')
        order_items = pd.read_csv(self.raw_path / 'olist_order_items_dataset.csv')
        order_reviews = pd.read_csv(self.raw_path / 'olist_order_reviews_dataset.csv')
        products = pd.read_csv(self.raw_path / 'olist_products_dataset.csv')

        # Convert date columns
        orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
        order_reviews['review_creation_date'] = pd.to_datetime(order_reviews['review_creation_date'])

        print(f"Loaded {len(orders)} orders, {len(order_items)} items, "
              f"{len(order_reviews)} reviews, {len(products)} products")

        return orders, order_items, order_reviews, products

    def join_tables(self, orders: pd.DataFrame, order_items: pd.DataFrame,
                   order_reviews: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
        """Join all tables to create unified dataset."""
        print("Joining tables...")

        # Join orders with items
        df = order_items.merge(orders[['order_id', 'order_purchase_timestamp', 'order_status']],
                               on='order_id', how='left')

        # Keep only delivered orders
        df = df[df['order_status'] == 'delivered'].copy()

        # Join with reviews (optional, some orders may not have reviews)
        reviews_agg = order_reviews.groupby('order_id').agg({
            'review_score': 'mean',
            'review_creation_date': 'first'
        }).reset_index()

        df = df.merge(reviews_agg, on='order_id', how='left')

        # Join with products
        df = df.merge(products[['product_id', 'product_category_name']],
                     on='product_id', how='left')

        print(f"Joined dataset: {len(df)} rows")
        return df

    def aggregate_to_product_day(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate to product-day level."""
        print("Aggregating to product-day level...")

        # Extract date (remove time)
        df['date'] = df['order_purchase_timestamp'].dt.date
        df['date'] = pd.to_datetime(df['date'])

        # Aggregate sales per product per day
        sales_agg = df.groupby(['product_id', 'date']).agg({
            'order_id': 'count',  # Number of orders (proxy for daily sales)
            'price': 'sum'  # Total revenue
        }).reset_index()

        sales_agg.columns = ['product_id', 'date', 'daily_sales', 'daily_revenue']

        # Aggregate reviews per product per day
        reviews_df = df[df['review_score'].notna()].copy()
        reviews_agg = reviews_df.groupby(['product_id', 'date']).agg({
            'review_score': ['mean', 'count']
        }).reset_index()

        reviews_agg.columns = ['product_id', 'date', 'avg_rating', 'review_count']

        # Merge sales and reviews
        product_day = sales_agg.merge(reviews_agg, on=['product_id', 'date'], how='left')

        # Fill missing review data with zeros
        product_day['avg_rating'] = product_day['avg_rating'].fillna(0)
        product_day['review_count'] = product_day['review_count'].fillna(0)

        print(f"Product-day panel: {len(product_day)} rows, "
              f"{product_day['product_id'].nunique()} unique products")

        return product_day

    def create_continuous_timeline(self, product_day: pd.DataFrame,
                                   min_history_days: int = 60) -> pd.DataFrame:
        """Create continuous timeline with zero-filled missing days."""
        print(f"Creating continuous timeline (filtering products with >={min_history_days} days)...")

        # Get product-level date ranges
        product_dates = product_day.groupby('product_id')['date'].agg(['min', 'max', 'count'])
        product_dates['days_active'] = (product_dates['max'] - product_dates['min']).dt.days + 1

        # Filter products with sufficient history
        valid_products = product_dates[product_dates['days_active'] >= min_history_days].index
        product_day = product_day[product_day['product_id'].isin(valid_products)].copy()

        print(f"Filtered to {len(valid_products)} products with >={min_history_days} days history")

        # Create continuous timeline for each product
        all_product_days = []

        for product_id in tqdm(valid_products, desc="Filling missing days"):
            product_data = product_day[product_day['product_id'] == product_id]

            # Create full date range
            date_range = pd.date_range(
                start=product_data['date'].min(),
                end=product_data['date'].max(),
                freq='D'
            )

            # Create full timeline
            full_timeline = pd.DataFrame({
                'product_id': product_id,
                'date': date_range
            })

            # Merge with actual data
            full_data = full_timeline.merge(product_data, on=['product_id', 'date'], how='left')

            # Fill missing values with zeros
            full_data['daily_sales'] = full_data['daily_sales'].fillna(0)
            full_data['daily_revenue'] = full_data['daily_revenue'].fillna(0)
            full_data['avg_rating'] = full_data['avg_rating'].fillna(0)
            full_data['review_count'] = full_data['review_count'].fillna(0)

            all_product_days.append(full_data)

        continuous_panel = pd.concat(all_product_days, ignore_index=True)
        print(f"Continuous panel: {len(continuous_panel)} rows")

        return continuous_panel

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features (day of week, weekend, etc.)."""
        print("Adding temporal features...")

        df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day

        # Add Brazilian holidays (simplified - major holidays only)
        df['is_holiday'] = 0
        # Christmas
        df.loc[(df['month'] == 12) & (df['day'] == 25), 'is_holiday'] = 1
        # New Year
        df.loc[(df['month'] == 1) & (df['day'] == 1), 'is_holiday'] = 1
        # Black Friday (approximate - last Friday of November)
        df.loc[(df['month'] == 11) & (df['day_of_week'] == 4) & (df['day'] >= 23), 'is_holiday'] = 1

        return df

    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize sales features per product."""
        print("Normalizing features per product...")

        # Pre-compute product-level stats (much faster than apply)
        product_stats = df.groupby('product_id').agg({
            'daily_sales': ['min', 'max'],
            'review_count': 'max'
        })

        product_stats.columns = ['sales_min', 'sales_max', 'review_max']
        product_stats = product_stats.reset_index()

        # Merge stats back to main df
        df = df.merge(product_stats, on='product_id', how='left')

        # Vectorized normalization (much faster than apply)
        # Min-max normalization for sales
        sales_range = df['sales_max'] - df['sales_min']
        df['daily_sales_norm'] = np.where(
            sales_range > 0,
            (df['daily_sales'] - df['sales_min']) / sales_range,
            0.0
        )

        # Normalize ratings to [0, 1]
        df['avg_rating_norm'] = df['avg_rating'] / 5.0

        # Log-transform and normalize review counts
        df['review_count_log'] = np.log1p(df['review_count'])
        df['review_count_norm'] = np.where(
            df['review_max'] > 0,
            df['review_count_log'] / np.log1p(df['review_max']),
            0.0
        )

        # Drop temporary columns
        df = df.drop(['sales_min', 'sales_max', 'review_max'], axis=1)

        return df

    def save_processed_data(self, df: pd.DataFrame):
        """Save processed data to parquet."""
        output_path = self.processed_path / 'product_daily_panel.parquet'
        df.to_parquet(output_path, index=False)
        print(f"\nSaved processed data to: {output_path}")
        print(f"Final dataset shape: {df.shape}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Number of products: {df['product_id'].nunique()}")

    def run_full_pipeline(self, min_history_days: int = 60):
        """Run the complete preprocessing pipeline."""
        print("="*60)
        print("OLIST DATA PREPROCESSING PIPELINE")
        print("="*60)

        # Load data
        orders, order_items, order_reviews, products = self.load_raw_data()

        # Join tables
        df = self.join_tables(orders, order_items, order_reviews, products)

        # Aggregate to product-day
        product_day = self.aggregate_to_product_day(df)

        # Create continuous timeline
        continuous_panel = self.create_continuous_timeline(product_day, min_history_days)

        # Add temporal features
        continuous_panel = self.add_temporal_features(continuous_panel)

        # Normalize features
        continuous_panel = self.normalize_features(continuous_panel)

        # Save
        self.save_processed_data(continuous_panel)

        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE!")
        print("="*60)

        return continuous_panel


if __name__ == "__main__":
    # Example usage
    preprocessor = OlistPreprocessor(
        raw_data_path="data/raw",
        processed_data_path="data/processed"
    )

    df = preprocessor.run_full_pipeline(min_history_days=60)

    # Print sample
    print("\nSample of processed data:")
    print(df.head(10))
    print("\nColumn names:")
    print(df.columns.tolist())
