"""
Script to download and preprocess Olist dataset.

Usage:
    python scripts/preprocess_data.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessor import OlistPreprocessor


def download_olist_data():
    """Download Olist dataset from Kaggle."""
    import os
    import zipfile

    print("Downloading Olist dataset from Kaggle...")
    print("NOTE: Make sure you have kaggle API configured (~/.kaggle/kaggle.json)")

    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download using kaggle API
    try:
        os.system(f"kaggle datasets download -d olistbr/brazilian-ecommerce -p {data_dir}")

        # Unzip
        zip_path = data_dir / "brazilian-ecommerce.zip"
        if zip_path.exists():
            print(f"Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            print("Extraction complete!")

            # Remove zip file
            zip_path.unlink()
        else:
            print("Warning: Downloaded file not found. Dataset may already exist.")

    except Exception as e:
        print(f"Error downloading data: {e}")
        print("\nAlternative: Download manually from:")
        print("https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce")
        print(f"And extract to: {data_dir.absolute()}")
        return False

    return True


def main():
    """Main preprocessing pipeline."""

    # Check if data already exists
    raw_dir = Path("data/raw")
    required_files = [
        'olist_orders_dataset.csv',
        'olist_order_items_dataset.csv',
        'olist_order_reviews_dataset.csv',
        'olist_products_dataset.csv'
    ]

    all_exist = all((raw_dir / f).exists() for f in required_files)

    if not all_exist:
        print("Raw data files not found. Attempting to download...")
        success = download_olist_data()
        if not success:
            print("\nPlease download the dataset manually and run again.")
            return

    # Run preprocessing
    preprocessor = OlistPreprocessor(
        raw_data_path="data/raw",
        processed_data_path="data/processed"
    )

    df = preprocessor.run_full_pipeline(min_history_days=60)

    print("\n✓ Preprocessing complete!")
    print(f"✓ Processed data saved to: data/processed/product_daily_panel.parquet")


if __name__ == "__main__":
    main()
