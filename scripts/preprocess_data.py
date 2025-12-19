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
from src.utils.kaggle_utils import (
    is_kaggle_environment,
    get_olist_data_path,
    get_kaggle_paths,
    print_environment_info
)


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

    # Print environment info
    print("="*60)
    print("OLIST DATA PREPROCESSING PIPELINE")
    print("="*60)
    print_environment_info()

    # Get paths based on environment
    try:
        raw_dir = get_olist_data_path()
        print(f"✓ Using raw data from: {raw_dir}")
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        if not is_kaggle_environment():
            print("\nAttempting to download data locally...")
            success = download_olist_data()
            if not success:
                print("\nPlease download the dataset manually and run again.")
                return
            raw_dir = Path("data/raw")
        else:
            return

    # Get output path
    paths = get_kaggle_paths()
    output_dir = paths['processed_data']
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"✓ Output directory: {output_dir}")

    # Check if data already exists
    required_files = [
        'olist_orders_dataset.csv',
        'olist_order_items_dataset.csv',
        'olist_order_reviews_dataset.csv',
        'olist_products_dataset.csv'
    ]

    all_exist = all((raw_dir / f).exists() for f in required_files)

    if not all_exist:
        print(f"\n❌ Required files not found in {raw_dir}")
        print("Please ensure the Olist dataset is added to your Kaggle notebook inputs.")
        return

    # Run preprocessing
    print(f"\nStarting preprocessing...")
    preprocessor = OlistPreprocessor(
        raw_data_path=str(raw_dir),
        processed_data_path=str(output_dir)
    )

    df = preprocessor.run_full_pipeline(min_history_days=60)

    output_file = output_dir / "product_daily_panel.parquet"
    print("\n✓ Preprocessing complete!")
    print(f"✓ Processed data saved to: {output_file}")
    print(f"✓ Shape: {df.shape}")


if __name__ == "__main__":
    main()
