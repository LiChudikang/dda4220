"""
Utility functions for Kaggle environment detection and path management.
"""

import os
from pathlib import Path
from typing import Dict


def is_kaggle_environment() -> bool:
    """
    Detect if code is running on Kaggle.

    Returns:
        bool: True if running on Kaggle, False otherwise
    """
    return os.path.exists('/kaggle/input')


def get_kaggle_paths() -> Dict[str, Path]:
    """
    Get appropriate paths based on environment (Kaggle or local).

    Returns:
        Dict with keys: data_dir, output_dir, checkpoints, logs
    """
    if is_kaggle_environment():
        return {
            'data_dir': Path('/kaggle/input'),
            'output_dir': Path('/kaggle/working'),
            'checkpoints': Path('/kaggle/working/checkpoints'),
            'logs': Path('/kaggle/working/logs'),
            'processed_data': Path('/kaggle/working/processed')
        }
    else:
        return {
            'data_dir': Path('data'),
            'output_dir': Path('.'),
            'checkpoints': Path('checkpoints'),
            'logs': Path('logs'),
            'processed_data': Path('data/processed')
        }


def setup_kaggle_directories():
    """Create necessary directories for Kaggle environment."""
    paths = get_kaggle_paths()

    # Create output directories
    for key in ['checkpoints', 'logs', 'processed_data']:
        paths[key].mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {paths[key]}")


def get_olist_data_path() -> Path:
    """
    Get the path to Olist dataset based on environment.

    On Kaggle, assumes the dataset is added as input.
    Locally, looks in data/raw.

    Returns:
        Path to Olist raw data directory
    """
    if is_kaggle_environment():
        # Check common Kaggle input paths
        possible_paths = [
            Path('/kaggle/input/brazilian-ecommerce'),
            Path('/kaggle/input/olistbr-brazilian-ecommerce'),
            Path('/kaggle/input/olist-ecommerce'),
        ]

        for path in possible_paths:
            if path.exists():
                return path

        # If not found, list available inputs
        input_dir = Path('/kaggle/input')
        if input_dir.exists():
            available = list(input_dir.iterdir())
            print(f"Available Kaggle inputs: {[p.name for p in available]}")

            # Try to find any directory containing olist data
            for d in available:
                if d.is_dir():
                    files = list(d.glob('*.csv'))
                    if any('olist' in f.name.lower() for f in files):
                        return d

        raise FileNotFoundError(
            "Olist dataset not found in /kaggle/input/. "
            "Please add the 'Brazilian E-Commerce Public Dataset by Olist' "
            "to your Kaggle notebook inputs."
        )
    else:
        return Path('data/raw')


def print_environment_info():
    """Print information about the current environment."""
    print("="*60)
    print("ENVIRONMENT INFORMATION")
    print("="*60)

    if is_kaggle_environment():
        print("✓ Running on Kaggle")

        # Print GPU info
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠ No GPU available")

        # Print available inputs
        input_dir = Path('/kaggle/input')
        if input_dir.exists():
            inputs = [d.name for d in input_dir.iterdir() if d.is_dir()]
            print(f"\nAvailable inputs:")
            for inp in inputs:
                print(f"  - {inp}")
    else:
        print("✓ Running locally")

    print()


def get_processed_data_path(filename: str = "product_daily_panel.parquet") -> Path:
    """
    Get the path to processed data file.

    Args:
        filename: Name of the processed data file

    Returns:
        Full path to the processed data file
    """
    paths = get_kaggle_paths()

    if is_kaggle_environment():
        # First check if preprocessed data is available as input
        preprocessed_input = Path('/kaggle/input/brazilian-ecommerce-preprocessed')
        if preprocessed_input.exists():
            data_file = preprocessed_input / filename
            if data_file.exists():
                print(f"✓ Using preprocessed data from input: {data_file}")
                return data_file

        # Otherwise use working directory
        return paths['processed_data'] / filename
    else:
        return paths['processed_data'] / filename
