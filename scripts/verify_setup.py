"""
Verify that the environment is set up correctly.

Usage:
    python scripts/verify_setup.py
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))


def check_imports():
    """Check that all required packages can be imported."""
    print("Checking imports...")

    required_packages = [
        ('torch', 'PyTorch'),
        ('pytorch_lightning', 'PyTorch Lightning'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('hydra', 'Hydra'),
        ('wandb', 'Weights & Biases'),
        ('transformers', 'Transformers'),
    ]

    failed = []
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT FOUND")
            failed.append(name)

    if failed:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All packages installed!")
        return True


def check_cuda():
    """Check CUDA availability."""
    print("\nChecking CUDA...")
    import torch

    if torch.cuda.is_available():
        print(f"  ✓ CUDA available")
        print(f"  ✓ CUDA version: {torch.version.cuda}")
        print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print(f"  ⚠ CUDA not available - will use CPU")
        return False


def check_project_structure():
    """Check that project directories exist."""
    print("\nChecking project structure...")

    required_dirs = [
        'data',
        'data/raw',
        'data/processed',
        'data/synthetic',
        'src',
        'src/data',
        'src/models',
        'src/models/gan',
        'src/models/baselines',
        'scripts',
        'configs',
        'checkpoints'
    ]

    missing = []
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} - MISSING")
            missing.append(dir_path)

    if missing:
        print(f"\n❌ Missing directories: {', '.join(missing)}")
        return False
    else:
        print("\n✓ All directories exist!")
        return True


def check_modules():
    """Check that custom modules can be imported."""
    print("\nChecking custom modules...")

    modules = [
        'src.data.preprocessor',
        'src.data.dataset',
        'src.data.datamodule',
        'src.models.gan.encoders',
        'src.models.gan.generator',
        'src.models.gan.discriminator',
        'src.models.gan.wgan_gp',
        'src.models.baselines.lstm_forecaster',
        'src.evaluation.metrics'
    ]

    failed = []
    for module in modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError as e:
            print(f"  ✗ {module} - {str(e)}")
            failed.append(module)

    if failed:
        print(f"\n❌ Failed to import: {', '.join(failed)}")
        return False
    else:
        print("\n✓ All modules can be imported!")
        return True


def test_model_instantiation():
    """Test that models can be instantiated."""
    print("\nTesting model instantiation...")

    try:
        from src.models.gan.wgan_gp import WGANGP
        model = WGANGP()
        print(f"  ✓ WGAN-GP model created")

        from src.models.baselines.lstm_forecaster import LSTMForecaster
        forecaster = LSTMForecaster()
        print(f"  ✓ LSTM forecaster created")

        print("\n✓ Models instantiated successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Error instantiating models: {str(e)}")
        return False


def check_data():
    """Check if data exists."""
    print("\nChecking data...")

    raw_files = [
        'data/raw/olist_orders_dataset.csv',
        'data/raw/olist_order_items_dataset.csv',
        'data/raw/olist_order_reviews_dataset.csv',
        'data/raw/olist_products_dataset.csv'
    ]

    all_exist = all(Path(f).exists() for f in raw_files)

    if all_exist:
        print("  ✓ Raw Olist data found")
    else:
        print("  ⚠ Raw data not found - run: python scripts/preprocess_data.py")

    processed_file = Path('data/processed/product_daily_panel.parquet')
    if processed_file.exists():
        print("  ✓ Processed data found")
    else:
        print("  ⚠ Processed data not found - run: python scripts/preprocess_data.py")

    return True


def main():
    """Run all verification checks."""
    print("="*60)
    print("VERIFYING SETUP FOR CONDITIONAL GAN SALES PREDICTION")
    print("="*60)

    results = []

    # Run checks
    results.append(("Imports", check_imports()))
    results.append(("CUDA", check_cuda()))
    results.append(("Project Structure", check_project_structure()))
    results.append(("Modules", check_modules()))
    results.append(("Model Instantiation", test_model_instantiation()))
    results.append(("Data", check_data()))

    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)

    all_passed = all(passed for _, passed in results)

    for check, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check:.<30} {status}")

    if all_passed:
        print("\n" + "="*60)
        print("✓ SETUP COMPLETE! YOU'RE READY TO GO!")
        print("="*60)
        print("\nNext steps:")
        print("  1. python scripts/preprocess_data.py  # Download & preprocess data")
        print("  2. python scripts/train_gan.py         # Train cGAN")
        print("  3. python scripts/generate_samples.py  # Generate synthetic data")
        print("  4. python scripts/train_baseline.py --augmented  # Train & evaluate")
    else:
        print("\n" + "="*60)
        print("❌ SETUP INCOMPLETE - SEE ERRORS ABOVE")
        print("="*60)


if __name__ == "__main__":
    main()
