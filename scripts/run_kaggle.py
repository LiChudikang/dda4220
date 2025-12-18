"""
Main script to run the entire pipeline on Kaggle.

This script:
1. Detects Kaggle environment
2. Preprocesses data (if needed)
3. Trains the cGAN
4. Generates synthetic samples
5. Trains and compares baseline models

Usage:
    python scripts/run_kaggle.py
    python scripts/run_kaggle.py --quick  # Fast run for testing
    python scripts/run_kaggle.py --skip-preprocess  # Skip data preprocessing
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.kaggle_utils import (
    is_kaggle_environment,
    print_environment_info,
    setup_kaggle_directories,
    get_olist_data_path,
    get_processed_data_path
)


def parse_args():
    parser = argparse.ArgumentParser(description='Run GAN training pipeline on Kaggle')
    parser.add_argument('--quick', action='store_true',
                        help='Run in quick mode (reduced epochs and data)')
    parser.add_argument('--skip-preprocess', action='store_true',
                        help='Skip data preprocessing step')
    parser.add_argument('--skip-baseline', action='store_true',
                        help='Skip baseline training')
    parser.add_argument('--max-epochs', type=int, default=None,
                        help='Maximum training epochs (overrides config)')
    return parser.parse_args()


def step_preprocess_data():
    """Step 1: Preprocess Olist data."""
    print("\n" + "="*60)
    print("STEP 1: DATA PREPROCESSING")
    print("="*60)

    from src.data.preprocessor import OlistPreprocessor

    # Get paths
    raw_path = get_olist_data_path()
    processed_path = get_processed_data_path()

    print(f"\nRaw data path: {raw_path}")
    print(f"Processed data will be saved to: {processed_path}")

    # Check if already preprocessed
    if processed_path.exists():
        print(f"\n✓ Preprocessed data already exists at {processed_path}")
        response = input("Reprocess data? [y/N]: ").strip().lower()
        if response != 'y':
            print("Skipping preprocessing.")
            return processed_path

    # Create output directory
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    # Run preprocessing
    preprocessor = OlistPreprocessor(
        raw_data_path=str(raw_path),
        processed_data_path=str(processed_path.parent)
    )

    df = preprocessor.run_full_pipeline(min_history_days=60)

    print(f"\n✓ Preprocessing complete!")
    print(f"✓ Processed {len(df)} product-day records")
    print(f"✓ Data saved to: {processed_path}")

    return processed_path


def step_train_gan(data_path: Path, quick_mode: bool = False, max_epochs: int = None):
    """Step 2: Train the conditional GAN."""
    print("\n" + "="*60)
    print("STEP 2: TRAINING CONDITIONAL GAN")
    print("="*60)

    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger

    from src.data.datamodule import SalesDataModule
    from src.models.gan.wgan_gp import WGANGP
    from src.utils.kaggle_utils import get_kaggle_paths

    paths = get_kaggle_paths()

    # Configuration
    config = {
        'seed': 42,
        'batch_size': 64 if quick_mode else 128,
        'max_epochs': max_epochs or (5 if quick_mode else 50),
        'num_workers': 2,
        'history_window': 30,
        'forecast_horizon': 7,
    }

    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Set seed
    pl.seed_everything(config['seed'])

    # Initialize data module
    datamodule = SalesDataModule(
        data_path=str(data_path),
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        history_window=config['history_window'],
        forecast_horizon=config['forecast_horizon'],
        train_ratio=0.7,
        val_ratio=0.15,
    )

    # Initialize model
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

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='g_loss',
        mode='min',
        save_top_k=3,
        dirpath=str(paths['checkpoints']),
        filename='gan-{epoch:02d}-{g_loss:.2f}',
        verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Logger
    logger = TensorBoardLogger(
        save_dir=str(paths['logs']),
        name='cgan_sales'
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        accelerator='auto',
        devices=1,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
        deterministic=False,
        limit_train_batches=0.2 if quick_mode else 1.0,
        limit_val_batches=0.5 if quick_mode else 1.0,
    )

    # Train
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)

    trainer.fit(model, datamodule)

    # Save final model
    final_path = paths['checkpoints'] / 'final_model.ckpt'
    trainer.save_checkpoint(final_path)

    print(f"\n✓ Training complete!")
    print(f"✓ Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"✓ Final model: {final_path}")

    return checkpoint_callback.best_model_path


def step_generate_samples(checkpoint_path: str, data_path: Path, num_samples: int = 5):
    """Step 3: Generate synthetic samples."""
    print("\n" + "="*60)
    print("STEP 3: GENERATING SYNTHETIC SAMPLES")
    print("="*60)

    import torch
    import pandas as pd
    from src.models.gan.wgan_gp import WGANGP
    from src.data.dataset import SalesDataset
    from src.utils.kaggle_utils import get_kaggle_paths

    paths = get_kaggle_paths()
    output_path = paths['output_dir'] / 'synthetic_samples.parquet'

    print(f"\nLoading model from: {checkpoint_path}")
    model = WGANGP.load_from_checkpoint(checkpoint_path)
    model.eval()

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(f"Using device: {device}")

    # Load data
    print(f"\nLoading data from: {data_path}")
    dataset = SalesDataset(str(data_path), history_window=30, forecast_horizon=7)

    print(f"Generating {num_samples} samples per real sample...")

    all_synthetic = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            if idx % 100 == 0:
                print(f"  Progress: {idx}/{len(dataset)}")

            sample = dataset[idx]
            history = sample['history_sales'].unsqueeze(0).to(device)
            review_features = sample['review_features'].unsqueeze(0).to(device)
            temporal_features = sample['temporal_features'].unsqueeze(0).to(device)

            # Generate multiple samples
            for _ in range(num_samples):
                noise = torch.randn(1, model.noise_dim).to(device)
                synthetic = model.generator(noise, history, review_features, temporal_features)

                all_synthetic.append({
                    'product_id': sample.get('product_id', f'product_{idx}'),
                    'synthetic_sales': synthetic.cpu().numpy().flatten()
                })

    # Save
    df_synthetic = pd.DataFrame(all_synthetic)
    df_synthetic.to_parquet(output_path)

    print(f"\n✓ Generated {len(df_synthetic)} synthetic samples")
    print(f"✓ Saved to: {output_path}")

    return output_path


def step_train_baseline(data_path: Path, augmented: bool = False):
    """Step 4: Train baseline LSTM model."""
    print("\n" + "="*60)
    print(f"STEP 4: TRAINING {'AUGMENTED' if augmented else 'BASELINE'} LSTM")
    print("="*60)

    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

    from src.data.datamodule import SalesDataModule
    from src.models.baselines.lstm_forecaster import LSTMForecaster
    from src.utils.kaggle_utils import get_kaggle_paths

    paths = get_kaggle_paths()

    # Data module
    datamodule = SalesDataModule(
        data_path=str(data_path),
        batch_size=128,
        num_workers=2,
        history_window=30,
        forecast_horizon=7,
        use_augmentation=augmented,
        synthetic_data_path=str(paths['output_dir'] / 'synthetic_samples.parquet') if augmented else None,
        synthetic_ratio=1.0
    )

    # Model
    model = LSTMForecaster(
        input_dim=1,
        hidden_dim=128,
        num_layers=2,
        output_len=7,
        lr=0.001
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_mae',
        mode='min',
        save_top_k=1,
        dirpath=str(paths['checkpoints']),
        filename=f'lstm-{"aug" if augmented else "base"}-{{epoch:02d}}-{{val_mae:.2f}}'
    )

    # Logger
    logger = TensorBoardLogger(
        save_dir=str(paths['logs']),
        name=f'lstm_{"augmented" if augmented else "baseline"}'
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator='auto',
        devices=1,
        callbacks=[checkpoint_callback],
        logger=logger,
        enable_progress_bar=True
    )

    # Train
    trainer.fit(model, datamodule)

    # Test
    results = trainer.test(model, datamodule)

    print(f"\n✓ {'Augmented' if augmented else 'Baseline'} LSTM training complete!")
    print(f"  Test MAE: {results[0]['test_mae']:.4f}")
    print(f"  Test RMSE: {results[0]['test_rmse']:.4f}")

    return results[0]


def main():
    """Main execution pipeline."""
    args = parse_args()

    # Print environment info
    print_environment_info()

    # Setup directories
    if is_kaggle_environment():
        setup_kaggle_directories()

    # Step 1: Preprocess data
    if not args.skip_preprocess:
        data_path = step_preprocess_data()
    else:
        data_path = get_processed_data_path()
        if not data_path.exists():
            print(f"Error: Processed data not found at {data_path}")
            print("Please run without --skip-preprocess first.")
            return

    # Step 2: Train GAN
    checkpoint_path = step_train_gan(
        data_path,
        quick_mode=args.quick,
        max_epochs=args.max_epochs
    )

    # Step 3: Generate synthetic samples
    synthetic_path = step_generate_samples(checkpoint_path, data_path, num_samples=5)

    # Step 4: Train baselines
    if not args.skip_baseline:
        print("\n" + "="*60)
        print("COMPARING MODELS")
        print("="*60)

        results_baseline = step_train_baseline(data_path, augmented=False)
        results_augmented = step_train_baseline(data_path, augmented=True)

        # Compare
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(f"\nBaseline LSTM:")
        print(f"  MAE:  {results_baseline['test_mae']:.4f}")
        print(f"  RMSE: {results_baseline['test_rmse']:.4f}")

        print(f"\nAugmented LSTM:")
        print(f"  MAE:  {results_augmented['test_mae']:.4f}")
        print(f"  RMSE: {results_augmented['test_rmse']:.4f}")

        improvement = (results_baseline['test_mae'] - results_augmented['test_mae']) / results_baseline['test_mae'] * 100
        print(f"\nImprovement: {improvement:.2f}%")

    print("\n" + "="*60)
    print("ALL DONE! 🚀")
    print("="*60)


if __name__ == "__main__":
    main()
