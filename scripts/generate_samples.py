"""
Generate synthetic sales sequences using trained WGAN-GP.

Usage:
    python scripts/generate_samples.py --checkpoint checkpoints/best_model.ckpt --num_samples_per_real 5
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import torch
from tqdm import tqdm

from src.models.gan.wgan_gp import WGANGP
from src.data.datamodule import SalesDataModule


def generate_samples(checkpoint_path: str,
                    data_path: str,
                    output_path: str,
                    num_samples_per_real: int = 5,
                    batch_size: int = 128):
    """
    Generate synthetic samples using trained GAN.

    Args:
        checkpoint_path: Path to trained model checkpoint
        data_path: Path to processed data
        output_path: Path to save synthetic samples
        num_samples_per_real: Number of synthetic samples per real sample
        batch_size: Batch size for generation
    """
    print("="*60)
    print("GENERATING SYNTHETIC SAMPLES")
    print("="*60)

    # Load trained model
    print(f"\nLoading model from: {checkpoint_path}")
    model = WGANGP.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.freeze()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Model loaded on device: {device}")

    # Load data
    print(f"\nLoading data from: {data_path}")
    datamodule = SalesDataModule(
        data_path=data_path,
        batch_size=batch_size,
        num_workers=0  # Auto-adjusted for Kaggle
    )
    datamodule.setup('fit')

    train_loader = datamodule.train_dataloader()
    print(f"Number of training batches: {len(train_loader)}")

    # Generate synthetic samples
    print(f"\nGenerating {num_samples_per_real} synthetic samples per real sample...")
    synthetic_data = []

    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Generating"):
            # Move batch to device
            history = batch['history'].to(device)
            temporal = batch['temporal'].to(device)
            reviews = batch['reviews'].to(device)
            target = batch['target'].to(device)

            current_batch_size = history.size(0)

            # Generate multiple samples for each real sample
            for _ in range(num_samples_per_real):
                # Encode condition
                condition = model.generator.encode_condition(history, reviews, temporal)

                # Generate fake sales
                fake_sales = model.generator(condition)

                # Store synthetic samples with real conditions
                for i in range(current_batch_size):
                    synthetic_data.append({
                        'history': history[i].cpu(),
                        'temporal': temporal[i].cpu(),
                        'reviews': reviews[i].cpu(),
                        'target': fake_sales[i].cpu(),
                        'real_target': target[i].cpu()
                    })

    print(f"\nGenerated {len(synthetic_data)} synthetic samples")

    # Save synthetic data
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(synthetic_data, output_path)
    print(f"\n✓ Synthetic data saved to: {output_path}")

    # Print statistics
    print("\n" + "="*60)
    print("GENERATION STATISTICS")
    print("="*60)

    real_count = len(datamodule.train_dataset)
    synthetic_count = len(synthetic_data)
    print(f"Real samples: {real_count}")
    print(f"Synthetic samples: {synthetic_count}")
    print(f"Augmentation ratio: {synthetic_count / real_count:.2f}x")

    return synthetic_data


def main():
    # Auto-detect Kaggle and set default paths
    from src.utils.kaggle_utils import is_kaggle_environment, get_kaggle_paths, get_processed_data_path

    if is_kaggle_environment():
        paths = get_kaggle_paths()
        default_data = str(get_processed_data_path())
        default_output = str(paths['output'] / 'synthetic_samples.pt')
        default_checkpoint = None  # Will use latest
    else:
        default_data = 'data/processed/product_daily_panel.parquet'
        default_output = 'data/synthetic/gan_samples.pt'
        default_checkpoint = 'checkpoints/best_model.ckpt'

    parser = argparse.ArgumentParser(description='Generate synthetic sales sequences')
    parser.add_argument('--checkpoint', type=str, default=default_checkpoint,
                       help='Path to trained model checkpoint (auto-detect latest if not provided)')
    parser.add_argument('--data_path', type=str, default=default_data,
                       help='Path to processed data')
    parser.add_argument('--output_path', type=str, default=default_output,
                       help='Path to save synthetic samples')
    parser.add_argument('--num_samples_per_real', type=int, default=5,
                       help='Number of synthetic samples to generate per real sample')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for generation')

    args = parser.parse_args()

    # Auto-find latest checkpoint if not provided
    checkpoint_path = args.checkpoint
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        if is_kaggle_environment():
            ckpt_dir = get_kaggle_paths()['checkpoints']
            checkpoints = sorted(ckpt_dir.glob('gan-epoch*.ckpt'),
                               key=lambda x: x.stat().st_mtime)
            if checkpoints:
                checkpoint_path = str(checkpoints[-1])
                print(f"Auto-selected checkpoint: {checkpoints[-1].name}")
            else:
                print("❌ No checkpoints found!")
                sys.exit(1)
        else:
            checkpoint_path = args.checkpoint

    generate_samples(
        checkpoint_path=checkpoint_path,
        data_path=args.data_path,
        output_path=args.output_path,
        num_samples_per_real=args.num_samples_per_real,
        batch_size=args.batch_size
    )

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
