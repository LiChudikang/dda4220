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
        num_workers=4
    )
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    print(f"Number of training batches: {len(train_loader)}")

    # Generate synthetic samples
    print(f"\nGenerating {num_samples_per_real} synthetic samples per real sample...")
    synthetic_data = []

    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Generating"):
            # Move batch to device
            sales_history = batch['sales_history'].to(device)
            temporal = batch['temporal_features'].to(device)
            review = batch['review_features'].to(device)

            current_batch_size = sales_history.size(0)

            # Generate multiple samples for each real sample
            for _ in range(num_samples_per_real):
                # Sample noise
                z = torch.randn(current_batch_size, model.noise_dim, device=device)

                # Generate fake sales
                fake_sales = model.generator(z, sales_history, temporal, review)

                # Store synthetic samples
                for i in range(current_batch_size):
                    synthetic_data.append({
                        'sales_history': sales_history[i].cpu(),
                        'temporal_features': temporal[i].cpu(),
                        'review_features': review[i].cpu(),
                        'target_sales': fake_sales[i].cpu()
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
    parser = argparse.ArgumentParser(description='Generate synthetic sales sequences')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.ckpt',
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_path', type=str, default='data/processed/product_daily_panel.parquet',
                       help='Path to processed data')
    parser.add_argument('--output_path', type=str, default='data/synthetic/gan_samples.pt',
                       help='Path to save synthetic samples')
    parser.add_argument('--num_samples_per_real', type=int, default=5,
                       help='Number of synthetic samples to generate per real sample')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for generation')

    args = parser.parse_args()

    generate_samples(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        output_path=args.output_path,
        num_samples_per_real=args.num_samples_per_real,
        batch_size=args.batch_size
    )

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
