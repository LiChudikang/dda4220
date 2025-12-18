"""
Run full evaluation pipeline: generate samples → train baseline → train augmented → compare

Usage in Kaggle:
    !python scripts/run_full_evaluation.py
    !python scripts/run_full_evaluation.py --quick  # Fast version (10 epochs)
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.kaggle_utils import is_kaggle_environment, get_kaggle_paths


def run_full_evaluation(quick: bool = False):
    """Run complete evaluation pipeline"""

    epochs = 10 if quick else 50
    num_samples = 3 if quick else 5

    print("="*60)
    print("FULL EVALUATION PIPELINE")
    print("="*60)
    print(f"Mode: {'QUICK' if quick else 'FULL'}")
    print(f"Epochs: {epochs}")
    print(f"Samples per real: {num_samples}")
    print("="*60)

    # Step 1: Generate synthetic samples
    print("\n" + "="*60)
    print("STEP 1/3: Generating synthetic samples...")
    print("="*60)

    import subprocess

    result = subprocess.run([
        sys.executable, 'scripts/generate_samples.py',
        '--num_samples_per_real', str(num_samples)
    ], capture_output=False, text=True)

    if result.returncode != 0:
        print("❌ Failed to generate samples")
        sys.exit(1)

    print("✓ Synthetic samples generated!")

    # Step 2: Train baseline (real only)
    print("\n" + "="*60)
    print("STEP 2/3: Training baseline model (real data only)...")
    print("="*60)

    result = subprocess.run([
        sys.executable, 'scripts/train_baseline.py',
        '--max_epochs', str(epochs)
    ], capture_output=False, text=True)

    if result.returncode != 0:
        print("❌ Failed to train baseline")
        sys.exit(1)

    print("✓ Baseline model trained!")

    # Step 3: Train augmented
    print("\n" + "="*60)
    print("STEP 3/3: Training augmented model (real + synthetic)...")
    print("="*60)

    result = subprocess.run([
        sys.executable, 'scripts/train_baseline.py',
        '--augmented',
        '--max_epochs', str(epochs)
    ], capture_output=False, text=True)

    if result.returncode != 0:
        print("❌ Failed to train augmented model")
        sys.exit(1)

    print("✓ Augmented model trained!")

    # Step 4: Compare results
    print("\n" + "="*60)
    print("COMPARING RESULTS")
    print("="*60)

    if is_kaggle_environment():
        results_dir = get_kaggle_paths()['output']
    else:
        results_dir = Path('results')

    baseline_file = results_dir / 'real_only_results.csv'
    augmented_file = results_dir / 'augmented_results.csv'

    if baseline_file.exists() and augmented_file.exists():
        baseline_df = pd.read_csv(baseline_file)
        augmented_df = pd.read_csv(augmented_file)

        print("\n【Baseline - Real Data Only】")
        print(baseline_df.to_string(index=False))

        print("\n【Augmented - Real + Synthetic】")
        print(augmented_df.to_string(index=False))

        # Calculate improvement
        test_loss_baseline = baseline_df['test_loss'].values[0]
        test_loss_augmented = augmented_df['test_loss'].values[0]

        improvement = (test_loss_baseline - test_loss_augmented) / test_loss_baseline * 100

        print("\n" + "="*60)
        print(f"📊 Performance change: {improvement:+.2f}%")
        if improvement > 0:
            print(f"✅ Augmented model is better! Loss reduced by {improvement:.2f}%")
        else:
            print(f"⚠️  Baseline model performs better")
        print("="*60)

    else:
        print("⚠️  Could not find result files")

    # Summary
    print("\n" + "="*60)
    print("✅ EVALUATION COMPLETE!")
    print("="*60)

    if is_kaggle_environment():
        paths = get_kaggle_paths()
        print(f"\n📁 All results saved to:")
        print(f"  - Checkpoints: {paths['checkpoints']}")
        print(f"  - Logs: {paths['logs']}")
        print(f"  - Results: {paths['output']}")
        print(f"\n💾 Download from Kaggle Output tab")
    else:
        print("\nResults saved to local directories")


def main():
    parser = argparse.ArgumentParser(description='Run full evaluation pipeline')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode (10 epochs, 3 samples per real)')

    args = parser.parse_args()

    try:
        run_full_evaluation(quick=args.quick)
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
