"""
Helper script to extract training results and format them for the report.

Usage:
    python report/extract_results.py

This script will:
1. Read training logs from TensorBoard or checkpoints
2. Extract key metrics (d_loss, g_loss, real_score, fake_score)
3. Generate LaTeX tables and figure data
4. Print formatted output to copy into main.tex
"""

import sys
from pathlib import Path
import pandas as pd
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def extract_validation_metrics(log_dir='/kaggle/working/logs'):
    """
    Extract validation metrics from training logs.

    Returns formatted LaTeX table rows.
    """
    print("="*60)
    print("EXTRACTING VALIDATION METRICS")
    print("="*60)

    # Try to find tensorboard event files
    log_path = Path(log_dir)

    if not log_path.exists():
        print(f"⚠️  Log directory not found: {log_dir}")
        print("Please update log_dir path or run this script on Kaggle after training.")
        return None

    # Parse TensorBoard logs
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        event_files = list(log_path.rglob("events.out.tfevents.*"))
        if not event_files:
            print("⚠️  No TensorBoard event files found")
            return None

        print(f"Found {len(event_files)} event file(s)")

        # Load first event file
        ea = EventAccumulator(str(event_files[0]))
        ea.Reload()

        # Extract metrics
        metrics = {}
        for tag in ['val_real_score', 'val_fake_score', 'wasserstein_dist']:
            if tag in ea.Tags()['scalars']:
                events = ea.Scalars(tag)
                metrics[tag] = [(e.step, e.value) for e in events]

        # Format as LaTeX table
        print("\nLaTeX Table Rows (copy to Table 2 in main.tex):")
        print("-"*60)

        if all(tag in metrics for tag in ['val_real_score', 'val_fake_score', 'wasserstein_dist']):
            max_len = max(len(metrics[tag]) for tag in metrics)

            for i in range(min(max_len, 5)):  # Up to 5 epochs
                epoch = i + 1
                real_score = metrics['val_real_score'][i][1] if i < len(metrics['val_real_score']) else 0
                fake_score = metrics['val_fake_score'][i][1] if i < len(metrics['val_fake_score']) else 0
                w_dist = metrics['wasserstein_dist'][i][1] if i < len(metrics['wasserstein_dist']) else 0

                print(f"{epoch} & {real_score:+.4f} & {fake_score:+.4f} & {w_dist:.4f} \\\\")

        return metrics

    except ImportError:
        print("⚠️  tensorboard package not found. Install with: pip install tensorboard")
        return None
    except Exception as e:
        print(f"⚠️  Error reading logs: {e}")
        return None


def extract_baseline_results(results_dir='/kaggle/working/output'):
    """
    Extract baseline vs augmented LSTM results.

    Returns formatted LaTeX table.
    """
    print("\n" + "="*60)
    print("EXTRACTING BASELINE COMPARISON")
    print("="*60)

    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"⚠️  Results directory not found: {results_dir}")
        print("Run baseline training first:")
        print("  !python scripts/train_baseline.py")
        print("  !python scripts/train_baseline.py --augmented")
        return None

    # Look for result CSV files
    baseline_file = results_path / 'real_only_results.csv'
    augmented_file = results_path / 'augmented_results.csv'

    if not baseline_file.exists() or not augmented_file.exists():
        print("⚠️  Result files not found")
        print(f"  Expected: {baseline_file}")
        print(f"  Expected: {augmented_file}")
        return None

    try:
        baseline_df = pd.read_csv(baseline_file)
        augmented_df = pd.read_csv(augmented_file)

        # Extract metrics
        baseline_mae = baseline_df['test_mae'].values[0] if 'test_mae' in baseline_df else baseline_df['val_mae'].values[0]
        baseline_rmse = baseline_df['test_rmse'].values[0] if 'test_rmse' in baseline_df else 0

        augmented_mae = augmented_df['test_mae'].values[0] if 'test_mae' in augmented_df else augmented_df['val_mae'].values[0]
        augmented_rmse = augmented_df['test_rmse'].values[0] if 'test_rmse' in augmented_df else 0

        # Calculate improvement
        mae_improvement = (baseline_mae - augmented_mae) / baseline_mae * 100
        rmse_improvement = (baseline_rmse - augmented_rmse) / baseline_rmse * 100 if baseline_rmse > 0 else 0

        # Format as LaTeX
        print("\nLaTeX Table Rows (copy to Table 3 in main.tex):")
        print("-"*60)
        print(f"Baseline (Real only) & {baseline_mae:.4f} & {baseline_rmse:.4f} \\\\")
        print(f"Augmented (Real + Synthetic 5:1) & {augmented_mae:.4f} & {augmented_rmse:.4f} \\\\")
        print("\\midrule")
        print(f"Improvement & {mae_improvement:+.2f}\\% & {rmse_improvement:+.2f}\\% \\\\")

        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"MAE Improvement: {mae_improvement:+.2f}%")
        print(f"RMSE Improvement: {rmse_improvement:+.2f}%")

        if mae_improvement > 0:
            print(f"✅ Augmented model is {mae_improvement:.2f}% better!")
        else:
            print(f"⚠️  Baseline model is {abs(mae_improvement):.2f}% better")

        return {
            'baseline_mae': baseline_mae,
            'baseline_rmse': baseline_rmse,
            'augmented_mae': augmented_mae,
            'augmented_rmse': augmented_rmse,
            'mae_improvement': mae_improvement,
            'rmse_improvement': rmse_improvement
        }

    except Exception as e:
        print(f"⚠️  Error reading results: {e}")
        return None


def extract_dataset_stats(data_path='/kaggle/working/processed/product_daily_panel.parquet'):
    """
    Extract dataset statistics for reporting.
    """
    print("\n" + "="*60)
    print("EXTRACTING DATASET STATISTICS")
    print("="*60)

    data_file = Path(data_path)

    if not data_file.exists():
        print(f"⚠️  Data file not found: {data_path}")
        return None

    try:
        df = pd.read_parquet(data_file)

        num_products = df['product_id'].nunique()
        num_records = len(df)
        date_range = f"{df['date'].min()} to {df['date'].max()}" if 'date' in df.columns else "N/A"

        print(f"\nDataset Statistics:")
        print(f"  Number of unique products: {num_products:,}")
        print(f"  Total product-day records: {num_records:,}")
        print(f"  Date range: {date_range}")

        print(f"\nLaTeX snippet (copy to Section 4.1):")
        print("-"*60)
        print(f"\\item \\textbf{{Full dataset}}: {num_products:,} products, {num_records:,} product-day records")

        # Check for small dataset
        small_path = Path(data_path).parent / 'product_daily_panel_small.parquet'
        if small_path.exists():
            df_small = pd.read_parquet(small_path)
            num_products_small = df_small['product_id'].nunique()
            num_records_small = len(df_small)
            print(f"\\item \\textbf{{Small dataset}}: {num_products_small} products, {num_records_small:,} product-day records")

        return {
            'num_products': num_products,
            'num_records': num_records,
            'date_range': date_range
        }

    except Exception as e:
        print(f"⚠️  Error reading data: {e}")
        return None


def generate_result_summary():
    """
    Generate complete summary of all results for the report.
    """
    print("\n" + "="*60)
    print("GENERATING COMPLETE RESULT SUMMARY")
    print("="*60)

    # Extract all results
    val_metrics = extract_validation_metrics()
    baseline_results = extract_baseline_results()
    dataset_stats = extract_dataset_stats()

    # Save to JSON for reference
    results = {
        'validation_metrics': val_metrics,
        'baseline_comparison': baseline_results,
        'dataset_stats': dataset_stats
    }

    output_file = Path(__file__).parent / 'extracted_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✅ Results saved to: {output_file}")
    print("\nNext steps:")
    print("  1. Copy the LaTeX snippets above into main.tex")
    print("  2. Generate training curve plots (see plot_results.py)")
    print("  3. Compile main.tex to PDF")


if __name__ == '__main__':
    generate_result_summary()
