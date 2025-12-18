"""
快速查看训练结果 - 一键运行
Quick view training results - one command run

在 Kaggle 运行:
    !python scripts/quick_view_results.py
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.gan.wgan_gp import WGANGP
from src.data.datamodule import SalesDataModule
from src.utils.kaggle_utils import get_kaggle_paths, get_processed_data_path, is_kaggle_environment


def plot_training_curves():
    """Plot training loss curves from CSV logs"""
    paths = get_kaggle_paths()

    # Find CSV log file
    log_dir = paths['logs'] / 'cgan_sales'
    csv_files = list(log_dir.glob('**/metrics.csv'))

    if not csv_files:
        print("⚠️  No training logs found")
        return

    # Load metrics
    df = pd.read_csv(csv_files[0])

    # Plot losses
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # D_loss
    if 'd_loss' in df.columns:
        axes[0].plot(df['step'], df['d_loss'], alpha=0.7, label='D_loss')
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Discriminator Loss')
        axes[0].set_title('判别器损失 (Discriminator Loss)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # G_loss
    if 'g_loss' in df.columns:
        axes[1].plot(df['step'], df['g_loss'], alpha=0.7, label='G_loss', color='orange')
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Generator Loss')
        axes[1].set_title('生成器损失 (Generator Loss)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = paths['output'] / 'training_curves.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 训练曲线已保存: {output_path}")

    if not is_kaggle_environment():
        plt.show()

    return df


def generate_samples():
    """Generate and visualize sample predictions"""
    paths = get_kaggle_paths()

    # Find latest checkpoint
    ckpt_dir = paths['checkpoints']
    checkpoints = sorted(ckpt_dir.glob('gan-epoch*.ckpt'),
                        key=lambda x: x.stat().st_mtime)

    if not checkpoints:
        print("⚠️  No checkpoints found")
        return

    latest_ckpt = checkpoints[-1]
    print(f"Loading checkpoint: {latest_ckpt.name}")

    # Load model
    model = WGANGP.load_from_checkpoint(str(latest_ckpt))
    model.eval()

    # Load some real data for conditioning
    data_path = get_processed_data_path()
    if not data_path.exists():
        print("⚠️  Processed data not found")
        return

    datamodule = SalesDataModule(
        data_path=str(data_path),
        batch_size=8,
        num_workers=0,
    )
    datamodule.setup('fit')

    # Get a batch
    val_loader = datamodule.val_dataloader()
    batch = next(iter(val_loader))

    # Generate samples
    with torch.no_grad():
        history = batch['history'][:8]
        reviews = batch['reviews'][:8]
        temporal = batch['temporal'][:8]
        real_target = batch['target'][:8]

        fake_sales = model.generator(
            model.generator.encode_condition(history, reviews, temporal)
        )

    # Visualize comparisons
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i in range(8):
        real = real_target[i].cpu().numpy()
        fake = fake_sales[i].cpu().numpy()

        axes[i].plot(real, 'o-', label='Real', linewidth=2, markersize=6)
        axes[i].plot(fake, 's--', label='Generated', linewidth=2, markersize=6, alpha=0.7)
        axes[i].set_title(f'Sample {i+1}')
        axes[i].set_xlabel('Day')
        axes[i].set_ylabel('Sales (normalized)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.suptitle('真实 vs 生成样本对比 (Real vs Generated Samples)', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_path = paths['output'] / 'sample_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 样本对比已保存: {output_path}")

    if not is_kaggle_environment():
        plt.show()


def print_summary():
    """Print training summary"""
    paths = get_kaggle_paths()

    print("\n" + "="*60)
    print("训练结果摘要 (Training Summary)")
    print("="*60)

    # Checkpoint info
    ckpt_dir = paths['checkpoints']
    if ckpt_dir.exists():
        checkpoints = list(ckpt_dir.glob('*.ckpt'))
        if checkpoints:
            print(f"\n📁 Checkpoints ({len(checkpoints)}):")
            for ckpt in sorted(checkpoints, key=lambda x: x.stat().st_mtime)[-5:]:
                size_mb = ckpt.stat().st_size / 1e6
                print(f"  ✓ {ckpt.name} ({size_mb:.1f} MB)")

    # Logs info
    log_dir = paths['logs'] / 'cgan_sales'
    csv_files = list(log_dir.glob('**/metrics.csv'))
    if csv_files:
        df = pd.read_csv(csv_files[0])

        print(f"\n📊 训练统计:")
        print(f"  Total steps: {len(df)}")

        if 'd_loss' in df.columns:
            final_d = df['d_loss'].dropna().iloc[-1]
            print(f"  Final D_loss: {final_d:.4f}")

        if 'g_loss' in df.columns:
            final_g = df['g_loss'].dropna().iloc[-1]
            print(f"  Final G_loss: {final_g:.4f}")

        if 'val_mae' in df.columns:
            val_mae = df['val_mae'].dropna()
            if len(val_mae) > 0:
                print(f"  Best VAL MAE: {val_mae.min():.4f}")

    # Output files
    output_dir = paths['output']
    if output_dir.exists():
        output_files = list(output_dir.glob('*.png'))
        if output_files:
            print(f"\n🖼️  生成的图表:")
            for f in output_files:
                print(f"  ✓ {f.name}")

    print("\n" + "="*60)


def main():
    print("="*60)
    print("快速查看训练结果")
    print("Quick View Training Results")
    print("="*60)
    print()

    try:
        # 1. Print summary
        print_summary()

        # 2. Plot training curves
        print("\n📈 绘制训练曲线...")
        df = plot_training_curves()

        # 3. Generate sample comparisons
        print("\n🎨 生成样本对比...")
        generate_samples()

        print("\n" + "="*60)
        print("✅ 完成! Results saved to /kaggle/working/output/")
        print("="*60)

        print("\n下一步 (Next steps):")
        print("  1. 查看生成的图片 (Check generated images)")
        print("  2. 下载结果 (Download results from Output tab)")
        print("  3. 生成更多合成数据: !python scripts/generate_samples.py")
        print("  4. 训练基线模型对比: !python scripts/train_baseline.py")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
