"""
Training script for conditional WGAN-GP.

Usage:
    python scripts/train_gan.py
    python scripts/train_gan.py training.max_epochs=200
    python scripts/train_gan.py data.batch_size=256 training.lr_g=0.0002
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from src.data.datamodule import SalesDataModule
from src.models.gan.wgan_gp import WGANGP


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def train(cfg: DictConfig):
    """Main training function."""

    print("="*60)
    print("TRAINING CONDITIONAL WGAN-GP FOR SALES FORECASTING")
    print("="*60)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    quick_cfg = cfg.get("quick_run")
    quick_enabled = bool(quick_cfg and quick_cfg.enabled)
    effective_batch_size = (
        quick_cfg.batch_size if quick_enabled and quick_cfg.batch_size else cfg.data.batch_size
    )
    effective_max_epochs = (
        quick_cfg.max_epochs if quick_enabled and quick_cfg.max_epochs else cfg.training.max_epochs
    )

    # Set seed for reproducibility
    pl.seed_everything(cfg.seed, workers=True)

    # Initialize data module
    print("\n" + "="*60)
    print("INITIALIZING DATA MODULE")
    print("="*60)
    datamodule = SalesDataModule(
        data_path=cfg.data.path,
        batch_size=effective_batch_size,
        num_workers=cfg.data.num_workers,
        history_window=cfg.data.history_window,
        forecast_horizon=cfg.data.forecast_horizon,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        use_augmentation=cfg.data.use_augmentation,
        synthetic_data_path=cfg.data.synthetic_data_path,
        synthetic_ratio=cfg.data.synthetic_ratio
    )

    # Initialize model
    print("\n" + "="*60)
    print("INITIALIZING WGAN-GP MODEL")
    print("="*60)
    model = WGANGP(
        noise_dim=cfg.model.noise_dim,
        condition_dim=cfg.model.condition_dim,
        hidden_dim=cfg.model.hidden_dim,
        output_len=cfg.model.output_len,
        lambda_gp=cfg.training.lambda_gp,
        n_critic=cfg.training.n_critic,
        lr_g=cfg.training.lr_g,
        lr_d=cfg.training.lr_d
    )

    # Print model summary
    total_params_g = sum(p.numel() for p in model.generator.parameters())
    total_params_d = sum(p.numel() for p in model.discriminator.parameters())
    print(f"\nModel parameters:")
    print(f"  Generator: {total_params_g:,}")
    print(f"  Discriminator: {total_params_d:,}")
    print(f"  Total: {total_params_g + total_params_d:,}")

    # Setup callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
        save_top_k=cfg.checkpoint.save_top_k,
        dirpath=cfg.checkpoint.dirpath,
        filename=cfg.checkpoint.filename,
        verbose=True
    )
    callbacks.append(checkpoint_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    # Logger
    if cfg.wandb.mode != 'disabled':
        logger = WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.experiment_name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
        print(f"\n✓ Weights & Biases logging enabled")
        print(f"  Project: {cfg.wandb.project}")
        print(f"  Experiment: {cfg.experiment_name}")
    else:
        logger = TensorBoardLogger(
            save_dir=cfg.paths.logs,
            name=cfg.experiment_name
        )
        print(f"\n✓ TensorBoard logging enabled")

    # Initialize trainer
    print("\n" + "="*60)
    print("INITIALIZING TRAINER")
    print("="*60)

    trainer_kwargs = dict(
        max_epochs=effective_max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=cfg.training.gradient_clip_val,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        enable_model_summary=cfg.trainer.enable_model_summary,
        deterministic=cfg.trainer.deterministic,
    )

    if quick_enabled:
        # Use only a fraction of data to iterate quickly during experiments
        if quick_cfg.limit_train_batches is not None:
            trainer_kwargs["limit_train_batches"] = quick_cfg.limit_train_batches
        if quick_cfg.limit_val_batches is not None:
            trainer_kwargs["limit_val_batches"] = quick_cfg.limit_val_batches
        if quick_cfg.limit_test_batches is not None:
            trainer_kwargs["limit_test_batches"] = quick_cfg.limit_test_batches

    trainer = pl.Trainer(**trainer_kwargs)

    print(f"\nTrainer configuration:")
    print(f"  Max epochs: {effective_max_epochs}")
    print(f"  Accelerator: {cfg.trainer.accelerator}")
    print(f"  Devices: {cfg.trainer.devices}")
    print(f"  Gradient clipping: {cfg.training.gradient_clip_val}")
    if quick_enabled:
        print("  Quick run: ENABLED")
        print(f"    Batch size: {effective_batch_size}")
        print(f"    limit_train_batches: {trainer_kwargs.get('limit_train_batches', 'full')}")
        print(f"    limit_val_batches: {trainer_kwargs.get('limit_val_batches', 'full')}")

    # Train model
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)

    trainer.fit(model, datamodule)

    # Save final model
    final_model_path = Path(cfg.checkpoint.dirpath) / "final_model.ckpt"
    trainer.save_checkpoint(final_model_path)

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\n✓ Best model checkpoint: {checkpoint_callback.best_model_path}")
    print(f"✓ Final model saved to: {final_model_path}")

    # Test on validation set
    print("\n" + "="*60)
    print("VALIDATING FINAL MODEL")
    print("="*60)

    trainer.validate(model, datamodule)

    print("\n✓ All done!")


if __name__ == "__main__":
    train()
