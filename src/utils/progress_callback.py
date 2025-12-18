"""
Custom progress callback for better visibility in Kaggle notebooks.
"""

import sys
import time
from pytorch_lightning.callbacks import Callback


class KaggleProgressCallback(Callback):
    """
    Custom callback to display training progress clearly in Kaggle.

    Shows:
    - Current epoch and step
    - Loss values
    - Time per epoch
    - ETA for completion
    """

    def __init__(self):
        super().__init__()
        self.epoch_start_time = None
        self.epoch_losses = []

    def on_train_epoch_start(self, trainer, pl_module):
        """Called when the training epoch begins."""
        self.epoch_start_time = time.time()
        self.epoch_losses = []

        print(f"\n{'='*60}")
        print(f"EPOCH {trainer.current_epoch + 1}/{trainer.max_epochs}")
        print(f"{'='*60}")
        sys.stdout.flush()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called when the training batch ends."""
        # Log every N batches
        if batch_idx % 100 == 0 and batch_idx > 0:
            # Get losses from outputs
            if isinstance(outputs, dict):
                d_loss = outputs.get('d_loss', 0)
                g_loss = outputs.get('g_loss', 0)

                total_batches = trainer.num_training_batches
                progress = (batch_idx / total_batches) * 100

                print(f"  Step {batch_idx}/{total_batches} ({progress:.1f}%) | "
                      f"D_loss: {d_loss:.4f} | G_loss: {g_loss:.4f}")
                sys.stdout.flush()

    def on_train_epoch_end(self, trainer, pl_module):
        """Called when the training epoch ends."""
        epoch_time = time.time() - self.epoch_start_time

        # Get metrics
        metrics = trainer.callback_metrics
        d_loss = metrics.get('d_loss', 0)
        g_loss = metrics.get('g_loss', 0)

        print(f"\n{'─'*60}")
        print(f"Epoch {trainer.current_epoch + 1} completed in {epoch_time/60:.1f} minutes")
        print(f"  Final D_loss: {d_loss:.4f}")
        print(f"  Final G_loss: {g_loss:.4f}")

        # Estimate remaining time
        remaining_epochs = trainer.max_epochs - trainer.current_epoch - 1
        if remaining_epochs > 0:
            eta = (epoch_time * remaining_epochs) / 60
            print(f"  ETA for completion: {eta:.1f} minutes ({eta/60:.1f} hours)")

        sys.stdout.flush()

    def on_validation_epoch_start(self, trainer, pl_module):
        """Called when the validation epoch begins."""
        print(f"\n  Running validation...")
        sys.stdout.flush()

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the validation epoch ends."""
        metrics = trainer.callback_metrics
        val_mae = metrics.get('val_mae', None)

        if val_mae is not None:
            print(f"  ✓ Validation MAE: {val_mae:.4f}")

        sys.stdout.flush()

    def on_train_end(self, trainer, pl_module):
        """Called when training ends."""
        print(f"\n{'='*60}")
        print(f"✅ TRAINING COMPLETED!")
        print(f"{'='*60}")
        print(f"Total epochs: {trainer.current_epoch + 1}")
        print(f"Best checkpoint saved")
        sys.stdout.flush()
