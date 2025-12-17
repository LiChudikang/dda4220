"""
LSTM baseline forecaster for comparison.

Simple LSTM model that predicts future sales based on historical sales.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class LSTMForecaster(pl.LightningModule):
    """
    Simple LSTM baseline for sales forecasting.

    Takes historical sales as input and predicts future sales.
    """

    def __init__(self,
                 input_dim: int = 30,
                 hidden_dim: int = 128,
                 output_len: int = 7,
                 num_layers: int = 2,
                 lr: float = 1e-3):
        """
        Args:
            input_dim: Length of input sequence (default: 30 days)
            hidden_dim: LSTM hidden dimension (default: 128)
            output_len: Length of forecast (default: 7 days)
            num_layers: Number of LSTM layers (default: 2)
            lr: Learning rate (default: 1e-3)
        """
        super().__init__()
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_len = output_len
        self.lr = lr

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=1,  # Sales value at each timestep
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )

        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_len)
        )

    def forward(self, sales_history):
        """
        Predict future sales.

        Args:
            sales_history: (batch, 30) - historical sales

        Returns:
            prediction: (batch, 7) - predicted future sales
        """
        # Add channel dimension
        x = sales_history.unsqueeze(-1)  # (batch, 30, 1)

        # LSTM encoding
        _, (h, c) = self.lstm(x)

        # Use last hidden state
        h_last = h[-1]  # (batch, hidden_dim)

        # Predict output
        output = self.fc(h_last)  # (batch, output_len)

        # Ensure non-negative
        output = F.relu(output)

        return output

    def training_step(self, batch, batch_idx):
        """Training step."""
        sales_history = batch['sales_history']
        target_sales = batch['target_sales']

        # Predict
        pred = self(sales_history)

        # Compute loss (MAE)
        loss = F.l1_loss(pred, target_sales)

        # Log metrics
        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        sales_history = batch['sales_history']
        target_sales = batch['target_sales']

        # Predict
        pred = self(sales_history)

        # Compute loss
        loss = F.l1_loss(pred, target_sales)

        # Log metrics
        self.log('val_loss', loss, prog_bar=True)

        # Additional metrics
        mae = torch.abs(pred - target_sales).mean()
        rmse = torch.sqrt(F.mse_loss(pred, target_sales))

        self.log('val_mae', mae)
        self.log('val_rmse', rmse)

        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        sales_history = batch['sales_history']
        target_sales = batch['target_sales']

        # Predict
        pred = self(sales_history)

        # Compute metrics
        mae = torch.abs(pred - target_sales).mean()
        rmse = torch.sqrt(F.mse_loss(pred, target_sales))

        self.log('test_mae', mae)
        self.log('test_rmse', rmse)

        return {'mae': mae, 'rmse': rmse}

    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }


if __name__ == "__main__":
    # Test LSTM forecaster
    print("Testing LSTM Forecaster...")

    model = LSTMForecaster(
        input_dim=30,
        hidden_dim=128,
        output_len=7,
        num_layers=2
    )

    # Create dummy batch
    batch = {
        'sales_history': torch.randn(16, 30),
        'target_sales': torch.randn(16, 7)
    }

    # Forward pass
    pred = model(batch['sales_history'])

    print(f"\nInput: {batch['sales_history'].shape}")
    print(f"Prediction: {pred.shape}")
    print(f"Target: {batch['target_sales'].shape}")

    # Check non-negativity
    assert (pred >= 0).all(), "Predictions must be non-negative!"
    print(f"\n✓ All predictions are non-negative")

    # Test training step
    loss = model.training_step(batch, 0)
    print(f"\n✓ Training step successful, loss: {loss.item():.4f}")

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    print(f"\n✓ All tests passed!")
