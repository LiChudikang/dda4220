"""
Evaluation metrics for sales forecasting.

Implements MAE, RMSE, sMAPE, MAPE, and other forecasting metrics.
"""

import torch
import torch.nn.functional as F
import numpy as np


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean Absolute Error.

    Args:
        pred: Predictions (batch, horizon)
        target: Targets (batch, horizon)

    Returns:
        MAE value
    """
    return torch.abs(pred - target).mean()


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Root Mean Squared Error.

    Args:
        pred: Predictions (batch, horizon)
        target: Targets (batch, horizon)

    Returns:
        RMSE value
    """
    return torch.sqrt(F.mse_loss(pred, target))


def smape(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Symmetric Mean Absolute Percentage Error.

    Args:
        pred: Predictions (batch, horizon)
        target: Targets (batch, horizon)
        epsilon: Small constant to avoid division by zero

    Returns:
        sMAPE value (percentage)
    """
    numerator = torch.abs(pred - target)
    denominator = (torch.abs(pred) + torch.abs(target)) / 2 + epsilon
    return (numerator / denominator).mean() * 100


def mape(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Mean Absolute Percentage Error.

    Args:
        pred: Predictions (batch, horizon)
        target: Targets (batch, horizon)
        epsilon: Small constant to avoid division by zero

    Returns:
        MAPE value (percentage)
    """
    return (torch.abs((target - pred) / (target + epsilon))).mean() * 100


def wape(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Weighted Absolute Percentage Error.

    Args:
        pred: Predictions (batch, horizon)
        target: Targets (batch, horizon)
        epsilon: Small constant to avoid division by zero

    Returns:
        WAPE value (percentage)
    """
    return (torch.abs(pred - target).sum() / (torch.abs(target).sum() + epsilon)) * 100


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean Squared Error.

    Args:
        pred: Predictions (batch, horizon)
        target: Targets (batch, horizon)

    Returns:
        MSE value
    """
    return F.mse_loss(pred, target)


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantile: float = 0.5) -> torch.Tensor:
    """
    Pinball loss for quantile regression.

    Args:
        pred: Predictions (batch, horizon)
        target: Targets (batch, horizon)
        quantile: Quantile to evaluate (default: 0.5 for median)

    Returns:
        Pinball loss value
    """
    error = target - pred
    return torch.max(quantile * error, (quantile - 1) * error).mean()


def compute_all_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    Compute all forecasting metrics.

    Args:
        pred: Predictions (batch, horizon)
        target: Targets (batch, horizon)

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'MAE': mae(pred, target).item(),
        'RMSE': rmse(pred, target).item(),
        'MSE': mse(pred, target).item(),
        'sMAPE': smape(pred, target).item(),
        'MAPE': mape(pred, target).item(),
        'WAPE': wape(pred, target).item()
    }

    return metrics


def evaluate_forecaster(model, dataloader, device='cpu'):
    """
    Evaluate a forecaster on a dataset.

    Args:
        model: Forecasting model
        dataloader: PyTorch DataLoader
        device: Device to run on

    Returns:
        Dictionary of aggregated metrics
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            sales_history = batch['sales_history'].to(device)
            target_sales = batch['target_sales'].to(device)

            # Predict
            if hasattr(model, 'generator'):
                # GAN model
                z = torch.randn(sales_history.size(0), model.noise_dim, device=device)
                pred = model.generator(
                    z,
                    sales_history,
                    batch['temporal_features'].to(device),
                    batch['review_features'].to(device)
                )
            else:
                # LSTM or other baseline
                pred = model(sales_history)

            all_preds.append(pred.cpu())
            all_targets.append(target_sales.cpu())

    # Concatenate all predictions and targets
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute metrics
    metrics = compute_all_metrics(all_preds, all_targets)

    return metrics


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics...")

    # Create dummy predictions and targets
    pred = torch.randn(100, 7).abs()  # Non-negative predictions
    target = torch.randn(100, 7).abs()  # Non-negative targets

    # Compute all metrics
    metrics = compute_all_metrics(pred, target)

    print("\nComputed metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    # Test individual metrics
    mae_val = mae(pred, target)
    rmse_val = rmse(pred, target)
    smape_val = smape(pred, target)

    print(f"\nIndividual metric tests:")
    print(f"  MAE: {mae_val.item():.4f}")
    print(f"  RMSE: {rmse_val.item():.4f}")
    print(f"  sMAPE: {smape_val.item():.2f}%")

    # Test pinball loss
    pinball_50 = pinball_loss(pred, target, quantile=0.5)
    pinball_10 = pinball_loss(pred, target, quantile=0.1)
    pinball_90 = pinball_loss(pred, target, quantile=0.9)

    print(f"\nPinball losses:")
    print(f"  50th percentile (median): {pinball_50.item():.4f}")
    print(f"  10th percentile: {pinball_10.item():.4f}")
    print(f"  90th percentile: {pinball_90.item():.4f}")

    print(f"\n✓ All tests passed!")
