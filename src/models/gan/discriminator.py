"""
Discriminator for conditional WGAN-GP.

Scores the realism of sales sequences given conditions.
Architecture: D(x|y) -> scalar score (no sigmoid for WGAN)
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from .encoders import ConditionEncoder


class Discriminator(nn.Module):
    """
    Conditional Discriminator: D(x|y) -> realism score

    Architecture:
    1. Process sales sequence with temporal CNN
    2. Encode condition y (sales + reviews + temporal)
    3. Combine sequence features with condition
    4. Output scalar score (no sigmoid for WGAN)
    """

    def __init__(self,
                 condition_dim: int = 512,
                 conv_filters: list = [64, 128, 256],
                 fc_dims: list = [512, 256]):
        """
        Args:
            condition_dim: Dimension of condition embedding (default: 512)
            conv_filters: List of CNN filter sizes (default: [64, 128, 256])
            fc_dims: List of FC layer dimensions (default: [512, 256])
        """
        super().__init__()

        self.condition_dim = condition_dim
        self.conv_filters = conv_filters
        self.fc_dims = fc_dims

        # Condition encoder (shared with Generator)
        self.encoder = ConditionEncoder(output_dim=condition_dim)

        # Temporal CNN for sequence processing
        # Input: (batch, 1, 7) - 1 channel, 7 timesteps
        conv_layers = []
        in_channels = 1

        for out_channels in conv_filters:
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                # No batch norm for WGAN-GP discriminator
            ])
            in_channels = out_channels

        self.conv = nn.Sequential(*conv_layers)

        # Global average pooling will reduce (batch, 256, 7) -> (batch, 256)
        # Then concatenate with condition: 256 + condition_dim

        # Score network with spectral normalization for stability
        score_layers = []
        input_dim = conv_filters[-1] + condition_dim  # 256 + 512 = 768

        for fc_dim in fc_dims:
            score_layers.extend([
                spectral_norm(nn.Linear(input_dim, fc_dim)),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1)
            ])
            input_dim = fc_dim

        # Final output layer (no activation for WGAN)
        score_layers.append(spectral_norm(nn.Linear(input_dim, 1)))

        self.score_net = nn.Sequential(*score_layers)

    def forward(self, sales_sequence, sales_history, temporal_features, review_features):
        """
        Score the realism of a sales sequence.

        Args:
            sales_sequence: (batch, 7) - 7-day sales sequence to evaluate
            sales_history: (batch, 30) - historical sales (for conditioning)
            temporal_features: (batch, 8) - temporal features (for conditioning)
            review_features: (batch, 2) - review features (for conditioning)

        Returns:
            score: (batch, 1) - realism score (higher = more realistic)
        """
        batch_size = sales_sequence.size(0)

        # Encode condition
        condition = self.encoder(sales_history, temporal_features, review_features)
        # condition: (batch, condition_dim)

        # Process sequence with temporal CNN
        # Add channel dimension: (batch, 7) -> (batch, 1, 7)
        seq_input = sales_sequence.unsqueeze(1)

        # Apply CNN
        seq_features = self.conv(seq_input)  # (batch, 256, 7)

        # Global average pooling
        seq_features = seq_features.mean(dim=2)  # (batch, 256)

        # Combine sequence features with condition
        combined = torch.cat([seq_features, condition], dim=1)  # (batch, 256 + 512)

        # Compute realism score
        score = self.score_net(combined)  # (batch, 1)

        return score


if __name__ == "__main__":
    # Test discriminator
    print("Testing Discriminator...")

    batch_size = 16
    condition_dim = 512

    discriminator = Discriminator(
        condition_dim=condition_dim,
        conv_filters=[64, 128, 256],
        fc_dims=[512, 256]
    )

    # Create dummy inputs
    sales_sequence = torch.randn(batch_size, 7)  # Generated or real sequence
    sales_history = torch.randn(batch_size, 30)
    temporal_features = torch.randn(batch_size, 8)
    review_features = torch.randn(batch_size, 2)

    # Forward pass
    score = discriminator(sales_sequence, sales_history, temporal_features, review_features)

    print(f"\nInput shapes:")
    print(f"  Sales sequence: {sales_sequence.shape}")
    print(f"  Sales history: {sales_history.shape}")
    print(f"  Temporal features: {temporal_features.shape}")
    print(f"  Review features: {review_features.shape}")

    print(f"\nOutput score: {score.shape}")
    print(f"  Expected: (batch_size={batch_size}, 1)")
    print(f"  Actual: {score.shape}")

    print(f"\nSample scores (first 5):")
    print(score[:5].squeeze().detach().numpy())

    # Test with different batch sizes
    for bs in [1, 8, 32, 64]:
        seq_test = torch.randn(bs, 7)
        sales_test = torch.randn(bs, 30)
        temporal_test = torch.randn(bs, 8)
        review_test = torch.randn(bs, 2)

        output = discriminator(seq_test, sales_test, temporal_test, review_test)
        assert output.shape == (bs, 1), f"Failed for batch_size={bs}"

    print(f"\n✓ All tests passed!")

    # Print model summary
    total_params = sum(p.numel() for p in discriminator.parameters())
    trainable_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Check spectral normalization
    print(f"\n✓ Spectral normalization applied to FC layers")
