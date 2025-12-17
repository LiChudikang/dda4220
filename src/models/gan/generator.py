"""
Generator for conditional GAN.

Generates realistic 7-day sales sequences conditioned on multi-modal features.
Architecture: G(z|y) where z ~ N(0,1) is noise and y is the condition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoders import ConditionEncoder


class Generator(nn.Module):
    """
    Conditional Generator: G(z|y) -> 7-day sales sequence

    Architecture:
    1. Encode condition y (sales + reviews + temporal)
    2. Concatenate noise z with condition
    3. Decode to 7-day sequence using GRU
    4. Output non-negative sales values
    """

    def __init__(self,
                 noise_dim: int = 128,
                 condition_dim: int = 512,
                 hidden_dim: int = 256,
                 output_len: int = 7,
                 num_layers: int = 3):
        """
        Args:
            noise_dim: Dimension of noise vector z (default: 128)
            condition_dim: Dimension of condition embedding (default: 512)
            hidden_dim: Hidden dimension for GRU decoder (default: 256)
            output_len: Length of output sequence (default: 7 days)
            num_layers: Number of GRU layers (default: 3)
        """
        super().__init__()

        self.noise_dim = noise_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.output_len = output_len
        self.num_layers = num_layers

        # Condition encoder
        self.encoder = ConditionEncoder(output_dim=condition_dim)

        # Input projection: concatenate noise + condition
        self.input_projection = nn.Sequential(
            nn.Linear(noise_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # GRU decoder
        self.decoder_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )

        # Output layer: project GRU output to sales values
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, z, sales_history, temporal_features, review_features):
        """
        Generate 7-day sales sequence.

        Args:
            z: (batch, noise_dim) - random noise sampled from N(0,1)
            sales_history: (batch, 30) - historical sales
            temporal_features: (batch, 8) - temporal features
            review_features: (batch, 2) - review features

        Returns:
            generated_sales: (batch, 7) - generated 7-day sales sequence
        """
        batch_size = z.size(0)

        # Encode condition
        condition = self.encoder(sales_history, temporal_features, review_features)
        # condition: (batch, condition_dim)

        # Concatenate noise with condition
        gen_input = torch.cat([z, condition], dim=1)  # (batch, noise_dim + condition_dim)

        # Project to decoder input size
        gen_input = self.input_projection(gen_input)  # (batch, hidden_dim)

        # Replicate for sequence length
        # (batch, hidden_dim) -> (batch, output_len, hidden_dim)
        decoder_input = gen_input.unsqueeze(1).repeat(1, self.output_len, 1)

        # Decode with GRU
        decoder_out, _ = self.decoder_gru(decoder_input)
        # decoder_out: (batch, output_len, hidden_dim)

        # Project to sales values
        sales_pred = self.output_layer(decoder_out)  # (batch, output_len, 1)
        sales_pred = sales_pred.squeeze(-1)  # (batch, output_len)

        # Apply ReLU to ensure non-negative sales
        generated_sales = F.relu(sales_pred)

        return generated_sales


if __name__ == "__main__":
    # Test generator
    print("Testing Generator...")

    batch_size = 16
    noise_dim = 128
    condition_dim = 512

    generator = Generator(
        noise_dim=noise_dim,
        condition_dim=condition_dim,
        hidden_dim=256,
        output_len=7,
        num_layers=3
    )

    # Create dummy inputs
    z = torch.randn(batch_size, noise_dim)
    sales_history = torch.randn(batch_size, 30)
    temporal_features = torch.randn(batch_size, 8)
    review_features = torch.randn(batch_size, 2)

    # Forward pass
    generated_sales = generator(z, sales_history, temporal_features, review_features)

    print(f"\nInput shapes:")
    print(f"  Noise z: {z.shape}")
    print(f"  Sales history: {sales_history.shape}")
    print(f"  Temporal features: {temporal_features.shape}")
    print(f"  Review features: {review_features.shape}")

    print(f"\nGenerated sales: {generated_sales.shape}")
    print(f"  Expected: (batch_size={batch_size}, output_len=7)")
    print(f"  Actual: {generated_sales.shape}")

    # Check non-negativity
    assert (generated_sales >= 0).all(), "Sales must be non-negative!"
    print(f"\n✓ All generated sales are non-negative")

    # Test with different batch sizes
    for bs in [1, 8, 32, 64]:
        z_test = torch.randn(bs, noise_dim)
        sales_test = torch.randn(bs, 30)
        temporal_test = torch.randn(bs, 8)
        review_test = torch.randn(bs, 2)

        output = generator(z_test, sales_test, temporal_test, review_test)
        assert output.shape == (bs, 7), f"Failed for batch_size={bs}"
        assert (output >= 0).all(), f"Negative values for batch_size={bs}"

    print(f"\n✓ All tests passed!")

    # Print model summary
    total_params = sum(p.numel() for p in generator.parameters())
    trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
