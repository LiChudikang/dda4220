"""
Condition encoders for the conditional GAN.

Encodes multi-modal conditions (sales history, reviews, temporal features) into
a unified embedding vector that conditions the generator and discriminator.
"""

import torch
import torch.nn as nn


class ConditionEncoder(nn.Module):
    """
    Encodes multi-modal conditions into unified embedding.

    Combines:
    - Sales history (30 days) -> LSTM encoding
    - Temporal features (8 dim: day_of_week one-hot + is_weekend) -> FC projection
    - Review features (2 dim: avg_rating + review_count) -> FC projection

    Output: Fixed-size condition vector (default 512-dim)
    """

    def __init__(self,
                 sales_dim: int = 30,
                 temporal_dim: int = 8,
                 review_dim: int = 2,
                 hidden_dim: int = 256,
                 output_dim: int = 512):
        """
        Args:
            sales_dim: Length of sales history (default: 30 days)
            temporal_dim: Dimension of temporal features (default: 8)
            review_dim: Dimension of review features (default: 2)
            hidden_dim: Hidden dimension for LSTM (default: 256)
            output_dim: Output embedding dimension (default: 512)
        """
        super().__init__()

        self.sales_dim = sales_dim
        self.temporal_dim = temporal_dim
        self.review_dim = review_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Sales encoder: Bidirectional LSTM
        self.sales_lstm = nn.LSTM(
            input_size=1,  # Sales value at each timestep
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        # Feature projections
        self.temporal_fc = nn.Sequential(
            nn.Linear(temporal_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.review_fc = nn.Sequential(
            nn.Linear(review_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Fusion layer
        # Sales LSTM output: hidden_dim * 2 (bidirectional)
        # Temporal: 64, Review: 64
        fusion_input_dim = hidden_dim * 2 + 64 + 64

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, sales_history, temporal_features, review_features):
        """
        Args:
            sales_history: (batch, 30) - historical sales
            temporal_features: (batch, 8) - temporal features
            review_features: (batch, 2) - review features

        Returns:
            condition: (batch, output_dim) - unified condition embedding
        """
        batch_size = sales_history.size(0)

        # Encode sales history with LSTM
        # Add channel dimension: (batch, 30) -> (batch, 30, 1)
        sales_input = sales_history.unsqueeze(-1)

        # LSTM forward pass
        lstm_out, (h, c) = self.sales_lstm(sales_input)
        # h: (num_layers * num_directions, batch, hidden_dim)

        # Concatenate final hidden states from both directions
        # h[-2]: forward direction last layer
        # h[-1]: backward direction last layer
        sales_emb = torch.cat([h[-2], h[-1]], dim=1)  # (batch, hidden_dim * 2)

        # Project temporal and review features
        temporal_emb = self.temporal_fc(temporal_features)  # (batch, 64)
        review_emb = self.review_fc(review_features)  # (batch, 64)

        # Fuse all embeddings
        combined = torch.cat([sales_emb, temporal_emb, review_emb], dim=1)
        condition = self.fusion(combined)  # (batch, output_dim)

        return condition


if __name__ == "__main__":
    # Test encoder
    batch_size = 16
    encoder = ConditionEncoder(
        sales_dim=30,
        temporal_dim=8,
        review_dim=2,
        hidden_dim=256,
        output_dim=512
    )

    # Create dummy inputs
    sales_history = torch.randn(batch_size, 30)
    temporal_features = torch.randn(batch_size, 8)
    review_features = torch.randn(batch_size, 2)

    # Forward pass
    condition = encoder(sales_history, temporal_features, review_features)

    print(f"Input shapes:")
    print(f"  Sales history: {sales_history.shape}")
    print(f"  Temporal features: {temporal_features.shape}")
    print(f"  Review features: {review_features.shape}")
    print(f"\nOutput condition: {condition.shape}")
    print(f"Successfully encoded to {condition.size(1)}-dim vector!")

    # Test with different batch sizes
    for bs in [1, 32, 64]:
        sales = torch.randn(bs, 30)
        temporal = torch.randn(bs, 8)
        review = torch.randn(bs, 2)
        cond = encoder(sales, temporal, review)
        assert cond.shape == (bs, 512), f"Failed for batch_size={bs}"

    print(f"\n✓ All tests passed!")
