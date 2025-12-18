"""
WGAN-GP (Wasserstein GAN with Gradient Penalty) Lightning Module.

Implements conditional GAN training with gradient penalty for stable training.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict
from .generator import Generator
from .discriminator import Discriminator


class WGANGP(pl.LightningModule):
    """
    Conditional WGAN-GP for sales forecasting.

    Trains Generator and Discriminator with Wasserstein loss and gradient penalty.
    """

    def __init__(self,
                 noise_dim: int = 128,
                 condition_dim: int = 512,
                 hidden_dim: int = 256,
                 output_len: int = 7,
                 lambda_gp: float = 10.0,
                 n_critic: int = 5,
                 lr_g: float = 1e-4,
                 lr_d: float = 4e-4):
        """
        Args:
            noise_dim: Dimension of noise vector (default: 128)
            condition_dim: Dimension of condition embedding (default: 512)
            hidden_dim: Hidden dimension for models (default: 256)
            output_len: Length of generated sequence (default: 7)
            lambda_gp: Gradient penalty coefficient (default: 10.0)
            n_critic: Number of discriminator updates per generator update (default: 5)
            lr_g: Generator learning rate (default: 1e-4)
            lr_d: Discriminator learning rate (default: 4e-4)
        """
        super().__init__()
        self.save_hyperparameters()

        self.noise_dim = noise_dim
        self.condition_dim = condition_dim
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic
        self.lr_g = lr_g
        self.lr_d = lr_d

        # Initialize models
        self.generator = Generator(
            noise_dim=noise_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            output_len=output_len
        )

        self.discriminator = Discriminator(
            condition_dim=condition_dim
        )

        # Automatic optimization for multiple optimizers
        self.automatic_optimization = False

    def forward(self, z, sales_history, temporal_features, review_features):
        """Generate samples (for inference)."""
        return self.generator(z, sales_history, temporal_features, review_features)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """
        Training step with alternating updates.

        Trains discriminator n_critic times, then generator once.
        """
        # Get optimizers
        opt_d, opt_g = self.optimizers()

        sales_history = batch['sales_history']
        temporal = batch['temporal_features']
        review = batch['review_features']
        real_sales = batch['target_sales']

        batch_size = real_sales.size(0)

        # ==================
        # Train Discriminator
        # ==================
        for _ in range(self.n_critic):
            opt_d.zero_grad()

            # Sample noise
            z = torch.randn(batch_size, self.noise_dim, device=self.device)

            # Generate fake samples
            fake_sales = self.generator(z, sales_history, temporal, review)

            # Discriminator scores
            real_score = self.discriminator(real_sales, sales_history, temporal, review)
            fake_score = self.discriminator(fake_sales.detach(), sales_history, temporal, review)

            # Wasserstein loss
            d_loss = fake_score.mean() - real_score.mean()

            # Gradient penalty
            gp = self.gradient_penalty(real_sales, fake_sales, sales_history, temporal, review)

            # Total discriminator loss
            d_loss_total = d_loss + self.lambda_gp * gp

            # Backward and optimize
            self.manual_backward(d_loss_total)
            opt_d.step()

        # Log discriminator metrics
        self.log('d_loss', d_loss, prog_bar=True)
        self.log('gp', gp, prog_bar=True)
        self.log('real_score', real_score.mean())
        self.log('fake_score', fake_score.mean())

        # ================
        # Train Generator
        # ================
        opt_g.zero_grad()

        # Sample new noise
        z = torch.randn(batch_size, self.noise_dim, device=self.device)

        # Generate fake samples
        fake_sales = self.generator(z, sales_history, temporal, review)

        # Discriminator score for fake samples
        fake_score = self.discriminator(fake_sales, sales_history, temporal, review)

        # Generator loss (maximize discriminator score for fake samples)
        g_loss = -fake_score.mean()

        # Backward and optimize
        self.manual_backward(g_loss)
        opt_g.step()

        # Log generator metrics
        self.log('g_loss', g_loss, prog_bar=True)

        # Log Wasserstein distance (estimate)
        wasserstein_dist = real_score.mean() - fake_score.mean()
        self.log('wasserstein_dist', wasserstein_dist)

    def gradient_penalty(self, real, fake, sales_history, temporal, review):
        """
        Compute gradient penalty for WGAN-GP.

        Enforces 1-Lipschitz constraint on discriminator.
        """
        batch_size = real.size(0)

        # Random interpolation coefficient
        epsilon = torch.rand(batch_size, 1, device=self.device)

        # Interpolate between real and fake samples
        interpolated = epsilon * real + (1 - epsilon) * fake
        interpolated.requires_grad_(True)

        # Discriminator score for interpolated samples
        score = self.discriminator(interpolated, sales_history, temporal, review)

        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=score,
            inputs=interpolated,
            grad_outputs=torch.ones_like(score),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Flatten gradients
        gradients = gradients.view(batch_size, -1)

        # Compute gradient norm
        gradient_norm = gradients.norm(2, dim=1)

        # Gradient penalty: (||gradient|| - 1)^2
        penalty = ((gradient_norm - 1) ** 2).mean()

        return penalty

    def configure_optimizers(self):
        """Configure optimizers for Generator and Discriminator."""
        # Discriminator optimizer (higher learning rate)
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr_d,
            betas=(0.0, 0.9)  # Special betas for WGAN
        )

        # Generator optimizer
        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.lr_g,
            betas=(0.0, 0.9)  # Special betas for WGAN
        )

        return [opt_d, opt_g], []

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Validation step."""
        sales_history = batch['sales_history']
        temporal = batch['temporal_features']
        review = batch['review_features']
        real_sales = batch['target_sales']

        batch_size = real_sales.size(0)

        # Generate samples
        z = torch.randn(batch_size, self.noise_dim, device=self.device)
        fake_sales = self.generator(z, sales_history, temporal, review)

        # Compute scores
        real_score = self.discriminator(real_sales, sales_history, temporal, review)
        fake_score = self.discriminator(fake_sales, sales_history, temporal, review)

        # Log validation metrics
        self.log('val_real_score', real_score.mean())
        self.log('val_fake_score', fake_score.mean())

        # Compute quality metrics
        mae = torch.abs(fake_sales - real_sales).mean()
        self.log('val_mae', mae)

    def on_train_epoch_end(self):
        """Hook called at the end of each training epoch."""
        # Sample and log generated sequences for visualization
        if self.current_epoch % 10 == 0:
            self.sample_and_visualize()

    @torch.no_grad()
    def sample_and_visualize(self, num_samples: int = 5):
        """Sample and visualize generated sequences."""
        # Get a batch from validation
        val_loader = self.trainer.val_dataloaders
        if val_loader is not None:
            batch = next(iter(val_loader))

            # Move to device
            sales_history = batch['sales_history'][:num_samples].to(self.device)
            temporal = batch['temporal_features'][:num_samples].to(self.device)
            review = batch['review_features'][:num_samples].to(self.device)
            real_sales = batch['target_sales'][:num_samples].to(self.device)

            # Generate samples
            z = torch.randn(num_samples, self.noise_dim, device=self.device)
            fake_sales = self.generator(z, sales_history, temporal, review)

            # Log to tensorboard (avoid print statements in Jupyter to prevent recursion)
            self.logger.experiment.add_scalars(
                'sample_comparison',
                {
                    'real_mean': real_sales.mean().item(),
                    'fake_mean': fake_sales.mean().item(),
                    'real_std': real_sales.std().item(),
                    'fake_std': fake_sales.std().item(),
                },
                self.current_epoch
            )


if __name__ == "__main__":
    # Test WGAN-GP module
    print("Testing WGAN-GP module...")

    model = WGANGP(
        noise_dim=128,
        condition_dim=512,
        hidden_dim=256,
        output_len=7,
        lambda_gp=10.0,
        n_critic=5
    )

    print(f"\n✓ WGAN-GP module initialized successfully!")

    # Create dummy batch
    batch = {
        'sales_history': torch.randn(16, 30),
        'temporal_features': torch.randn(16, 8),
        'review_features': torch.randn(16, 2),
        'target_sales': torch.randn(16, 7)
    }

    # Test gradient penalty
    real = batch['target_sales']
    z = torch.randn(16, 128)
    fake = model.generator(z, batch['sales_history'], batch['temporal_features'], batch['review_features'])
    gp = model.gradient_penalty(real, fake, batch['sales_history'], batch['temporal_features'], batch['review_features'])

    print(f"\nGradient penalty: {gp.item():.4f}")
    print(f"✓ Gradient penalty computed successfully!")

    # Print model summary
    total_params_g = sum(p.numel() for p in model.generator.parameters())
    total_params_d = sum(p.numel() for p in model.discriminator.parameters())
    print(f"\nModel parameters:")
    print(f"  Generator: {total_params_g:,}")
    print(f"  Discriminator: {total_params_d:,}")
    print(f"  Total: {total_params_g + total_params_d:,}")
