# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Conditional GAN-based Sales Prediction Model using WGAN-GP. Two-stage system:
1. Train conditional GAN to generate synthetic sales sequences conditioned on historical sales, reviews, and temporal features
2. Augment downstream forecasting models (LSTM) with synthetic data and measure improvement

**Tech Stack**: PyTorch Lightning, Hydra config management, designed for both local and Kaggle environments.

---

## Essential Commands

### Development Workflow

```bash
# Quick test (5 epochs, 20% data, ~5 minutes)
python scripts/train_gan.py quick_run.enabled=true

# Full training locally
python scripts/train_gan.py

# Override config from CLI (Hydra syntax)
python scripts/train_gan.py training.max_epochs=200 data.batch_size=256 training.lambda_gp=15.0

# Full pipeline (preprocess → train GAN → generate samples → train baselines)
python scripts/preprocess_data.py
python scripts/train_gan.py
python scripts/generate_samples.py --checkpoint checkpoints/best_model.ckpt --num_samples_per_real 5
python scripts/train_baseline.py
python scripts/train_baseline.py --augmented
```

### Kaggle-Specific Commands

```python
# Quick test on Kaggle (~10-15 minutes on P100)
!python kaggle_quickstart.py

# Full Kaggle pipeline (~2-3 hours on P100)
!python scripts/run_kaggle.py

# Custom Kaggle runs
!python scripts/run_kaggle.py --max-epochs 30
!python scripts/run_kaggle.py --skip-preprocess  # If data already processed
!python scripts/run_kaggle.py --skip-baseline    # Skip LSTM comparison

# View training logs
%load_ext tensorboard
%tensorboard --logdir /kaggle/working/logs  # On Kaggle
%tensorboard --logdir logs                   # Locally
```

### Environment Setup

```bash
# Local setup
pip install -r requirements.txt
python scripts/verify_setup.py

# Kaggle setup (in notebook)
!git clone https://github.com/YOUR_USERNAME/dda4220.git
%cd dda4220
!pip install -q -r requirements.txt
```

---

## Architecture Overview

### Two-Stage Pipeline

```
Stage 1: Conditional GAN Training
Raw Olist Data → OlistPreprocessor → Product-Day Panel → SalesDataModule
→ WGANGP (Lightning) → Trained Generator → generate_samples.py → Synthetic Data

Stage 2: Baseline Comparison
Real Data → LSTM Baseline
Real + Synthetic → LSTM Augmented → Compare MAE/RMSE improvements
```

### Key Components

**Data Pipeline** (`src/data/`):
- `preprocessor.py`: `OlistPreprocessor` - Joins Olist CSVs, aggregates to product-day level, fills missing days, adds temporal features, normalizes per-product
- `datamodule.py`: `SalesDataModule` (Lightning DataModule) - Handles train/val/test splits, supports augmentation
- `dataset.py`: `SalesDataset` (windowed sequences: 30-day history → 7-day target), `AugmentedSalesDataset` (real + synthetic)

**GAN Models** (`src/models/gan/`):
- `wgan_gp.py`: `WGANGP` - Main Lightning module, implements manual optimization with alternating discriminator/generator updates, Wasserstein loss + gradient penalty
- `encoders.py`: `ConditionEncoder` - Fuses sales history (LSTM), temporal features (FC), review features (FC) into 512-dim embedding
- `generator.py`: `Generator` - Takes noise + condition → GRU decoder → 7-day sales sequence (with ReLU for non-negative)
- `discriminator.py`: `Discriminator` - Conv1D on sequence + condition fusion → realism score (spectral norm for stability)

**Baselines** (`src/models/baselines/`):
- `lstm_forecaster.py`: `LSTMForecaster` - Simple LSTM for comparison, trained on real or augmented data

**Utilities** (`src/utils/`):
- `kaggle_utils.py`: Environment detection (`is_kaggle_environment()`), path management (`get_kaggle_paths()`), dataset discovery

### Configuration (Hydra)

All hyperparameters in `configs/config.yaml` (local) or `configs/config_kaggle.yaml` (Kaggle).

**Critical settings**:
- `training.n_critic: 5` - Discriminator updates per generator update
- `training.lambda_gp: 10.0` - Gradient penalty weight
- `training.lr_d: 0.0004` (discriminator) vs `training.lr_g: 0.0001` (generator) - Imbalanced learning rates for stability
- `data.history_window: 30` and `data.forecast_horizon: 7` - Sequence lengths
- `quick_run.enabled: true` - Activates fast testing mode (reduces epochs, limits batches)

**Override via CLI**: `python scripts/train_gan.py training.lambda_gp=15.0 data.batch_size=64`

### Manual Optimization in WGANGP

`src/models/gan/wgan_gp.py` uses `automatic_optimization=False` because of alternating updates:

```python
def training_step(self, batch, batch_idx):
    opt_d, opt_g = self.optimizers()

    # Train discriminator (5 steps)
    for _ in range(self.n_critic):
        d_loss = wasserstein_loss + lambda_gp * gradient_penalty
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()

    # Train generator (1 step)
    g_loss = -discriminator_score_on_fake
    opt_g.zero_grad()
    self.manual_backward(g_loss)
    opt_g.step()
```

---

## Environment Detection & Path Management

The codebase automatically adapts to local vs Kaggle:

```python
from src.utils.kaggle_utils import is_kaggle_environment, get_kaggle_paths

if is_kaggle_environment():
    # Uses /kaggle/input/ and /kaggle/working/
    paths = get_kaggle_paths()
else:
    # Uses data/, checkpoints/, logs/
    paths = get_kaggle_paths()
```

**Kaggle paths**:
- Input data: `/kaggle/input/brazilian-ecommerce/` (or similar, auto-discovered)
- Checkpoints: `/kaggle/working/checkpoints/`
- Logs: `/kaggle/working/logs/`
- Processed data: `/kaggle/working/processed/`

**Local paths**:
- Input data: `data/raw/`
- Checkpoints: `checkpoints/`
- Logs: `logs/`
- Processed data: `data/processed/`

---

## Important Implementation Details

### WGAN-GP Specifics

1. **Gradient Penalty**: Enforces 1-Lipschitz constraint on discriminator
   - Computed on interpolated samples between real and fake
   - `lambda_gp=10.0` is standard value from paper

2. **Critic Updates**: `n_critic=5` means discriminator trains 5x more than generator
   - Prevents generator from overpowering discriminator

3. **No Sigmoid**: Discriminator outputs raw scores (not probabilities) for Wasserstein loss

4. **Special Optimizer Config**: `Adam(betas=(0.0, 0.9))` instead of default `(0.9, 0.999)`
   - From WGAN-GP paper recommendations

5. **Spectral Normalization**: Applied to all discriminator linear layers for additional stability

### Conditional Generation

ALL components are fully conditional:
- Generator: `G(z | sales_history, temporal, reviews)`
- Discriminator: `D(x | sales_history, temporal, reviews)`
- Shared condition encoder ensures G and D see same context

**Why conditioning matters**: Unconditional GANs would generate arbitrary sales sequences. Conditioning on history and context ensures generated sequences are realistic for the specific product/time context.

### Data Normalization

**Per-product min-max normalization**: Each product's sales are normalized to [0,1] independently. This prevents high-volume products from dominating the loss function.

**Temporal features**: One-hot encoded day_of_week (7 dims) + is_weekend (1 dim) = 8 dims

**Review features**: avg_rating (normalized [0,1]) + log(review_count + 1) (normalized)

### Chronological Splits

**Critical**: Data is split chronologically, NOT randomly:
- Train: First 70% of timeline
- Val: Next 15%
- Test: Final 15%

This prevents information leakage (training on future data to predict past).

---

## Common Development Tasks

### Adding New Hyperparameters

1. Add to `configs/config.yaml`:
   ```yaml
   training:
     new_param: value
   ```

2. Access in code:
   ```python
   @hydra.main(config_path="../configs", config_name="config")
   def train(cfg: DictConfig):
       new_param = cfg.training.new_param
   ```

3. Override from CLI:
   ```bash
   python scripts/train_gan.py training.new_param=new_value
   ```

### Modifying Generator/Discriminator

Edit `src/models/gan/generator.py` or `src/models/gan/discriminator.py`.

**Important**: Keep input/output signatures compatible with `WGANGP`:
- Generator: `forward(noise, history, review_features, temporal_features) -> generated_sequence`
- Discriminator: `forward(sequence, history, review_features, temporal_features) -> score`

### Adding New Metrics

Add to `src/evaluation/metrics.py`:
```python
def new_metric(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    # Implementation
    return metric_value
```

Use in training:
```python
from src.evaluation.metrics import new_metric
val_metric = new_metric(y_true, y_pred)
self.log('val_new_metric', val_metric)
```

### Debugging Training Issues

**GAN training instability**:
1. Increase `lambda_gp` (try 15.0 or 20.0)
2. Decrease `lr_d` (try 2e-4)
3. Increase `n_critic` (try 7)
4. Check discriminator isn't outputting extreme values (add gradient clipping)

**Out of memory**:
1. Reduce `data.batch_size` (try 64 or 32)
2. Reduce `model.hidden_dim` (try 128)
3. Use gradient accumulation in Lightning

**Loss not decreasing**:
1. Check data normalization (should be [0,1])
2. Verify generator outputs non-negative values (ReLU activation)
3. Enable quick_run mode to iterate faster

---

## Testing on Kaggle Before Full Training

Always run quick test first:

```python
# ~10-15 minutes, verifies everything works
!python kaggle_quickstart.py

# If successful, run full training
!python scripts/run_kaggle.py --max-epochs 30  # ~1 hour on P100
```

**Expected Kaggle timeline** (P100 GPU):
- Data preprocessing: ~5 minutes
- GAN training (50 epochs): ~2 hours
- Sample generation: ~10 minutes
- Baseline training: ~30 minutes each
- Total: ~3-4 hours for complete pipeline

---

## Project-Specific Conventions

1. **All scripts in `scripts/` are entry points**: They use `if __name__ == "__main__"` and can be run directly

2. **All modules in `src/` are importable**: They never run directly, only imported by scripts

3. **Hydra configs control everything**: Avoid hardcoded hyperparameters, always use `cfg.section.param`

4. **Lightning modules for all models**: Use `pl.LightningModule` and `pl.LightningDataModule` for consistency

5. **Kaggle compatibility required**: Any new feature must work on both local and Kaggle environments (use `kaggle_utils`)

6. **Reproducibility**: Always set seeds via `pl.seed_everything(cfg.seed)` at script start

7. **Checkpointing**: Lightning automatically saves checkpoints, configure via `checkpoint` section in config

8. **Logging**: Use `self.log()` in Lightning modules, logged to TensorBoard automatically

---

## File Locations Reference

**Key files for modifications**:
- GAN architecture: `src/models/gan/{generator,discriminator,encoders}.py`
- GAN training loop: `src/models/gan/wgan_gp.py` (training_step method)
- Data preprocessing: `src/data/preprocessor.py`
- Configuration: `configs/config.yaml` (local) or `configs/config_kaggle.yaml` (Kaggle)
- Entry points: `scripts/train_gan.py`, `scripts/run_kaggle.py`

**Generated outputs**:
- Checkpoints: `checkpoints/` (local) or `/kaggle/working/checkpoints/` (Kaggle)
- Logs: `logs/` (local) or `/kaggle/working/logs/` (Kaggle)
- Preprocessed data: `data/processed/product_daily_panel.parquet`
- Synthetic samples: `data/synthetic/gan_samples.pt` (or specified path)

---

## Dependencies

Main libraries (see `requirements.txt`):
- `pytorch-lightning>=2.0.0` - Training framework
- `torch>=2.0.0` - Deep learning
- `hydra-core>=1.3.0` - Configuration management
- `pandas>=2.0.0`, `pyarrow>=12.0.0` - Data processing
- `wandb>=0.15.0` - Experiment tracking (optional)
- `kaggle>=1.5.0` - Kaggle API for data download

Install all: `pip install -r requirements.txt`

---

## Kaggle Setup Requirements

**Notebook Configuration**:
1. GPU: P100 or T4 x2 (Settings → Accelerator)
2. Internet: On (Settings → Internet)
3. Dataset: "Brazilian E-Commerce Public Dataset by Olist" added as input

**GitHub Integration**:
- Either use "File → Link to GitHub" or manually clone repo
- Repo must be public or Kaggle authorized to access private repos

**Expected Kaggle Inputs**:
- Dataset will be at `/kaggle/input/brazilian-ecommerce/` (or similar)
- Code automatically discovers correct path using `get_olist_data_path()`

---

## Common Errors and Solutions

**Error: "No module named 'src'"**
- Solution: Make sure you're in the repo root directory: `%cd dda4220`

**Error: "Dataset not found"**
- Solution: Add "Brazilian E-Commerce Public Dataset by Olist" to Kaggle inputs

**Error: "CUDA out of memory"**
- Solution: Use `--quick` mode or reduce batch size: `python scripts/train_gan.py data.batch_size=32`

**Error: "Discriminator loss explodes"**
- Solution: Increase gradient penalty: `python scripts/train_gan.py training.lambda_gp=15.0`

**Error: "Generator loss not decreasing"**
- Solution: Check data normalization, ensure generator outputs are non-negative, try reducing learning rate

**Error: "Kaggle session timeout"**
- Solution: Kaggle notebooks have 12-hour limit. Use `--max-epochs 30` for faster completion

---

## Notes for Future Developers

1. **Manual optimization in WGANGP**: This is intentional for alternating D/G updates. Don't switch to automatic optimization without understanding implications.

2. **Chronological splits**: Never use random train/test split for time series data in this project.

3. **Condition encoder architecture**: The 512-dim bottleneck is carefully tuned. Changes may affect training stability.

4. **Kaggle paths**: Always use `get_kaggle_paths()` for new file I/O, never hardcode `/kaggle/` paths.

5. **WGAN-GP hyperparameters**: The current values (lambda_gp=10, n_critic=5, lr_d=4e-4, lr_g=1e-4) are from the paper and work well. Changes should be tested carefully.

6. **Hydra working directory**: `hydra.run.dir: .` in config is critical for Kaggle. Don't remove this.

7. **Per-product normalization**: This is essential for handling products with vastly different sales volumes. Don't switch to global normalization.
