# Conditional GAN-based Sales Prediction Model

**Authors:** Li Siqi, Li Chudikang, Yichun Wang

A novel approach to sales forecasting using Conditional Generative Adversarial Networks (cGAN) with WGAN-GP. This system generates realistic future sales sequences conditioned on historical sales, customer reviews, and temporal features, then uses the synthetic data to augment training of downstream forecasting models.

---

## Quick Start

### Option A: Run on Kaggle (Recommended)

**The easiest way to run this project is on Kaggle with GPU support!**

1. **Fork this GitHub repository** or upload it to your GitHub account

2. **Create a new Kaggle Notebook**
   - Go to https://www.kaggle.com/code
   - Click "New Notebook"
   - Enable GPU: Settings → Accelerator → GPU T4 x2 (or P100)

3. **Connect to GitHub**
   - In the notebook, click "File" → "Link to GitHub"
   - Authorize Kaggle to access your repository
   - Select this repository

4. **Add the Olist dataset**
   - Click "Add Input" → "Datasets"
   - Search for "Brazilian E-Commerce Public Dataset by Olist"
   - Add the dataset

5. **Run the pipeline**
   ```python
   # Clone the repo (if not using GitHub integration)
   !git clone https://github.com/YOUR_USERNAME/dda4220.git
   %cd dda4220

   # Install dependencies
   !pip install -q -r requirements.txt

   # Run the complete pipeline
   !python scripts/run_kaggle.py

   # Or for a quick test run (5 epochs, 20% data)
   !python scripts/run_kaggle.py --quick
   ```

**Advantages of Kaggle:**
- Free GPU access (30+ hours/week)
- Olist dataset already available
- No local setup required
- Easy to share and reproduce results

---

### Option B: Run Locally

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch, pytorch_lightning; print('✓ Installation successful!')"
```

### 2. Download and Preprocess Data

```bash
# Download Olist dataset from Kaggle (requires Kaggle API configured)
python scripts/preprocess_data.py
```

**Note:** If automatic download fails, manually download from [Kaggle Olist Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and extract to `data/raw/`.

### 3. Train the cGAN

```bash
# Train WGAN-GP with default configuration
python scripts/train_gan.py

# Or customize parameters
python scripts/train_gan.py training.max_epochs=200 data.batch_size=256
```

### 4. Generate Synthetic Samples

```bash
# Generate 5 synthetic samples per real sample
python scripts/generate_samples.py \
    --checkpoint checkpoints/best_model.ckpt \
    --num_samples_per_real 5
```

### 5. Train and Compare Baselines

```bash
# Train LSTM on real data only
python scripts/train_baseline.py

# Train LSTM on real + synthetic data
python scripts/train_baseline.py --augmented
```

---

## Project Structure

```
dda4220/
├── data/
│   ├── raw/                          # Olist dataset (downloaded)
│   ├── processed/                    # Preprocessed product-day panel
│   └── synthetic/                    # GAN-generated samples
├── src/
│   ├── data/
│   │   ├── preprocessor.py          # Data preprocessing pipeline
│   │   ├── dataset.py               # PyTorch Dataset classes
│   │   └── datamodule.py            # Lightning DataModule
│   ├── models/
│   │   ├── gan/
│   │   │   ├── encoders.py          # Condition encoder (sales + reviews + temporal)
│   │   │   ├── generator.py         # Generator G(z|y)
│   │   │   ├── discriminator.py     # Discriminator D(x|y)
│   │   │   └── wgan_gp.py           # WGAN-GP training module
│   │   └── baselines/
│   │       └── lstm_forecaster.py   # LSTM baseline
│   └── evaluation/
│       └── metrics.py               # MAE, RMSE, sMAPE, etc.
├── scripts/
│   ├── preprocess_data.py           # Download & preprocess
│   ├── train_gan.py                 # Train cGAN
│   ├── generate_samples.py          # Generate synthetic data
│   └── train_baseline.py            # Train baselines
├── configs/
│   └── config.yaml                  # Hydra configuration
└── requirements.txt
```

---

## Model Architecture

### Conditional Generator G(z|y)

Generates 7-day sales sequences conditioned on multi-modal features:

```
Input:
  - Noise z ~ N(0,1) (128-dim)
  - Sales history (30 days) → LSTM encoder
  - Review features (avg_rating, review_count) → FC projection
  - Temporal features (day_of_week, is_weekend) → FC projection

Architecture:
  1. Encode conditions → 512-dim embedding
  2. Concatenate [z, condition]
  3. GRU decoder (3 layers, 256 hidden)
  4. Output 7-day sequence with ReLU

Output: (batch, 7) - generated sales sequence
```

### Conditional Discriminator D(x|y)

Scores realism of sales sequences given conditions:

```
Input:
  - Sales sequence (7 days)
  - Conditions (sales + reviews + temporal)

Architecture:
  1. Temporal CNN (filters: 64, 128, 256)
  2. Encode conditions → 512-dim
  3. Concatenate [sequence_features, condition]
  4. FC layers with spectral normalization
  5. Output scalar score (no sigmoid for WGAN)

Output: (batch, 1) - realism score
```

### WGAN-GP Training

- **Wasserstein loss** for stable training
- **Gradient penalty** (λ=10) for 1-Lipschitz constraint
- **Critic updates:** 5 discriminator updates per generator update
- **Optimizers:** Adam with betas=(0.0, 0.9)
- **Learning rates:** lr_d=4e-4, lr_g=1e-4

---

## Configuration

All hyperparameters can be configured via `configs/config.yaml` or command-line:

```bash
# Change batch size
python scripts/train_gan.py data.batch_size=256

# Change learning rates
python scripts/train_gan.py training.lr_g=0.0002 training.lr_d=0.0008

# Change gradient penalty coefficient
python scripts/train_gan.py training.lambda_gp=15.0

# Disable W&B logging
python scripts/train_gan.py wandb.mode=disabled
```

---

## Evaluation

### Compare Models

Train both real-only and augmented baselines, then compare:

```bash
# Real-only baseline
python scripts/train_baseline.py

# Augmented baseline
python scripts/train_baseline.py --augmented

# Results will be logged to TensorBoard
tensorboard --logdir logs
```

### Metrics

All models are evaluated using:
- **MAE** (Mean Absolute Error) ↓
- **RMSE** (Root Mean Squared Error) ↓
- **sMAPE** (Symmetric Mean Absolute Percentage Error) ↓
- **MAPE** (Mean Absolute Percentage Error) ↓
- **WAPE** (Weighted Absolute Percentage Error) ↓

**Expected improvement:** 5-15% reduction in MAE with augmentation vs. real-only.

---

## Troubleshooting

### Data Download Issues

If `kaggle datasets download` fails:
1. Manually download from https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
2. Extract all CSV files to `data/raw/`
3. Run preprocessing: `python scripts/preprocess_data.py`

### CUDA/GPU Issues

If CUDA errors occur:
```bash
# Train on CPU
python scripts/train_gan.py trainer.accelerator=cpu
```

### GAN Training Instability

If generator loss explodes or discriminator dominates:
- Increase gradient penalty: `training.lambda_gp=15.0`
- Decrease discriminator learning rate: `training.lr_d=0.0002`
- Increase n_critic: `training.n_critic=7`

### Out of Memory

If batch size too large:
```bash
# Reduce batch size
python scripts/train_gan.py data.batch_size=64
```

---

## Key Implementation Details

### WGAN-GP Stability Tips
1. ✓ **Gradient penalty essential:** λ=10 prevents mode collapse
2. ✓ **Discriminator advantage:** 5 critic updates per generator update
3. ✓ **No batch normalization in discriminator:** Uses LayerNorm
4. ✓ **Learning rate imbalance:** lr_d (4e-4) > lr_g (1e-4)
5. ✓ **Spectral normalization:** Applied to all discriminator FC layers
6. ✓ **Special Adam betas:** (0.0, 0.9) for WGAN stability

### Data Processing
- **Normalization:** Min-max per product → [0, 1]
- **Missing days:** Filled with zeros
- **Minimum history:** 60 days required
- **Time-based split:** 70% train / 15% val / 15% test (chronological)

---

## Expected Results

### Minimum Viable Product (MVP)
- ✓ Working cGAN generating plausible 7-day sales sequences
- ✓ Synthetic data improves LSTM forecaster MAE by 5-10%
- ✓ Evidence that conditioning on reviews/temporal helps

### Success Criteria

**Quantitative:**
- MAE reduction ≥5% with augmentation vs baseline
- Generated samples pass visual inspection (realistic distributions)
- WGAN training stable (no mode collapse)

**Qualitative:**
- Generated sequences respect temporal patterns (weekday/weekend)
- Negative sentiment correlates with lower sales
- Model captures holiday effects

---

## Advanced Usage

### Training with Custom Data

1. Prepare data in the same format as Olist:
   - `product_id`, `date`, `daily_sales`, `avg_rating`, `review_count`
   - Temporal features: `day_of_week`, `is_weekend`, etc.

2. Save as parquet: `your_data.parquet`

3. Train:
```bash
python scripts/train_gan.py data.path=your_data.parquet
```

### Hyperparameter Tuning

Key hyperparameters to tune:
- `model.hidden_dim`: 128, 256, 512
- `training.lambda_gp`: 5, 10, 15
- `training.n_critic`: 3, 5, 7
- `model.condition_dim`: 256, 512, 1024

### Extending the Model

To add BERT sentiment embeddings (instead of simple ratings):

1. Extract BERT embeddings:
```python
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained('neuralmind/bert-base-portuguese-cased')
# Extract [CLS] token embeddings for reviews
```

2. Modify `src/features/review_encoder.py` to load precomputed embeddings

3. Update `src/data/dataset.py` to include BERT features

---

## Citation

If you use this code for your research, please cite:

```bibtex
@article{li2024conditional,
  title={Conditional GAN-based Sales Prediction Model Using Reviews and Sales Data},
  author={Li, Siqi and Li, Chudikang and Wang, Yichun},
  year={2024}
}
```

---

## References

- **WGAN-GP:** Gulrajani et al. (2017). "Improved Training of Wasserstein GANs"
- **Time-series GAN:** Yoon et al. (2019). "Time-series Generative Adversarial Networks"
- **Olist Dataset:** Brazilian E-Commerce Public Dataset by Olist

---

## License

This project is for academic use only.

---

## Running on Kaggle - Detailed Guide

### Step-by-Step Kaggle Setup

#### 1. Prepare Your GitHub Repository

```bash
# Make sure all files are committed
git add .
git commit -m "Ready for Kaggle"
git push origin main
```

#### 2. Create Kaggle Notebook

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click **"New Notebook"**
3. Change settings:
   - **Notebook Type**: Notebook (not Script)
   - **Language**: Python
   - **Accelerator**: GPU T4 x2 or GPU P100

#### 3. Add Input Data

In your Kaggle notebook:
1. Click **"Add Input"** (right sidebar)
2. Search for **"Brazilian E-Commerce Public Dataset by Olist"**
3. Click **"Add"** on the dataset by olistbr

#### 4. Setup Code in Kaggle Notebook

Create a new cell and run:

```python
# Option 1: If using GitHub integration
# (File → Link to GitHub → Select your repo)
# Your code will already be available

# Option 2: Clone from GitHub manually
!git clone https://github.com/YOUR_USERNAME/dda4220.git
%cd dda4220

# Install dependencies
!pip install -q -r requirements.txt

# Verify GPU
import torch
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

#### 5. Run the Complete Pipeline

**Full Training (recommended):**
```python
!python scripts/run_kaggle.py
```

**Quick Test (5 epochs, 20% data):**
```python
!python scripts/run_kaggle.py --quick
```

**Custom Configuration:**
```python
# Set specific number of epochs
!python scripts/run_kaggle.py --max-epochs 30

# Skip preprocessing if data is already preprocessed
!python scripts/run_kaggle.py --skip-preprocess

# Skip baseline comparison to save time
!python scripts/run_kaggle.py --skip-baseline
```

### Kaggle-Specific Features

The code automatically detects Kaggle environment and adjusts:

1. **Data Paths**: Uses `/kaggle/input/` for datasets
2. **Output Paths**: Saves to `/kaggle/working/` (accessible after run)
3. **GPU Settings**: Auto-configures for Kaggle GPUs
4. **Resource Optimization**: Adjusts batch size and workers for Kaggle

### Expected Runtime on Kaggle

- **Quick mode** (`--quick`): ~10-15 minutes
- **Full training** (50 epochs): ~2-3 hours with GPU
- **Complete pipeline** (with baselines): ~3-4 hours

### Downloading Results from Kaggle

After training completes, download your results:

1. Click **"Output"** in the right sidebar
2. Download the generated files:
   - `checkpoints/` - Trained model weights
   - `logs/` - TensorBoard logs
   - `synthetic_samples.parquet` - Generated data

### Viewing TensorBoard on Kaggle

```python
# Load TensorBoard in Kaggle notebook
%load_ext tensorboard
%tensorboard --logdir /kaggle/working/logs
```

### Troubleshooting on Kaggle

**Out of Memory Error:**
```python
# Reduce batch size
!python scripts/run_kaggle.py --quick
```

**Session Timeout:**
- Kaggle notebooks have 12-hour limit
- Use `--max-epochs 30` for faster training
- Save checkpoints frequently (already configured)

**Dataset Not Found:**
- Make sure you added "Brazilian E-Commerce Public Dataset by Olist" as input
- Check dataset name in `/kaggle/input/` directory

**GPU Not Available:**
- Go to Settings (right sidebar)
- Change Accelerator to GPU T4 x2 or GPU P100
- Click "Save"

---

## Contact

For questions or issues, please contact:
- Li Chudikang: 122040057@link.cuhk.edu.cn

---

**Happy Training! 🚀**
