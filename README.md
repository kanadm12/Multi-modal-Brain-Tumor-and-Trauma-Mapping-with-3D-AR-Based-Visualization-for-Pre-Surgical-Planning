# BraTS 3D Brain Tumor Segmentation

**Optimized for RunPod: 4x NVIDIA A100 80GB GPUs**

High-performance 3D U-Net with Transformer bottleneck for BraTS brain tumor segmentation.  
Target: **90-95% Dice Score**

---

## 📁 Directory Structure

```
BraTS_Optimized_Solution/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── train.py                       # Main training script (renamed from optimized_brats_final.py)
├── download_from_azure.py         # Download dataset from Azure Blob Storage
├── redownload_corrupted_files.py  # Fix corrupted/incomplete downloads
├── .env.example                   # Template for environment variables
├── .env                           # Your Azure credentials (DO NOT COMMIT)
└── .gitignore                     # Git ignore rules

# Created at runtime on RunPod:
/workspace/
├── dataset/                       # BraTS dataset (downloaded from Azure)
│   ├── BraTS2021_00000/
│   │   ├── BraTS2021_00000_t1.nii.gz
│   │   ├── BraTS2021_00000_t1ce.nii.gz
│   │   ├── BraTS2021_00000_t2.nii.gz
│   │   ├── BraTS2021_00000_flair.nii.gz
│   │   └── BraTS2021_00000_seg.nii.gz
│   └── ...
├── outputs/                       # Training logs and metrics
├── checkpoints/                   # Model checkpoints
└── tensorboard/                   # TensorBoard logs
```

---

## 🚀 Quick Start (RunPod)

### 1. Launch RunPod Instance

- **Pod Type:** GPU Pod
- **GPU:** 4x NVIDIA A100 80GB
- **Container Image:** `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`
- **Volume:** 500GB+ (for dataset and checkpoints)

### 2. Clone Repository

```bash
cd /workspace
git clone <your-repo-url> brats
cd brats
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Azure Credentials

```bash
# Option 1: Environment variable
export AZURE_STORAGE_CONNECTION_STRING='your_connection_string_here'

# Option 2: Create .env file
cp .env.example .env
nano .env  # Add your connection string
```

### 5. Download Dataset

```bash
python download_from_azure.py --output-dir /workspace/dataset
```

### 6. Start Training

```bash
# Multi-GPU training with 4x A100
python train.py
```

---

## ⚙️ Configuration

### Hardware-Optimized Settings (4x A100 80GB)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `CROP_SIZE` | (192, 224, 192) | Input volume size |
| `MODEL_FILTERS` | [64, 128, 256, 512, 1024] | Channel progression |
| `BATCH_SIZE` | 4 | Per GPU (16 total) |
| `ACCUMULATION_STEPS` | 2 | Effective batch = 32 |
| `TRANSFORMER_DEPTH` | 3 | Attention layers |
| `NUM_WORKERS` | 8 | DataLoader workers |

### Training Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| `EPOCHS` | 500 | Maximum epochs |
| `INITIAL_LR` | 2e-4 | Initial learning rate |
| `PATIENCE` | 100 | Early stopping patience |
| `WARMUP_EPOCHS` | 30 | LR warmup period |
| `N_FOLDS` | 3 | Cross-validation folds |

### Loss Function Weights

| Component | Weight | Purpose |
|-----------|--------|---------|
| Dice Loss | 0.45 | Overlap optimization |
| Boundary Loss | 0.20 | Edge refinement |
| Tversky Loss | 0.15 | Class imbalance |
| Lovasz Loss | 0.10 | IoU optimization |
| CrossEntropy | 0.10 | Pixel classification |

---

## 📊 Model Architecture

```
3D U-Net with Transformer Bottleneck
├── Encoder: 5 levels [64 → 128 → 256 → 512 → 1024]
├── Bottleneck: Multi-Head Self-Attention (8 heads, depth 3)
├── Decoder: 5 levels with skip connections
├── Deep Supervision: Auxiliary outputs at each level
└── Output: 4-class segmentation (BG, NCR, ED, ET)
```

### Key Features

- **Mixed Precision (AMP):** 2x faster training, 50% memory reduction
- **Distributed Data Parallel (DDP):** Efficient multi-GPU scaling
- **Gradient Checkpointing:** Optional memory optimization
- **Test-Time Augmentation (TTA):** 12 transforms for robust inference
- **Learning Rate Warmup:** 30 epochs for transformer stability
- **Adaptive Post-processing:** Size-based tumor refinement

---

## 🔧 Utility Scripts

### Download Dataset

```bash
# Full download
python download_from_azure.py --output-dir /workspace/dataset

# With progress logging
python download_from_azure.py --output-dir /workspace/dataset --verbose
```

### Fix Corrupted Files

```bash
# Scan and redownload corrupted files
python redownload_corrupted_files.py --output-dir /workspace/dataset

# Dry run (check without downloading)
python redownload_corrupted_files.py --output-dir /workspace/dataset --dry-run
```

---

## 📈 Expected Performance

| Metric | Target | Notes |
|--------|--------|-------|
| Mean Dice | 90-95% | Across all tumor regions |
| NCR Dice | 85%+ | Necrotic core |
| ED Dice | 90%+ | Edema |
| ET Dice | 88%+ | Enhancing tumor |
| HD95 | < 5mm | 95th percentile Hausdorff |

### Training Time Estimates (4x A100 80GB)

| Phase | Duration |
|-------|----------|
| Per Epoch | ~3-5 min |
| Full Training (500 epochs) | ~25-40 hours |
| 3-Fold CV | ~75-120 hours |

---

## 🔍 Monitoring Training

### TensorBoard

```bash
tensorboard --logdir /workspace/tensorboard --port 6006 --bind_all
```

Access at: `http://<pod-ip>:6006`

### Logs

```bash
# Real-time training logs
tail -f /workspace/outputs/training.log
```

---

## 💾 Checkpoints

Checkpoints are saved to `/workspace/checkpoints/`:

- `fold_X_best.pth` - Best model for each fold
- `fold_X_epoch_Y.pth` - Periodic checkpoints (every 25 epochs)

### Resume Training

Edit `train.py`:
```python
RESUME_TRAINING = True
RESUME_CHECKPOINT_PATH = "/workspace/checkpoints/fold_0_epoch_100.pth"
```

---

## ⚠️ Troubleshooting

### Out of Memory (OOM)

Reduce settings in `train.py`:
```python
CROP_SIZE = (160, 192, 160)  # Smaller input
BATCH_SIZE = 2               # Smaller batch
USE_GRADIENT_CHECKPOINTING = True  # Save memory
```

### Slow Data Loading

```python
NUM_WORKERS = 12  # Increase workers
USE_PREPROCESSED = True  # Use NPZ format
```

### Azure Download Fails

```bash
# Check connection
python -c "from azure.storage.blob import BlobServiceClient; print('OK')"

# Retry failed downloads
python redownload_corrupted_files.py
```

---

## 📝 Environment Variables

| Variable | Description |
|----------|-------------|
| `AZURE_STORAGE_CONNECTION_STRING` | Azure Blob Storage connection string |
| `CUDA_VISIBLE_DEVICES` | GPU selection (default: all) |
| `MASTER_ADDR` | DDP master address (default: localhost) |
| `MASTER_PORT` | DDP master port (default: 12355) |

---

## 📚 References

- [BraTS Challenge](https://www.med.upenn.edu/cbica/brats/)
- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)
- [PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

---

## License

MIT License
