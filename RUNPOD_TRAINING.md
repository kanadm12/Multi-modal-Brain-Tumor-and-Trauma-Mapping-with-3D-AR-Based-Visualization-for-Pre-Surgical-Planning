# BraTS Training on RunPod - Quick Start Guide

## 🚀 Getting Started on RunPod

### 1. Create a Pod

Select the following configuration:
- **GPU Type**: 4x AMD MI300X (192GB each)
- **Template**: PyTorch ROCm (latest)
- **Disk**: Minimum 500GB (for dataset + checkpoints)
- **Expose Ports**: `6006` (TensorBoard), `8888` (optional Jupyter)

### 2. Upload Your Data

SSH into your pod and upload your BraTS dataset:

```bash
# Your dataset should be structured as:
/workspace/dataset/
├── BraTS2021_00000/
│   ├── BraTS2021_00000_t1.nii.gz
│   ├── BraTS2021_00000_t1ce.nii.gz
│   ├── BraTS2021_00000_t2.nii.gz
│   ├── BraTS2021_00000_flair.nii.gz
│   └── BraTS2021_00000_seg.nii.gz
├── BraTS2021_00001/
│   └── ...
```

### 3. Clone/Upload Training Code

```bash
cd /workspace
git clone <your-repo> .  # or upload files directly
```

### 4. Install Dependencies

```bash
# PyTorch with ROCm should already be installed
pip install -r requirements_training.txt
```

### 5. Start Training

```bash
bash launch_training.sh
```

Or manually with torchrun:

```bash
export CLOUD_PLATFORM="runpod"
torchrun --nproc_per_node=4 train.py
```

---

## 📊 Monitoring Training

### TensorBoard

Access TensorBoard at: `http://<your-pod-ip>:6006`

Logged metrics:
- **Training**: Total loss + individual components (Dice, Boundary, Tversky, Lovasz, CE)
- **Validation**: Mean Dice, Mean HD95
- **Per-Class**: NCR/ED/ET Dice and HD95 scores
- **Learning Rate**: Current LR with warmup visualization

### Log Files

```bash
# Real-time training logs
tail -f /workspace/outputs/training.log

# GPU utilization
watch -n 1 rocm-smi
```

---

## 💾 Checkpointing

### Automatic Checkpoints

- **Best model**: Saved whenever validation Dice improves
- **Periodic**: Every 25 epochs
- **Resume**: Automatic on pod restart (RESUME_TRAINING=True)

### Checkpoint Locations

```
/workspace/checkpoints/
├── fold_0_best.pth       # Best model for fold 0
├── fold_0_epoch_25.pth   # Periodic checkpoint
├── fold_0_epoch_50.pth
├── fold_1_best.pth       # Best model for fold 1
└── ...
```

### Manual Resume

If training was interrupted:

```bash
# Training automatically resumes from latest checkpoint
bash launch_training.sh

# Or specify a specific checkpoint
export RESUME_CHECKPOINT_PATH="/workspace/checkpoints/fold_0_epoch_100.pth"
python train.py
```

---

## ⚙️ Configuration

Key settings in `train.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `BATCH_SIZE` | 12 | Per GPU (48 total with 4 GPUs) |
| `ACCUMULATION_STEPS` | 2 | Effective batch = 96 |
| `EPOCHS` | 300 | Max epochs (early stopping at 75) |
| `CROP_SIZE` | (224, 256, 224) | Input volume size |
| `MODEL_FILTERS` | [64, 128, 256, 512, 1024] | U-Net filter sizes |
| `TRANSFORMER_DEPTH` | 4 | Bottleneck transformer layers |
| `USE_TTA` | True | 12-point test time augmentation |
| `N_FOLDS` | 3 | Cross-validation folds |

---

## 📈 Expected Results

With proper training:
- **Mean Dice Score**: 90-95%
- **Mean HD95**: < 5mm
- **Training Time**: ~20-30 hours (4x MI300X)

---

## 🔧 Troubleshooting

### Out of Memory
Reduce batch size:
```python
BATCH_SIZE = 8  # Instead of 12
ACCUMULATION_STEPS = 3  # Keep effective batch ~96
```

### Slow Data Loading
Enable preprocessed data:
```python
USE_PREPROCESSED = True  # Requires running preprocessing script first
```

### GPU Not Detected
```bash
# Check ROCm installation
rocm-smi

# Check PyTorch ROCm
python -c "import torch; print(torch.cuda.is_available())"
```

### NCCL Errors
```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
```

---

## 📦 Output Files

After training completes:

```
/workspace/
├── outputs/
│   ├── training.log        # Full training log
│   └── cv_summary.json     # Cross-validation results
├── checkpoints/
│   ├── fold_0_best.pth     # Best models per fold
│   ├── fold_1_best.pth
│   └── fold_2_best.pth
└── tensorboard/
    ├── fold_0/             # TensorBoard events
    ├── fold_1/
    └── fold_2/
```

---

## 🔄 Download Results

```bash
# Compress checkpoints
tar -czvf best_models.tar.gz /workspace/checkpoints/fold_*_best.pth

# Download via RunPod UI or scp
scp -P <port> root@<pod-ip>:/workspace/best_models.tar.gz ./
```
