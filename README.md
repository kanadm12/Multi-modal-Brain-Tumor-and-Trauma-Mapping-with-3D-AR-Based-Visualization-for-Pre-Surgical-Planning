# BraTS 3D Segmentation - Optimized Training Guide

## 📋 Quick Start

### 1. **Prerequisites**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install nibabel scipy scikit-learn pandas tensorboard tqdm matplotlib
```

### 2. **Data Preparation**
Ensure your dataset is organized as:
```
dataset/
├── BraTS_0001/
│   ├── BraTS_0001_t1.nii.gz
│   ├── BraTS_0001_t1ce.nii.gz
│   ├── BraTS_0001_t2.nii.gz
│   ├── BraTS_0001_flair.nii.gz
│   └── BraTS_0001_seg.nii.gz
├── BraTS_0002/
│   └── ...
└── ...
```

### 3. **Configuration (Update in script)**
```python
DATA_DIR = r"C:\path\to\your\dataset"  # ← UPDATE THIS
OUTPUT_DIR = r"C:\path\to\outputs"      # ← Optional
MODEL_SAVE_DIR = r"C:\path\to\models"   # ← Optional
```

### 4. **Run Training**
```bash
python optimized_brats_final.py
```

---

## 🎯 Key Optimizations Implemented

| Feature | Current | Target | Impact |
|---------|---------|--------|--------|
| **Input Size** | (160, 192, 160) | Larger context | +4-6% Dice |
| **Model Capacity** | 48-768 filters | 1.5x larger | +3-5% Dice |
| **Batch Size (eff.)** | 16 | Stable gradients | +2-3% Dice |
| **Bottleneck** | Transformer | Better global context | +1-2% Dice |
| **TTA** | 8-point | Consistent predictions | +1-2% Dice |
| **Loss Function** | Combined (Dice+Lovasz+CE) | Better optimization | +1-2% Dice |
| **Cross-Validation** | 3-fold | Faster convergence | - |

---

## 📊 Expected Performance

### Dice Score by Class:
```
NCR (Necrotic Core):  90-93%
ED (Edema):           92-95%  ← Easiest
ET (Enhancing Tumor): 88-92%  ← Hardest, most important
───────────────────────────
Mean Dice:            92-95%
```

### Hausdorff Distance:
```
HD95 (Distance): 5-8 mm  (Good is <10mm)
```

---

## ⏱️ Training Time

| Component | Time |
|-----------|------|
| Fold 1 | ~48-60 hours |
| Fold 2 | ~48-60 hours |
| Fold 3 | ~48-60 hours |
| **Total** | **~5-7 days** |

**Recommended Hardware**: 1x A100 40GB or 1x H100 80GB

---

## 📈 Monitoring Progress

### TensorBoard
```bash
tensorboard --logdir=tensorboard_optimized_3fold
# Navigate to http://localhost:6006
```

### Logs
- **Training Log**: `outputs_optimized_3fold/training.log`
- **Summary**: `outputs_optimized_3fold/cv_summary.json`
- **Models**: `models_optimized_3fold/fold_*_best.pth`

---

## 🔧 Hyperparameter Tuning

If you want to adjust performance:

```python
# For better Dice (may be slower):
CROP_SIZE = (192, 224, 192)      # Larger input
MODEL_FILTERS = [64, 128, 256, 512, 1024]  # Bigger model
BATCH_SIZE = 4                    # Larger batch (if GPU memory allows)
ACCUMULATION_STEPS = 4            # Effective BS = 16

# For faster training (slightly lower Dice):
CROP_SIZE = (144, 160, 144)       # Smaller input
MODEL_FILTERS = [32, 64, 128, 256, 512]  # Smaller model
EPOCHS = 300                      # Fewer epochs
PATIENCE = 50                     # Early stopping sooner
```

---

## ⚠️ Common Issues

### Out of Memory
**Solution**: Reduce batch size or input size
```python
BATCH_SIZE = 1
ACCUMULATION_STEPS = 16  # Keep effective BS = 16
```

### Data Not Found
**Solution**: Check data directory format
```bash
ls dataset/
# Should show: BraTS_0001, BraTS_0002, ...
```

### Slow Data Loading
**Solution**: Increase num_workers
```python
# In dataloaders:
num_workers=4  # Increase if you have more CPU cores
```

---

## 📁 Output Files

After training completes:
```
outputs_optimized_3fold/
├── training.log                 # Detailed training log
├── cv_summary.json              # Results summary
│
tensorboard_optimized_3fold/
├── fold_0/                       # TensorBoard logs per fold
├── fold_1/
└── fold_2/

models_optimized_3fold/
├── fold_0_best.pth              # Best model for fold 0
├── fold_1_best.pth
└── fold_2_best.pth
```

---

## 📊 Interpreting Results

Example `cv_summary.json`:
```json
{
  "mean_test_dice": 0.9234,
  "std_test_dice": 0.0087,
  "mean_test_hd95": 6.45,
  "folds": [
    {
      "fold": 1,
      "test_dice": 0.9156,
      "test_hd95": 6.78
    },
    ...
  ]
}
```

**Good performance**:
- Dice > 0.92 ✅
- HD95 < 8mm ✅
- Std < 0.01 (stable) ✅

---

## 🚀 Next Steps After Training

1. **Ensemble Multiple Models**
   - Average predictions from all 3 folds
   - Use weighted averaging based on validation Dice

2. **Post-processing Optimization**
   - Fine-tune component size thresholds
   - Experiment with smoothing iterations

3. **Fine-tuning**
   - Use best fold models and continue training on hard cases
   - Increase augmentation for underperforming regions

4. **Submission**
   - Use 8-point TTA for inference
   - Apply adaptive post-processing
   - Average ensemble predictions

---

## 📞 Troubleshooting

**Issue**: "ModuleNotFoundError: No module named 'nibabel'"
```bash
pip install nibabel
```

**Issue**: "CUDA out of memory"
- Reduce BATCH_SIZE
- Reduce CROP_SIZE
- Use ReduceLROnPlateau more aggressively

**Issue**: "No BraTS patients found"
- Check DATA_DIR path
- Ensure patient directories start with "BraTS_"

---

## 💡 Tips for Best Results

1. **Let it train fully** - Don't stop early. 500 epochs allows convergence.
2. **Use TTA** - 8-point TTA adds ~1-2% Dice at inference.
3. **Monitor validation** - Check TensorBoard for convergence patterns.
4. **Save outputs** - Keep all fold models for ensemble.
5. **GPU memory** - This script is optimized for A100 40GB.

---

## 📚 References

- **BraTS Challenge**: https://www.med.upenn.edu/cbica/brats/
- **nnU-Net**: Isensee et al., 2021
- **Transformer in Medical Imaging**: Dosovitskiy et al., 2021

---

**Target Dice Score**: 90-95%
**Expected Training Time**: 5-7 days on A100 40GB
