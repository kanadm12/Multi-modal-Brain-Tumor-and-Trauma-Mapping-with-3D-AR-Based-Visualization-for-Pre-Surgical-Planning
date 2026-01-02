# 🚀 Multi-GPU Training Guide (4x RTX 4090)

## Quick Start

### Enable Multi-GPU Training

In [optimized_brats_final.py](optimized_brats_final.py), set:
```python
# Multi-GPU Settings (around line 176)
USE_MULTI_GPU = True  # Enable 4x RTX 4090
WORLD_SIZE = 4        # Number of GPUs
```

### Run Training

```powershell
# Single command - DDP will spawn 4 processes automatically
python optimized_brats_final.py
```

That's it! The script automatically:
- Spawns 4 GPU processes (rank 0-3)
- Synchronizes gradients across GPUs
- Saves models only from rank 0
- Shows progress bars only on rank 0

---

## ⏱️ Training Time Estimates

### Architecture Configuration:
- **Input size**: (160, 192, 160)
- **Model**: ~40-45M parameters
- **Batch size**: 2 per GPU × 4 GPUs = 8 images/step
- **Accumulation**: 8 steps = **Effective BS: 64**
- **Epochs**: 500 (early stopping ~150-250)
- **3-fold cross-validation**

### Single RTX 4090 (24GB):
| Metric | Time |
|--------|------|
| Per epoch | ~8-12 min |
| Per fold (200 epochs) | ~27-40 hours |
| **Total (3 folds)** | **3.5-5 days** |

### 4x RTX 4090 Cluster (96GB):
| Metric | Time |
|--------|------|
| Per epoch | ~2-3 min (3-4x speedup) |
| Per fold (200 epochs) | ~7-10 hours |
| **Total (3 folds)** | **🚀 21-30 hours (0.9-1.3 days)** |

**Speedup**: 3-4x faster (not 4x due to communication overhead)

---

## 📊 Expected Performance

| Metric | Value |
|--------|-------|
| **Mean Dice** | 92-95% |
| **NCR Dice** | 90-93% |
| **ED Dice** | 92-95% |
| **ET Dice** | 88-92% |
| **HD95** | <5mm |

---

## 🔧 Configuration Options

### Option 1: Use Balanced Preset (Default)
```python
# No changes needed - already optimized for 4x RTX 4090
# Effective batch size: 2 × 4 GPUs × 8 accumulation = 64
```

### Option 2: Use 4x RTX 4090 Preset
```python
from config import Config
Config.preset_4xRTX4090()

# This sets:
# - Crop size: (176, 192, 176) - Larger input
# - Batch size: 2 per GPU = 8 total
# - Accumulation: 4 steps
# - Effective BS: 32
# - Better memory utilization
```

### Option 3: Custom Configuration
```python
# In optimized_brats_final.py
CROP_SIZE = (192, 224, 192)  # Larger for more context
BATCH_SIZE = 2                # Per GPU
ACCUMULATION_STEPS = 4        # Total effective BS = 2×4×4 = 32
```

---

## 💾 Memory Usage

### Per GPU (RTX 4090 24GB):

| Component | Memory |
|-----------|--------|
| Model | ~2.5 GB |
| Batch (BS=2) | ~8-10 GB |
| Gradients | ~2.5 GB |
| Optimizer | ~2.5 GB |
| Misc | ~2 GB |
| **Total** | **~17-19 GB** |

**Available**: ~5-7 GB headroom per GPU ✅

---

## 🎯 DDP Features Implemented

- ✅ **DistributedDataParallel (DDP)**: Efficient multi-GPU training
- ✅ **DistributedSampler**: Data distributed across GPUs
- ✅ **Gradient Synchronization**: Automatic all-reduce
- ✅ **Single-node multi-GPU**: All 4 GPUs in same machine
- ✅ **Rank 0 logging**: Only main process logs/saves
- ✅ **Metric broadcasting**: Validation metrics shared
- ✅ **Model unwrapping**: Proper state dict saving

---

## 🐛 Troubleshooting

### Issue: "Connection timeout" or "NCCL error"
**Solution**: Check GPU availability
```powershell
nvidia-smi
# Should show 4 GPUs available
```

### Issue: "Out of memory"
**Solution**: Reduce batch size
```python
BATCH_SIZE = 1  # Change from 2 to 1
```

### Issue: "Port already in use"
**Solution**: Change DDP port in script
```python
os.environ['MASTER_PORT'] = '12356'  # Change from 12355
```

### Issue: Slower than expected
**Check**:
- All 4 GPUs are being utilized: `nvidia-smi`
- No other processes using GPUs
- Data loading not bottleneck (increase `num_workers`)

---

## 📈 Monitoring

### Terminal Output
```
STARTING 3-FOLD CROSS-VALIDATION
Multi-GPU: True (4 GPUs)
Batch Size: 2 x 8 x 4 = 64

FOLD 1/3
Model parameters: 42.35M (wrapped with DDP)

E001 | Loss: 0.4523 | Val Dice: 0.7845 | HD95: 12.34 | Time: 165.3s
```

### TensorBoard
```powershell
tensorboard --logdir=tensorboard_optimized_3fold
```

### GPU Usage
```powershell
# Watch GPU utilization
nvidia-smi -l 1
```

---

## ✅ Verification Checklist

Before starting training:

- [ ] 4x RTX 4090 GPUs available (`nvidia-smi`)
- [ ] `USE_MULTI_GPU = True` in script
- [ ] `WORLD_SIZE = 4` in script
- [ ] Dataset path correctly set
- [ ] CUDA version compatible (11.8+)
- [ ] PyTorch with CUDA support installed
- [ ] At least 200GB free disk space (models + logs)

---

## 🚀 Performance Tips

1. **Use SSD for dataset**: 3-5x faster data loading
2. **Increase num_workers**: Set to 4-8 if CPU allows
3. **Monitor GPU usage**: Should be ~95-100% during training
4. **Use mixed precision**: Already enabled (USE_AMP = True)
5. **Optimal batch size**: 2 per GPU (current setting) ✅

---

## 📝 Notes

- **Single node only**: All 4 GPUs must be in same machine
- **NCCL backend**: Required for NVIDIA GPUs (automatic)
- **Reproducibility**: Seed set but DDP may have minor variations
- **Model loading**: Load with `model.load_state_dict()` (unwrapped)
- **Inference**: Can use single GPU (model saved unwrapped)

---

## 🎓 Advanced: Multi-Node Training

If you want to scale to multiple machines (e.g., 2 nodes × 4 GPUs = 8 GPUs):

```python
# Node 0 (Master)
os.environ['MASTER_ADDR'] = '192.168.1.100'  # IP of node 0
os.environ['MASTER_PORT'] = '12355'
os.environ['NODE_RANK'] = '0'

# Node 1 (Worker)
os.environ['MASTER_ADDR'] = '192.168.1.100'  # Same master IP
os.environ['MASTER_PORT'] = '12355'
os.environ['NODE_RANK'] = '1'

# Update
WORLD_SIZE = 8  # Total GPUs across all nodes
```

---

## 📞 Support

For issues or questions:
1. Check error logs in `outputs_optimized_3fold/training.log`
2. Verify GPU setup with `nvidia-smi`
3. Test with single GPU first (`USE_MULTI_GPU = False`)
4. Check PyTorch DDP documentation
