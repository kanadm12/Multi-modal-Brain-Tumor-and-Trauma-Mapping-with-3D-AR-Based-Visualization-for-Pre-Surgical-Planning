# 🚀 RunPod Training Setup Guide

## Quick Start on RunPod (4x RTX 4090)

### Step 1: Create RunPod Instance

1. Go to https://www.runpod.io/
2. Select **Secure Cloud** or **Community Cloud**
3. Choose **4x RTX 4090** GPU configuration
4. Select template: **PyTorch 2.1**
5. Storage: **200GB+ Volume** (for datasets + models)
6. Click **Deploy**

**Cost Estimate**: ~$2.50-3.00/hour for 4x RTX 4090

---

## Step 2: Connect to Pod

### Via Web Terminal
```bash
# Click "Connect" → "Start Web Terminal"
```

### Via SSH (Recommended)
```bash
# Get SSH command from RunPod dashboard
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519
```

---

## Step 3: Clone Repository

```bash
# Clone your repo
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd BraTS_Optimized_Solution

# Verify you're in the right directory
ls -la
```

---

## Step 4: Install Dependencies

```bash
# Update pip
pip install --upgrade pip

# Install PyTorch (if not pre-installed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install required packages
pip install nibabel scipy scikit-learn pandas tensorboard tqdm matplotlib

# Verify CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"
```

**Expected output**:
```
CUDA available: True
GPUs: 4
```

---

## Step 5: Download Dataset

### Option A: From Azure Blob Storage (if already uploaded)
```bash
# Install Azure CLI
pip install azure-storage-blob

# Download dataset (create a script)
cat > download_from_azure.py << 'EOF'
from azure.storage.blob import BlobServiceClient
import os

connection_string = "YOUR_AZURE_CONNECTION_STRING"
container_name = "beproject"
local_path = "./dataset"

blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)

os.makedirs(local_path, exist_ok=True)

blobs = container_client.list_blobs(name_starts_with="dataset/dataset/")
for blob in blobs:
    local_file = os.path.join(local_path, blob.name)
    os.makedirs(os.path.dirname(local_file), exist_ok=True)
    with open(local_file, 'wb') as f:
        f.write(container_client.download_blob(blob.name).readall())
    print(f"Downloaded: {blob.name}")
EOF

python download_from_azure.py
```

### Option B: From Kaggle
```bash
# Setup Kaggle API
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json << 'EOF'
{
  "username": "YOUR_KAGGLE_USERNAME",
  "key": "YOUR_KAGGLE_KEY"
}
EOF
chmod 600 ~/.kaggle/kaggle.json

# Download using provided script
pip install kaggle
python brats_kaggle_to_azure.py  # (Modify to save locally instead)
```

### Option C: Direct Upload (Small Datasets)
```bash
# From your local machine, use SCP
scp -r dataset/ root@<pod-ip>:/workspace/BraTS_Optimized_Solution/dataset/
```

---

## Step 6: Configure Training Script

```bash
# Edit the script
nano optimized_brats_final.py

# Or use vim/vi
vim optimized_brats_final.py
```

**Changes needed**:
1. Set data path (around line 70):
```python
DATA_DIR = "/workspace/BraTS_Optimized_Solution/dataset"
```

2. Enable multi-GPU (around line 176):
```python
USE_MULTI_GPU = True
WORLD_SIZE = 4  # For 4x RTX 4090
```

3. Verify output paths:
```python
WORKSPACE_DIR = "/workspace/BraTS_Optimized_Solution"
```

**Save changes**: `Ctrl+O`, `Enter`, `Ctrl+X` (nano) or `:wq` (vim)

---

## Step 7: Verify Setup

```bash
# Check GPU status
nvidia-smi

# Quick test run
python verify_setup.py
```

---

## Step 8: Start Training

```bash
# Run in background with nohup
nohup python optimized_brats_final.py > training.log 2>&1 &

# Get process ID
echo $!

# Or use screen for interactive monitoring
screen -S brats_training
python optimized_brats_final.py

# Detach: Ctrl+A, then D
# Reattach: screen -r brats_training
```

---

## Step 9: Monitor Training

### Option 1: View Logs
```bash
# Real-time log viewing
tail -f training.log

# Or check output logs
tail -f outputs_optimized_3fold/training.log
```

### Option 2: TensorBoard
```bash
# In a new terminal/screen session
tensorboard --logdir=tensorboard_optimized_3fold --host=0.0.0.0 --port=6006

# Access via RunPod:
# Click "Connect" → "HTTP Service [6006]"
```

### Option 3: GPU Monitoring
```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Or use gpustat
pip install gpustat
gpustat -i 1
```

---

## Step 10: Check Progress

```bash
# Check if training is running
ps aux | grep python

# View latest metrics
tail -20 outputs_optimized_3fold/training.log

# Check model files
ls -lh models_optimized_3fold/
```

---

## 💾 Saving Results

### Sync to Cloud During Training
```bash
# Install rclone (for cloud sync)
curl https://rclone.org/install.sh | sudo bash

# Configure rclone for your cloud storage
rclone config

# Sync periodically (add to cron)
rclone sync /workspace/BraTS_Optimized_Solution/models_optimized_3fold/ mycloud:brats_models/
```

### Download After Training
```bash
# From your local machine
scp -r root@<pod-ip>:/workspace/BraTS_Optimized_Solution/models_optimized_3fold/ ./local_models/
```

---

## ⏱️ Training Timeline

| Fold | Time (4x RTX 4090) |
|------|-------------------|
| Fold 1 | ~7-10 hours |
| Fold 2 | ~7-10 hours |
| Fold 3 | ~7-10 hours |
| **Total** | **21-30 hours** |

**Total Cost**: ~$50-90 for complete training

---

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Edit script and reduce batch size
nano optimized_brats_final.py
# Change: BATCH_SIZE = 1 (from 2)
```

### Training Stopped Unexpectedly
```bash
# Check logs
cat outputs_optimized_3fold/training.log

# Check if process died
ps aux | grep python

# Restart from checkpoint (script automatically loads best model)
python optimized_brats_final.py
```

### SSH Connection Lost
```bash
# Training continues if using nohup or screen
# Reconnect and check:
screen -r brats_training  # If using screen
tail -f training.log      # If using nohup
```

### Slow Data Loading
```bash
# Increase workers in script
nano optimized_brats_final.py
# Change: num_workers=4 (from 2)
```

---

## ✅ Verification Checklist

Before starting:
- [ ] 4 GPUs detected (`nvidia-smi`)
- [ ] Dataset downloaded (~500GB)
- [ ] Data path set correctly
- [ ] `USE_MULTI_GPU = True`
- [ ] `WORLD_SIZE = 4`
- [ ] Screen/nohup for background running
- [ ] TensorBoard accessible
- [ ] Backup strategy configured

---

## 📊 Expected Results

After 21-30 hours:
- ✅ 3 model checkpoints saved
- ✅ Mean Dice: 92-95%
- ✅ Training logs complete
- ✅ TensorBoard data available

---

## 💡 Pro Tips

1. **Use Screen**: Better than nohup for interactive monitoring
2. **Watch GPU Usage**: Should be ~95-100% during training
3. **Sync Frequently**: Backup models every 12 hours
4. **Monitor Costs**: Check RunPod dashboard regularly
5. **Stop When Done**: Don't forget to terminate pod after downloading results

---

## 🛑 Stopping & Cleanup

```bash
# Stop training gracefully
pkill -f optimized_brats_final.py

# Download results
scp -r root@<pod-ip>:/workspace/BraTS_Optimized_Solution/models_optimized_3fold/ ./

# Terminate pod from RunPod dashboard
# Or use API
runpodctl stop pod <pod-id>
```

---

## 📞 Support

If issues arise:
1. Check `outputs_optimized_3fold/training.log`
2. Run `nvidia-smi` to verify GPUs
3. Test with single GPU first (`USE_MULTI_GPU = False`)
4. Contact RunPod support for infrastructure issues

---

**Quick Commands Reference**:
```bash
# Clone repo
git clone <your-repo-url> && cd BraTS_Optimized_Solution

# Setup
pip install torch torchvision nibabel scipy scikit-learn pandas tensorboard tqdm matplotlib

# Verify
nvidia-smi && python verify_setup.py

# Train
screen -S brats && python optimized_brats_final.py

# Monitor
# Ctrl+A, D (detach)
# screen -r brats (reattach)
# tail -f outputs_optimized_3fold/training.log
```

**Estimated Total Cost**: $50-90 for complete 3-fold training on 4x RTX 4090
