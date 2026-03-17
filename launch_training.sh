#!/bin/bash
# =============================================================================
# RUNPOD MULTI-GPU TRAINING LAUNCHER
# 
# For 4x AMD MI300X 192GB GPUs
# Usage: bash launch_training.sh
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "BraTS 3D Segmentation Training - RunPod"
echo "=============================================="
echo ""

# =============================================================================
# CONFIGURATION
# =============================================================================
export CLOUD_PLATFORM="runpod"
export WORLD_SIZE=4  # Number of GPUs
export MASTER_ADDR="localhost"
export MASTER_PORT="12355"

# ROCm-specific settings for MI300X
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export ROCM_PATH=/opt/rocm
export HIP_VISIBLE_DEVICES=0,1,2,3

# Performance optimizations
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

# =============================================================================
# DIRECTORY SETUP
# =============================================================================
WORKSPACE_DIR="/workspace"
SCRIPT_DIR="$WORKSPACE_DIR"
DATA_DIR="$WORKSPACE_DIR/dataset"
OUTPUT_DIR="$WORKSPACE_DIR/outputs"
CHECKPOINT_DIR="$WORKSPACE_DIR/checkpoints"
TENSORBOARD_DIR="$WORKSPACE_DIR/tensorboard"

echo "📁 Directory Configuration:"
echo "  Workspace: $WORKSPACE_DIR"
echo "  Dataset: $DATA_DIR"
echo "  Outputs: $OUTPUT_DIR"
echo "  Checkpoints: $CHECKPOINT_DIR"
echo "  TensorBoard: $TENSORBOARD_DIR"
echo ""

# Create directories
mkdir -p "$OUTPUT_DIR" "$CHECKPOINT_DIR" "$TENSORBOARD_DIR"

# =============================================================================
# GPU CHECK
# =============================================================================
echo "🔍 Checking GPU availability..."
if command -v rocm-smi &> /dev/null; then
    echo "✅ ROCm detected"
    rocm-smi --showproductname
    echo ""
    rocm-smi --showmeminfo vram
    echo ""
else
    echo "⚠️  rocm-smi not found, checking with PyTorch..."
fi

python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'ROCm available: {torch.cuda.is_available()}')
print(f'Number of GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)')
"
echo ""

# =============================================================================
# DATASET CHECK
# =============================================================================
echo "📊 Checking dataset..."
if [ -d "$DATA_DIR" ]; then
    PATIENT_COUNT=$(find "$DATA_DIR" -maxdepth 1 -type d -name "BraTS*" | wc -l)
    echo "✅ Found $PATIENT_COUNT BraTS patients"
    
    # Check for preprocessed data
    if [ -d "$WORKSPACE_DIR/preprocessed_data" ]; then
        PREPROCESSED_COUNT=$(find "$WORKSPACE_DIR/preprocessed_data" -maxdepth 1 -type d -name "BraTS*" | wc -l)
        echo "✅ Found $PREPROCESSED_COUNT preprocessed patients"
    else
        echo "⚠️  Preprocessed data not found. Training will be slower."
        echo "   Consider running the preprocessing script first."
    fi
else
    echo "❌ Dataset directory not found: $DATA_DIR"
    echo "   Please upload your BraTS dataset to this directory."
    exit 1
fi
echo ""

# =============================================================================
# CHECK FOR EXISTING CHECKPOINTS
# =============================================================================
echo "💾 Checking for existing checkpoints..."
if ls "$CHECKPOINT_DIR"/fold_*_best.pth 1> /dev/null 2>&1; then
    echo "✅ Found existing checkpoints - training will resume"
    ls -la "$CHECKPOINT_DIR"/fold_*_best.pth
else
    echo "📝 No existing checkpoints - starting fresh training"
fi
echo ""

# =============================================================================
# START TENSORBOARD (Background)
# =============================================================================
echo "📈 Starting TensorBoard on port 6006..."
tensorboard --logdir="$TENSORBOARD_DIR" --host=0.0.0.0 --port=6006 &
TENSORBOARD_PID=$!
echo "✅ TensorBoard running (PID: $TENSORBOARD_PID)"
echo "   Access at: http://<your-pod-ip>:6006"
echo ""

# =============================================================================
# LAUNCH TRAINING
# =============================================================================
echo "🚀 Launching distributed training on $WORLD_SIZE GPUs..."
echo "=============================================="
echo ""

cd "$SCRIPT_DIR"

# Use torchrun for distributed training
torchrun \
    --nproc_per_node=$WORLD_SIZE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py

echo ""
echo "=============================================="
echo "✅ Training Complete!"
echo "=============================================="
echo ""
echo "📁 Results saved to:"
echo "  - Checkpoints: $CHECKPOINT_DIR"
echo "  - TensorBoard logs: $TENSORBOARD_DIR"
echo "  - Training logs: $OUTPUT_DIR/training.log"
echo "  - CV Summary: $OUTPUT_DIR/cv_summary.json"
echo ""

# Cleanup TensorBoard
kill $TENSORBOARD_PID 2>/dev/null || true
