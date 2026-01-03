"""
Quick Test Script for BraTS Training
Tests with 100 patients and 50 epochs to validate pipeline
"""

import os
import sys
import torch
import torch.multiprocessing as mp
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from main script
from optimized_brats_final import (
    OptimizedUNet3D, BraTSDataset3D, CombinedLoss,
    train_epoch, validate_epoch, WarmupScheduler,
    setup_ddp, cleanup_ddp, save_checkpoint
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ============================================================================
# TEST CONFIGURATION
# ============================================================================

WORKSPACE_DIR = "/workspace/Multi-modal-Brain-Tumor-and-Trauma-Mapping-with-3D-AR-Based-Visualization-for-Pre-Surgical-Planning"
DATA_DIR = os.path.join(WORKSPACE_DIR, "dataset")
PREPROCESSED_DIR = os.path.join(WORKSPACE_DIR, "preprocessed_data")
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "test_outputs")
MODEL_SAVE_DIR = os.path.join(WORKSPACE_DIR, "test_models")
TENSORBOARD_DIR = os.path.join(WORKSPACE_DIR, "test_tensorboard")

# Test Parameters
TEST_NUM_PATIENTS = 100  # Test with 100 patients
TEST_EPOCHS = 50  # Test for 50 epochs
TEST_PATIENCE = 15  # Early stopping after 15 epochs no improvement

# Model Configuration
CROP_SIZE = (160, 192, 160)
NUM_CLASSES = 4
IN_CHANNELS = 4
MODEL_FILTERS = [48, 96, 192, 384, 768]  # A100 config
USE_ATTENTION = True
ATTENTION_TYPE = 'transformer'
NUM_ATTENTION_HEADS = 8
TRANSFORMER_DEPTH = 2
DROPOUT_RATE = 0.2
USE_GRADIENT_CHECKPOINTING = False

# Training Configuration
BATCH_SIZE = 4  # A100 config
ACCUMULATION_STEPS = 4
INITIAL_LR = 2e-4
WEIGHT_DECAY = 1e-4
EPSILON = 1e-8

# Features
USE_WARMUP = True
WARMUP_EPOCHS = 5  # Reduced for test
USE_GRADIENT_CLIPPING = True
GRADIENT_CLIP_VALUE = 1.0
USE_PREPROCESSED = True
NUM_WORKERS = 6
USE_AMP = True
USE_TTA = False  # Disable TTA for faster testing
USE_ADAPTIVE_POSTPROCESSING = True

# Loss weights
CLASS_WEIGHTS = torch.tensor([0.0, 1.0, 1.0, 1.5])
LOSS_DICE_WEIGHT = 0.5
LOSS_SURFACE_WEIGHT = 0.25
LOSS_CE_WEIGHT = 0.15
LOSS_LOVASZ_WEIGHT = 0.1

# Augmentation
AUGMENTATION_PROBABILITY = 0.85
MIN_COMPONENT_SIZE = 150

# Multi-GPU
WORLD_SIZE = 4  # 4x A100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# TEST DATA PREPARATION
# ============================================================================

def prepare_test_data():
    """Select random 100 patients for testing"""
    import random
    from sklearn.model_selection import train_test_split
    
    print("=" * 80)
    print("PREPARING TEST DATA")
    print("=" * 80)
    
    # Get all preprocessed patients
    all_patients = sorted([d for d in os.listdir(PREPROCESSED_DIR) 
                          if os.path.isdir(os.path.join(PREPROCESSED_DIR, d)) 
                          and d.startswith('BraTS')])
    
    print(f"Total patients available: {len(all_patients)}")
    
    # Randomly select 100 patients
    random.seed(42)
    test_patients = random.sample(all_patients, min(TEST_NUM_PATIENTS, len(all_patients)))
    
    print(f"Selected {len(test_patients)} patients for testing")
    
    # Split: 70% train, 15% val, 15% test
    train_patients, temp = train_test_split(test_patients, test_size=0.3, random_state=42)
    val_patients, test_patients_split = train_test_split(temp, test_size=0.5, random_state=42)
    
    print(f"Train: {len(train_patients)}, Val: {len(val_patients)}, Test: {len(test_patients_split)}")
    
    return train_patients, val_patients, test_patients_split

# ============================================================================
# TEST TRAINING FUNCTION
# ============================================================================

def run_test_training(rank, world_size, train_patients, val_patients):
    """Run test training on single fold"""
    
    # Setup DDP
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    
    # Create output directories
    if rank == 0:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(TENSORBOARD_DIR, exist_ok=True)
    
    # Setup logging
    import logging
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(OUTPUT_DIR, f'test_training_rank{rank}.log')),
            logging.StreamHandler()
        ] if rank == 0 else []
    )
    logger = logging.getLogger(__name__)
    
    if rank == 0:
        logger.info("=" * 80)
        logger.info("QUICK TEST MODE: 100 Patients, 50 Epochs")
        logger.info("=" * 80)
        logger.info(f"Train: {len(train_patients)}, Val: {len(val_patients)}")
    
    # Create datasets
    train_dataset = BraTSDataset3D(
        PREPROCESSED_DIR, train_patients, split='train', 
        crop_size=CROP_SIZE, use_preprocessed=True
    )
    val_dataset = BraTSDataset3D(
        PREPROCESSED_DIR, val_patients, split='val',
        crop_size=CROP_SIZE, use_preprocessed=True
    )
    
    # Create samplers for DDP
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, sampler=val_sampler,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
    )
    
    # Create model
    model = OptimizedUNet3D(
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        filters=MODEL_FILTERS,
        use_attention=USE_ATTENTION,
        attention_type=ATTENTION_TYPE,
        num_heads=NUM_ATTENTION_HEADS,
        dropout=DROPOUT_RATE,
        use_checkpointing=USE_GRADIENT_CHECKPOINTING
    ).to(device)
    
    # Wrap with DDP
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], output_device=rank
    )
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {total_params/1e6:.2f}M")
    
    # Create loss, optimizer, scheduler
    loss_fn = CombinedLoss(
        num_classes=NUM_CLASSES,
        class_weights=CLASS_WEIGHTS.to(device),
        dice_weight=LOSS_DICE_WEIGHT,
        surface_weight=LOSS_SURFACE_WEIGHT,
        ce_weight=LOSS_CE_WEIGHT,
        lovasz_weight=LOSS_LOVASZ_WEIGHT
    )
    
    optimizer = AdamW(
        model.parameters(), lr=INITIAL_LR, 
        weight_decay=WEIGHT_DECAY, eps=EPSILON
    )
    
    # Create scheduler
    plateau_scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5,
        verbose=False, min_lr=1e-7
    )
    
    if USE_WARMUP:
        scheduler = WarmupScheduler(
            optimizer=optimizer,
            warmup_epochs=WARMUP_EPOCHS,
            initial_lr=INITIAL_LR,
            after_scheduler=plateau_scheduler
        )
    else:
        scheduler = plateau_scheduler
    
    scaler = torch.amp.GradScaler('cuda') if USE_AMP else None
    
    if rank == 0:
        logger.info(f"Using gradient clipping: max_norm={GRADIENT_CLIP_VALUE}")
        logger.info(f"Using LR warmup: {WARMUP_EPOCHS} epochs")
    
    # Training loop
    best_val_dice = 0.0
    patience_counter = 0
    
    for epoch in range(1, TEST_EPOCHS + 1):
        train_sampler.set_epoch(epoch)
        
        # Adjust LR for warmup
        if USE_WARMUP and epoch <= WARMUP_EPOCHS:
            warmup_lr = INITIAL_LR * (epoch / WARMUP_EPOCHS)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, scaler, 
            device, ACCUMULATION_STEPS, rank
        )
        
        # Validate
        val_dice, val_hd95 = validate_epoch(
            model, val_loader, device, use_tta=USE_TTA, rank=rank
        )
        
        # LR scheduling (after warmup)
        if not USE_WARMUP or epoch > WARMUP_EPOCHS:
            scheduler.step(val_dice)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        if rank == 0:
            logger.info(
                f"E{epoch:03d}/{TEST_EPOCHS:03d} | "
                f"Loss: {train_loss:.4f} | "
                f"Val Dice: {val_dice:.4f} | "
                f"Val HD95: {val_hd95:.2f} | "
                f"LR: {current_lr:.2e}"
            )
            
            # Save checkpoint
            is_best = val_dice > best_val_dice
            if is_best:
                best_val_dice = val_dice
                patience_counter = 0
                
                save_checkpoint(
                    model, optimizer, scheduler, scaler,
                    epoch, val_dice, val_hd95,
                    MODEL_SAVE_DIR, fold=0, is_best=True
                )
                logger.info(f"✓ New best model! Dice: {val_dice:.4f}")
            else:
                patience_counter += 1
            
            # Periodic checkpoint
            if epoch % 10 == 0:
                save_checkpoint(
                    model, optimizer, scheduler, scaler,
                    epoch, val_dice, val_hd95,
                    MODEL_SAVE_DIR, fold=0, is_best=False
                )
        
        # Early stopping
        if patience_counter >= TEST_PATIENCE:
            if rank == 0:
                logger.info(f"Early stopping at epoch {epoch}")
            break
    
    if rank == 0:
        logger.info("=" * 80)
        logger.info(f"TEST COMPLETE! Best Val Dice: {best_val_dice:.4f}")
        logger.info("=" * 80)
    
    cleanup_ddp()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Prepare test data
    train_patients, val_patients, test_patients = prepare_test_data()
    
    print("\nStarting test training...")
    print(f"Expected time: ~30-45 minutes (50 epochs with 70 train samples)")
    print("=" * 80)
    
    # Run test training
    mp.spawn(
        run_test_training,
        args=(WORLD_SIZE, train_patients, val_patients),
        nprocs=WORLD_SIZE,
        join=True
    )
    
    print("\n" + "=" * 80)
    print("TEST TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Check results in: {OUTPUT_DIR}")
    print(f"Best model saved in: {MODEL_SAVE_DIR}")
    print("\nIf test passed, run full training with:")
    print("  python optimized_brats_final.py")
    print("=" * 80)
