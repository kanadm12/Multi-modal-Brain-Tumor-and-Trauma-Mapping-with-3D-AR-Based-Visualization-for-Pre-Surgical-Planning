# Configuration file for BraTS 3D Segmentation Training
# Adjust settings here and import in the training script

import torch

class Config:
    """Training configuration"""
    
    # ==================== PATHS ====================
    WORKSPACE_DIR = r"C:\Users\Kanad\Desktop\BR_PROJECT\BEPROJECT-RUNPOD-DATA"
    DATA_DIR = f"{WORKSPACE_DIR}/dataset"
    OUTPUT_DIR = f"{WORKSPACE_DIR}/outputs_optimized_3fold"
    MODEL_SAVE_DIR = f"{WORKSPACE_DIR}/models_optimized_3fold"
    TENSORBOARD_DIR = f"{WORKSPACE_DIR}/tensorboard_optimized_3fold"
    
    # ==================== DATA ====================
    CROP_SIZE = (160, 192, 160)      # Input size for model
    NUM_CLASSES = 4                  # Background + 3 tumor regions
    IN_CHANNELS = 4                  # T1, T1c, T2, FLAIR
    NORMALIZATION = "nnunet"         # Normalization method
    
    # ==================== ARCHITECTURE ====================
    MODEL_FILTERS = [48, 96, 192, 384, 768]  # Channel progression
    USE_ATTENTION = True
    ATTENTION_TYPE = 'transformer'   # 'transformer' or 'lightweight'
    NUM_ATTENTION_HEADS = 8
    TRANSFORMER_DEPTH = 2
    DROPOUT_RATE = 0.2
    
    # ==================== TRAINING ====================
    N_FOLDS = 3                      # Cross-validation folds
    BATCH_SIZE = 2                   # Batch size per GPU
    ACCUMULATION_STEPS = 8           # Gradient accumulation
    EPOCHS = 500                     # Maximum epochs
    INITIAL_LR = 2e-4                # Initial learning rate
    WEIGHT_DECAY = 1e-4              # L2 regularization
    PATIENCE = 75                    # Early stopping patience
    
    # ==================== LOSS FUNCTION ====================
    CLASS_WEIGHTS = torch.tensor([0.0, 1.0, 1.0, 1.5])  # Emphasize ET
    LOSS_DICE_WEIGHT = 0.7           # Dice component
    LOSS_LOVASZ_WEIGHT = 0.1         # Lovasz component
    LOSS_CE_WEIGHT = 0.2             # CrossEntropy component
    
    # ==================== AUGMENTATION ====================
    AUGMENTATION_PROBABILITY = 0.85  # Probability of augmentation
    MIN_COMPONENT_SIZE = 150         # Minimum tumor component size
    
    # ==================== INFERENCE ====================
    USE_TTA = True                   # Test-time augmentation
    TTA_TRANSFORMS = 8               # Number of TTA transforms
    USE_ADAPTIVE_POSTPROCESSING = True
    
    # ==================== DEVICE & PRECISION ====================
    USE_AMP = True                   # Automatic mixed precision
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ==================== PRESETS ====================
    
    @staticmethod
    def preset_highperformance():
        """High performance configuration (needs A100 40GB+)"""
        Config.CROP_SIZE = (192, 224, 192)
        Config.MODEL_FILTERS = [64, 128, 256, 512, 1024]
        Config.BATCH_SIZE = 4
        Config.ACCUMULATION_STEPS = 4
        Config.TRANSFORMER_DEPTH = 3
        print("✅ High Performance Preset Applied")
    
    @staticmethod
    def preset_balanced():
        """Balanced configuration (A100 40GB)"""
        Config.CROP_SIZE = (160, 192, 160)
        Config.MODEL_FILTERS = [48, 96, 192, 384, 768]
        Config.BATCH_SIZE = 2
        Config.ACCUMULATION_STEPS = 8
        Config.TRANSFORMER_DEPTH = 2
        print("✅ Balanced Preset Applied")
    
    @staticmethod
    def preset_memory_efficient():
        """Memory efficient configuration (RTX 3090, 2x RTX 4090)"""
        Config.CROP_SIZE = (144, 160, 144)
        Config.MODEL_FILTERS = [32, 64, 128, 256, 512]
        Config.BATCH_SIZE = 1
        Config.ACCUMULATION_STEPS = 16
        Config.TRANSFORMER_DEPTH = 1
        Config.EPOCHS = 300
        Config.PATIENCE = 50
        print("✅ Memory Efficient Preset Applied")
    
    @staticmethod
    def preset_4xRTX4090():
        """Optimized configuration for 4x RTX 4090 (96GB total)"""
        Config.CROP_SIZE = (176, 192, 176)
        Config.MODEL_FILTERS = [48, 96, 192, 384, 768]
        Config.BATCH_SIZE = 2  # Per GPU = 8 total
        Config.ACCUMULATION_STEPS = 4  # Effective BS = 32
        Config.TRANSFORMER_DEPTH = 2
        Config.EPOCHS = 400
        Config.PATIENCE = 60
        print("✅ 4x RTX 4090 Preset Applied (Multi-GPU)")
    
    @staticmethod
    def preset_quick_test():
        """Quick test configuration for debugging"""
        Config.CROP_SIZE = (96, 96, 96)
        Config.MODEL_FILTERS = [16, 32, 64, 128, 256]
        Config.BATCH_SIZE = 1
        Config.ACCUMULATION_STEPS = 1
        Config.EPOCHS = 10
        Config.PATIENCE = 5
        print("✅ Quick Test Preset Applied")


# Usage instructions:
"""
# In your training script, use:

from config import Config

# Option 1: Use default balanced config
# (nothing to do, it's the default)

# Option 2: Use preset
Config.preset_highperformance()

# Option 3: Custom configuration
Config.CROP_SIZE = (192, 192, 192)
Config.BATCH_SIZE = 4
Config.EPOCHS = 600

# Then use Config values in your training script:
CROP_SIZE = Config.CROP_SIZE
BATCH_SIZE = Config.BATCH_SIZE
# ... etc
"""
