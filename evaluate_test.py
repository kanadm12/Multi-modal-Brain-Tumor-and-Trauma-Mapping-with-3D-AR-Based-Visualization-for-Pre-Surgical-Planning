#!/usr/bin/env python3
"""
BraTS Standalone Test Evaluation Script
========================================

Performs comprehensive evaluation on test set with:
- Best fold model loading
- 12-point Test Time Augmentation (TTA)
- Adaptive post-processing
- Full BraTS Challenge metrics (Dice + HD95 for WT/TC/ET)

Usage:
    python evaluate_test.py --checkpoint /path/to/fold_0_best.pth --data_dir /path/to/test_data
    
    # On RunPod:
    python evaluate_test.py --checkpoint /workspace/checkpoints/fold_0_best.pth --data_dir /workspace/dataset/beproject/dataset/dataset

Author: Auto-generated for BraTS Challenge evaluation
"""

import os
import sys
import argparse
import logging
import glob
import json
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.ndimage import (
    label as ndimage_label, binary_closing, binary_opening,
    gaussian_filter, binary_fill_holes, generate_binary_structure
)

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model Architecture - Must match training configuration
CROP_SIZE = (160, 192, 160)
NUM_CLASSES = 4  # Background + NCR + ED + ET
IN_CHANNELS = 4  # T1, T1c, T2, FLAIR
MODEL_FILTERS = [48, 96, 192, 384, 768]
USE_ATTENTION = True
ATTENTION_TYPE = 'transformer'
NUM_ATTENTION_HEADS = 8
TRANSFORMER_DEPTH = 4
DROPOUT_RATE = 0.12

# TTA Configuration
TTA_TRANSFORMS = 12  # 12-point TTA

# Post-processing
MIN_COMPONENT_SIZE = 100

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def nnunet_normalize(img):
    """nnU-Net style normalization"""
    nonzero_mask = img > 0
    if not np.any(nonzero_mask):
        return img
    
    # Percentile clipping
    p001, p999 = np.percentile(img[nonzero_mask], [0.1, 99.9])
    img = np.clip(img, p001, p999)
    
    # Standardization
    mean = img[nonzero_mask].mean()
    std = img[nonzero_mask].std()
    
    if std > 1e-8:
        img = np.where(nonzero_mask, (img - mean) / (std + 1e-8), 0)
    
    return img


def center_crop_or_pad(volume, target_shape):
    """Center crop or pad volume to target shape"""
    output = np.zeros(target_shape, dtype=volume.dtype)
    min_shape = np.minimum(volume.shape, target_shape)
    start_src = ((np.array(volume.shape) - min_shape) // 2).astype(int)
    start_dst = ((np.array(target_shape) - min_shape) // 2).astype(int)
    
    slice_src = tuple(slice(s, s + m) for s, m in zip(start_src, min_shape))
    slice_dst = tuple(slice(s, s + m) for s, m in zip(start_dst, min_shape))
    
    output[slice_dst] = volume[slice_src]
    return output


# ============================================================================
# BRATS METRICS
# ============================================================================

def compute_brats_regions(segmentation):
    """Convert class segmentation to BraTS challenge regions
    
    BraTS Challenge uses these tumor regions:
    - WT (Whole Tumor): NCR + ED + ET (classes 1, 2, 3)
    - TC (Tumor Core): NCR + ET (classes 1, 3)
    - ET (Enhancing Tumor): ET only (class 3)
    """
    if isinstance(segmentation, torch.Tensor):
        seg = segmentation.cpu().numpy()
    else:
        seg = segmentation
    
    ncr = (seg == 1)
    ed = (seg == 2)
    et = (seg == 3)
    
    return {
        'WT': ncr | ed | et,
        'TC': ncr | et,
        'ET': et
    }


def dice_coefficient_brats(pred, target, smooth=1e-6):
    """Calculate Dice coefficient for BraTS regions (WT/TC/ET)
    
    Returns:
        Dict with per-region dice scores and mean
    """
    pred_regions = compute_brats_regions(pred)
    target_regions = compute_brats_regions(target)
    
    brats_dice = {}
    for region_name in ['WT', 'TC', 'ET']:
        pred_r = pred_regions[region_name].flatten().astype(float)
        target_r = target_regions[region_name].flatten().astype(float)
        
        intersection = np.sum(pred_r * target_r)
        denominator = np.sum(pred_r) + np.sum(target_r)
        
        if denominator < smooth:
            dice = 1.0 if np.sum(target_r) < smooth else 0.0
        else:
            dice = (2.0 * intersection + smooth) / (denominator + smooth)
        
        brats_dice[region_name] = dice
    
    brats_dice['mean'] = np.mean([brats_dice['WT'], brats_dice['TC'], brats_dice['ET']])
    return brats_dice


def hausdorff_95_brats(pred, target, spacing=(1, 1, 1)):
    """Calculate 95th percentile Hausdorff distance for BraTS regions
    
    Returns:
        Dict with per-region HD95 scores and mean
    """
    pred_np = pred.cpu().numpy() if isinstance(pred, torch.Tensor) else pred
    target_np = target.cpu().numpy() if isinstance(target, torch.Tensor) else target
    
    def compute_hd95_for_mask(pred_mask, target_mask, spacing):
        """Compute HD95 for a single binary mask pair"""
        if not np.any(target_mask) and not np.any(pred_mask):
            return 0.0
        if not np.any(target_mask) or not np.any(pred_mask):
            return 373.13  # Maximum distance for BraTS (diagonal of max volume)
        
        try:
            # Get surface points
            pred_surface = pred_mask ^ binary_closing(pred_mask, generate_binary_structure(3, 1))
            target_surface = target_mask ^ binary_closing(target_mask, generate_binary_structure(3, 1))
            
            pred_points = np.array(np.where(pred_surface)).T * np.array(spacing)
            target_points = np.array(np.where(target_surface)).T * np.array(spacing)
            
            if len(pred_points) == 0 or len(target_points) == 0:
                return 373.13
            
            # Sample if too many points (for efficiency)
            max_points = 10000
            if len(pred_points) > max_points:
                idx = np.random.choice(len(pred_points), max_points, replace=False)
                pred_points = pred_points[idx]
            if len(target_points) > max_points:
                idx = np.random.choice(len(target_points), max_points, replace=False)
                target_points = target_points[idx]
            
            # Compute distances
            distances_pred_to_target = cdist(pred_points, target_points).min(axis=1)
            distances_target_to_pred = cdist(target_points, pred_points).min(axis=1)
            
            all_distances = np.concatenate([distances_pred_to_target, distances_target_to_pred])
            hd95 = np.percentile(all_distances, 95)
            
            return float(hd95)
        except Exception as e:
            logger.warning(f"HD95 computation error: {e}")
            return 373.13
    
    pred_regions = compute_brats_regions(pred_np)
    target_regions = compute_brats_regions(target_np)
    
    brats_hd95 = {}
    for region_name in ['WT', 'TC', 'ET']:
        brats_hd95[region_name] = compute_hd95_for_mask(
            pred_regions[region_name],
            target_regions[region_name],
            spacing
        )
    
    brats_hd95['mean'] = np.mean([brats_hd95['WT'], brats_hd95['TC'], brats_hd95['ET']])
    return brats_hd95


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class MultiHeadSelfAttention3D(nn.Module):
    """Multi-head self-attention for 3D"""
    def __init__(self, channels, num_heads=8):
        super().__init__()
        assert channels % num_heads == 0
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels)
        )
    
    def forward(self, x):
        B, C, D, H, W = x.shape
        x_flat = x.reshape(B, C, -1).permute(0, 2, 1)  # (B, N, C)
        
        # Self-attention
        qkv = self.qkv(x_flat)
        qkv = qkv.reshape(B, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        x_attn = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        
        x_attn = self.norm1(x_flat + self.proj(x_attn))
        x_attn = self.norm2(x_attn + self.mlp(x_attn))
        
        return x_attn.permute(0, 2, 1).reshape(B, C, D, H, W)


class TransformerBottleneck(nn.Module):
    """Transformer bottleneck with multi-head attention"""
    def __init__(self, channels, num_heads=8, depth=2):
        super().__init__()
        self.conv_in = nn.Conv3d(channels, channels, 1)
        
        self.layers = nn.ModuleList([
            MultiHeadSelfAttention3D(channels, num_heads) for _ in range(depth)
        ])
        
        self.conv_out = nn.Conv3d(channels, channels, 1)
        self.norm = nn.InstanceNorm3d(channels)
    
    def forward(self, x):
        x = self.conv_in(x)
        identity = x
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.conv_out(x)
        return self.norm(x + identity)


class LightweightAttention3D(nn.Module):
    """Lightweight channel and spatial attention with SE block"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
        self.channel_attn_max = nn.Sequential(
            nn.AdaptiveMaxPool3d(1),
            nn.Conv3d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction, channels, 1),
        )
        
        self.spatial_attn = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.InstanceNorm3d(1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        avg_attn = self.channel_attn(x)
        max_attn = torch.sigmoid(self.channel_attn_max(x))
        channel_attn = (avg_attn + max_attn) / 2
        x = x * channel_attn
        
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_attn = self.spatial_attn(spatial_input)
        
        return x * spatial_attn + x


class AttentionGate3D(nn.Module):
    """Attention Gate for skip connections"""
    def __init__(self, gate_channels, skip_channels, inter_channels=None):
        super().__init__()
        inter_channels = inter_channels or skip_channels // 2
        
        self.gate_conv = nn.Conv3d(gate_channels, inter_channels, 1, bias=False)
        self.skip_conv = nn.Conv3d(skip_channels, inter_channels, 1, bias=False)
        
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, 1, bias=False),
            nn.InstanceNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, gate, skip):
        if gate.shape[2:] != skip.shape[2:]:
            gate = F.interpolate(gate, size=skip.shape[2:], mode='trilinear', align_corners=False)
        
        g1 = self.gate_conv(gate)
        x1 = self.skip_conv(skip)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return skip * psi


class ConvBlock3D(nn.Module):
    """3D Convolutional block"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.GELU()
    
    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class EncoderBlock3D(nn.Module):
    """Encoder block"""
    def __init__(self, in_channels, out_channels, use_attention=True, attention_type='lightweight'):
        super().__init__()
        self.double_conv = nn.Sequential(
            ConvBlock3D(in_channels, out_channels),
            ConvBlock3D(out_channels, out_channels)
        )
        
        self.attention = LightweightAttention3D(out_channels) if use_attention else None
    
    def forward(self, x):
        x = self.double_conv(x)
        if self.attention is not None:
            x = self.attention(x)
        return x


class DecoderBlock3D(nn.Module):
    """Decoder block"""
    def __init__(self, in_channels, out_channels, use_attention=True, attention_type='lightweight'):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2)
        self.double_conv = nn.Sequential(
            ConvBlock3D(out_channels * 2, out_channels),
            ConvBlock3D(out_channels, out_channels)
        )
        
        self.attention = LightweightAttention3D(out_channels) if use_attention else None
    
    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.double_conv(x)
        if self.attention is not None:
            x = self.attention(x)
        return x


class OptimizedUNet3D(nn.Module):
    """Optimized 3D U-Net with transformer bottleneck and deep supervision"""
    def __init__(self, in_channels, num_classes, filters, use_attention=True, 
                 attention_type='transformer', num_heads=8, dropout=0.2, use_checkpointing=False):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.filters = filters
        self.use_attention = use_attention
        self.attention_type = attention_type
        self.use_checkpointing = use_checkpointing
        
        self.input_conv = nn.Sequential(
            ConvBlock3D(in_channels, filters[0]),
            ConvBlock3D(filters[0], filters[0])
        )
        
        self.encoder = nn.ModuleList([
            EncoderBlock3D(filters[i], filters[i + 1], use_attention, attention_type)
            for i in range(len(filters) - 1)
        ])
        
        self.bottleneck = TransformerBottleneck(filters[-1], num_heads=num_heads, depth=TRANSFORMER_DEPTH)
        
        self.attention_gates = nn.ModuleList([
            AttentionGate3D(
                gate_channels=filters[i + 1],
                skip_channels=filters[i],
                inter_channels=filters[i] // 2
            )
            for i in range(len(filters) - 2, -1, -1)
        ])
        
        self.decoder = nn.ModuleList([
            DecoderBlock3D(filters[i + 1], filters[i], use_attention, attention_type)
            for i in range(len(filters) - 2, -1, -1)
        ])
        
        self.output_conv = nn.Conv3d(filters[0], num_classes, 1)
        
        self.aux_outputs = nn.ModuleList([
            nn.Conv3d(filters[i], num_classes, 1) 
            for i in range(len(filters) - 2, -1, -1)
        ])
        
        self.dropout = nn.Dropout3d(dropout)
    
    def forward(self, x):
        x0 = self.input_conv(x)
        
        encoder_outputs = [x0]
        x = x0
        
        for encoder_block in self.encoder:
            x = F.max_pool3d(x, 2)
            x = encoder_block(x)
            x = self.dropout(x)
            encoder_outputs.append(x)
        
        x = self.bottleneck(x)
        
        aux_outputs = []
        for i, decoder_block in enumerate(self.decoder):
            skip = encoder_outputs[-(i + 2)]
            skip = self.attention_gates[i](x, skip)
            x = decoder_block(x, skip)
            x = self.dropout(x)
            
            if i < len(self.aux_outputs):
                aux = self.aux_outputs[i](x)
                aux = F.interpolate(aux, size=encoder_outputs[0].shape[2:], mode='trilinear', align_corners=False)
                aux_outputs.append(aux)
        
        out = self.output_conv(x)
        return out, aux_outputs


# ============================================================================
# TEST TIME AUGMENTATION (12-POINT)
# ============================================================================

def apply_tta_transform(image, transform_idx):
    """Apply TTA transform - 12-point version
    
    0: Original
    1-3: Axis flips (X, Y, Z)
    4-6: 2D rotations in XY plane (90°, 180°, 270°)
    7-8: 2D rotations in XZ plane (90°, 270°)
    9-10: 2D rotations in YZ plane (90°, 270°)
    11: Combined flip X + Y
    """
    if transform_idx == 0:
        return image
    elif transform_idx == 1:
        return torch.flip(image, dims=[2])
    elif transform_idx == 2:
        return torch.flip(image, dims=[3])
    elif transform_idx == 3:
        return torch.flip(image, dims=[4])
    elif transform_idx == 4:
        return torch.rot90(image, 1, dims=[3, 4])
    elif transform_idx == 5:
        return torch.rot90(image, 2, dims=[3, 4])
    elif transform_idx == 6:
        return torch.rot90(image, 3, dims=[3, 4])
    elif transform_idx == 7:
        return torch.rot90(image, 1, dims=[2, 4])
    elif transform_idx == 8:
        return torch.rot90(image, 3, dims=[2, 4])
    elif transform_idx == 9:
        return torch.rot90(image, 1, dims=[2, 3])
    elif transform_idx == 10:
        return torch.rot90(image, 3, dims=[2, 3])
    else:
        return torch.flip(torch.flip(image, dims=[2]), dims=[3])


def reverse_tta_transform(pred, transform_idx):
    """Reverse TTA transform - 12-point version"""
    is_5d = pred.dim() == 5
    offset = 1 if is_5d else 0
    
    if transform_idx == 0:
        return pred
    elif transform_idx == 1:
        return torch.flip(pred, dims=[1 + offset])
    elif transform_idx == 2:
        return torch.flip(pred, dims=[2 + offset])
    elif transform_idx == 3:
        return torch.flip(pred, dims=[3 + offset])
    elif transform_idx == 4:
        return torch.rot90(pred, -1, dims=[2 + offset, 3 + offset])
    elif transform_idx == 5:
        return torch.rot90(pred, -2, dims=[2 + offset, 3 + offset])
    elif transform_idx == 6:
        return torch.rot90(pred, -3, dims=[2 + offset, 3 + offset])
    elif transform_idx == 7:
        return torch.rot90(pred, -1, dims=[1 + offset, 3 + offset])
    elif transform_idx == 8:
        return torch.rot90(pred, -3, dims=[1 + offset, 3 + offset])
    elif transform_idx == 9:
        return torch.rot90(pred, -1, dims=[1 + offset, 2 + offset])
    elif transform_idx == 10:
        return torch.rot90(pred, -3, dims=[1 + offset, 2 + offset])
    else:
        return torch.flip(torch.flip(pred, dims=[2 + offset]), dims=[1 + offset])


# ============================================================================
# POST-PROCESSING
# ============================================================================

def adaptive_postprocessing(prediction, min_size=100):
    """Enhanced adaptive post-processing with boundary smoothing"""
    pred_np = prediction.cpu().numpy().astype(np.uint8)
    processed = np.zeros_like(pred_np)
    
    struct_small = generate_binary_structure(3, 1)
    struct_large = generate_binary_structure(3, 2)
    
    for class_id in range(1, 4):
        class_mask = (pred_np == class_id).astype(np.uint8)
        
        if not np.any(class_mask):
            continue
        
        # Fill holes
        try:
            class_mask = binary_fill_holes(class_mask).astype(np.uint8)
        except:
            pass
        
        # Morphological operations
        if class_id == 3:  # ET
            class_mask = binary_opening(class_mask, struct_small).astype(np.uint8)
            class_mask = binary_closing(class_mask, struct_small).astype(np.uint8)
        else:
            class_mask = binary_closing(class_mask, struct_large).astype(np.uint8)
            class_mask = binary_opening(class_mask, struct_small).astype(np.uint8)
        
        # Remove small components
        labeled, num_features = ndimage_label(class_mask)
        if num_features > 0:
            sizes = np.bincount(labeled.ravel())
            if len(sizes) > 1:
                sizes[0] = 0
                mask_sizes = sizes > min_size
                class_mask = mask_sizes[labeled].astype(np.uint8)
        
        # Smooth boundaries
        try:
            smoothed = gaussian_filter(class_mask.astype(float), sigma=0.5)
            class_mask = (smoothed > 0.3).astype(np.uint8)
        except:
            pass
        
        # Assign to output (handle overlaps by priority)
        if class_id == 3:  # ET highest priority
            processed[class_mask == 1] = class_id
        elif class_id == 1:  # NCR second priority
            processed[(class_mask == 1) & (processed != 3)] = class_id
        else:  # ED lowest priority
            processed[(class_mask == 1) & (processed == 0)] = class_id
    
    # Ensure ET is within TC region
    et_mask = (processed == 3)
    ncr_mask = (processed == 1)
    ed_mask = (processed == 2)
    
    if np.any(et_mask) and np.any(ncr_mask):
        tc_dilated = binary_closing(ncr_mask | et_mask, generate_binary_structure(3, 2))
        isolated_et = et_mask & ~tc_dilated
        if np.any(isolated_et):
            processed[isolated_et] = 2  # Convert isolated ET to ED
    
    return torch.tensor(processed, device=prediction.device, dtype=torch.long)


# ============================================================================
# DATASET
# ============================================================================

class BraTSTestDataset(Dataset):
    """BraTS test dataset loader"""
    def __init__(self, data_dir, patient_ids, crop_size=CROP_SIZE):
        self.data_dir = data_dir
        self.patient_ids = patient_ids
        self.crop_size = crop_size
    
    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        patient_dir = os.path.join(self.data_dir, patient_id)
        
        # Load modalities
        modality_mappings = [
            ['t1', 't1n'],
            ['t1ce', 't1c'],
            ['t2', 't2w'],
            ['flair', 't2f']
        ]
        
        img_data = []
        for mod_variants in modality_mappings:
            file_path = None
            for mod in mod_variants:
                patterns = [
                    os.path.join(patient_dir, f"*{mod}.nii.gz"),
                    os.path.join(patient_dir, f"*{mod}.nii"),
                    os.path.join(patient_dir, f"*_{mod}.nii.gz"),
                    os.path.join(patient_dir, f"*-{mod}.nii.gz"),
                ]
                for pattern in patterns:
                    files = glob.glob(pattern)
                    if files:
                        file_path = files[0]
                        break
                if file_path:
                    break
            
            if file_path:
                img = nib.load(file_path).get_fdata().astype(np.float32)
                img = nnunet_normalize(img)
                img_data.append(img)
        
        if len(img_data) < 4:
            raise ValueError(f"Missing modalities for {patient_id}")
        
        image = np.stack(img_data, axis=0)
        image = np.array([center_crop_or_pad(image[c], self.crop_size) for c in range(4)])
        
        # Load segmentation if available
        seg_files = glob.glob(os.path.join(patient_dir, "*seg.nii.gz"))
        if not seg_files:
            seg_files = glob.glob(os.path.join(patient_dir, "*seg.nii"))
        
        if seg_files and os.path.getsize(seg_files[0]) > 1024:
            seg = nib.load(seg_files[0]).get_fdata().astype(np.uint8)
            # Map labels: 1->NCR, 2->ED, 4->ET to 1, 2, 3
            seg_mapped = np.zeros_like(seg, dtype=np.uint8)
            seg_mapped[seg == 1] = 1
            seg_mapped[seg == 2] = 2
            seg_mapped[seg == 4] = 3
            seg = center_crop_or_pad(seg_mapped, self.crop_size)
        else:
            seg = np.zeros(self.crop_size, dtype=np.uint8)
        
        return torch.from_numpy(image).float(), torch.from_numpy(seg).long(), patient_id


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def load_model(checkpoint_path, device):
    """Load model from checkpoint"""
    logger.info(f"Loading model from: {checkpoint_path}")
    
    model = OptimizedUNet3D(
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        filters=MODEL_FILTERS,
        use_attention=USE_ATTENTION,
        attention_type=ATTENTION_TYPE,
        num_heads=NUM_ATTENTION_HEADS,
        dropout=DROPOUT_RATE,
        use_checkpointing=False
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle DDP state dict
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    # Log checkpoint info
    epoch = checkpoint.get('epoch', 'unknown')
    val_dice = checkpoint.get('val_dice', checkpoint.get('best_val_dice', 'unknown'))
    logger.info(f"Loaded checkpoint from epoch {epoch}, validation dice: {val_dice}")
    
    return model


def evaluate_patient_with_tta(model, image, device, num_tta=TTA_TRANSFORMS, use_amp=True):
    """Evaluate a single patient with TTA"""
    image = image.unsqueeze(0).to(device)  # Add batch dimension
    
    all_probs = []
    
    with torch.no_grad():
        for t in range(num_tta):
            img_t = apply_tta_transform(image, t)
            
            if use_amp:
                with autocast():
                    pred, _ = model(img_t)
            else:
                pred, _ = model(img_t)
            
            prob = F.softmax(pred, dim=1)
            prob = reverse_tta_transform(prob, t)
            all_probs.append(prob)
    
    # Average probabilities
    avg_prob = torch.stack(all_probs, dim=0).mean(dim=0)
    prediction = torch.argmax(avg_prob, dim=1).squeeze(0)
    
    return prediction


def run_evaluation(args):
    """Main evaluation function"""
    set_seed(42)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Get patient IDs
    patient_dirs = sorted([
        d for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d)) and d.startswith('BraTS')
    ])
    
    logger.info(f"Found {len(patient_dirs)} patients in {args.data_dir}")
    
    # Optionally limit patients
    if args.max_patients:
        patient_dirs = patient_dirs[:args.max_patients]
        logger.info(f"Limiting to {len(patient_dirs)} patients")
    
    # Create dataset
    dataset = BraTSTestDataset(args.data_dir, patient_dirs, CROP_SIZE)
    
    # Results storage
    all_results = []
    brats_dice_all = {'WT': [], 'TC': [], 'ET': []}
    brats_hd95_all = {'WT': [], 'TC': [], 'ET': []}
    
    # Evaluate each patient
    logger.info(f"\n{'='*80}")
    logger.info(f"Starting evaluation with {TTA_TRANSFORMS}-point TTA")
    logger.info(f"{'='*80}\n")
    
    for idx in tqdm(range(len(dataset)), desc="Evaluating"):
        image, target, patient_id = dataset[idx]
        
        # Run inference with TTA
        prediction = evaluate_patient_with_tta(
            model, image, device, 
            num_tta=args.num_tta if args.num_tta else TTA_TRANSFORMS,
            use_amp=args.use_amp
        )
        
        # Apply post-processing
        if args.use_postprocessing:
            prediction = adaptive_postprocessing(prediction, min_size=MIN_COMPONENT_SIZE)
        
        # Compute metrics
        pred_np = prediction.cpu().numpy()
        target_np = target.numpy()
        
        # Check if we have ground truth
        has_gt = np.any(target_np > 0)
        
        if has_gt:
            dice_scores = dice_coefficient_brats(pred_np, target_np)
            hd95_scores = hausdorff_95_brats(pred_np, target_np)
            
            result = {
                'patient_id': patient_id,
                'dice': dice_scores,
                'hd95': hd95_scores
            }
            all_results.append(result)
            
            for region in ['WT', 'TC', 'ET']:
                brats_dice_all[region].append(dice_scores[region])
                brats_hd95_all[region].append(hd95_scores[region])
            
            if args.verbose:
                logger.info(
                    f"{patient_id}: Dice(WT={dice_scores['WT']:.4f}, TC={dice_scores['TC']:.4f}, "
                    f"ET={dice_scores['ET']:.4f}) | HD95(WT={hd95_scores['WT']:.1f}, "
                    f"TC={hd95_scores['TC']:.1f}, ET={hd95_scores['ET']:.1f})"
                )
        else:
            logger.info(f"{patient_id}: No ground truth available, skipping metrics")
        
        # Save prediction if requested
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            
            # Map back to BraTS labels: 1->1(NCR), 2->2(ED), 3->4(ET)
            pred_brats = np.zeros_like(pred_np, dtype=np.uint8)
            pred_brats[pred_np == 1] = 1
            pred_brats[pred_np == 2] = 2
            pred_brats[pred_np == 3] = 4
            
            output_path = os.path.join(args.output_dir, f"{patient_id}_pred.nii.gz")
            nib.save(nib.Nifti1Image(pred_brats, np.eye(4)), output_path)
    
    # Compute summary statistics
    if all_results:
        logger.info(f"\n{'='*80}")
        logger.info("EVALUATION RESULTS - BRATS CHALLENGE FORMAT")
        logger.info(f"{'='*80}")
        
        mean_dice = {
            'WT': np.mean(brats_dice_all['WT']),
            'TC': np.mean(brats_dice_all['TC']),
            'ET': np.mean(brats_dice_all['ET'])
        }
        mean_dice['mean'] = np.mean([mean_dice['WT'], mean_dice['TC'], mean_dice['ET']])
        
        std_dice = {
            'WT': np.std(brats_dice_all['WT']),
            'TC': np.std(brats_dice_all['TC']),
            'ET': np.std(brats_dice_all['ET'])
        }
        
        mean_hd95 = {
            'WT': np.mean(brats_hd95_all['WT']),
            'TC': np.mean(brats_hd95_all['TC']),
            'ET': np.mean(brats_hd95_all['ET'])
        }
        mean_hd95['mean'] = np.mean([mean_hd95['WT'], mean_hd95['TC'], mean_hd95['ET']])
        
        std_hd95 = {
            'WT': np.std(brats_hd95_all['WT']),
            'TC': np.std(brats_hd95_all['TC']),
            'ET': np.std(brats_hd95_all['ET'])
        }
        
        logger.info(f"\nPatients evaluated: {len(all_results)}")
        logger.info(f"\n--- DICE SCORES ---")
        logger.info(f"  WT (Whole Tumor):  {mean_dice['WT']:.4f} ± {std_dice['WT']:.4f}")
        logger.info(f"  TC (Tumor Core):   {mean_dice['TC']:.4f} ± {std_dice['TC']:.4f}")
        logger.info(f"  ET (Enhancing):    {mean_dice['ET']:.4f} ± {std_dice['ET']:.4f}")
        logger.info(f"  Mean:              {mean_dice['mean']:.4f}")
        
        logger.info(f"\n--- HD95 SCORES (mm) ---")
        logger.info(f"  WT (Whole Tumor):  {mean_hd95['WT']:.2f} ± {std_hd95['WT']:.2f}")
        logger.info(f"  TC (Tumor Core):   {mean_hd95['TC']:.2f} ± {std_hd95['TC']:.2f}")
        logger.info(f"  ET (Enhancing):    {mean_hd95['ET']:.2f} ± {std_hd95['ET']:.2f}")
        logger.info(f"  Mean:              {mean_hd95['mean']:.2f}")
        
        logger.info(f"\n{'='*80}\n")
        
        # Save results to JSON
        if args.output_dir:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'checkpoint': args.checkpoint,
                'data_dir': args.data_dir,
                'num_patients': len(all_results),
                'num_tta': args.num_tta if args.num_tta else TTA_TRANSFORMS,
                'use_postprocessing': args.use_postprocessing,
                'dice': {
                    'WT': {'mean': float(mean_dice['WT']), 'std': float(std_dice['WT'])},
                    'TC': {'mean': float(mean_dice['TC']), 'std': float(std_dice['TC'])},
                    'ET': {'mean': float(mean_dice['ET']), 'std': float(std_dice['ET'])},
                    'mean': float(mean_dice['mean'])
                },
                'hd95': {
                    'WT': {'mean': float(mean_hd95['WT']), 'std': float(std_hd95['WT'])},
                    'TC': {'mean': float(mean_hd95['TC']), 'std': float(std_hd95['TC'])},
                    'ET': {'mean': float(mean_hd95['ET']), 'std': float(std_hd95['ET'])},
                    'mean': float(mean_hd95['mean'])
                },
                'per_patient': all_results
            }
            
            summary_path = os.path.join(args.output_dir, 'evaluation_results.json')
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Results saved to: {summary_path}")
    
    return all_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='BraTS Test Evaluation with TTA and Full Metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python evaluate_test.py --checkpoint /workspace/checkpoints/fold_0_best.pth --data_dir /workspace/dataset

  # Full evaluation with output
  python evaluate_test.py --checkpoint fold_0_best.pth --data_dir /data --output_dir /results --verbose

  # Quick test with fewer TTA
  python evaluate_test.py --checkpoint model.pth --data_dir /data --num_tta 4 --max_patients 10
        """
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (e.g., fold_0_best.pth)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to test data directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save predictions and results (optional)')
    parser.add_argument('--num_tta', type=int, default=12,
                        help='Number of TTA transforms (default: 12)')
    parser.add_argument('--use_postprocessing', action='store_true', default=True,
                        help='Apply post-processing (default: True)')
    parser.add_argument('--no_postprocessing', action='store_false', dest='use_postprocessing',
                        help='Disable post-processing')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use mixed precision (default: True)')
    parser.add_argument('--no_amp', action='store_false', dest='use_amp',
                        help='Disable mixed precision')
    parser.add_argument('--max_patients', type=int, default=None,
                        help='Maximum number of patients to evaluate (for testing)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-patient results')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    # Run evaluation
    run_evaluation(args)


if __name__ == '__main__':
    main()
