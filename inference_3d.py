#!/usr/bin/env python3
"""
BraTS 3D Inference and Visualization Script
============================================

Complete pipeline from MRI input to 3D interactive visualization:
1. Load patient MRI data (4 modalities: T1, T1ce, T2, FLAIR)
2. Run inference with trained model (with optional TTA)
3. Generate 3D meshes for each tumor region
4. Export to interactive HTML viewer with region toggle controls
5. Export VTK/VTP/VTI files for ParaView and 3D Slicer

Tumor Regions (with colors):
- NCR (Necrotic Core): Dark Red #8B0000 - Class 1
- ED (Edema): Yellow #FFD700 (semi-transparent) - Class 2  
- ET (Enhancing Tumor): Bright Red #FF0000 - Class 3
- Brain Surface: Light blue-gray #E5E5F2 (very transparent)

Output Formats:
- .vtp: VTK XML PolyData - Best for ParaView/3D Slicer with colors
- .vtk: Legacy VTK format - Maximum compatibility
- .vti: VTK ImageData - Volumetric segmentation
- .vtm: VTK MultiBlock - Combined file for easy loading
- .html: Interactive browser-based 3D viewer

Usage:
    python inference_3d.py --checkpoint model.pth --patient_dir /path/to/patient --output_dir /output
    
    # With TTA for better accuracy
    python inference_3d.py --checkpoint model.pth --patient_dir /path/to/patient --output_dir /output --use_tta

Author: BraTS 3D Pipeline
"""

import os
import sys
import argparse
import logging
import glob
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from scipy.ndimage import (
    gaussian_filter, binary_fill_holes, generate_binary_structure,
    binary_closing, binary_opening, label as ndimage_label
)

try:
    from skimage import measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("⚠️ scikit-image not installed. Install with: pip install scikit-image")

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Model Architecture
CROP_SIZE = (160, 192, 160)
NUM_CLASSES = 4
IN_CHANNELS = 4
MODEL_FILTERS = [48, 96, 192, 384, 768]
USE_ATTENTION = True
ATTENTION_TYPE = 'transformer'
NUM_ATTENTION_HEADS = 8
TRANSFORMER_DEPTH = 4
DROPOUT_RATE = 0.12

# Visualization Colors (RGBA, values 0-1)
TUMOR_COLORS = {
    1: {
        "name": "NCR", 
        "label": "Necrotic Core", 
        "color": [0.55, 0.0, 0.0, 1.0],  # Dark Red
        "hex": "#8B0000",
        "description": "Dead tissue at tumor center"
    },
    2: {
        "name": "ED", 
        "label": "Peritumoral Edema", 
        "color": [1.0, 0.84, 0.0, 0.6],  # Yellow (semi-transparent)
        "hex": "#FFD700",
        "description": "Swelling around tumor"
    },
    3: {
        "name": "ET", 
        "label": "Enhancing Tumor", 
        "color": [1.0, 0.0, 0.0, 1.0],  # Bright Red
        "hex": "#FF0000",
        "description": "Active tumor with contrast enhancement"
    },
}

BRAIN_COLOR = {
    "color": [0.9, 0.9, 0.95, 0.12],
    "hex": "#E5E5F2",
    "description": "Brain surface"
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def nnunet_normalize(img):
    """nnU-Net style normalization"""
    nonzero_mask = img > 0
    if not np.any(nonzero_mask):
        return img
    p001, p999 = np.percentile(img[nonzero_mask], [0.1, 99.9])
    img = np.clip(img, p001, p999)
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
# MODEL ARCHITECTURE (Must match training)
# ============================================================================

class MultiHeadSelfAttention3D(nn.Module):
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
        x_flat = x.reshape(B, C, -1).permute(0, 2, 1)
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
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.GELU()
    
    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class EncoderBlock3D(nn.Module):
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
# TTA FUNCTIONS
# ============================================================================

def apply_tta_transform(image, transform_idx):
    """Apply TTA transform"""
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
    """Reverse TTA transform"""
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
    """Post-processing for cleaner segmentation"""
    pred_np = prediction.cpu().numpy().astype(np.uint8) if torch.is_tensor(prediction) else prediction.astype(np.uint8)
    processed = np.zeros_like(pred_np)
    
    struct_small = generate_binary_structure(3, 1)
    struct_large = generate_binary_structure(3, 2)
    
    for class_id in range(1, 4):
        class_mask = (pred_np == class_id).astype(np.uint8)
        if not np.any(class_mask):
            continue
        
        try:
            class_mask = binary_fill_holes(class_mask).astype(np.uint8)
        except:
            pass
        
        if class_id == 3:
            class_mask = binary_opening(class_mask, struct_small).astype(np.uint8)
            class_mask = binary_closing(class_mask, struct_small).astype(np.uint8)
        else:
            class_mask = binary_closing(class_mask, struct_large).astype(np.uint8)
            class_mask = binary_opening(class_mask, struct_small).astype(np.uint8)
        
        labeled, num_features = ndimage_label(class_mask)
        if num_features > 0:
            sizes = np.bincount(labeled.ravel())
            if len(sizes) > 1:
                sizes[0] = 0
                mask_sizes = sizes > min_size
                class_mask = mask_sizes[labeled].astype(np.uint8)
        
        try:
            smoothed = gaussian_filter(class_mask.astype(float), sigma=0.5)
            class_mask = (smoothed > 0.3).astype(np.uint8)
        except:
            pass
        
        if class_id == 3:
            processed[class_mask == 1] = class_id
        elif class_id == 1:
            processed[(class_mask == 1) & (processed != 3)] = class_id
        else:
            processed[(class_mask == 1) & (processed == 0)] = class_id
    
    return processed


# ============================================================================
# 3D MESH GENERATION
# ============================================================================

class MeshGenerator:
    """Generate 3D meshes from segmentation"""
    
    def __init__(self, smoothing_sigma=1.0, step_size=2, decimate_ratio=0.3):
        self.smoothing_sigma = smoothing_sigma
        self.step_size = step_size
        self.decimate_ratio = decimate_ratio
        
        if not SKIMAGE_AVAILABLE:
            raise ImportError("scikit-image required: pip install scikit-image")
    
    def generate_all_meshes(self, prediction, brain_data, spacing=(1.0, 1.0, 1.0)):
        """Generate meshes for brain and all tumor regions"""
        result = {
            "brain": None,
            "regions": {},
            "stats": {},
            "total_tumor_volume_cm3": 0
        }
        
        # Brain surface
        brain_mesh = self._extract_brain_surface(brain_data, spacing)
        if brain_mesh:
            result["brain"] = brain_mesh
        
        # Tumor regions
        total_volume = 0
        for class_id, info in TUMOR_COLORS.items():
            mask = (prediction == class_id).astype(np.float32)
            if mask.sum() < 10:
                continue
            
            # Statistics
            volume_voxels = int(mask.sum())
            voxel_vol = spacing[0] * spacing[1] * spacing[2]
            volume_mm3 = volume_voxels * voxel_vol
            volume_cm3 = volume_mm3 / 1000.0
            total_volume += volume_cm3
            
            coords = np.argwhere(mask > 0)
            centroid = coords.mean(axis=0) * np.array(spacing) if len(coords) > 0 else [0, 0, 0]
            
            result["stats"][info["name"]] = {
                "class_id": class_id,
                "label": info["label"],
                "description": info["description"],
                "volume_voxels": volume_voxels,
                "volume_cm3": round(volume_cm3, 3),
                "centroid_mm": [round(c, 1) for c in centroid],
                "color": info["hex"]
            }
            
            # Mesh
            mesh = self._extract_surface(mask, spacing, info["color"])
            if mesh:
                mesh["name"] = info["name"]
                mesh["label"] = info["label"]
                mesh["hex"] = info["hex"]
                result["regions"][info["name"]] = mesh
        
        result["total_tumor_volume_cm3"] = round(total_volume, 3)
        return result
    
    def _extract_brain_surface(self, brain_data, spacing):
        """Extract brain surface mesh"""
        try:
            threshold = np.percentile(brain_data[brain_data > 0], 15) if np.any(brain_data > 0) else 0
            brain_mask = (brain_data > threshold).astype(np.float32)
            brain_mask = gaussian_filter(brain_mask, sigma=2.0)
            
            verts, faces, normals, _ = measure.marching_cubes(
                brain_mask, level=0.5, spacing=spacing, step_size=4
            )
            
            verts, faces = self._decimate_mesh(verts, faces, 0.15)
            
            return {
                "vertices": verts.tolist(),
                "faces": faces.tolist(),
                "color": BRAIN_COLOR["color"],
                "hex": BRAIN_COLOR["hex"],
                "opacity": 0.12,
                "name": "Brain",
                "vertex_count": len(verts),
                "face_count": len(faces)
            }
        except Exception as e:
            logger.warning(f"Brain surface extraction failed: {e}")
            return None
    
    def _extract_surface(self, mask, spacing, color):
        """Extract surface mesh from binary mask"""
        try:
            smoothed = gaussian_filter(mask, sigma=self.smoothing_sigma)
            verts, faces, normals, _ = measure.marching_cubes(
                smoothed, level=0.5, spacing=spacing, step_size=self.step_size
            )
            verts, faces = self._decimate_mesh(verts, faces, self.decimate_ratio)
            
            return {
                "vertices": verts.tolist(),
                "faces": faces.tolist(),
                "color": color,
                "opacity": color[3],
                "vertex_count": len(verts),
                "face_count": len(faces)
            }
        except Exception as e:
            logger.warning(f"Surface extraction failed: {e}")
            return None
    
    def _decimate_mesh(self, vertices, faces, ratio):
        """Simple mesh decimation"""
        if ratio >= 1.0 or len(vertices) < 100:
            return vertices, faces
        
        keep_every = max(1, int(1.0 / ratio))
        old_to_new = {}
        new_vertices = []
        
        for i in range(0, len(vertices), keep_every):
            old_to_new[i] = len(new_vertices)
            new_vertices.append(vertices[i])
        
        new_faces = []
        for face in faces:
            new_face = []
            valid = True
            for idx in face:
                nearest = (idx // keep_every) * keep_every
                if nearest >= len(vertices):
                    nearest = len(vertices) - 1
                if nearest not in old_to_new:
                    valid = False
                    break
                new_face.append(old_to_new[nearest])
            if valid and len(set(new_face)) == 3:
                new_faces.append(new_face)
        
        return np.array(new_vertices), np.array(new_faces)


# ============================================================================
# HTML VIEWER GENERATION
# ============================================================================

def generate_interactive_html(mesh_data, output_path, patient_id="Unknown"):
    """Generate interactive HTML viewer with Three.js"""
    
    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BraTS 3D Viewer - {patient_id}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            overflow: hidden;
        }}
        #container {{ width: 100vw; height: 100vh; }}
        
        /* Control Panel */
        #controls {{
            position: fixed;
            top: 20px;
            left: 20px;
            background: rgba(0, 0, 0, 0.85);
            padding: 20px;
            border-radius: 15px;
            min-width: 280px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            z-index: 1000;
        }}
        #controls h2 {{
            font-size: 18px;
            margin-bottom: 15px;
            color: #00d4ff;
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
            padding-bottom: 10px;
        }}
        #controls h3 {{
            font-size: 14px;
            margin: 15px 0 10px 0;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .region-toggle {{
            display: flex;
            align-items: center;
            margin: 8px 0;
            padding: 10px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        .region-toggle:hover {{
            background: rgba(255, 255, 255, 0.1);
        }}
        .region-toggle input {{
            display: none;
        }}
        .region-toggle .checkbox {{
            width: 24px;
            height: 24px;
            border: 2px solid #666;
            border-radius: 5px;
            margin-right: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s ease;
        }}
        .region-toggle input:checked + .checkbox {{
            border-color: var(--region-color);
            background: var(--region-color);
        }}
        .region-toggle input:checked + .checkbox::after {{
            content: '✓';
            color: #fff;
            font-weight: bold;
        }}
        .region-toggle .color-dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 10px;
        }}
        .region-toggle .region-info {{
            flex: 1;
        }}
        .region-toggle .region-name {{
            font-weight: 600;
            font-size: 14px;
        }}
        .region-toggle .region-volume {{
            font-size: 11px;
            color: #888;
            margin-top: 2px;
        }}
        
        /* Stats Panel */
        #stats {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.85);
            padding: 20px;
            border-radius: 15px;
            min-width: 220px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        #stats h3 {{
            font-size: 14px;
            color: #00d4ff;
            margin-bottom: 15px;
        }}
        .stat-row {{
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            font-size: 13px;
        }}
        .stat-value {{
            color: #00d4ff;
            font-weight: 600;
        }}
        
        /* Instructions */
        #instructions {{
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0, 0, 0, 0.7);
            padding: 12px 25px;
            border-radius: 25px;
            font-size: 13px;
            color: #aaa;
        }}
        
        /* Loading */
        #loading {{
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
            z-index: 2000;
        }}
        #loading.hidden {{ display: none; }}
        .spinner {{
            width: 50px;
            height: 50px;
            border: 3px solid rgba(255, 255, 255, 0.1);
            border-top-color: #00d4ff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
    </style>
</head>
<body>
    <div id="loading">
        <div class="spinner"></div>
        <p style="margin-top: 15px;">Loading 3D Model...</p>
    </div>
    
    <div id="container"></div>
    
    <div id="controls">
        <h2>🧠 Brain Tumor Viewer</h2>
        <p style="font-size: 12px; color: #888; margin-bottom: 10px;">Patient: {patient_id}</p>
        
        <h3>Toggle Regions</h3>
        <div id="region-toggles"></div>
        
        <h3 style="margin-top: 20px;">View Options</h3>
        <label class="region-toggle" style="--region-color: {brain_color};">
            <input type="checkbox" id="toggle-brain" checked onchange="toggleMesh('brain', this.checked)">
            <span class="checkbox"></span>
            <span class="color-dot" style="background: {brain_color};"></span>
            <span class="region-info">
                <div class="region-name">Brain Surface</div>
                <div class="region-volume">Semi-transparent outline</div>
            </span>
        </label>
        
        <button onclick="resetCamera()" style="
            width: 100%;
            margin-top: 15px;
            padding: 12px;
            background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
            border: none;
            border-radius: 8px;
            color: #fff;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
        " onmouseover="this.style.transform='scale(1.02)'" onmouseout="this.style.transform='scale(1)'">
            Reset View
        </button>
    </div>
    
    <div id="stats">
        <h3>📊 Tumor Statistics</h3>
        <div id="stats-content"></div>
    </div>
    
    <div id="instructions">
        🖱️ Left-click + drag to rotate | Scroll to zoom | Right-click + drag to pan
    </div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    
    <script>
        // Mesh data from Python
        const meshData = {mesh_data_json};
        
        // Three.js setup
        let scene, camera, renderer, controls;
        const meshes = {{}};
        
        function init() {{
            // Scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x1a1a2e);
            
            // Camera
            camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 2000);
            camera.position.set(200, 150, 200);
            
            // Renderer
            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            document.getElementById('container').appendChild(renderer.domElement);
            
            // Controls
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.autoRotate = false;
            
            // Lighting
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(100, 100, 100);
            scene.add(directionalLight);
            
            const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
            directionalLight2.position.set(-100, -50, -100);
            scene.add(directionalLight2);
            
            // Load meshes
            loadMeshes();
            
            // Generate UI
            generateRegionToggles();
            generateStats();
            
            // Hide loading
            document.getElementById('loading').classList.add('hidden');
            
            // Handle resize
            window.addEventListener('resize', onWindowResize);
            
            // Animate
            animate();
        }}
        
        function loadMeshes() {{
            // Center offset (to center the model)
            let centerOffset = [0, 0, 0];
            if (meshData.brain && meshData.brain.vertices.length > 0) {{
                const verts = meshData.brain.vertices;
                const minX = Math.min(...verts.map(v => v[0]));
                const maxX = Math.max(...verts.map(v => v[0]));
                const minY = Math.min(...verts.map(v => v[1]));
                const maxY = Math.max(...verts.map(v => v[1]));
                const minZ = Math.min(...verts.map(v => v[2]));
                const maxZ = Math.max(...verts.map(v => v[2]));
                centerOffset = [-(minX + maxX) / 2, -(minY + maxY) / 2, -(minZ + maxZ) / 2];
            }}
            
            // Brain mesh
            if (meshData.brain) {{
                const mesh = createMesh(meshData.brain, centerOffset);
                if (mesh) {{
                    meshes['brain'] = mesh;
                    scene.add(mesh);
                }}
            }}
            
            // Tumor region meshes
            for (const [name, data] of Object.entries(meshData.regions || {{}})) {{
                const mesh = createMesh(data, centerOffset);
                if (mesh) {{
                    meshes[name] = mesh;
                    scene.add(mesh);
                }}
            }}
        }}
        
        function createMesh(data, centerOffset) {{
            if (!data.vertices || data.vertices.length === 0) return null;
            
            const geometry = new THREE.BufferGeometry();
            
            // Vertices
            const vertices = [];
            for (const v of data.vertices) {{
                vertices.push(v[0] + centerOffset[0], v[1] + centerOffset[1], v[2] + centerOffset[2]);
            }}
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
            
            // Faces
            const indices = [];
            for (const f of data.faces) {{
                indices.push(f[0], f[1], f[2]);
            }}
            geometry.setIndex(indices);
            geometry.computeVertexNormals();
            
            // Material
            const color = Array.isArray(data.color) 
                ? new THREE.Color(data.color[0], data.color[1], data.color[2])
                : new THREE.Color(data.hex || '#ff0000');
            
            const material = new THREE.MeshPhongMaterial({{
                color: color,
                transparent: true,
                opacity: data.opacity || 1.0,
                side: THREE.DoubleSide,
                shininess: 30
            }});
            
            return new THREE.Mesh(geometry, material);
        }}
        
        function generateRegionToggles() {{
            const container = document.getElementById('region-toggles');
            const stats = meshData.stats || {{}};
            
            for (const [name, stat] of Object.entries(stats)) {{
                const div = document.createElement('div');
                div.className = 'region-toggle';
                div.style.setProperty('--region-color', stat.color);
                
                div.innerHTML = `
                    <input type="checkbox" id="toggle-${{name}}" checked onchange="toggleMesh('${{name}}', this.checked)">
                    <span class="checkbox"></span>
                    <span class="color-dot" style="background: ${{stat.color}};"></span>
                    <span class="region-info">
                        <div class="region-name">${{stat.label}}</div>
                        <div class="region-volume">${{stat.volume_cm3}} cm³</div>
                    </span>
                `;
                div.onclick = (e) => {{
                    if (e.target.tagName !== 'INPUT') {{
                        const checkbox = div.querySelector('input');
                        checkbox.checked = !checkbox.checked;
                        toggleMesh(name, checkbox.checked);
                    }}
                }};
                container.appendChild(div);
            }}
        }}
        
        function generateStats() {{
            const container = document.getElementById('stats-content');
            const stats = meshData.stats || {{}};
            
            let html = `
                <div class="stat-row">
                    <span>Total Tumor Volume</span>
                    <span class="stat-value">${{meshData.total_tumor_volume_cm3 || 0}} cm³</span>
                </div>
                <hr style="border: none; border-top: 1px solid rgba(255,255,255,0.1); margin: 10px 0;">
            `;
            
            for (const [name, stat] of Object.entries(stats)) {{
                html += `
                    <div class="stat-row">
                        <span style="color: ${{stat.color}};">● ${{name}}</span>
                        <span class="stat-value">${{stat.volume_cm3}} cm³</span>
                    </div>
                `;
            }}
            
            container.innerHTML = html;
        }}
        
        function toggleMesh(name, visible) {{
            if (meshes[name]) {{
                meshes[name].visible = visible;
            }}
        }}
        
        function resetCamera() {{
            camera.position.set(200, 150, 200);
            controls.target.set(0, 0, 0);
            controls.update();
        }}
        
        function onWindowResize() {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }}
        
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        
        init();
    </script>
</body>
</html>'''
    
    # Prepare mesh data for JSON
    mesh_json = json.dumps(mesh_data, cls=NumpyEncoder)
    
    html_content = html_template.format(
        patient_id=patient_id,
        mesh_data_json=mesh_json,
        brain_color=BRAIN_COLOR["hex"]
    )
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Interactive viewer saved to: {output_path}")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return super().default(obj)


# ============================================================================
# VTK/VTP EXPORT - Better for Medical Visualization
# ============================================================================

def export_vtp(mesh_data, output_dir, patient_id="tumor"):
    """Export meshes to VTK XML PolyData (.vtp) format with colors
    
    VTP format is ideal for:
    - ParaView visualization
    - 3D Slicer
    - Medical imaging applications
    - Preserves per-region colors
    
    Args:
        mesh_data: Dictionary with brain and regions meshes
        output_dir: Output directory
        patient_id: Patient identifier for filenames
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    def write_vtp_file(vertices, faces, color_rgb, opacity, filepath, name):
        """Write a single mesh to VTP format"""
        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.int32)
        
        if len(vertices) == 0 or len(faces) == 0:
            return False
        
        n_points = len(vertices)
        n_cells = len(faces)
        
        # Convert color to 0-255 range
        r, g, b = int(color_rgb[0] * 255), int(color_rgb[1] * 255), int(color_rgb[2] * 255)
        a = int(opacity * 255)
        
        # Create VTP XML content
        vtp_content = f'''<?xml version="1.0"?>
<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian">
  <PolyData>
    <Piece NumberOfPoints="{n_points}" NumberOfVerts="0" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="{n_cells}">
      <PointData>
        <DataArray type="UInt8" Name="Colors" NumberOfComponents="4" format="ascii">
'''
        # Add per-vertex colors (same color for all vertices in this region)
        for _ in range(n_points):
            vtp_content += f"          {r} {g} {b} {a}\n"
        
        vtp_content += '''        </DataArray>
      </PointData>
      <Points>
        <DataArray type="Float32" NumberOfComponents="3" format="ascii">
'''
        # Add vertices
        for v in vertices:
            vtp_content += f"          {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n"
        
        vtp_content += '''        </DataArray>
      </Points>
      <Polys>
        <DataArray type="Int32" Name="connectivity" format="ascii">
'''
        # Add face connectivity
        for f in faces:
            vtp_content += f"          {f[0]} {f[1]} {f[2]}\n"
        
        vtp_content += '''        </DataArray>
        <DataArray type="Int32" Name="offsets" format="ascii">
'''
        # Add offsets (each triangle has 3 vertices)
        for i in range(1, n_cells + 1):
            vtp_content += f"          {i * 3}\n"
        
        vtp_content += '''        </DataArray>
      </Polys>
    </Piece>
  </PolyData>
</VTKFile>
'''
        
        with open(filepath, 'w') as f:
            f.write(vtp_content)
        
        return True
    
    exported_files = []
    
    # Export brain mesh
    if mesh_data.get("brain") and mesh_data["brain"].get("vertices"):
        brain_path = output_dir / f"{patient_id}_brain.vtp"
        brain = mesh_data["brain"]
        color = brain.get("color", BRAIN_COLOR["color"])
        if write_vtp_file(brain["vertices"], brain["faces"], color[:3], color[3], brain_path, "Brain"):
            exported_files.append(brain_path)
            logger.info(f"Brain mesh saved to: {brain_path}")
    
    # Export tumor region meshes
    for name, data in mesh_data.get("regions", {}).items():
        if data and data.get("vertices"):
            region_path = output_dir / f"{patient_id}_{name}.vtp"
            color = data.get("color", [1, 0, 0, 1])
            if write_vtp_file(data["vertices"], data["faces"], color[:3], color[3], region_path, name):
                exported_files.append(region_path)
                logger.info(f"{name} mesh saved to: {region_path}")
    
    # Create a combined multi-block VTM file for easy loading
    vtm_path = output_dir / f"{patient_id}_combined.vtm"
    write_vtm_file(exported_files, vtm_path, patient_id)
    
    return exported_files


def write_vtm_file(vtp_files, output_path, patient_id):
    """Write VTK MultiBlock file (.vtm) that references all VTP files"""
    vtm_content = '''<?xml version="1.0"?>
<VTKFile type="vtkMultiBlockDataSet" version="1.0" byte_order="LittleEndian">
  <vtkMultiBlockDataSet>
'''
    for i, vtp_file in enumerate(vtp_files):
        name = Path(vtp_file).stem.replace(f"{patient_id}_", "")
        vtm_content += f'    <DataSet index="{i}" name="{name}" file="{Path(vtp_file).name}"/>\n'
    
    vtm_content += '''  </vtkMultiBlockDataSet>
</VTKFile>
'''
    
    with open(output_path, 'w') as f:
        f.write(vtm_content)
    
    logger.info(f"Combined VTM file saved to: {output_path}")


def export_vti(segmentation, output_path, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    """Export segmentation volume to VTK ImageData (.vti) format
    
    VTI format preserves:
    - Full 3D volumetric data
    - Spacing information
    - Can be loaded in ParaView/3D Slicer
    - Supports color mapping by label
    
    Args:
        segmentation: 3D numpy array with labels (0=BG, 1=NCR, 2=ED, 3=ET)
        output_path: Output file path
        spacing: Voxel spacing in mm
        origin: Volume origin coordinates
    """
    seg = np.asarray(segmentation, dtype=np.uint8)
    
    # VTI expects Fortran order (x, y, z) but numpy is C order (z, y, x)
    # We'll write in the native order and specify dimensions accordingly
    nz, ny, nx = seg.shape
    
    vti_content = f'''<?xml version="1.0"?>
<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian">
  <ImageData WholeExtent="0 {nx-1} 0 {ny-1} 0 {nz-1}" Origin="{origin[0]} {origin[1]} {origin[2]}" Spacing="{spacing[0]} {spacing[1]} {spacing[2]}">
    <Piece Extent="0 {nx-1} 0 {ny-1} 0 {nz-1}">
      <PointData Scalars="Labels">
        <DataArray type="UInt8" Name="Labels" format="ascii">
'''
    
    # Flatten and write data (VTK expects x-fastest order)
    # Transpose from (z,y,x) to (x,y,z) order
    seg_vtk = np.transpose(seg, (2, 1, 0)).flatten()
    
    # Write in chunks for readability
    chunk_size = 20
    for i in range(0, len(seg_vtk), chunk_size):
        chunk = seg_vtk[i:i+chunk_size]
        vti_content += "          " + " ".join(map(str, chunk)) + "\n"
    
    vti_content += '''        </DataArray>
      </PointData>
    </Piece>
  </ImageData>
</VTKFile>
'''
    
    with open(output_path, 'w') as f:
        f.write(vti_content)
    
    logger.info(f"VTI volume saved to: {output_path}")


def export_vtk_legacy(mesh_data, output_dir, patient_id="tumor"):
    """Export meshes to legacy VTK format (.vtk) - maximum compatibility
    
    Legacy VTK format works with almost all VTK-compatible software.
    Creates separate files for each region with embedded colors.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    def write_vtk_polydata(vertices, faces, color_rgb, filepath, name):
        """Write mesh to legacy VTK polydata format"""
        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.int32)
        
        if len(vertices) == 0 or len(faces) == 0:
            return False
        
        n_points = len(vertices)
        n_cells = len(faces)
        
        # Convert color to 0-255
        r, g, b = int(color_rgb[0] * 255), int(color_rgb[1] * 255), int(color_rgb[2] * 255)
        
        vtk_content = f'''# vtk DataFile Version 3.0
{name} - BraTS 3D Segmentation
ASCII
DATASET POLYDATA
POINTS {n_points} float
'''
        # Add vertices
        for v in vertices:
            vtk_content += f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n"
        
        vtk_content += f"\nPOLYGONS {n_cells} {n_cells * 4}\n"
        
        # Add faces (prefixed with vertex count = 3 for triangles)
        for f in faces:
            vtk_content += f"3 {f[0]} {f[1]} {f[2]}\n"
        
        # Add color data
        vtk_content += f"\nPOINT_DATA {n_points}\n"
        vtk_content += "COLOR_SCALARS colors 3\n"
        
        # Normalized RGB colors (0-1 range for COLOR_SCALARS)
        r_norm, g_norm, b_norm = color_rgb[0], color_rgb[1], color_rgb[2]
        for _ in range(n_points):
            vtk_content += f"{r_norm:.3f} {g_norm:.3f} {b_norm:.3f}\n"
        
        with open(filepath, 'w') as f:
            f.write(vtk_content)
        
        return True
    
    exported_files = []
    
    # Export brain
    if mesh_data.get("brain") and mesh_data["brain"].get("vertices"):
        brain_path = output_dir / f"{patient_id}_brain.vtk"
        brain = mesh_data["brain"]
        color = brain.get("color", BRAIN_COLOR["color"])
        if write_vtk_polydata(brain["vertices"], brain["faces"], color[:3], brain_path, "Brain"):
            exported_files.append(brain_path)
    
    # Export tumor regions
    for name, data in mesh_data.get("regions", {}).items():
        if data and data.get("vertices"):
            region_path = output_dir / f"{patient_id}_{name}.vtk"
            color = data.get("color", [1, 0, 0, 1])
            if write_vtk_polydata(data["vertices"], data["faces"], color[:3], region_path, name):
                exported_files.append(region_path)
    
    logger.info(f"Exported {len(exported_files)} VTK files to {output_dir}")
    return exported_files


def create_paraview_state(output_dir, patient_id, vtp_files):
    """Create a ParaView state file (.pvsm) for easy loading with preset colors"""
    
    # Color lookup table for tumor regions
    color_map = {
        "brain": {"rgb": [0.9, 0.9, 0.95], "opacity": 0.15},
        "NCR": {"rgb": [0.55, 0.0, 0.0], "opacity": 1.0},  # Dark red
        "ED": {"rgb": [1.0, 0.84, 0.0], "opacity": 0.6},   # Yellow
        "ET": {"rgb": [1.0, 0.0, 0.0], "opacity": 1.0},    # Bright red
    }
    
    # Create a simple Python script for ParaView instead of .pvsm
    script_content = f'''# ParaView Python Script - {patient_id}
# Run this script in ParaView: Tools > Python Shell > Run Script

from paraview.simple import *

# Set background
view = GetActiveViewOrCreate('RenderView')
view.Background = [0.1, 0.1, 0.15]

# Color definitions
colors = {{
    "brain": ([0.9, 0.9, 0.95], 0.15),
    "NCR": ([0.55, 0.0, 0.0], 1.0),
    "ED": ([1.0, 0.84, 0.0], 0.6),
    "ET": ([1.0, 0.0, 0.0], 1.0),
}}

# Load and display each region
import os
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else r"{output_dir}"

for name, (rgb, opacity) in colors.items():
    filepath = os.path.join(script_dir, f"{patient_id}_{{name}}.vtp")
    if os.path.exists(filepath):
        reader = XMLPolyDataReader(FileName=[filepath])
        display = Show(reader, view)
        display.Representation = 'Surface'
        display.DiffuseColor = rgb
        display.Opacity = opacity
        display.Specular = 0.3
        RenameSource(name, reader)
        print(f"Loaded: {{name}}")

# Reset camera
view.ResetCamera()
Render()

print("\\n=== BraTS 3D Visualization Loaded ===")
print("Use the Pipeline Browser to toggle regions on/off")
'''
    
    script_path = output_dir / f"{patient_id}_paraview_script.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    logger.info(f"ParaView script saved to: {script_path}")
    return script_path


# ============================================================================
# OBJ EXPORT - For Blender, Maya, 3ds Max
# ============================================================================

def export_obj(mesh_data, output_dir, patient_id="tumor"):
    """Export meshes to OBJ format with MTL material file
    
    OBJ+MTL format is ideal for:
    - Blender (native import with colors)
    - Maya, 3ds Max, Cinema 4D
    - Most 3D software
    - Each region is a separate object for easy toggling
    
    Args:
        mesh_data: Dictionary with brain and regions meshes
        output_dir: Output directory
        patient_id: Patient identifier for filenames
        
    Returns:
        Path to the combined OBJ file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    obj_path = output_dir / f"{patient_id}_brain_tumor.obj"
    mtl_path = output_dir / f"{patient_id}_brain_tumor.mtl"
    
    # Material definitions with colors
    materials = {
        "Brain": {"Kd": [0.9, 0.9, 0.95], "d": 0.15},      # Light gray, transparent
        "NCR": {"Kd": [0.55, 0.0, 0.0], "d": 1.0},         # Dark red (Necrotic Core)
        "ED": {"Kd": [1.0, 0.84, 0.0], "d": 0.6},          # Yellow (Edema)
        "ET": {"Kd": [1.0, 0.0, 0.0], "d": 1.0},           # Bright red (Enhancing Tumor)
    }
    
    # Write MTL file
    mtl_content = "# BraTS Brain Tumor Materials\n"
    mtl_content += f"# Patient: {patient_id}\n"
    mtl_content += "# Colors: NCR=Dark Red, ED=Yellow, ET=Bright Red, Brain=Gray\n\n"
    
    for mat_name, props in materials.items():
        mtl_content += f"newmtl {mat_name}\n"
        mtl_content += f"Kd {props['Kd'][0]:.3f} {props['Kd'][1]:.3f} {props['Kd'][2]:.3f}\n"
        mtl_content += f"Ka 0.1 0.1 0.1\n"  # Ambient
        mtl_content += f"Ks 0.3 0.3 0.3\n"  # Specular
        mtl_content += f"Ns 50.0\n"          # Shininess
        mtl_content += f"d {props['d']:.2f}\n"  # Dissolve (transparency)
        mtl_content += f"illum 2\n\n"
    
    with open(mtl_path, 'w') as f:
        f.write(mtl_content)
    
    # Write OBJ file
    obj_content = f"# BraTS Brain Tumor 3D Model\n"
    obj_content += f"# Patient: {patient_id}\n"
    obj_content += f"# Regions: Brain (transparent), NCR (dark red), ED (yellow), ET (bright red)\n"
    obj_content += f"# Import in Blender: File > Import > Wavefront (.obj)\n"
    obj_content += f"mtllib {mtl_path.name}\n\n"
    
    vertex_offset = 1  # OBJ indices are 1-based
    
    def add_mesh_to_obj(mesh_info, material_name):
        nonlocal vertex_offset, obj_content
        
        if not mesh_info or 'vertices' not in mesh_info:
            return
        
        vertices = np.array(mesh_info['vertices'], dtype=np.float32)
        faces = np.array(mesh_info['faces'], dtype=np.int32)
        
        if len(vertices) == 0 or len(faces) == 0:
            return
        
        obj_content += f"# {material_name}\n"
        obj_content += f"o {material_name}\n"
        obj_content += f"usemtl {material_name}\n"
        
        # Write vertices
        for v in vertices:
            obj_content += f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n"
        
        # Write faces (with offset for combined file)
        for f in faces:
            obj_content += f"f {f[0] + vertex_offset} {f[1] + vertex_offset} {f[2] + vertex_offset}\n"
        
        vertex_offset += len(vertices)
        obj_content += "\n"
    
    # Add brain mesh
    if mesh_data.get("brain") and mesh_data["brain"].get("vertices"):
        add_mesh_to_obj(mesh_data["brain"], "Brain")
    
    # Add tumor region meshes
    for name, data in mesh_data.get("regions", {}).items():
        if data and data.get("vertices"):
            add_mesh_to_obj(data, name)
    
    with open(obj_path, 'w') as f:
        f.write(obj_content)
    
    logger.info(f"OBJ model saved to: {obj_path}")
    logger.info(f"MTL materials saved to: {mtl_path}")
    
    return obj_path, mtl_path


# ============================================================================
# MAIN INFERENCE PIPELINE
# ============================================================================

def load_model(checkpoint_path, device):
    """Load trained model"""
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
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model


def load_patient_data(patient_dir):
    """Load patient MRI data"""
    patient_dir = Path(patient_dir)
    
    modality_mappings = [
        ['t1', 't1n'],
        ['t1ce', 't1c'],
        ['t2', 't2w'],
        ['flair', 't2f']
    ]
    
    img_data = []
    reference_img = None
    
    for mod_variants in modality_mappings:
        file_path = None
        for mod in mod_variants:
            patterns = [
                patient_dir / f"*{mod}.nii.gz",
                patient_dir / f"*{mod}.nii",
                patient_dir / f"*_{mod}.nii.gz",
            ]
            for pattern in patterns:
                files = list(patient_dir.glob(pattern.name))
                if files:
                    file_path = files[0]
                    break
            if file_path:
                break
        
        if file_path:
            nii = nib.load(str(file_path))
            img = nii.get_fdata().astype(np.float32)
            if reference_img is None:
                reference_img = nii
            img = nnunet_normalize(img)
            img_data.append(img)
    
    if len(img_data) < 4:
        raise ValueError(f"Missing modalities in {patient_dir}")
    
    image = np.stack(img_data, axis=0)
    original_shape = image.shape[1:]
    
    # Crop/pad for model
    image_cropped = np.array([center_crop_or_pad(image[c], CROP_SIZE) for c in range(4)])
    
    # Use FLAIR for brain visualization
    brain_data = img_data[3]  # FLAIR
    brain_cropped = center_crop_or_pad(brain_data, CROP_SIZE)
    
    return image_cropped, brain_cropped, original_shape, reference_img


def run_inference(model, image, device, use_tta=False, num_tta=12):
    """Run inference with optional TTA"""
    image_tensor = torch.from_numpy(image).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        if use_tta:
            all_probs = []
            for t in range(num_tta):
                img_t = apply_tta_transform(image_tensor, t)
                with autocast():
                    pred, _ = model(img_t)
                prob = F.softmax(pred, dim=1)
                prob = reverse_tta_transform(prob, t)
                all_probs.append(prob)
            avg_prob = torch.stack(all_probs, dim=0).mean(dim=0)
            prediction = torch.argmax(avg_prob, dim=1).squeeze(0)
        else:
            with autocast():
                pred, _ = model(image_tensor)
            prediction = torch.argmax(pred, dim=1).squeeze(0)
    
    return prediction.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='BraTS 3D Inference and Visualization')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--patient_dir', type=str, required=True, help='Path to patient folder with MRI files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--use_tta', action='store_true', help='Use test-time augmentation')
    parser.add_argument('--num_tta', type=int, default=12, help='Number of TTA transforms (default: 12)')
    parser.add_argument('--no_postprocess', action='store_true', help='Disable post-processing')
    parser.add_argument('--save_nifti', action='store_true', help='Save segmentation as NIfTI')
    
    args = parser.parse_args()
    
    set_seed(42)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    patient_id = Path(args.patient_dir).name
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Load patient data
    logger.info(f"Loading patient data from: {args.patient_dir}")
    image, brain_data, original_shape, reference_nii = load_patient_data(args.patient_dir)
    
    # Run inference
    logger.info(f"Running inference {'with TTA' if args.use_tta else 'without TTA'}...")
    prediction = run_inference(model, image, device, args.use_tta, args.num_tta)
    
    # Post-processing
    if not args.no_postprocess:
        logger.info("Applying post-processing...")
        prediction = adaptive_postprocessing(prediction)
    
    # Generate 3D meshes
    logger.info("Generating 3D meshes...")
    mesh_generator = MeshGenerator()
    mesh_data = mesh_generator.generate_all_meshes(prediction, brain_data)
    
    # Save mesh data as JSON
    json_path = output_dir / "mesh_data.json"
    with open(json_path, 'w') as f:
        json.dump(mesh_data, f, cls=NumpyEncoder, indent=2)
    logger.info(f"Mesh data saved to: {json_path}")
    
    # Generate interactive HTML viewer
    html_path = output_dir / "viewer.html"
    generate_interactive_html(mesh_data, html_path, patient_id)
    
    # Export VTK files (VTP format with colors)
    logger.info("Exporting VTK files...")
    vtp_files = export_vtp(mesh_data, output_dir, patient_id)
    
    # Export legacy VTK format for maximum compatibility
    export_vtk_legacy(mesh_data, output_dir, patient_id)
    
    # Export volumetric segmentation as VTI
    vti_path = output_dir / f"{patient_id}_segmentation.vti"
    export_vti(prediction, vti_path)
    
    # Create ParaView helper script
    create_paraview_state(output_dir, patient_id, vtp_files)
    
    # Export OBJ+MTL for Blender
    logger.info("Exporting OBJ for Blender...")
    obj_path, mtl_path = export_obj(mesh_data, output_dir, patient_id)
    
    # Save segmentation as NIfTI
    if args.save_nifti:
        # Map back to BraTS labels
        seg_brats = np.zeros_like(prediction, dtype=np.uint8)
        seg_brats[prediction == 1] = 1
        seg_brats[prediction == 2] = 2
        seg_brats[prediction == 3] = 4
        
        nifti_path = output_dir / f"{patient_id}_seg.nii.gz"
        if reference_nii:
            seg_nii = nib.Nifti1Image(seg_brats, reference_nii.affine, reference_nii.header)
        else:
            seg_nii = nib.Nifti1Image(seg_brats, np.eye(4))
        nib.save(seg_nii, str(nifti_path))
        logger.info(f"Segmentation saved to: {nifti_path}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("INFERENCE COMPLETE")
    logger.info("="*60)
    logger.info(f"Patient: {patient_id}")
    logger.info(f"Total tumor volume: {mesh_data['total_tumor_volume_cm3']} cm³")
    for name, stat in mesh_data['stats'].items():
        logger.info(f"  {stat['label']}: {stat['volume_cm3']} cm³")
    logger.info(f"\nOutputs:")
    logger.info(f"  - Interactive HTML viewer: {html_path}")
    logger.info(f"  - OBJ+MTL (Blender): {obj_path}")
    logger.info(f"  - VTP meshes (ParaView/3D Slicer): {output_dir}/*.vtp")
    logger.info(f"  - VTK meshes (legacy format): {output_dir}/*.vtk")
    logger.info(f"  - VTI volume: {vti_path}")
    logger.info(f"  - ParaView script: {output_dir}/{patient_id}_paraview_script.py")
    logger.info(f"  - Mesh data JSON: {json_path}")
    logger.info("="*60)
    logger.info("\nTo view in Blender:")
    logger.info(f"  1. File > Import > Wavefront (.obj)")
    logger.info(f"  2. Select {patient_id}_brain_tumor.obj")
    logger.info(f"  3. Materials/colors load automatically from .mtl")
    logger.info(f"  4. Toggle regions in Outliner (eye icon)")
    logger.info("\nTo view in ParaView:")
    logger.info(f"  1. File > Open > Select {patient_id}_combined.vtm")


if __name__ == '__main__':
    main()
