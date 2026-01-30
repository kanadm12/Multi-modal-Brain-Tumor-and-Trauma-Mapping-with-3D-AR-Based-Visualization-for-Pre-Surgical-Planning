# =============================================================================
# BRATS INFERENCE MODULE
# 
# Loads trained model and runs prediction on new MRI scans
# Supports Test-Time Augmentation (TTA) for better accuracy
# =============================================================================

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import SimpleITK as sitk

# Import model architecture from train.py
# We'll define a minimal version here to avoid circular imports


# =============================================================================
# CONFIGURATION (must match training)
# =============================================================================

CROP_SIZE = (192, 224, 192)
NUM_CLASSES = 4
IN_CHANNELS = 4
MODEL_FILTERS = [64, 128, 256, 512, 1024]
NUM_ATTENTION_HEADS = 8
TRANSFORMER_DEPTH = 3
DROPOUT_RATE = 0.15
TTA_TRANSFORMS = 12


# =============================================================================
# MODEL COMPONENTS (copied from train.py for standalone use)
# =============================================================================

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
    """Lightweight channel and spatial attention"""
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
    """Optimized 3D U-Net with transformer bottleneck"""
    def __init__(self, in_channels=4, num_classes=4, filters=None, use_attention=True, 
                 attention_type='transformer', num_heads=8, dropout=0.2, use_checkpointing=False):
        super().__init__()
        
        if filters is None:
            filters = MODEL_FILTERS
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.filters = filters
        self.use_attention = use_attention
        self.use_checkpointing = use_checkpointing
        
        # Input convolution
        self.input_conv = nn.Sequential(
            ConvBlock3D(in_channels, filters[0]),
            ConvBlock3D(filters[0], filters[0])
        )
        
        # Encoder
        self.encoder = nn.ModuleList([
            EncoderBlock3D(filters[i], filters[i + 1], use_attention, attention_type)
            for i in range(len(filters) - 1)
        ])
        
        # Bottleneck
        self.bottleneck = TransformerBottleneck(filters[-1], num_heads=num_heads, depth=TRANSFORMER_DEPTH)
        
        # Attention gates
        self.attention_gates = nn.ModuleList([
            AttentionGate3D(
                gate_channels=filters[i + 1],
                skip_channels=filters[i],
                inter_channels=filters[i] // 2
            )
            for i in range(len(filters) - 2, -1, -1)
        ])
        
        # Decoder
        self.decoder = nn.ModuleList([
            DecoderBlock3D(filters[i + 1], filters[i], use_attention, attention_type)
            for i in range(len(filters) - 2, -1, -1)
        ])
        
        # Output
        self.output_conv = nn.Conv3d(filters[0], num_classes, 1)
        
        # Deep supervision
        self.aux_outputs = nn.ModuleList([
            nn.Conv3d(filters[i], num_classes, 1) 
            for i in range(len(filters) - 2, -1, -1)
        ])
        
        self.dropout = nn.Dropout3d(dropout)
    
    def forward(self, x):
        # Input
        x0 = self.input_conv(x)
        
        # Encoder
        encoder_outputs = [x0]
        x = x0
        
        for encoder_block in self.encoder:
            x = F.max_pool3d(x, 2)
            x = encoder_block(x)
            x = self.dropout(x)
            encoder_outputs.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
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
        
        # Output
        out = self.output_conv(x)
        
        return out, aux_outputs


# =============================================================================
# PREPROCESSING UTILITIES
# =============================================================================

def nnunet_normalize(img: np.ndarray) -> np.ndarray:
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


def center_crop_or_pad(volume: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Center crop or pad volume to target shape"""
    output = np.zeros(target_shape, dtype=volume.dtype)
    min_shape = np.minimum(volume.shape, target_shape)
    start_src = ((np.array(volume.shape) - min_shape) // 2).astype(int)
    start_dst = ((np.array(target_shape) - min_shape) // 2).astype(int)
    
    slice_src = tuple(slice(s, s + m) for s, m in zip(start_src, min_shape))
    slice_dst = tuple(slice(s, s + m) for s, m in zip(start_dst, min_shape))
    
    output[slice_dst] = volume[slice_src]
    return output


# =============================================================================
# TTA FUNCTIONS
# =============================================================================

def apply_tta_transform(image: torch.Tensor, transform_idx: int) -> torch.Tensor:
    """Apply TTA transform"""
    if transform_idx == 0:
        return image
    elif transform_idx == 1:
        return torch.flip(image, dims=[4])
    elif transform_idx == 2:
        return torch.flip(image, dims=[3])
    elif transform_idx == 3:
        return torch.flip(image, dims=[2])
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
        return torch.flip(torch.flip(image, dims=[4]), dims=[3])


def reverse_tta_transform(pred: torch.Tensor, transform_idx: int) -> torch.Tensor:
    """Reverse TTA transform"""
    if transform_idx == 0:
        return pred
    elif transform_idx == 1:
        return torch.flip(pred, dims=[3])
    elif transform_idx == 2:
        return torch.flip(pred, dims=[2])
    elif transform_idx == 3:
        return torch.flip(pred, dims=[1])
    elif transform_idx == 4:
        return torch.rot90(pred, 3, dims=[2, 3])
    elif transform_idx == 5:
        return torch.rot90(pred, 2, dims=[2, 3])
    elif transform_idx == 6:
        return torch.rot90(pred, 1, dims=[2, 3])
    elif transform_idx == 7:
        return torch.rot90(pred, 3, dims=[1, 3])
    elif transform_idx == 8:
        return torch.rot90(pred, 1, dims=[1, 3])
    elif transform_idx == 9:
        return torch.rot90(pred, 3, dims=[1, 2])
    elif transform_idx == 10:
        return torch.rot90(pred, 1, dims=[1, 2])
    else:
        return torch.flip(torch.flip(pred, dims=[2]), dims=[3])


# =============================================================================
# INFERENCE CLASS
# =============================================================================

class BraTSInference:
    """BraTS Inference Pipeline"""
    
    def __init__(self, checkpoint_path: str, device: str = None):
        """
        Initialize inference engine.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = self._load_model(checkpoint_path)
        self.use_amp = torch.cuda.is_available()
        
        print(f"✅ Inference engine initialized on {self.device}")
    
    def _load_model(self, checkpoint_path: str) -> nn.Module:
        """Load model from checkpoint"""
        print(f"Loading model from: {checkpoint_path}")
        
        model = OptimizedUNet3D(
            in_channels=IN_CHANNELS,
            num_classes=NUM_CLASSES,
            filters=MODEL_FILTERS,
            use_attention=True,
            attention_type='transformer',
            num_heads=NUM_ATTENTION_HEADS,
            dropout=DROPOUT_RATE,
            use_checkpointing=False
        ).to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Handle DDP wrapped models
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model.eval()
        
        return model
    
    def load_patient_data(self, patient_dir: Path) -> Tuple[np.ndarray, Dict]:
        """
        Load MRI modalities from patient directory.
        
        Args:
            patient_dir: Directory containing NIfTI files
            
        Returns:
            Tuple of (stacked image data, metadata dict)
        """
        modality_mappings = [
            ['t1', 't1n'],
            ['t1ce', 't1c'],
            ['t2', 't2w'],
            ['flair', 't2f']
        ]
        
        img_data = []
        metadata = {'files': [], 'original_shape': None, 'spacing': None}
        
        for mod_variants in modality_mappings:
            file_path = None
            for mod in mod_variants:
                files = list(patient_dir.glob(f"*{mod}*.nii*"))
                if files:
                    file_path = files[0]
                    break
            
            if file_path and file_path.stat().st_size > 1024:
                img_sitk = sitk.ReadImage(str(file_path))
                img = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
                
                if metadata['original_shape'] is None:
                    metadata['original_shape'] = img.shape
                    metadata['spacing'] = img_sitk.GetSpacing()
                
                img = nnunet_normalize(img)
                img_data.append(img)
                metadata['files'].append(str(file_path))
            else:
                # Missing modality - use zeros
                if img_data:
                    img_data.append(np.zeros_like(img_data[0]))
                else:
                    img_data.append(np.zeros(CROP_SIZE))
        
        # Stack and crop/pad
        img = np.stack(img_data, axis=0)
        img = np.stack([center_crop_or_pad(img[i], CROP_SIZE) for i in range(img.shape[0])])
        
        return img, metadata
    
    @torch.no_grad()
    def predict(self, patient_dir: Path, use_tta: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run segmentation prediction on patient data.
        
        Args:
            patient_dir: Directory containing NIfTI files
            use_tta: Whether to use Test-Time Augmentation
            
        Returns:
            Tuple of (segmentation prediction, first modality data for visualization)
        """
        # Load data
        img_data, metadata = self.load_patient_data(patient_dir)
        
        # Convert to tensor
        img_tensor = torch.tensor(img_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        if use_tta:
            # Test-Time Augmentation
            pred_list = []
            
            for transform_idx in range(TTA_TRANSFORMS):
                img_tta = apply_tta_transform(img_tensor, transform_idx)
                
                with autocast(enabled=self.use_amp):
                    outputs, _ = self.model(img_tta)
                    probs = F.softmax(outputs, dim=1)
                    pred_tta = torch.argmax(probs, dim=1)
                
                pred_tta = reverse_tta_transform(pred_tta, transform_idx)
                pred_list.append(pred_tta.float())
            
            # Ensemble by voting
            pred_stack = torch.stack(pred_list, dim=0)
            pred = torch.mode(pred_stack, dim=0).values.squeeze()
        else:
            with autocast(enabled=self.use_amp):
                outputs, _ = self.model(img_tensor)
                pred = torch.argmax(outputs, dim=1).squeeze()
        
        prediction = pred.cpu().numpy().astype(np.uint8)
        
        # Return prediction and first modality for visualization
        return prediction, img_data[0]
    
    def predict_with_probabilities(self, patient_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run prediction and return class probabilities.
        
        Returns:
            Tuple of (segmentation, probabilities [C, D, H, W], input data)
        """
        img_data, metadata = self.load_patient_data(patient_dir)
        img_tensor = torch.tensor(img_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            with autocast(enabled=self.use_amp):
                outputs, _ = self.model(img_tensor)
                probs = F.softmax(outputs, dim=1).squeeze(0)
                pred = torch.argmax(probs, dim=0)
        
        return pred.cpu().numpy(), probs.cpu().numpy(), img_data[0]


# =============================================================================
# CLI USAGE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BraTS Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Path to patient directory")
    parser.add_argument("--output", type=str, required=True, help="Output path for segmentation")
    parser.add_argument("--no-tta", action="store_true", help="Disable TTA")
    
    args = parser.parse_args()
    
    # Run inference
    engine = BraTSInference(args.checkpoint)
    prediction, _ = engine.predict(Path(args.input), use_tta=not args.no_tta)
    
    # Save result
    img = sitk.GetImageFromArray(prediction.astype(np.int16))
    sitk.WriteImage(img, args.output)
    
    print(f"✅ Segmentation saved to: {args.output}")
