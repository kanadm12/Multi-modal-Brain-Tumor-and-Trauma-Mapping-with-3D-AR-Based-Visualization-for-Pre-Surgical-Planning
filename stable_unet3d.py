"""
StableUNet3D - 3D U-Net with Attention Gates and ASPP Bottleneck
================================================================

A 3D U-Net architecture for brain tumor segmentation (BraTS dataset).

Features:
- Encoder-Decoder architecture with skip connections
- Attention Gates for improved feature selection
- ASPP (Atrous Spatial Pyramid Pooling) bottleneck
- Residual blocks in encoder and decoder
- Deep supervision outputs
- Instance Normalization for stability

Input: 4-channel 3D volume (T1, T1ce, T2, FLAIR modalities)
Output: 4-class segmentation (background, NCR/NET, ED, ET)

Compatible with checkpoint: unet_modified_83_38.pth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with two 3D convolutions and skip connection."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.bn1 = nn.InstanceNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.bn2 = nn.InstanceNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out


class EncoderBlock(nn.Module):
    """Encoder block with two convolutions followed by a residual block."""
    
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.InstanceNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.InstanceNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.residual = ResidualBlock(out_ch)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.residual(x)
        return x


class DecoderBlock(nn.Module):
    """Decoder block with two convolutions followed by a residual block."""
    
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.InstanceNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.InstanceNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.residual = ResidualBlock(out_ch)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.residual(x)
        return x


class AttentionGate(nn.Module):
    """
    Attention Gate for focusing on relevant spatial regions.
    
    Args:
        F_g: Number of channels in gating signal (from decoder)
        F_l: Number of channels in skip connection (from encoder)
        F_int: Number of intermediate channels
    """
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, 1, bias=False),
            nn.InstanceNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, 1, bias=False),
            nn.InstanceNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, 1, bias=False),
            nn.InstanceNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: Gating signal from decoder path
            x: Skip connection from encoder path
        Returns:
            Attention-weighted skip connection
        """
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='trilinear', align_corners=False)
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module.
    
    Captures multi-scale context using parallel dilated convolutions
    with different dilation rates plus global average pooling.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # 1x1 convolution
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels // 4, 1),
            nn.InstanceNorm3d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 convolution with dilation=2
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels // 4, 3, padding=2, dilation=2),
            nn.InstanceNorm3d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 convolution with dilation=4
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels // 4, 3, padding=4, dilation=4),
            nn.InstanceNorm3d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 convolution with dilation=6
        self.conv4 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels // 4, 3, padding=6, dilation=6),
            nn.InstanceNorm3d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # Global average pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, out_channels // 4, 1),
            nn.ReLU(inplace=True)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv3d(out_channels + out_channels // 4, out_channels, 1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=size, mode='trilinear', align_corners=False)
        
        out = torch.cat([feat1, feat2, feat3, feat4, global_feat], dim=1)
        out = self.fusion(out)
        
        return out


class StableUNet3D(nn.Module):
    """
    3D U-Net with Attention Gates and ASPP Bottleneck.
    
    Architecture:
        - 4 encoder blocks with increasing filters [32, 64, 128, 256]
        - ASPP bottleneck with 512 filters
        - 4 decoder blocks with attention-gated skip connections
        - Deep supervision outputs at multiple scales
    
    Args:
        in_channels: Number of input channels (default: 4 for BraTS)
        out_channels: Number of output classes (default: 4 for BraTS)
    
    Example:
        >>> model = StableUNet3D(in_channels=4, out_channels=4)
        >>> x = torch.randn(1, 4, 144, 144, 144)
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([1, 4, 144, 144, 144])
    """
    
    def __init__(self, in_channels: int = 4, out_channels: int = 4):
        super().__init__()
        filters = [32, 64, 128, 256, 512]
        
        # Encoder path
        self.enc1 = EncoderBlock(in_channels, filters[0])
        self.enc2 = EncoderBlock(filters[0], filters[1])
        self.enc3 = EncoderBlock(filters[1], filters[2])
        self.enc4 = EncoderBlock(filters[2], filters[3])
        
        self.pool = nn.MaxPool3d(2)
        
        # Bottleneck with ASPP
        self.bottleneck = ASPP(filters[3], filters[4])
        
        # Attention gates
        self.att4 = AttentionGate(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.att3 = AttentionGate(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.att2 = AttentionGate(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.att1 = AttentionGate(F_g=filters[0], F_l=filters[0], F_int=filters[0] // 2)
        
        # Decoder path with transposed convolutions for upsampling
        self.up4 = nn.ConvTranspose3d(filters[4], filters[3], 2, stride=2)
        self.dec4 = DecoderBlock(filters[4], filters[3])
        
        self.up3 = nn.ConvTranspose3d(filters[3], filters[2], 2, stride=2)
        self.dec3 = DecoderBlock(filters[3], filters[2])
        
        self.up2 = nn.ConvTranspose3d(filters[2], filters[1], 2, stride=2)
        self.dec2 = DecoderBlock(filters[2], filters[1])
        
        self.up1 = nn.ConvTranspose3d(filters[1], filters[0], 2, stride=2)
        self.dec1 = DecoderBlock(filters[1], filters[0])
        
        # Final output layer
        self.final = nn.Conv3d(filters[0], out_channels, 1)
        
        # Deep supervision heads
        self.ds4 = nn.Conv3d(filters[3], out_channels, 1)
        self.ds3 = nn.Conv3d(filters[2], out_channels, 1)
        self.ds2 = nn.Conv3d(filters[1], out_channels, 1)
        
    def forward(self, x: torch.Tensor, deep_supervision: bool = False):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            deep_supervision: If True and training, return auxiliary outputs
            
        Returns:
            If deep_supervision=True and training: tuple of (main_output, ds4, ds3, ds2)
            Otherwise: main output tensor
        """
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with attention-gated skip connections
        d4 = self.up4(b)
        e4_att = self.att4(d4, e4)
        d4 = torch.cat([d4, e4_att], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        e3_att = self.att3(d3, e3)
        d3 = torch.cat([d3, e3_att], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        e2_att = self.att2(d2, e2)
        d2 = torch.cat([d2, e2_att], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        e1_att = self.att1(d1, e1)
        d1 = torch.cat([d1, e1_att], dim=1)
        d1 = self.dec1(d1)
        
        # Final output
        out = self.final(d1)
        
        # Deep supervision outputs (only during training)
        if deep_supervision and self.training:
            ds4_out = F.interpolate(self.ds4(d4), size=x.shape[2:], mode='trilinear', align_corners=False)
            ds3_out = F.interpolate(self.ds3(d3), size=x.shape[2:], mode='trilinear', align_corners=False)
            ds2_out = F.interpolate(self.ds2(d2), size=x.shape[2:], mode='trilinear', align_corners=False)
            return out, ds4_out, ds3_out, ds2_out
        
        return out


def load_model(checkpoint_path: str, device: str = 'cuda') -> StableUNet3D:
    """
    Load a pretrained StableUNet3D model from checkpoint.
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        device: Device to load the model on ('cuda' or 'cpu')
        
    Returns:
        Loaded StableUNet3D model in eval mode
    """
    model = StableUNet3D(in_channels=4, out_channels=4)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test the model
    print("StableUNet3D Architecture Test")
    print("=" * 50)
    
    model = StableUNet3D(in_channels=4, out_channels=4)
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Test with dummy input
    x = torch.randn(1, 4, 144, 144, 144)
    print(f"Input shape: {x.shape}")
    
    # Inference mode
    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"Output shape: {out.shape}")
    
    # Training mode with deep supervision
    model.train()
    out, ds4, ds3, ds2 = model(x, deep_supervision=True)
    print(f"Deep supervision shapes: {out.shape}, {ds4.shape}, {ds3.shape}, {ds2.shape}")
    
    print("\n✓ Model test passed!")
