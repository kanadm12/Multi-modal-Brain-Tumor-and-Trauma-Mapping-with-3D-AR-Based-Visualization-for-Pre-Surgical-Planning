# =============================================================================
# BRATS INFERENCE MODULE - StableUNet3D
# 
# Inference pipeline using the StableUNet3D model architecture
# Compatible with: unet_modified_83_38.pth (83.38% Dice)
# 
# Features:
# - Test-Time Augmentation (TTA) for improved accuracy
# - Mixed precision inference for speed
# - 3D mesh generation for visualization
# - Clinical metrics calculation
# =============================================================================

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Any
import SimpleITK as sitk
from scipy.ndimage import (
    label as ndimage_label, binary_closing, binary_opening,
    binary_fill_holes, binary_erosion, binary_dilation,
    generate_binary_structure
)

# Import the StableUNet3D model
from stable_unet3d import StableUNet3D, load_model


# =============================================================================
# CONFIGURATION
# =============================================================================

# Input size that the model was trained on
CROP_SIZE = (144, 144, 144)  # StableUNet3D training size
NUM_CLASSES = 4  # Background + NCR + ED + ET
IN_CHANNELS = 4  # T1, T1ce, T2, FLAIR

# TTA transforms
TTA_TRANSFORMS = 8  # Use 8-point TTA for good balance of speed/accuracy

# Class names for BraTS
CLASS_NAMES = {
    0: 'Background',
    1: 'NCR/NET',  # Necrotic and Non-Enhancing Tumor Core
    2: 'ED',       # Peritumoral Edema
    3: 'ET'        # Enhancing Tumor
}


# =============================================================================
# PREPROCESSING UTILITIES
# =============================================================================

def nnunet_normalize(img: np.ndarray) -> np.ndarray:
    """nnU-Net style normalization for MRI data"""
    nonzero_mask = img > 0
    if not np.any(nonzero_mask):
        return img
    
    # Percentile clipping to remove outliers
    p001, p999 = np.percentile(img[nonzero_mask], [0.1, 99.9])
    img = np.clip(img, p001, p999)
    
    # Z-score normalization on non-zero voxels
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


def restore_original_size(prediction: np.ndarray, original_shape: tuple) -> np.ndarray:
    """Restore prediction to original image size"""
    return center_crop_or_pad(prediction, original_shape)


# =============================================================================
# TEST-TIME AUGMENTATION
# =============================================================================

def apply_tta_transform(image: torch.Tensor, transform_idx: int) -> torch.Tensor:
    """
    Apply TTA transform to input image.
    
    Transforms:
        0: Original
        1-3: Axis flips (X, Y, Z)
        4-6: 2D rotations (90°, 180°, 270°)
        7: Combined flip
    """
    if transform_idx == 0:
        return image  # Original
    elif transform_idx == 1:
        return torch.flip(image, dims=[4])  # Flip X
    elif transform_idx == 2:
        return torch.flip(image, dims=[3])  # Flip Y
    elif transform_idx == 3:
        return torch.flip(image, dims=[2])  # Flip Z
    elif transform_idx == 4:
        return torch.rot90(image, 1, dims=[3, 4])  # Rotate 90° in XY
    elif transform_idx == 5:
        return torch.rot90(image, 2, dims=[3, 4])  # Rotate 180° in XY
    elif transform_idx == 6:
        return torch.rot90(image, 3, dims=[3, 4])  # Rotate 270° in XY
    else:  # 7: Combined flip
        return torch.flip(torch.flip(image, dims=[4]), dims=[3])  # Flip X + Y


def reverse_tta_transform(pred: torch.Tensor, transform_idx: int) -> torch.Tensor:
    """Reverse TTA transform on prediction"""
    if transform_idx == 0:
        return pred
    elif transform_idx == 1:
        return torch.flip(pred, dims=[3])  # Reverse flip X
    elif transform_idx == 2:
        return torch.flip(pred, dims=[2])  # Reverse flip Y
    elif transform_idx == 3:
        return torch.flip(pred, dims=[1])  # Reverse flip Z
    elif transform_idx == 4:
        return torch.rot90(pred, 3, dims=[2, 3])  # Reverse 90° XY
    elif transform_idx == 5:
        return torch.rot90(pred, 2, dims=[2, 3])  # Reverse 180° XY
    elif transform_idx == 6:
        return torch.rot90(pred, 1, dims=[2, 3])  # Reverse 270° XY
    else:  # 7: Reverse combined flip
        return torch.flip(torch.flip(pred, dims=[2]), dims=[3])  # Reverse X + Y


# =============================================================================
# POST-PROCESSING
# =============================================================================

def postprocess_segmentation(prediction: np.ndarray, min_size: int = 100) -> np.ndarray:
    """
    Post-process segmentation for cleaner boundaries and reduced noise.
    
    Steps:
    1. Fill holes in each class
    2. Morphological smoothing (closing + opening)
    3. Remove small connected components
    4. Ensure anatomical consistency
    """
    processed = np.zeros_like(prediction)
    
    struct_small = generate_binary_structure(3, 1)  # 6-connectivity
    struct_large = generate_binary_structure(3, 2)  # 18-connectivity
    
    for class_id in range(1, 4):  # NCR=1, ED=2, ET=3
        mask = (prediction == class_id).astype(bool)
        
        if not np.any(mask):
            continue
        
        # Fill holes
        try:
            mask = binary_fill_holes(mask)
        except:
            pass
        
        # Morphological smoothing
        try:
            if class_id == 3:  # ET - more aggressive smoothing
                mask = binary_closing(mask, structure=struct_large, iterations=2)
                mask = binary_opening(mask, structure=struct_small, iterations=1)
            else:
                mask = binary_closing(mask, structure=struct_small, iterations=1)
                mask = binary_opening(mask, structure=struct_small, iterations=1)
        except:
            pass
        
        # Connected components - remove small regions
        labeled, num_features = ndimage_label(mask)
        
        if num_features > 0:
            component_sizes = np.bincount(labeled.ravel())
            
            for feature_id in range(1, num_features + 1):
                if component_sizes[feature_id] >= min_size:
                    processed[labeled == feature_id] = class_id
    
    return processed


# =============================================================================
# METRICS CALCULATION
# =============================================================================

def calculate_dice(pred: np.ndarray, target: np.ndarray, class_id: int) -> float:
    """Calculate Dice coefficient for a specific class"""
    pred_c = (pred == class_id).astype(float)
    target_c = (target == class_id).astype(float)
    
    intersection = np.sum(pred_c * target_c)
    denominator = np.sum(pred_c) + np.sum(target_c)
    
    if denominator == 0:
        return 1.0 if np.sum(pred_c) == 0 else 0.0
    
    return (2.0 * intersection + 1e-6) / (denominator + 1e-6)


def calculate_tumor_volumes(segmentation: np.ndarray, spacing: tuple = (1.0, 1.0, 1.0)) -> Dict[str, float]:
    """
    Calculate tumor volumes in cm³.
    
    Returns:
        Dictionary with volumes for each region:
        - NCR: Necrotic core
        - ED: Edema
        - ET: Enhancing tumor
        - TC: Tumor core (NCR + ET)
        - WT: Whole tumor (NCR + ED + ET)
    """
    voxel_volume = np.prod(spacing) / 1000  # mm³ to cm³
    
    ncr_voxels = np.sum(segmentation == 1)
    ed_voxels = np.sum(segmentation == 2)
    et_voxels = np.sum(segmentation == 3)
    
    return {
        'NCR': float(ncr_voxels * voxel_volume),
        'ED': float(ed_voxels * voxel_volume),
        'ET': float(et_voxels * voxel_volume),
        'TC': float((ncr_voxels + et_voxels) * voxel_volume),  # Tumor Core
        'WT': float((ncr_voxels + ed_voxels + et_voxels) * voxel_volume)  # Whole Tumor
    }


def get_tumor_statistics(segmentation: np.ndarray, spacing: tuple = (1.0, 1.0, 1.0)) -> Dict[str, Any]:
    """
    Get comprehensive tumor statistics for reporting.
    """
    volumes = calculate_tumor_volumes(segmentation, spacing)
    
    # Find tumor bounding box
    tumor_mask = segmentation > 0
    if np.any(tumor_mask):
        coords = np.argwhere(tumor_mask)
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)
        dimensions = (max_coords - min_coords + 1) * np.array(spacing)
    else:
        dimensions = np.array([0, 0, 0])
    
    # Calculate enhancing ratio
    total_tumor = volumes['WT']
    enhancing_ratio = volumes['ET'] / total_tumor if total_tumor > 0 else 0
    
    return {
        'volumes': volumes,
        'dimensions_mm': {
            'depth': float(dimensions[0]),
            'height': float(dimensions[1]),
            'width': float(dimensions[2])
        },
        'enhancing_ratio': float(enhancing_ratio),
        'has_necrosis': volumes['NCR'] > 0,
        'has_edema': volumes['ED'] > 0,
        'has_enhancement': volumes['ET'] > 0
    }


# =============================================================================
# INFERENCE CLASS
# =============================================================================

class StableUNet3DInference:
    """
    Inference pipeline for StableUNet3D model.
    
    Compatible with: unet_modified_83_38.pth
    
    Usage:
        engine = StableUNet3DInference('unet_modified_83_38.pth')
        prediction, volumes, stats = engine.predict('/path/to/patient')
    """
    
    def __init__(self, checkpoint_path: str, device: str = None):
        """
        Initialize inference engine.
        
        Args:
            checkpoint_path: Path to trained model checkpoint (.pth file)
            device: Device to use ('cuda' or 'cpu'). Auto-detect if None.
        """
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.checkpoint_path = checkpoint_path
        self.model = self._load_model(checkpoint_path)
        self.use_amp = torch.cuda.is_available()
        
        print(f"✅ StableUNet3D inference engine initialized on {self.device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def _load_model(self, checkpoint_path: str) -> nn.Module:
        """Load StableUNet3D model from checkpoint"""
        print(f"Loading model from: {checkpoint_path}")
        
        model = StableUNet3D(in_channels=IN_CHANNELS, out_channels=NUM_CLASSES)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Handle DDP wrapped models (remove 'module.' prefix)
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"✅ Model loaded ({total_params:.2f}M parameters)")
        
        return model
    
    def load_patient_data(self, patient_dir: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load MRI modalities from patient directory.
        
        Expects NIfTI files with naming convention:
        - *t1.nii.gz or *t1n.nii.gz (T1 native)
        - *t1ce.nii.gz or *t1c.nii.gz (T1 contrast-enhanced)
        - *t2.nii.gz or *t2w.nii.gz (T2 weighted)
        - *flair.nii.gz or *t2f.nii.gz (T2 FLAIR)
        
        Args:
            patient_dir: Path to directory containing NIfTI files
            
        Returns:
            Tuple of (stacked_modalities [4, D, H, W], metadata_dict)
        """
        patient_dir = Path(patient_dir)
        
        modality_mappings = [
            ['t1.nii', 't1n.nii', '_t1_', '_t1n_'],      # T1
            ['t1ce.nii', 't1c.nii', '_t1ce_', '_t1c_'],  # T1ce
            ['t2.nii', 't2w.nii', '_t2_', '_t2w_'],      # T2
            ['flair.nii', 't2f.nii', '_flair_', '_t2f_'] # FLAIR
        ]
        
        img_data = []
        metadata = {
            'files': [],
            'original_shape': None,
            'spacing': (1.0, 1.0, 1.0),
            'patient_id': patient_dir.name
        }
        
        for mod_patterns in modality_mappings:
            file_path = None
            
            # Try each pattern
            for pattern in mod_patterns:
                files = list(patient_dir.glob(f"*{pattern}*"))
                if files:
                    file_path = files[0]
                    break
            
            if file_path and file_path.stat().st_size > 1024:
                try:
                    img_sitk = sitk.ReadImage(str(file_path))
                    img = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
                    
                    if metadata['original_shape'] is None:
                        metadata['original_shape'] = img.shape
                        metadata['spacing'] = img_sitk.GetSpacing()
                    
                    img = nnunet_normalize(img)
                    img_data.append(img)
                    metadata['files'].append(str(file_path))
                except Exception as e:
                    print(f"Warning: Failed to load {file_path}: {e}")
                    if img_data:
                        img_data.append(np.zeros_like(img_data[0]))
                    else:
                        img_data.append(np.zeros(CROP_SIZE))
            else:
                # Missing modality - use zeros
                if img_data:
                    img_data.append(np.zeros_like(img_data[0]))
                else:
                    img_data.append(np.zeros(CROP_SIZE))
        
        # Stack modalities
        img = np.stack(img_data, axis=0)
        
        # Crop/pad to model input size
        img = np.stack([center_crop_or_pad(img[i], CROP_SIZE) for i in range(img.shape[0])])
        
        return img, metadata
    
    @torch.no_grad()
    def predict(
        self, 
        patient_dir: Path, 
        use_tta: bool = True,
        use_postprocessing: bool = True,
        return_probabilities: bool = False
    ) -> Tuple[np.ndarray, Dict[str, float], Dict[str, Any]]:
        """
        Run segmentation prediction on patient data.
        
        Args:
            patient_dir: Directory containing NIfTI files
            use_tta: Whether to use Test-Time Augmentation (slower but more accurate)
            use_postprocessing: Whether to apply morphological post-processing
            return_probabilities: If True, also return class probabilities
            
        Returns:
            Tuple of:
                - segmentation: np.ndarray of shape [D, H, W] with class labels 0-3
                - volumes: Dict with tumor volumes in cm³
                - stats: Dict with comprehensive tumor statistics
        """
        patient_dir = Path(patient_dir)
        
        # Load data
        img_data, metadata = self.load_patient_data(patient_dir)
        
        # Convert to tensor
        img_tensor = torch.tensor(img_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        if use_tta:
            # Test-Time Augmentation with voting
            pred_list = []
            
            for transform_idx in range(TTA_TRANSFORMS):
                img_tta = apply_tta_transform(img_tensor, transform_idx)
                
                with autocast(enabled=self.use_amp):
                    outputs = self.model(img_tta)
                    probs = F.softmax(outputs, dim=1)
                    pred_tta = torch.argmax(probs, dim=1)
                
                pred_tta = reverse_tta_transform(pred_tta, transform_idx)
                pred_list.append(pred_tta.float())
            
            # Ensemble by majority voting
            pred_stack = torch.stack(pred_list, dim=0)
            prediction = torch.mode(pred_stack, dim=0).values.squeeze()
            
            # Get probabilities from non-augmented prediction for confidence
            with autocast(enabled=self.use_amp):
                outputs = self.model(img_tensor)
                probabilities = F.softmax(outputs, dim=1).squeeze(0)
        else:
            with autocast(enabled=self.use_amp):
                outputs = self.model(img_tensor)
                probabilities = F.softmax(outputs, dim=1).squeeze(0)
                prediction = torch.argmax(probabilities, dim=0)
        
        prediction = prediction.cpu().numpy().astype(np.uint8)
        probabilities = probabilities.cpu().numpy()
        
        # Post-processing
        if use_postprocessing:
            prediction = postprocess_segmentation(prediction)
        
        # Calculate volumes and statistics
        spacing = metadata.get('spacing', (1.0, 1.0, 1.0))
        volumes = calculate_tumor_volumes(prediction, spacing)
        stats = get_tumor_statistics(prediction, spacing)
        stats['metadata'] = metadata
        
        if return_probabilities:
            return prediction, volumes, stats, probabilities
        
        return prediction, volumes, stats
    
    def predict_from_arrays(
        self,
        modalities: np.ndarray,
        spacing: tuple = (1.0, 1.0, 1.0),
        use_tta: bool = True,
        use_postprocessing: bool = True
    ) -> Tuple[np.ndarray, Dict[str, float], Dict[str, Any]]:
        """
        Run prediction directly from numpy arrays.
        
        Args:
            modalities: np.ndarray of shape [4, D, H, W] (already normalized)
            spacing: Voxel spacing in mm
            use_tta: Whether to use TTA
            use_postprocessing: Whether to post-process
            
        Returns:
            Same as predict()
        """
        # Ensure correct shape
        if modalities.shape[0] != 4:
            raise ValueError(f"Expected 4 modalities, got {modalities.shape[0]}")
        
        # Crop/pad to model input size
        modalities = np.stack([center_crop_or_pad(modalities[i], CROP_SIZE) for i in range(4)])
        
        # Convert to tensor
        img_tensor = torch.tensor(modalities, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if use_tta:
                pred_list = []
                for transform_idx in range(TTA_TRANSFORMS):
                    img_tta = apply_tta_transform(img_tensor, transform_idx)
                    with autocast(enabled=self.use_amp):
                        outputs = self.model(img_tta)
                        pred_tta = torch.argmax(outputs, dim=1)
                    pred_tta = reverse_tta_transform(pred_tta, transform_idx)
                    pred_list.append(pred_tta.float())
                
                pred_stack = torch.stack(pred_list, dim=0)
                prediction = torch.mode(pred_stack, dim=0).values.squeeze()
            else:
                with autocast(enabled=self.use_amp):
                    outputs = self.model(img_tensor)
                    prediction = torch.argmax(outputs, dim=1).squeeze()
        
        prediction = prediction.cpu().numpy().astype(np.uint8)
        
        if use_postprocessing:
            prediction = postprocess_segmentation(prediction)
        
        volumes = calculate_tumor_volumes(prediction, spacing)
        stats = get_tumor_statistics(prediction, spacing)
        
        return prediction, volumes, stats
    
    def save_prediction(
        self,
        prediction: np.ndarray,
        output_path: str,
        reference_image_path: str = None
    ) -> str:
        """
        Save prediction as NIfTI file.
        
        Args:
            prediction: Segmentation array
            output_path: Output file path
            reference_image_path: Reference image for copying metadata (optional)
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create SimpleITK image
        seg_img = sitk.GetImageFromArray(prediction.astype(np.int16))
        
        # Copy metadata from reference if provided
        if reference_image_path and Path(reference_image_path).exists():
            ref_img = sitk.ReadImage(str(reference_image_path))
            seg_img.SetSpacing(ref_img.GetSpacing())
            seg_img.SetOrigin(ref_img.GetOrigin())
            seg_img.SetDirection(ref_img.GetDirection())
        
        sitk.WriteImage(seg_img, str(output_path))
        print(f"✅ Segmentation saved to: {output_path}")
        
        return str(output_path)


# =============================================================================
# CONVENIENCE FUNCTION FOR API
# =============================================================================

def create_inference_engine(
    checkpoint_path: str = None,
    device: str = None
) -> StableUNet3DInference:
    """
    Create inference engine with default or specified checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint. If None, uses default.
        device: Device to use. If None, auto-detect.
        
    Returns:
        Initialized StableUNet3DInference engine
    """
    if checkpoint_path is None:
        # Default checkpoint location
        default_paths = [
            Path(__file__).parent / "unet_modified_83_38.pth",
            Path("/workspace/checkpoints/unet_modified_83_38.pth"),
            Path("./unet_modified_83_38.pth")
        ]
        
        for path in default_paths:
            if path.exists():
                checkpoint_path = str(path)
                break
        
        if checkpoint_path is None:
            raise FileNotFoundError(
                "No checkpoint found. Please provide checkpoint_path or ensure "
                "unet_modified_83_38.pth exists in the project directory."
            )
    
    return StableUNet3DInference(checkpoint_path, device)


# =============================================================================
# CLI USAGE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BraTS Inference with StableUNet3D")
    parser.add_argument("--checkpoint", type=str, default="unet_modified_83_38.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to patient directory containing NIfTI files")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for segmentation (default: input_dir/prediction.nii.gz)")
    parser.add_argument("--no-tta", action="store_true",
                        help="Disable Test-Time Augmentation (faster but less accurate)")
    parser.add_argument("--no-postprocess", action="store_true",
                        help="Disable post-processing")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = StableUNet3DInference(args.checkpoint, args.device)
    
    # Run inference
    print(f"\n📂 Processing: {args.input}")
    prediction, volumes, stats = engine.predict(
        args.input,
        use_tta=not args.no_tta,
        use_postprocessing=not args.no_postprocess
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("TUMOR VOLUMES (cm³)")
    print("=" * 50)
    print(f"  NCR (Necrotic Core):    {volumes['NCR']:.2f} cm³")
    print(f"  ED (Edema):             {volumes['ED']:.2f} cm³")
    print(f"  ET (Enhancing Tumor):   {volumes['ET']:.2f} cm³")
    print(f"  TC (Tumor Core):        {volumes['TC']:.2f} cm³")
    print(f"  WT (Whole Tumor):       {volumes['WT']:.2f} cm³")
    print("=" * 50)
    print(f"  Enhancing Ratio:        {stats['enhancing_ratio']:.1%}")
    print("=" * 50)
    
    # Save prediction
    if args.output:
        output_path = args.output
    else:
        output_path = Path(args.input) / "prediction.nii.gz"
    
    # Get reference image for metadata
    ref_files = list(Path(args.input).glob("*.nii.gz"))
    ref_path = str(ref_files[0]) if ref_files else None
    
    engine.save_prediction(prediction, str(output_path), ref_path)
    
    print("\n✅ Inference complete!")
