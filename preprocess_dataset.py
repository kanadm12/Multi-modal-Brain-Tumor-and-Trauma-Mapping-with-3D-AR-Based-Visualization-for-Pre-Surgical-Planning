"""
Preprocess BraTS dataset to NPY format for fast loading during training.

This script:
1. Loads raw NIfTI files
2. Applies nnU-Net normalization
3. Crops/pads to target size
4. Saves as compressed NPY files

Run once before training to dramatically speed up data loading.

Usage:
    python preprocess_dataset.py
"""

import os
import glob
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# Configuration
DATA_DIR = "/workspace/Multi-modal-Brain-Tumor-and-Trauma-Mapping-with-3D-AR-Based-Visualization-for-Pre-Surgical-Planning/dataset"
OUTPUT_DIR = "/workspace/Multi-modal-Brain-Tumor-and-Trauma-Mapping-with-3D-AR-Based-Visualization-for-Pre-Surgical-Planning/preprocessed_data"
CROP_SIZE = (160, 192, 160)
NUM_WORKERS = 16  # Parallel preprocessing


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


def preprocess_patient(patient_id, data_dir, output_dir, crop_size):
    """Preprocess a single patient's data"""
    try:
        patient_dir = os.path.join(data_dir, patient_id)
        output_patient_dir = os.path.join(output_dir, patient_id)
        os.makedirs(output_patient_dir, exist_ok=True)
        
        # Check if already preprocessed
        img_file = os.path.join(output_patient_dir, "image.npz")
        seg_file = os.path.join(output_patient_dir, "segmentation.npz")
        
        if os.path.exists(img_file) and os.path.exists(seg_file):
            return patient_id, "skipped"
        
        # Load imaging modalities
        modality_mappings = [
            ['t1', 't1n'],      # T1 native
            ['t1ce', 't1c'],    # T1 contrast-enhanced
            ['t2', 't2w'],      # T2 weighted
            ['flair', 't2f']    # T2 FLAIR
        ]
        
        img_data = []
        for mod_variants in modality_mappings:
            file_path = None
            for mod in mod_variants:
                file_path = glob.glob(os.path.join(patient_dir, f"*{mod}.nii.gz"))
                if not file_path:
                    file_path = glob.glob(os.path.join(patient_dir, f"*{mod}.nii"))
                if file_path and os.path.getsize(file_path[0]) > 1024:
                    break
            
            if file_path:
                img = nib.load(file_path[0]).get_fdata().astype(np.float32)
                img = nnunet_normalize(img)
                img_data.append(img)
            else:
                # Skip patients with missing modalities
                return patient_id, "missing_modality"
        
        if len(img_data) < 4:
            return patient_id, "incomplete"
        
        # Ensure all modalities have same shape
        shapes = [m.shape for m in img_data]
        if len(set(shapes)) > 1:
            reference_shape = img_data[0].shape
            for i in range(1, len(img_data)):
                if img_data[i].shape != reference_shape:
                    img_data[i] = center_crop_or_pad(img_data[i], reference_shape)
        
        # Stack modalities
        img = np.stack(img_data, axis=0)
        
        # Load segmentation
        seg_file_path = glob.glob(os.path.join(patient_dir, "*seg.nii.gz"))
        if not seg_file_path:
            seg_file_path = glob.glob(os.path.join(patient_dir, "*seg.nii"))
        
        if seg_file_path and os.path.getsize(seg_file_path[0]) > 1024:
            seg = nib.load(seg_file_path[0]).get_fdata().astype(np.uint8)
        else:
            return patient_id, "missing_segmentation"
        
        # Map labels: 1->1 (NCR), 2->2 (ED), 4->3 (ET)
        seg_new = np.zeros_like(seg, dtype=np.uint8)
        seg_new[seg == 1] = 1
        seg_new[seg == 2] = 2
        seg_new[seg == 4] = 3
        seg = seg_new
        
        # Crop/pad to target size
        img = np.stack([center_crop_or_pad(img[i], crop_size) for i in range(img.shape[0])])
        seg = center_crop_or_pad(seg, crop_size)
        
        # Save as compressed NPZ
        np.savez_compressed(img_file, data=img)
        np.savez_compressed(seg_file, data=seg)
        
        return patient_id, "success"
        
    except Exception as e:
        return patient_id, f"error: {str(e)}"


def main():
    """Main preprocessing function"""
    print("="*70)
    print("BraTS Dataset Preprocessing")
    print("="*70)
    print(f"Input:  {DATA_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Crop size: {CROP_SIZE}")
    print(f"Workers: {NUM_WORKERS}")
    print("="*70)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all patient IDs
    patient_ids = [
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith('BraTS')
    ]
    patient_ids.sort()
    
    print(f"\nFound {len(patient_ids)} patients")
    print("\nStarting preprocessing...")
    
    # Preprocess in parallel
    preprocess_fn = partial(
        preprocess_patient,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        crop_size=CROP_SIZE
    )
    
    results = {"success": 0, "skipped": 0, "error": 0, "missing": 0}
    
    with mp.Pool(NUM_WORKERS) as pool:
        for patient_id, status in tqdm(
            pool.imap(preprocess_fn, patient_ids),
            total=len(patient_ids),
            desc="Preprocessing",
            unit="patient"
        ):
            if status == "success":
                results["success"] += 1
            elif status == "skipped":
                results["skipped"] += 1
            elif "missing" in status or "incomplete" in status:
                results["missing"] += 1
            else:
                results["error"] += 1
    
    # Summary
    print("\n" + "="*70)
    print("Preprocessing Complete")
    print("="*70)
    print(f"Successfully preprocessed: {results['success']}")
    print(f"Already preprocessed:      {results['skipped']}")
    print(f"Missing/incomplete data:   {results['missing']}")
    print(f"Errors:                    {results['error']}")
    print(f"Total valid:               {results['success'] + results['skipped']}")
    print("="*70)
    
    # Calculate disk usage
    total_size = 0
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for f in files:
            total_size += os.path.getsize(os.path.join(root, f))
    
    print(f"Disk usage: {total_size / (1024**3):.2f} GB")
    print("\nYou can now run training with preprocessed data!")
    print("="*70)


if __name__ == "__main__":
    main()
