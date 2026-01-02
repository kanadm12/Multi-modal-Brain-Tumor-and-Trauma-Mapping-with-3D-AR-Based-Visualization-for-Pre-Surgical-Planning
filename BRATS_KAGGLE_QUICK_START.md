# BraTS Kaggle to Azure - Quick Start

## What This Does

Downloads 3 BraTS datasets from Kaggle and uploads all patient folders directly to your Azure Blob Storage at:
```
beproject/dataset/dataset/[patient_folder_name]/
```

## Datasets Included

1. **BRATS2023 Part 1** - All folders
   - https://www.kaggle.com/datasets/aiocta/brats2023-part-1
   - Downloads: All folders

2. **BRATS2019** - HGG Only
   - https://www.kaggle.com/datasets/aryashah2k/brain-tumor-segmentation-brats-2019
   - Downloads: Only folders inside HGG/

3. **BRATS2020** - MICCAI Training Data Only
   - https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation
   - Downloads: Only folders inside MICCAI_BraTS2020_TrainingData/

## Prerequisites

### 1. Kaggle API Setup
```powershell
# Download kaggle.json from https://www.kaggle.com/settings/account
# Place it at: C:\Users\YourUsername\.kaggle\kaggle.json
Test-Path $env:USERPROFILE\.kaggle\kaggle.json
```

### 2. Install Dependencies
```powershell
pip install kaggle azure-storage-blob
```

### 3. Azure Setup (Already Done)
- Connection String: `DefaultEndpointsProtocol=https;AccountName=spartis9488473038;...`
- Container: `beproject`
- Destination: `dataset/dataset/`

## Usage

### Run the script:
```powershell
python brats_kaggle_to_azure.py
```

That's it! The script will:

1. ✓ Download BRATS2023 Part 1 → Extract → Upload all patient folders
2. ✓ Download BRATS2019 → Extract HGG → Upload all patient folders from HGG
3. ✓ Download BRATS2020 → Extract MICCAI_BraTS2020_TrainingData → Upload all patient folders

## What Happens

```
STEP 1: BRATS2023 Part 1
├─ Download from Kaggle
├─ Extract zip files
├─ Find all patient folders (BraTS_###, etc.)
└─ Upload to: beproject/dataset/dataset/[patient_name]/

STEP 2: BRATS2019 (HGG)
├─ Download from Kaggle
├─ Extract zip files
├─ Find HGG subfolder
├─ Find all patient folders inside HGG/
└─ Upload to: beproject/dataset/dataset/[patient_name]/

STEP 3: BRATS2020 (MICCAI Training Data)
├─ Download from Kaggle
├─ Extract zip files
├─ Find MICCAI_BraTS2020_TrainingData subfolder
├─ Find all patient folders inside it
└─ Upload to: beproject/dataset/dataset/[patient_name]/
```

## Output Structure in Azure

After completion, your `beproject` container will have:

```
beproject/
└── dataset/
    └── dataset/
        ├── BraTS_001/
        │   ├── BraTS_001_t1.nii.gz
        │   ├── BraTS_001_t1ce.nii.gz
        │   ├── BraTS_001_t2.nii.gz
        │   ├── BraTS_001_flair.nii.gz
        │   └── BraTS_001_seg.nii.gz
        ├── BraTS_002/
        │   ├── ...
        ├── BraTS19_TCIA_001_1_0/
        │   ├── ...
        ├── BraTS20_Training_001/
        │   ├── ...
        └── [All other patient folders...]
```

## Monitoring Progress

The script outputs:
- Dataset download progress
- Number of patient folders found
- File upload count and total size in GB
- Success/failure status for each dataset

```
2025-11-25 10:30:15 - INFO - Processing: BRATS2023 Part 1
2025-11-25 10:30:20 - INFO - Downloading Kaggle dataset: aiocta/brats2023-part-1
2025-11-25 10:35:45 - INFO - ✓ Successfully downloaded aiocta/brats2023-part-1
2025-11-25 10:35:50 - INFO - Found 125 patient folders
2025-11-25 10:35:55 - INFO - Uploading patient folder: BraTS_001
2025-11-25 11:45:20 - INFO - ✓ Upload complete: 2340 files, 45.32 GB
...
```

## Estimated Time & Size

- **BRATS2023 Part 1**: ~50-100 patients, ~100-150 GB
- **BRATS2019 (HGG)**: ~210 patients, ~150-200 GB
- **BRATS2020 (Training)**: ~369 patients, ~250-300 GB

**Total**: ~500+ GB, 1-2 hours (depends on internet speed)

## Temporary Storage

The script uses temporary folders:
- `./brats_downloads/` - Downloaded zip files
- `./brats_temp/` - Extracted files

These are **automatically deleted** after upload to save disk space.

## Troubleshooting

### "Kaggle credentials not found"
```powershell
# Solution: Download from https://www.kaggle.com/settings/account
# Save to: C:\Users\YourUsername\.kaggle\kaggle.json
```

### "Failed to connect to Azure"
```powershell
# Check connection string is correct
# Verify container "beproject" exists
# Ensure firewall allows Azure connections
```

### Script interrupted?
Just re-run it! Azure will skip already uploaded files and continue from where it stopped.

### Check Azure upload
```powershell
# Use Azure Storage Explorer to view uploaded files
# Or check in Azure Portal → Storage Account → Containers → beproject
```

## Notes

- Files are uploaded to: `beproject/dataset/dataset/`
- All imaging modalities (t1, t1ce, t2, flair, seg) are included
- Patient folder names are preserved from original datasets
- Duplicate patient names across datasets will overwrite (rename if needed)
- Internet connection must remain stable during upload

## Success Indicators

When complete, you'll see:
```
======================================================================
FINAL SUMMARY
======================================================================
✓ SUCCESS: BRATS2023
✓ SUCCESS: BRATS2019
✓ SUCCESS: BRATS2020

✓ All datasets processed
Data uploaded to Azure at: beproject/dataset/dataset/
```
