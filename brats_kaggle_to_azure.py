"""
Download specific BraTS datasets from Kaggle and upload to Azure Blob Storage.

This script:
1. Downloads BraTS datasets from Kaggle with specific folder structures
2. Extracts only patient folders (HGG, MICCAI_BraTS2020_TrainingData)
3. Uploads to Azure Blob Storage at: beproject/dataset/dataset/
4. Cleans up local files

Datasets:
- BRATS2023 Part 1: all folders
- BRATS2019: HGG subfolder only
- BRATS2020: MICCAI_BraTS2020_TrainingData subfolder only
"""

import os
import sys
import zipfile
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BraTS_AzureUploader:
    """Handle BraTS dataset downloads and Azure uploads."""
    
    def __init__(self, azure_connection_string: str, container_name: str):
        """Initialize with Azure credentials."""
        self.azure_conn_str = azure_connection_string
        self.container_name = container_name
        self.local_download_dir = Path("./brats_downloads")
        self.local_temp_dir = Path("./brats_temp")
        
        self._verify_dependencies()
        self._setup_azure_client()
    
    def _verify_dependencies(self):
        """Verify required packages are installed."""
        required_packages = {
            'kaggle': 'kaggle',
            'azure.storage.blob': 'azure-storage-blob'
        }
        
        missing_packages = []
        for import_name, pip_name in required_packages.items():
            try:
                __import__(import_name)
                logger.info(f"✓ {pip_name} is installed")
            except ImportError:
                missing_packages.append(pip_name)
        
        if missing_packages:
            logger.info(f"Installing missing packages: {missing_packages}")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                *missing_packages, "-q"
            ])
            logger.info("✓ Packages installed successfully")
    
    def _setup_azure_client(self):
        """Initialize Azure Blob Storage client."""
        try:
            from azure.storage.blob import BlobServiceClient
            self.blob_service_client = BlobServiceClient.from_connection_string(
                self.azure_conn_str
            )
            self.container_client = self.blob_service_client.get_container_client(
                self.container_name
            )
            logger.info(f"✓ Connected to Azure container: {self.container_name}")
        except Exception as e:
            logger.error(f"Failed to connect to Azure: {e}")
            raise
    
    def _verify_kaggle_credentials(self):
        """Verify Kaggle credentials are configured."""
        kaggle_config_dir = Path.home() / ".kaggle"
        kaggle_json = kaggle_config_dir / "kaggle.json"
        
        if not kaggle_json.exists():
            logger.error(
                "Kaggle credentials not found!\n"
                "Please download kaggle.json from https://www.kaggle.com/settings/account\n"
                f"and place it at: {kaggle_json}"
            )
            raise FileNotFoundError(f"Kaggle credentials missing at {kaggle_json}")
        
        os.chmod(kaggle_json, 0o600)
        logger.info("✓ Kaggle credentials verified")
    
    def download_dataset(self, dataset_name: str) -> Path:
        """Download dataset from Kaggle."""
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError:
            logger.error("Kaggle package not installed")
            raise
        
        self._verify_kaggle_credentials()
        self.local_download_dir.mkdir(parents=True, exist_ok=True)
        
        download_path = self.local_download_dir / dataset_name.replace('/', '_')
        download_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading Kaggle dataset: {dataset_name}")
        
        try:
            api = KaggleApi()
            api.authenticate()
            
            api.dataset_download_files(
                dataset_name,
                path=download_path,
                unzip=False  # Keep as zip for now
            )
            
            logger.info(f"✓ Successfully downloaded {dataset_name}")
            return download_path
            
        except Exception as e:
            logger.error(f"Failed to download dataset {dataset_name}: {e}")
            raise
    
    def extract_and_get_patient_folders(
        self,
        download_path: Path,
        subfolder_filter: Optional[str] = None
    ) -> Path:
        """
        Extract dataset and get patient folders.
        
        Args:
            download_path: Path where dataset was downloaded
            subfolder_filter: Specific subfolder to extract (e.g., 'HGG', 'MICCAI_BraTS2020_TrainingData')
        
        Returns:
            Path to extracted patient folders
        """
        logger.info(f"Extracting dataset from {download_path}")
        
        self.local_temp_dir.mkdir(parents=True, exist_ok=True)
        extract_path = self.local_temp_dir / download_path.name
        extract_path.mkdir(parents=True, exist_ok=True)
        
        # Find and extract zip files
        zip_files = list(download_path.glob('**/*.zip'))
        logger.info(f"Found {len(zip_files)} zip files")
        
        for zip_file in zip_files:
            logger.info(f"Extracting: {zip_file.name}")
            try:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                logger.info(f"✓ Extracted {zip_file.name}")
            except Exception as e:
                logger.warning(f"Failed to extract {zip_file}: {e}")
        
        # Find patient folders
        if subfolder_filter:
            # Look for specific subfolder
            target_folder = extract_path / subfolder_filter
            if not target_folder.exists():
                # Search recursively
                for folder in extract_path.rglob(subfolder_filter):
                    if folder.is_dir():
                        target_folder = folder
                        break
            
            if target_folder.exists():
                logger.info(f"✓ Found target subfolder: {subfolder_filter}")
                return target_folder
            else:
                logger.warning(f"Could not find subfolder: {subfolder_filter}")
                return extract_path
        
        return extract_path
    
    def get_patient_folders(self, base_path: Path) -> List[Path]:
        """Get all patient folders (those containing imaging modalities)."""
        patient_folders = []
        
        # Patient folders typically contain .nii.gz files or have specific naming
        for item in base_path.iterdir():
            if item.is_dir():
                # Check if it's a patient folder
                # Patient folders usually have names like: BraTS_001, BraTS19_TCIA_001_1_0, etc.
                nii_files = list(item.glob('**/*.nii.gz'))
                if nii_files or item.name.startswith('BraTS'):
                    patient_folders.append(item)
                else:
                    # Recursively check subfolders
                    sub_patients = self.get_patient_folders(item)
                    patient_folders.extend(sub_patients)
        
        return sorted(patient_folders)
    
    def upload_to_azure(self, local_path: Path, blob_prefix: str = ""):
        """Recursively upload directory to Azure Blob Storage."""
        if not local_path.exists():
            logger.error(f"Local path does not exist: {local_path}")
            raise FileNotFoundError(f"Path not found: {local_path}")
        
        logger.info(f"Uploading to Azure at: {blob_prefix}")
        
        file_count = 0
        total_size = 0
        
        for file_path in local_path.rglob('*'):
            if file_path.is_file():
                try:
                    relative_path = file_path.relative_to(local_path)
                    blob_name = f"{blob_prefix}{relative_path}".replace("\\", "/")
                    
                    file_size = file_path.stat().st_size
                    
                    with open(file_path, 'rb') as data:
                        self.container_client.upload_blob(
                            blob_name,
                            data,
                            overwrite=True
                        )
                    
                    total_size += file_size
                    file_count += 1
                    
                    # Log larger files and checkpoints
                    if file_count % 50 == 0:
                        logger.info(f"Progress: {file_count} files uploaded, {total_size / (1024**3):.2f} GB")
                    
                except Exception as e:
                    logger.error(f"Failed to upload {file_path}: {e}")
                    raise
        
        logger.info(
            f"✓ Upload complete: {file_count} files, {total_size / (1024**3):.2f} GB"
        )
        return file_count, total_size
    
    def cleanup_local(self, path: Path):
        """Delete local files."""
        try:
            if path.exists():
                shutil.rmtree(path)
                logger.info(f"✓ Cleaned up: {path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup {path}: {e}")
    
    def process_brats2023_part1(self):
        """
        Process BRATS2023 Part 1 - download all folders.
        """
        dataset_name = "aiocta/brats2023-part-1"
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing: BRATS2023 Part 1")
        logger.info(f"{'='*70}")
        
        try:
            # Download
            download_path = self.download_dataset(dataset_name)
            
            # Extract
            extracted_path = self.extract_and_get_patient_folders(download_path)
            
            # Upload all patient folders
            patient_folders = self.get_patient_folders(extracted_path)
            logger.info(f"Found {len(patient_folders)} patient folders")
            
            for patient_folder in patient_folders:
                blob_prefix = f"dataset/dataset/{patient_folder.name}/"
                logger.info(f"Uploading patient folder: {patient_folder.name}")
                self.upload_to_azure(patient_folder, blob_prefix)
            
            # Cleanup
            self.cleanup_local(download_path)
            self.cleanup_local(extracted_path)
            
            logger.info(f"✓ BRATS2023 Part 1 completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process BRATS2023 Part 1: {e}")
            return False
    
    def process_brats2019_hgg(self):
        """
        Process BRATS2019 - HGG folder only.
        """
        dataset_name = "aryashah2k/brain-tumor-segmentation-brats-2019"
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing: BRATS2019 (HGG only)")
        logger.info(f"{'='*70}")
        
        try:
            # Download
            download_path = self.download_dataset(dataset_name)
            
            # Extract and get HGG subfolder
            extracted_path = self.extract_and_get_patient_folders(download_path, "HGG")
            
            # Get patient folders from HGG
            patient_folders = self.get_patient_folders(extracted_path)
            logger.info(f"Found {len(patient_folders)} patient folders in HGG")
            
            for patient_folder in patient_folders:
                blob_prefix = f"dataset/dataset/{patient_folder.name}/"
                logger.info(f"Uploading patient folder: {patient_folder.name}")
                self.upload_to_azure(patient_folder, blob_prefix)
            
            # Cleanup
            self.cleanup_local(download_path)
            self.cleanup_local(extracted_path)
            
            logger.info(f"✓ BRATS2019 (HGG) completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process BRATS2019 HGG: {e}")
            return False
    
    def process_brats2020_training(self):
        """
        Process BRATS2020 - MICCAI_BraTS2020_TrainingData folder only.
        """
        dataset_name = "awsaf49/brats20-dataset-training-validation"
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing: BRATS2020 (MICCAI_BraTS2020_TrainingData only)")
        logger.info(f"{'='*70}")
        
        try:
            # Download
            download_path = self.download_dataset(dataset_name)
            
            # Extract and get MICCAI subfolder
            extracted_path = self.extract_and_get_patient_folders(
                download_path, 
                "MICCAI_BraTS2020_TrainingData"
            )
            
            # Get patient folders from MICCAI_BraTS2020_TrainingData
            patient_folders = self.get_patient_folders(extracted_path)
            logger.info(f"Found {len(patient_folders)} patient folders in MICCAI_BraTS2020_TrainingData")
            
            for patient_folder in patient_folders:
                blob_prefix = f"dataset/dataset/{patient_folder.name}/"
                logger.info(f"Uploading patient folder: {patient_folder.name}")
                self.upload_to_azure(patient_folder, blob_prefix)
            
            # Cleanup
            self.cleanup_local(download_path)
            self.cleanup_local(extracted_path)
            
            logger.info(f"✓ BRATS2020 completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process BRATS2020: {e}")
            return False
    
    def process_all_datasets(self):
        """Process all three BraTS datasets."""
        results = {
            'BRATS2023': False,
            'BRATS2019': False,
            'BRATS2020': False
        }
        
        logger.info(f"\n{'#'*70}")
        logger.info(f"# BRATS Datasets Download and Upload to Azure")
        logger.info(f"# Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'#'*70}")
        
        # Process each dataset
        results['BRATS2023'] = self.process_brats2023_part1()
        results['BRATS2019'] = self.process_brats2019_hgg()
        results['BRATS2020'] = self.process_brats2020_training()
        
        # Print summary
        logger.info(f"\n{'='*70}")
        logger.info("FINAL SUMMARY")
        logger.info(f"{'='*70}")
        for dataset, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            logger.info(f"{status}: {dataset}")
        
        all_success = all(results.values())
        logger.info(f"\nOverall: {'✓ All datasets processed' if all_success else '✗ Some datasets failed'}")
        logger.info(f"Data uploaded to Azure at: beproject/dataset/dataset/")
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'#'*70}\n")
        
        return results


def main():
    """Main entry point."""
    # Azure credentials
    connection_string = "DefaultEndpointsProtocol=https;AccountName=spartis9488473038;AccountKey=WxiLwTEm+WEut0AIFRTLiWcXgHhDixXtYtF5gbbGIKLMWANt5wHOVwg/QzRgz2uG1CHcazDil58i+ASttN+yaA==;EndpointSuffix=core.windows.net"
    container_name = "beproject"
    
    try:
        uploader = BraTS_AzureUploader(connection_string, container_name)
        results = uploader.process_all_datasets()
        
        # Exit with appropriate code
        sys.exit(0 if all(results.values()) else 1)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
