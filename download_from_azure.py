"""
Download BraTS dataset from Azure Blob Storage.

This script downloads the complete BraTS dataset from Azure to local storage.
Useful for RunPod or other cloud GPU instances.

Usage:
    python download_from_azure.py [--output-dir ./dataset]
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AzureDatasetDownloader:
    """Download BraTS dataset from Azure Blob Storage."""
    
    # Azure credentials
    BLOB_CONNECTION_STRING: str = (
        "DefaultEndpointsProtocol=https;AccountName=spartis9488473038;"
        "AccountKey=WxiLwTEm+WEut0AIFRTLiWcXgHhDixXtYtF5gbbGIKLMWANt5wHOVwg/"
        "QzRgz2uG1CHcazDil58i+ASttN+yaA==;EndpointSuffix=core.windows.net"
    )
    CONTAINER_NAME: str = "beproject"
    BLOB_PREFIX: str = "dataset/dataset/"
    
    def __init__(self, output_dir: str = "./dataset"):
        """
        Initialize downloader.
        
        Args:
            output_dir: Local directory to save dataset
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._install_dependencies()
        self._setup_azure_client()
    
    def _install_dependencies(self):
        """Install required Azure package if not present."""
        try:
            import azure.storage.blob
            logger.info("✓ azure-storage-blob is already installed")
        except ImportError:
            logger.info("Installing azure-storage-blob...")
            import subprocess
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "azure-storage-blob", "-q"
            ])
            logger.info("✓ azure-storage-blob installed successfully")
    
    def _setup_azure_client(self):
        """Initialize Azure Blob Storage client."""
        try:
            from azure.storage.blob import BlobServiceClient
            
            self.blob_service_client = BlobServiceClient.from_connection_string(
                self.BLOB_CONNECTION_STRING
            )
            self.container_client = self.blob_service_client.get_container_client(
                self.CONTAINER_NAME
            )
            
            # Test connection
            self.container_client.get_container_properties()
            logger.info(f"✓ Connected to Azure container: {self.CONTAINER_NAME}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Azure: {e}")
            raise
    
    def list_patient_folders(self) -> list:
        """List all patient folders in Azure."""
        logger.info(f"Listing patient folders from {self.BLOB_PREFIX}...")
        
        patient_folders = set()
        blob_list = self.container_client.list_blobs(name_starts_with=self.BLOB_PREFIX)
        
        for blob in blob_list:
            # Extract patient folder name
            # Path format: dataset/dataset/BraTS_001/file.nii.gz
            relative_path = blob.name[len(self.BLOB_PREFIX):]
            if '/' in relative_path:
                patient_folder = relative_path.split('/')[0]
                patient_folders.add(patient_folder)
        
        patient_folders = sorted(patient_folders)
        logger.info(f"✓ Found {len(patient_folders)} patient folders")
        
        return patient_folders
    
    def download_patient_folder(self, patient_name: str) -> tuple:
        """
        Download a single patient folder.
        
        Args:
            patient_name: Name of patient folder (e.g., 'BraTS_001')
        
        Returns:
            Tuple of (file_count, total_bytes)
        """
        logger.info(f"Downloading: {patient_name}")
        
        blob_prefix = f"{self.BLOB_PREFIX}{patient_name}/"
        patient_dir = self.output_dir / patient_name
        patient_dir.mkdir(parents=True, exist_ok=True)
        
        file_count = 0
        total_bytes = 0
        
        blob_list = self.container_client.list_blobs(name_starts_with=blob_prefix)
        
        for blob in blob_list:
            try:
                # Get relative path within patient folder
                relative_path = blob.name[len(blob_prefix):]
                local_file = patient_dir / relative_path
                
                # Create subdirectories if needed
                local_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Download blob
                blob_client = self.container_client.get_blob_client(blob.name)
                
                with open(local_file, 'wb') as f:
                    download_stream = blob_client.download_blob()
                    data = download_stream.readall()
                    f.write(data)
                
                file_count += 1
                total_bytes += len(data)
                
                # Log progress for larger files
                if len(data) > 10 * 1024 * 1024:  # > 10MB
                    logger.info(f"  Downloaded: {relative_path} ({len(data) / (1024**2):.1f} MB)")
                
            except Exception as e:
                logger.error(f"  Failed to download {blob.name}: {e}")
                continue
        
        return file_count, total_bytes
    
    def download_all(self, max_patients: Optional[int] = None):
        """
        Download all patient folders.
        
        Args:
            max_patients: Optional limit on number of patients to download
        """
        start_time = datetime.now()
        
        logger.info("="*70)
        logger.info("Starting BraTS Dataset Download from Azure")
        logger.info(f"Output directory: {self.output_dir.absolute()}")
        logger.info("="*70)
        
        # Get patient list
        patient_folders = self.list_patient_folders()
        
        if max_patients:
            patient_folders = patient_folders[:max_patients]
            logger.info(f"Limiting to first {max_patients} patients")
        
        # Download each patient
        total_files = 0
        total_bytes = 0
        success_count = 0
        
        for i, patient_name in enumerate(patient_folders, 1):
            try:
                logger.info(f"\n[{i}/{len(patient_folders)}] Processing: {patient_name}")
                
                file_count, byte_count = self.download_patient_folder(patient_name)
                
                total_files += file_count
                total_bytes += byte_count
                success_count += 1
                
                logger.info(
                    f"  ✓ Downloaded {file_count} files "
                    f"({byte_count / (1024**2):.1f} MB)"
                )
                
                # Progress summary every 10 patients
                if i % 10 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    rate = i / elapsed * 3600  # patients per hour
                    logger.info(
                        f"\nProgress: {i}/{len(patient_folders)} patients | "
                        f"{total_bytes / (1024**3):.2f} GB downloaded | "
                        f"Rate: {rate:.1f} patients/hour"
                    )
                
            except Exception as e:
                logger.error(f"Failed to download {patient_name}: {e}")
                continue
        
        # Final summary
        elapsed_time = datetime.now() - start_time
        
        logger.info("\n" + "="*70)
        logger.info("DOWNLOAD COMPLETE")
        logger.info("="*70)
        logger.info(f"Successfully downloaded: {success_count}/{len(patient_folders)} patients")
        logger.info(f"Total files: {total_files}")
        logger.info(f"Total size: {total_bytes / (1024**3):.2f} GB")
        logger.info(f"Time elapsed: {elapsed_time}")
        logger.info(f"Output directory: {self.output_dir.absolute()}")
        logger.info("="*70)
        
        return success_count == len(patient_folders)
    
    def verify_download(self) -> dict:
        """Verify downloaded dataset structure."""
        logger.info("\nVerifying downloaded dataset...")
        
        stats = {
            'patient_folders': 0,
            'total_files': 0,
            'total_size_gb': 0,
            'modalities': {
                't1': 0,
                't1ce': 0,
                't2': 0,
                'flair': 0,
                'seg': 0
            }
        }
        
        # Count patient folders
        patient_folders = [d for d in self.output_dir.iterdir() if d.is_dir()]
        stats['patient_folders'] = len(patient_folders)
        
        # Count files and modalities
        for patient_dir in patient_folders:
            for file in patient_dir.rglob('*.nii.gz'):
                stats['total_files'] += 1
                stats['total_size_gb'] += file.stat().st_size / (1024**3)
                
                # Check modality
                filename = file.name.lower()
                if '_t1.' in filename or '_t1_' in filename:
                    stats['modalities']['t1'] += 1
                elif '_t1ce.' in filename or '_t1ce_' in filename:
                    stats['modalities']['t1ce'] += 1
                elif '_t2.' in filename or '_t2_' in filename:
                    stats['modalities']['t2'] += 1
                elif '_flair.' in filename or '_flair_' in filename:
                    stats['modalities']['flair'] += 1
                elif '_seg.' in filename or '_seg_' in filename:
                    stats['modalities']['seg'] += 1
        
        # Print verification report
        logger.info("\n" + "="*70)
        logger.info("DATASET VERIFICATION")
        logger.info("="*70)
        logger.info(f"Patient folders: {stats['patient_folders']}")
        logger.info(f"Total files: {stats['total_files']}")
        logger.info(f"Total size: {stats['total_size_gb']:.2f} GB")
        logger.info("\nModality counts:")
        for modality, count in stats['modalities'].items():
            logger.info(f"  {modality.upper()}: {count} files")
        
        # Check if dataset looks complete
        expected_files_per_patient = 5  # t1, t1ce, t2, flair, seg
        expected_total = stats['patient_folders'] * expected_files_per_patient
        
        if stats['total_files'] >= expected_total * 0.95:  # Allow 5% tolerance
            logger.info("\n✓ Dataset appears complete")
        else:
            logger.warning(
                f"\n⚠ Dataset may be incomplete. "
                f"Expected ~{expected_total} files, found {stats['total_files']}"
            )
        
        logger.info("="*70)
        
        return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download BraTS dataset from Azure Blob Storage"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./dataset',
        help='Output directory for dataset (default: ./dataset)'
    )
    parser.add_argument(
        '--max-patients',
        type=int,
        default=None,
        help='Maximum number of patients to download (for testing)'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing download without downloading'
    )
    
    args = parser.parse_args()
    
    try:
        downloader = AzureDatasetDownloader(output_dir=args.output_dir)
        
        if args.verify_only:
            # Just verify
            downloader.verify_download()
        else:
            # Download all
            success = downloader.download_all(max_patients=args.max_patients)
            
            # Verify
            downloader.verify_download()
            
            sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("\n\nDownload interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
