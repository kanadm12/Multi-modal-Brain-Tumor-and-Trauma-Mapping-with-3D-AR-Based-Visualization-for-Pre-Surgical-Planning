"""
Download BraTS dataset from Azure Blob Storage.

This script downloads the complete BraTS dataset from Azure to local storage.
Useful for RunPod or other cloud GPU instances.

Setup:
    1. Set the AZURE_STORAGE_CONNECTION_STRING environment variable:
       
       Linux/Mac:
         export AZURE_STORAGE_CONNECTION_STRING='your_connection_string'
       
       Windows (PowerShell):
         $env:AZURE_STORAGE_CONNECTION_STRING='your_connection_string'
       
       Or create a .env file (see .env.example)

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
    
    # Azure credentials - loaded from environment variables
    BLOB_CONNECTION_STRING: str = os.getenv(
        'AZURE_STORAGE_CONNECTION_STRING',
        ''  # Empty default - will be validated in __init__
    )
    CONTAINER_NAME: str = "beproject"
    BLOB_PREFIX: str = "dataset/dataset/"
    
    def __init__(self, output_dir: str = "./dataset"):
        """
        Initialize downloader.
        
        Args:
            output_dir: Local directory to save dataset
        """
        # Validate connection string
        if not self.BLOB_CONNECTION_STRING:
            raise ValueError(
                "Azure Storage connection string not found!\n"
                "Please set the AZURE_STORAGE_CONNECTION_STRING environment variable.\n\n"
                "On Linux/Mac:\n"
                "  export AZURE_STORAGE_CONNECTION_STRING='your_connection_string'\n\n"
                "On Windows (PowerShell):\n"
                "  $env:AZURE_STORAGE_CONNECTION_STRING='your_connection_string'\n\n"
                "Or create a .env file with:\n"
                "  AZURE_STORAGE_CONNECTION_STRING=your_connection_string"
            )
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._install_dependencies()
        self._setup_azure_client()
    
    def _install_dependencies(self):
        """Install required Azure package if not present."""
        required_packages = {
            'azure.storage.blob': 'azure-storage-blob',
            'tqdm': 'tqdm'
        }
        
        for import_name, pip_name in required_packages.items():
            try:
                __import__(import_name)
                logger.info(f"✓ {pip_name} is already installed")
            except ImportError:
                logger.info(f"Installing {pip_name}...")
                import subprocess
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    pip_name, "-q"
                ])
                logger.info(f"✓ {pip_name} installed successfully")
    
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
    
    def download_patient_folder(self, patient_name: str, pbar: Optional[object] = None) -> tuple:
        """
        Download a single patient folder.
        
        Args:
            patient_name: Name of patient folder (e.g., 'BraTS_001')
            pbar: Optional tqdm progress bar to update
        
        Returns:
            Tuple of (file_count, total_bytes, skipped_count)
        """
        from tqdm import tqdm
        
        blob_prefix = f"{self.BLOB_PREFIX}{patient_name}/"
        patient_dir = self.output_dir / patient_name
        patient_dir.mkdir(parents=True, exist_ok=True)
        
        file_count = 0
        total_bytes = 0
        skipped_count = 0
        downloaded_count = 0
        
        # Get list of blobs first
        blob_list = list(self.container_client.list_blobs(name_starts_with=blob_prefix))
        
        # Quick check: if all files exist with correct size, skip entire folder
        all_files_exist = True
        for blob in blob_list:
            relative_path = blob.name[len(blob_prefix):]
            local_file = patient_dir / relative_path
            if not local_file.exists() or local_file.stat().st_size != blob.size:
                all_files_exist = False
                break
        
        if all_files_exist and len(blob_list) > 0:
            # All files already downloaded - skip entire folder
            for blob in blob_list:
                file_count += 1
                total_bytes += blob.size
                skipped_count += 1
            
            if pbar:
                pbar.set_postfix_str(f"✓ Skipped {patient_name} (complete)")
            return file_count, total_bytes, skipped_count
        
        # Create progress bar for files within this patient folder
        file_pbar = tqdm(
            blob_list,
            desc=f"  Files for {patient_name}",
            unit="file",
            leave=False,
            disable=len(blob_list) < 3  # Only show for folders with 3+ files
        )
        
        for blob in file_pbar:
            try:
                # Get relative path within patient folder
                relative_path = blob.name[len(blob_prefix):]
                local_file = patient_dir / relative_path
                
                # Skip if file already exists and has same size
                if local_file.exists() and local_file.stat().st_size == blob.size:
                    file_count += 1
                    total_bytes += blob.size
                    skipped_count += 1
                    file_pbar.set_postfix_str(f"✓ Skipped (exists)")
                    continue
                
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
                downloaded_count += 1
                
                # Update progress bar postfix with current file
                file_pbar.set_postfix_str(f"⬇ {len(data) / (1024**2):.1f} MB")
                
            except Exception as e:
                logger.error(f"  Failed to download {blob.name}: {e}")
                continue
        
        file_pbar.close()
        
        if pbar:
            if downloaded_count > 0:
                pbar.set_postfix_str(f"⬇ {patient_name} ({downloaded_count} new, {skipped_count} skip)")
            else:
                pbar.set_postfix_str(f"✓ {patient_name} (all cached)")
        
        return file_count, total_bytes, skipped_count
    
    def download_all(self, max_patients: Optional[int] = None):
        """
        Download all patient folders.
        
        Args:
            max_patients: Optional limit on number of patients to download
        """
        from tqdm import tqdm
        
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
        
        # Download each patient with progress bar
        total_files = 0
        total_bytes = 0
        total_skipped = 0
        total_downloaded = 0
        success_count = 0
        
        # Create main progress bar for patients
        pbar = tqdm(
            patient_folders,
            desc="Downloading patients",
            unit="patient",
            colour="green",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        )
        
        for patient_name in pbar:
            try:
                file_count, byte_count, skipped_count = self.download_patient_folder(patient_name, pbar)
                
                total_files += file_count
                total_bytes += byte_count
                total_skipped += skipped_count
                total_downloaded += (file_count - skipped_count)
                success_count += 1
                
                # Update progress bar with statistics
                pbar.set_postfix({
                    'Files': f'{total_files} ({total_downloaded}⬇/{total_skipped}✓)',
                    'Size': f'{total_bytes / (1024**3):.1f}GB',
                    'Avg': f'{total_bytes / success_count / (1024**2):.0f}MB/pt'
                })
                
            except Exception as e:
                logger.error(f"Failed to download {patient_name}: {e}")
                pbar.set_postfix_str(f"ERROR: {patient_name}")
                continue
        
        pbar.close()
        
        # Final summary
        elapsed_time = datetime.now() - start_time
        
        logger.info("\n" + "="*70)
        logger.info("DOWNLOAD COMPLETE")
        logger.info("="*70)
        logger.info(f"Successfully processed: {success_count}/{len(patient_folders)} patients")
        logger.info(f"Total files: {total_files} ({total_downloaded} downloaded, {total_skipped} skipped)")
        logger.info(f"Total size: {total_bytes / (1024**3):.2f} GB")
        logger.info(f"Time elapsed: {elapsed_time}")
        logger.info(f"Average speed: {total_bytes / (1024**2) / elapsed_time.total_seconds():.2f} MB/s")
        if total_skipped > 0:
            logger.info(f"⚡ Resume detected: {total_skipped} files were already downloaded")
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
