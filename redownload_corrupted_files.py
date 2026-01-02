"""
Re-download corrupted/empty files from Azure Blob Storage.

This script scans the dataset for files with 0 bytes or < 1KB
and re-downloads only those files from Azure.

Usage:
    python redownload_corrupted_files.py [--output-dir ./dataset] [--dry-run]
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CorruptedFileRedownloader:
    """Identify and re-download corrupted files from Azure."""
    
    # Azure credentials - loaded from environment variables
    BLOB_CONNECTION_STRING: str = os.getenv(
        'AZURE_STORAGE_CONNECTION_STRING',
        ''
    )
    CONTAINER_NAME: str = "beproject"
    BLOB_PREFIX: str = "dataset/dataset/"
    MIN_FILE_SIZE: int = 1024  # Files smaller than 1KB are considered corrupted
    
    def __init__(self, dataset_dir: str = "./dataset"):
        """
        Initialize redownloader.
        
        Args:
            dataset_dir: Directory containing the dataset
        """
        # Validate connection string
        if not self.BLOB_CONNECTION_STRING:
            raise ValueError(
                "Azure Storage connection string not found!\n"
                "Please set the AZURE_STORAGE_CONNECTION_STRING environment variable."
            )
        
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.exists():
            raise ValueError(f"Dataset directory not found: {self.dataset_dir}")
        
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
            except ImportError:
                logger.info(f"Installing {pip_name}...")
                import subprocess
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    pip_name, "-q"
                ])
    
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
    
    def scan_corrupted_files(self) -> List[Tuple[Path, int]]:
        """
        Scan dataset for corrupted files.
        
        Returns:
            List of (file_path, file_size) tuples
        """
        logger.info(f"Scanning {self.dataset_dir} for corrupted files...")
        logger.info(f"Files smaller than {self.MIN_FILE_SIZE} bytes will be flagged")
        
        corrupted_files = []
        
        # Scan all .nii and .nii.gz files
        for pattern in ['**/*.nii', '**/*.nii.gz']:
            for file_path in self.dataset_dir.glob(pattern):
                file_size = file_path.stat().st_size
                if file_size < self.MIN_FILE_SIZE:
                    corrupted_files.append((file_path, file_size))
        
        corrupted_files.sort(key=lambda x: x[0])
        
        logger.info(f"✓ Found {len(corrupted_files)} corrupted files")
        
        return corrupted_files
    
    def get_blob_path(self, local_file: Path) -> str:
        """
        Convert local file path to Azure blob path.
        
        Args:
            local_file: Local file path
            
        Returns:
            Azure blob path
        """
        # Get relative path from dataset directory
        relative_path = local_file.relative_to(self.dataset_dir)
        
        # Convert to Azure blob path
        blob_path = self.BLOB_PREFIX + str(relative_path).replace('\\', '/')
        
        return blob_path
    
    def download_file(self, local_file: Path, blob_path: str) -> bool:
        """
        Download a single file from Azure.
        
        Args:
            local_file: Local file path to save to
            blob_path: Azure blob path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            blob_client = self.container_client.get_blob_client(blob_path)
            
            # Get blob properties to check size
            props = blob_client.get_blob_properties()
            expected_size = props.size
            
            # Download
            with open(local_file, 'wb') as f:
                download_stream = blob_client.download_blob()
                data = download_stream.readall()
                f.write(data)
            
            # Verify downloaded size
            actual_size = local_file.stat().st_size
            
            if actual_size != expected_size:
                logger.warning(f"Size mismatch: expected {expected_size}, got {actual_size}")
                return False
            
            logger.info(f"  ✓ Downloaded {local_file.name} ({actual_size / 1024:.1f} KB)")
            return True
            
        except Exception as e:
            logger.error(f"  ✗ Failed to download {local_file.name}: {e}")
            return False
    
    def redownload_corrupted_files(self, dry_run: bool = False):
        """
        Re-download all corrupted files.
        
        Args:
            dry_run: If True, only list files without downloading
        """
        from tqdm import tqdm
        
        start_time = datetime.now()
        
        logger.info("="*70)
        logger.info("Corrupted File Re-download Tool")
        logger.info(f"Dataset directory: {self.dataset_dir.absolute()}")
        if dry_run:
            logger.info("DRY RUN MODE - No files will be downloaded")
        logger.info("="*70)
        
        # Scan for corrupted files
        corrupted_files = self.scan_corrupted_files()
        
        if not corrupted_files:
            logger.info("✓ No corrupted files found!")
            return
        
        # Print summary
        logger.info("\nCorrupted files by patient:")
        patient_counts = {}
        for file_path, file_size in corrupted_files:
            patient_name = file_path.parent.name
            patient_counts[patient_name] = patient_counts.get(patient_name, 0) + 1
        
        for patient, count in sorted(patient_counts.items()):
            logger.info(f"  {patient}: {count} files")
        
        total_size = sum(size for _, size in corrupted_files)
        logger.info(f"\nTotal: {len(corrupted_files)} files ({total_size} bytes)")
        
        if dry_run:
            logger.info("\n✓ Dry run complete. Run without --dry-run to download.")
            return
        
        # Download files
        logger.info("\nStarting re-download...")
        success_count = 0
        failed_count = 0
        
        pbar = tqdm(
            corrupted_files,
            desc="Re-downloading files",
            unit="file",
            colour="cyan"
        )
        
        for local_file, old_size in pbar:
            blob_path = self.get_blob_path(local_file)
            pbar.set_postfix_str(f"{local_file.name}")
            
            if self.download_file(local_file, blob_path):
                success_count += 1
            else:
                failed_count += 1
        
        pbar.close()
        
        # Final summary
        elapsed_time = datetime.now() - start_time
        
        logger.info("\n" + "="*70)
        logger.info("Re-download Complete")
        logger.info("="*70)
        logger.info(f"Success: {success_count} files")
        logger.info(f"Failed:  {failed_count} files")
        logger.info(f"Time:    {elapsed_time}")
        logger.info("="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Re-download corrupted files from Azure Blob Storage"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./dataset',
        help='Dataset directory (default: ./dataset)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='List corrupted files without downloading'
    )
    
    args = parser.parse_args()
    
    try:
        redownloader = CorruptedFileRedownloader(args.output_dir)
        redownloader.redownload_corrupted_files(dry_run=args.dry_run)
        
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Download interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
