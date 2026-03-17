#!/usr/bin/env python3
"""
GCP Vertex AI Training Job Launcher for BraTS 3D Segmentation

This script submits a custom container training job to Vertex AI.

Prerequisites:
1. Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install
2. Authenticate: gcloud auth login && gcloud auth application-default login
3. Enable APIs: gcloud services enable aiplatform.googleapis.com
4. Create a GCS bucket for data and outputs
5. Upload BraTS dataset to GCS
6. Build and push Docker container to GCR/Artifact Registry

Usage:
    python vertex_training.py --project YOUR_PROJECT_ID --bucket YOUR_BUCKET_NAME

Example:
    python vertex_training.py \
        --project my-gcp-project \
        --bucket brats-training-data \
        --region us-central1 \
        --machine-type a2-ultragpu-4g
"""

import argparse
import os
from datetime import datetime


def submit_training_job(
    project_id: str,
    bucket_name: str,
    region: str = "us-central1",
    machine_type: str = "a2-ultragpu-4g",
    accelerator_type: str = "NVIDIA_TESLA_A100",
    accelerator_count: int = 4,
    container_uri: str = None,
    display_name: str = None,
    tensorboard_name: str = None,
    use_spot: bool = False,
):
    """Submit a Vertex AI custom training job.
    
    Args:
        project_id: GCP project ID
        bucket_name: GCS bucket name (without gs:// prefix)
        region: GCP region (default: us-central1)
        machine_type: Vertex AI machine type
        accelerator_type: GPU type
        accelerator_count: Number of GPUs
        container_uri: Custom container URI (gcr.io/project/image:tag)
        display_name: Job display name
        tensorboard_name: Vertex AI TensorBoard instance name (optional)
        use_spot: Use spot/preemptible VMs for cost savings
    """
    from google.cloud import aiplatform
    
    # Initialize Vertex AI
    aiplatform.init(
        project=project_id,
        location=region,
        staging_bucket=f"gs://{bucket_name}",
    )
    
    # Generate display name if not provided
    if display_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        display_name = f"brats-segmentation-{timestamp}"
    
    # Default container URI
    if container_uri is None:
        container_uri = f"gcr.io/{project_id}/brats-training:latest"
    
    print(f"\n{'='*60}")
    print("BraTS 3D Segmentation - Vertex AI Training Job")
    print(f"{'='*60}")
    print(f"Project:       {project_id}")
    print(f"Region:        {region}")
    print(f"Machine Type:  {machine_type}")
    print(f"GPUs:          {accelerator_count}x {accelerator_type}")
    print(f"Container:     {container_uri}")
    print(f"Data Bucket:   gs://{bucket_name}")
    print(f"Spot VMs:      {'Yes (70% cheaper)' if use_spot else 'No'}")
    print(f"{'='*60}\n")
    
    # Define environment variables for the training container
    environment_variables = {
        "CLOUD_PLATFORM": "vertex_ai",
        "WORLD_SIZE": str(accelerator_count),
        "USE_PREPROCESSED": "true",  # Use preprocessed NPZ for faster loading
    }
    
    # GCS paths
    base_output_dir = f"gs://{bucket_name}/outputs/{display_name}"
    training_data_uri = f"gs://{bucket_name}/dataset"
    
    # Create custom container training job
    job = aiplatform.CustomContainerTrainingJob(
        display_name=display_name,
        container_uri=container_uri,
    )
    
    # Configure worker pool specs for multi-GPU
    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": machine_type,
                "accelerator_type": accelerator_type,
                "accelerator_count": accelerator_count,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": container_uri,
                "env": [{"name": k, "value": v} for k, v in environment_variables.items()],
            },
        }
    ]
    
    # Add spot VM configuration if requested
    if use_spot:
        worker_pool_specs[0]["machine_spec"]["spot"] = True
    
    print("Submitting training job...")
    
    # Submit the job
    job.run(
        replica_count=1,
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        base_output_dir=base_output_dir,
        environment_variables=environment_variables,
        tensorboard=tensorboard_name,
        sync=False,  # Don't wait for completion
    )
    
    print(f"\n✅ Training job submitted successfully!")
    print(f"\nMonitor progress at:")
    print(f"  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={project_id}")
    print(f"\nOutput artifacts will be saved to:")
    print(f"  {base_output_dir}")
    print(f"\nTo view TensorBoard logs:")
    print(f"  tensorboard --logdir={base_output_dir}/tensorboard")
    
    return job


def setup_gcs_data(bucket_name: str, local_data_path: str = None):
    """Upload BraTS dataset to GCS if not already present.
    
    Args:
        bucket_name: GCS bucket name
        local_data_path: Local path to BraTS dataset
    """
    from google.cloud import storage
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # Check if dataset exists
    blobs = list(bucket.list_blobs(prefix="dataset/", max_results=10))
    
    if blobs:
        print(f"✅ Dataset already exists in gs://{bucket_name}/dataset/")
        return
    
    if local_data_path is None:
        print("\n⚠️  Dataset not found in GCS bucket.")
        print("Please upload your BraTS dataset using one of these methods:\n")
        print("1. From local machine:")
        print(f"   gsutil -m cp -r /path/to/BraTS_dataset gs://{bucket_name}/dataset/\n")
        print("2. From Azure Blob Storage (using azcopy + gsutil):")
        print("   azcopy copy 'https://storage.blob.core.windows.net/container/BraTS' ./BraTS --recursive")
        print(f"   gsutil -m cp -r ./BraTS gs://{bucket_name}/dataset/\n")
        return
    
    print(f"Uploading dataset from {local_data_path} to gs://{bucket_name}/dataset/...")
    # Use gsutil for efficient parallel upload
    import subprocess
    subprocess.run([
        "gsutil", "-m", "cp", "-r",
        local_data_path,
        f"gs://{bucket_name}/dataset/"
    ], check=True)
    print("✅ Dataset uploaded successfully!")


def build_and_push_container(project_id: str, tag: str = "latest"):
    """Build and push the Docker container to Google Container Registry.
    
    Args:
        project_id: GCP project ID
        tag: Container tag (default: latest)
    """
    import subprocess
    
    container_uri = f"gcr.io/{project_id}/brats-training:{tag}"
    
    print(f"\nBuilding container: {container_uri}")
    
    # Build the container
    subprocess.run([
        "docker", "build",
        "-f", "Dockerfile.vertexai",
        "-t", container_uri,
        "."
    ], check=True)
    
    print(f"\nPushing container to GCR...")
    
    # Configure Docker for GCR
    subprocess.run(["gcloud", "auth", "configure-docker", "--quiet"], check=True)
    
    # Push the container
    subprocess.run(["docker", "push", container_uri], check=True)
    
    print(f"✅ Container pushed successfully: {container_uri}")
    return container_uri


def create_tensorboard_instance(project_id: str, region: str = "us-central1"):
    """Create a Vertex AI TensorBoard instance for monitoring.
    
    Args:
        project_id: GCP project ID
        region: GCP region
        
    Returns:
        TensorBoard resource name
    """
    from google.cloud import aiplatform
    
    aiplatform.init(project=project_id, location=region)
    
    # Check if TensorBoard instance exists
    tensorboards = aiplatform.Tensorboard.list(filter='display_name="brats-training-tb"')
    
    if tensorboards:
        print(f"✅ Using existing TensorBoard instance")
        return tensorboards[0].resource_name
    
    print("Creating Vertex AI TensorBoard instance...")
    tensorboard = aiplatform.Tensorboard.create(
        display_name="brats-training-tb",
        description="TensorBoard for BraTS 3D Segmentation Training"
    )
    
    print(f"✅ TensorBoard created: {tensorboard.resource_name}")
    return tensorboard.resource_name


def main():
    parser = argparse.ArgumentParser(
        description="Submit BraTS training job to GCP Vertex AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with 4x A100 80GB
  python vertex_training.py --project my-project --bucket my-bucket

  # Use spot VMs for 70% cost savings
  python vertex_training.py --project my-project --bucket my-bucket --spot

  # Build and push container first
  python vertex_training.py --project my-project --bucket my-bucket --build-container

  # Use H100 instead of A100
  python vertex_training.py --project my-project --bucket my-bucket \\
      --machine-type a3-highgpu-1g --accelerator-type NVIDIA_H100_80GB --accelerator-count 1
        """
    )
    
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--bucket", required=True, help="GCS bucket name (without gs://)")
    parser.add_argument("--region", default="us-central1", help="GCP region (default: us-central1)")
    parser.add_argument("--machine-type", default="a2-ultragpu-4g", 
                       help="Machine type (default: a2-ultragpu-4g for 4x A100 80GB)")
    parser.add_argument("--accelerator-type", default="NVIDIA_TESLA_A100",
                       help="GPU type (default: NVIDIA_TESLA_A100)")
    parser.add_argument("--accelerator-count", type=int, default=4,
                       help="Number of GPUs (default: 4)")
    parser.add_argument("--container-uri", default=None,
                       help="Custom container URI (default: gcr.io/PROJECT/brats-training:latest)")
    parser.add_argument("--display-name", default=None, help="Job display name")
    parser.add_argument("--spot", action="store_true", 
                       help="Use spot/preemptible VMs (70%% cheaper but may be interrupted)")
    parser.add_argument("--build-container", action="store_true",
                       help="Build and push container before submitting job")
    parser.add_argument("--setup-tensorboard", action="store_true",
                       help="Create Vertex AI TensorBoard instance")
    parser.add_argument("--upload-data", default=None,
                       help="Local path to BraTS dataset to upload to GCS")
    
    args = parser.parse_args()
    
    # Check for required dependencies
    try:
        from google.cloud import aiplatform
    except ImportError:
        print("Error: google-cloud-aiplatform not installed.")
        print("Install with: pip install google-cloud-aiplatform")
        return 1
    
    # Build and push container if requested
    container_uri = args.container_uri
    if args.build_container:
        container_uri = build_and_push_container(args.project)
    
    # Setup TensorBoard if requested
    tensorboard_name = None
    if args.setup_tensorboard:
        tensorboard_name = create_tensorboard_instance(args.project, args.region)
    
    # Upload data if path provided
    if args.upload_data:
        setup_gcs_data(args.bucket, args.upload_data)
    else:
        # Just check if data exists
        setup_gcs_data(args.bucket)
    
    # Submit the training job
    submit_training_job(
        project_id=args.project,
        bucket_name=args.bucket,
        region=args.region,
        machine_type=args.machine_type,
        accelerator_type=args.accelerator_type,
        accelerator_count=args.accelerator_count,
        container_uri=container_uri,
        display_name=args.display_name,
        tensorboard_name=tensorboard_name,
        use_spot=args.spot,
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
