#!/usr/bin/env python
# Verification script to ensure all dependencies are installed and setup is correct

import sys
import os

def check_python_version():
    """Check Python version"""
    print("✓ Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"  ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ❌ Python {version.major}.{version.minor} (Need 3.8+)")
        return False

def check_dependencies():
    """Check all required packages"""
    print("\n✓ Checking dependencies...")
    
    required = {
        'torch': 'PyTorch',
        'nibabel': 'NiBabel',
        'scipy': 'SciPy',
        'sklearn': 'scikit-learn',
        'pandas': 'Pandas',
        'tensorboard': 'TensorBoard',
        'tqdm': 'tqdm',
        'matplotlib': 'Matplotlib'
    }
    
    all_ok = True
    for package, name in required.items():
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} (missing)")
            all_ok = False
    
    return all_ok

def check_gpu():
    """Check GPU availability"""
    print("\n✓ Checking GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  ✅ GPU: {gpu_name}")
            print(f"  ✅ VRAM: {gpu_mem:.1f} GB")
            
            if gpu_mem < 20:
                print(f"  ⚠️  Warning: Less than 20GB VRAM. May need memory optimizations.")
                return True
            return True
        else:
            print(f"  ❌ No GPU detected (CPU only - training will be VERY slow)")
            return False
    except Exception as e:
        print(f"  ❌ Error checking GPU: {e}")
        return False

def check_data_dir():
    """Check if data directory structure is correct"""
    print("\n✓ Checking data directory...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "dataset")
    
    if os.path.exists(data_dir):
        patients = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('BraTS')]
        
        if patients:
            print(f"  ✅ Data directory: {data_dir}")
            print(f"  ✅ Found {len(patients)} BraTS patient(s)")
            
            # Check first patient
            first_patient_dir = os.path.join(data_dir, patients[0])
            required_files = ['_t1.nii.gz', '_t1ce.nii.gz', '_t2.nii.gz', '_flair.nii.gz', '_seg.nii.gz']
            
            found_files = []
            for file in os.listdir(first_patient_dir):
                for req in required_files:
                    if req in file:
                        found_files.append(req)
            
            if len(found_files) == 5:
                print(f"  ✅ First patient ({patients[0]}) has all required files")
                return True
            else:
                print(f"  ⚠️  First patient missing some files. Found: {found_files}")
                return False
        else:
            print(f"  ⚠️  No BraTS patients found in {data_dir}")
            print(f"     Please create dataset/ folder with BraTS_* subdirectories")
            return False
    else:
        print(f"  ⚠️  Data directory not found: {data_dir}")
        print(f"     Please create dataset/ folder in {script_dir}")
        return False

def check_script_exists():
    """Check if main script exists"""
    print("\n✓ Checking training script...")
    
    script_path = os.path.join(os.path.dirname(__file__), "optimized_brats_final.py")
    
    if os.path.exists(script_path):
        size_mb = os.path.getsize(script_path) / 1e6
        print(f"  ✅ optimized_brats_final.py ({size_mb:.1f} MB)")
        return True
    else:
        print(f"  ❌ optimized_brats_final.py not found")
        return False

def check_config():
    """Check if config file exists"""
    print("\n✓ Checking configuration file...")
    
    config_path = os.path.join(os.path.dirname(__file__), "config.py")
    
    if os.path.exists(config_path):
        print(f"  ✅ config.py (presets available)")
        return True
    else:
        print(f"  ⚠️  config.py not found (but not critical)")
        return False

def main():
    """Run all checks"""
    print("\n" + "="*70)
    print("BraTS 3D SEGMENTATION - SETUP VERIFICATION")
    print("="*70 + "\n")
    
    checks = [
        ("Python Version", check_python_version()),
        ("Dependencies", check_dependencies()),
        ("GPU", check_gpu()),
        ("Script File", check_script_exists()),
        ("Config File", check_config()),
        ("Data Directory", check_data_dir()),
    ]
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in checks if result)
    total = len(checks)
    
    for check_name, result in checks:
        status = "✅ PASS" if result else "⚠️  WARN"
        print(f"{status:8} | {check_name}")
    
    print("="*70 + "\n")
    
    if passed == total:
        print("✅ ALL CHECKS PASSED! You're ready to train.")
        print("\nNext steps:")
        print("1. Update DATA_DIR in optimized_brats_final.py if needed")
        print("2. Run: python optimized_brats_final.py")
        print("3. Monitor: tensorboard --logdir=tensorboard_optimized_3fold")
        return 0
    elif passed >= total - 2:
        print("⚠️  MOST CHECKS PASSED - You can proceed with caution")
        print("\nMissing items:")
        for check_name, result in checks:
            if not result:
                print(f"  - {check_name}")
        return 1
    else:
        print("❌ CRITICAL ISSUES FOUND - Please fix before training")
        print("\nFailing checks:")
        for check_name, result in checks:
            if not result:
                print(f"  - {check_name}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
