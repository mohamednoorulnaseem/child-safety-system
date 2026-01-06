"""
Dataset Download Helper Script
Instructions for downloading UTKFace and FER2013 datasets
"""
import os
import sys
from pathlib import Path

def print_banner(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")

def check_dataset_exists(path):
    return os.path.exists(path) and len(os.listdir(path)) > 0

def main():
    print_banner("Dataset Download Instructions")
    
    # Check current datasets
    utkface_path = Path("data/datasets/UTKFace")
    fer2013_path = Path("data/datasets/FER2013")
    
    utkface_exists = check_dataset_exists(utkface_path)
    fer2013_exists = check_dataset_exists(fer2013_path)
    
    print("Current Status:")
    print(f"  ✓ UTKFace: {'FOUND' if utkface_exists else 'NOT FOUND'}")
    print(f"  ✓ FER2013: {'FOUND' if fer2013_exists else 'NOT FOUND'}")
    
    # UTKFace instructions
    if not utkface_exists:
        print_banner("UTKFace Dataset (Age Classification)")
        print("📦 Dataset: UTKFace")
        print("📊 Size: ~3 GB")
        print("🔗 URL: https://susanqq.github.io/UTKFace/")
        print("\n📝 Download Instructions:")
        print("  1. Visit the official website:")
        print("     https://susanqq.github.io/UTKFace/")
        print("\n  2. Click on 'Download' to get UTKFace.tar.gz")
        print("\n  3. Extract the archive:")
        print("     - Windows: Use 7-Zip or WinRAR")
        print("     - Linux/Mac: tar -xzf UTKFace.tar.gz")
        print(f"\n  4. Move extracted files to: {utkface_path.absolute()}")
        print("\n📁 Expected Structure:")
        print(f"  {utkface_path}/")
        print("    ├── 1_0_0_20161219140623618.jpg")
        print("    ├── 2_0_1_20161219203650988.jpg")
        print("    └── ... (20,000+ images)")
        print("\n💡 Filename Format: [age]_[gender]_[race]_[date&time].jpg")
    
    # FER2013 instructions
    if not fer2013_exists:
        print_banner("FER2013 Dataset (Emotion Detection)")
        print("📦 Dataset: FER2013")
        print("📊 Size: ~300 MB")
        print("🔗 URL: https://www.kaggle.com/datasets/msambare/fer2013")
        print("\n📝 Download Instructions:")
        print("  1. Visit Kaggle (requires account):")
        print("     https://www.kaggle.com/datasets/msambare/fer2013")
        print("\n  2. Click 'Download' to get fer2013.zip")
        print("\n  3. Extract the archive")
        print(f"\n  4. Move extracted files to: {fer2013_path.absolute()}")
        print("\n📁 Expected Structure (Option 1 - CSV):")
        print(f"  {fer2013_path}/")
        print("    └── fer2013.csv")
        print("\n📁 Expected Structure (Option 2 - Folders):")
        print(f"  {fer2013_path}/")
        print("    ├── train/")
        print("    │   ├── angry/")
        print("    │   ├── disgust/")
        print("    │   ├── fear/")
        print("    │   ├── happy/")
        print("    │   ├── sad/")
        print("    │   ├── surprise/")
        print("    │   └── neutral/")
        print("    └── test/")
        print("        └── ... (same structure)")
        print("\n💡 Either format works - training script handles both")
    
    # Create directories
    print_banner("Setting Up Directory Structure")
    os.makedirs(utkface_path, exist_ok=True)
    os.makedirs(fer2013_path, exist_ok=True)
    print(f"✓ Created: {utkface_path}")
    print(f"✓ Created: {fer2013_path}")
    
    # Next steps
    print_banner("Next Steps")
    if not utkface_exists or not fer2013_exists:
        print("After downloading datasets:")
        print("\n1. Verify dataset structure matches above")
        print("\n2. Train Age Classifier:")
        print("   python training/train_age_model.py")
        print("\n3. Train Emotion Detector:")
        print("   python training/train_emotion_model.py")
        print("\n⏱️  Estimated Training Time:")
        print("   - Age Classifier: 6-8 hours (CPU) or 1-2 hours (GPU)")
        print("   - Emotion Detector: 6-8 hours (CPU) or 2-3 hours (GPU)")
    else:
        print("✅ All datasets are ready!")
        print("\nYou can now train the models:")
        print("  python training/train_age_model.py")
        print("  python training/train_emotion_model.py")
    
    print("\n" + "=" * 70)

if __name__ == '__main__':
    main()
