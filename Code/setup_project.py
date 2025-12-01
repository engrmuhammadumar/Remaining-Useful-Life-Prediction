"""
SETUP SCRIPT FOR C-RLM PROJECT
===============================
Copies necessary files from previous work and sets up the project structure
"""

import os
import shutil

print("="*70)
print("SETTING UP C-RLM PROJECT")
print("="*70)

# Define paths
NEW_CODE_DIR = r'E:\4 Paper\Implemenatation\new_code'
CRLM_DIR = r'E:\4 Paper\Implemenatation\caus-reinfo'

# Files to copy
FILES_TO_COPY = [
    'step1_data_loader.py',
    'step2_models.py',
    'best_causal_optimized.pth',
    'best_baseline_optimized.pth'
]

print("\nStep 1: Copying necessary files from previous work...")
print(f"From: {NEW_CODE_DIR}")
print(f"To: {CRLM_DIR}")

for filename in FILES_TO_COPY:
    src = os.path.join(NEW_CODE_DIR, filename)
    dst = os.path.join(CRLM_DIR, filename)
    
    if os.path.exists(src):
        try:
            shutil.copy2(src, dst)
            print(f"  ✓ Copied: {filename}")
        except Exception as e:
            print(f"  ✗ Failed to copy {filename}: {e}")
    else:
        print(f"  ⚠ Not found: {filename}")

print("\nStep 2: Verifying C-RLM files...")
CRLM_FILES = [
    'PhD_Research_Architecture.py',
    'step1_maintenance_environment.py',
    'test_environment.py'
]

for filename in CRLM_FILES:
    filepath = os.path.join(CRLM_DIR, filename)
    if os.path.exists(filepath):
        print(f"  ✓ Found: {filename}")
    else:
        print(f"  ✗ Missing: {filename}")

print("\n" + "="*70)
print("PROJECT STRUCTURE:")
print("="*70)

print("""
caus-reinfo/
├── step1_data_loader.py           (copied from new_code)
├── step2_models.py                (copied from new_code)
├── best_causal_optimized.pth      (copied from new_code)
├── best_baseline_optimized.pth    (copied from new_code)
├── PhD_Research_Architecture.py   (new - research blueprint)
├── step1_maintenance_environment.py (new - RL environment)
└── test_environment.py            (new - test script)
""")

print("="*70)
print("✅ SETUP COMPLETE!")
print("="*70)

print("\nNext: Run test_environment.py to verify everything works!")
