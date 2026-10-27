# ============================================================
# SAVE ALL RESULTS TO ZIP & DOWNLOAD
# Run this AFTER Cells 13-17 finish in your notebook
# ============================================================
import os
import shutil
from google.colab import files

# Create a master folder with everything
save_dir = '/content/DL_Assignment_Backup'
os.makedirs(save_dir, exist_ok=True)

# 1. Model weights (the most important files)
if os.path.exists('/content/runs'):
    shutil.copytree('/content/runs', f'{save_dir}/runs', dirs_exist_ok=True)
    print("✅ Model weights & training logs saved")

# 2. Generated plots
plots = ['class_distribution.png', 'precision_recall_comparison.png',
         'detection_comparison.png', 'cbam_attention.png']
for p in plots:
    if os.path.exists(p):
        shutil.copy(p, save_dir)
print("✅ Plots saved")

# 3. Config files
for cfg in ['cadi.yaml', 'yolov8n-cbam.yaml']:
    if os.path.exists(cfg):
        shutil.copy(cfg, save_dir)
print("✅ Config files saved")

# 4. Dataset structure (labels only — images are too large, we re-download those)
for split in ['train', 'val', 'test']:
    lbl_src = f'/content/datasets/CADI-AI/labels/{split}'
    lbl_dst = f'{save_dir}/labels/{split}'
    if os.path.exists(lbl_src):
        shutil.copytree(lbl_src, lbl_dst, dirs_exist_ok=True)
print("✅ Label files saved")

# Zip everything
shutil.make_archive('/content/DL_Assignment_Complete', 'zip', save_dir)
print(f"\n📦 ZIP created: DL_Assignment_Complete.zip")

# Download to your laptop
files.download('/content/DL_Assignment_Complete.zip')
print("⬇️  Download started — save it to your laptop!")
