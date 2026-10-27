# ============================================================
# VIVA DAY: Upload & Load Everything Back
# Run this in a FRESH Colab notebook
# Then run Cells 1-2 (imports), Cells 8-9 (CBAM), then Cells 13-17
# ============================================================
import os
import shutil
import zipfile
from google.colab import files

# Step 1: Upload the zip from your laptop
print("📤 Select DL_Assignment_Complete.zip from your laptop...")
uploaded = files.upload()

# Step 2: Unzip
zip_name = list(uploaded.keys())[0]
with zipfile.ZipFile(zip_name, 'r') as z:
    z.extractall('/content/DL_Backup')
print("✅ Unzipped")

# Step 3: Restore runs folder (model weights)
if os.path.exists('/content/DL_Backup/runs'):
    shutil.copytree('/content/DL_Backup/runs', '/content/runs', dirs_exist_ok=True)
print("✅ Model weights restored")

# Step 4: Restore config files
for cfg in ['cadi.yaml', 'yolov8n-cbam.yaml']:
    src = f'/content/DL_Backup/{cfg}'
    if os.path.exists(src):
        shutil.copy(src, f'/content/{cfg}')
print("✅ Configs restored")

# Step 5: Restore plots
for p in ['class_distribution.png', 'precision_recall_comparison.png',
          'detection_comparison.png', 'cbam_attention.png']:
    src = f'/content/DL_Backup/{p}'
    if os.path.exists(src):
        shutil.copy(src, f'/content/{p}')
print("✅ Plots restored")

# Step 6: Load both models — NO RETRAINING NEEDED
from ultralytics import YOLO
baseline_model = YOLO('/content/runs/detect/baseline_yolov8n/weights/best.pt')
modified_model = YOLO('/content/runs/detect/modified_yolov8n_cbam/weights/best.pt')

print("\n🚀 Both models loaded! Now run Cells 13-17 for live results.")
