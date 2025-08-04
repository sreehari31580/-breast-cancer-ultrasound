import os

for cls in ["benign", "malignant", "normal"]:
    roi_folder = f"tumor_masks/{cls}"
    print(f"\nROI masks in {roi_folder}:")
    if os.path.exists(roi_folder):
        print(os.listdir(roi_folder))
    else:
        print("Folder not found")