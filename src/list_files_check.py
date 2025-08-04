import os

for cls in ["benign", "malignant", "normal"]:
    print(f"\n--- {cls.upper()} ---")
    img_dir = f"data/{cls}"
    mask_dir = f"tumor_masks/{cls}"
    print(f"Images in {img_dir}:")
    print(os.listdir(img_dir) if os.path.exists(img_dir) else "Folder not found")
    print(f"Masks in {mask_dir}:")
    print(os.listdir(mask_dir) if os.path.exists(mask_dir) else "Folder not found")