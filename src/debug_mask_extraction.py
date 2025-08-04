import os

data_dir = "data"
for cls in ["benign", "malignant"]:
    class_dir = os.path.join(data_dir, cls)
    img_files = [f for f in os.listdir(class_dir) if f.endswith(".png") and '_mask' not in f]
    print(f"\n--- Checking {cls.upper()} ---")
    for img_name in img_files:
        img_path = os.path.join(class_dir, img_name)
        mask_name = img_name.replace('.png', '_mask.png')
        mask_path = os.path.join(class_dir, mask_name)
        if not os.path.exists(mask_path):
            print(f"NO MASK FOUND for image: {img_name}")
        else:
            print(f"Image: {img_name}  |  Mask: {mask_name}  |  FOUND")