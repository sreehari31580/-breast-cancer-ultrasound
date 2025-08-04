import os
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
from tqdm import tqdm
import pandas as pd

IMG_SIZE = 224  # Set to 224 for ResNet compatibility

data_dir = "Dataset_BUSI_with_GT"  # Use the new dataset as the source
masks_dir = "Dataset_BUSI_with_GT"  # Masks are in the same folders as images
out_dir = "cnn_data"
csv_file = "cnn_data/labels.csv"

os.makedirs(out_dir, exist_ok=True)
os.makedirs(os.path.join(out_dir, "debug"), exist_ok=True)

label_map = {"benign": 0, "malignant": 1, "normal": 2}
rows = []

debug_count = {"benign": 0, "malignant": 0, "normal": 0}

for cls in ["benign", "malignant", "normal"]:
    class_dir = os.path.join(data_dir, cls)
    out_class_dir = os.path.join(out_dir, cls)
    os.makedirs(out_class_dir, exist_ok=True)
    img_files = [f for f in os.listdir(class_dir) if f.endswith(".png") and '_mask' not in f]
    for img_name in tqdm(img_files, desc=f"Processing {cls}"):
        img_path = os.path.join(class_dir, img_name)
        mask_name = img_name.replace('.png', '_mask.png')
        mask_path = os.path.join(class_dir, mask_name)
        img = imread(img_path)
        img = resize(img, (IMG_SIZE, IMG_SIZE), preserve_range=True, anti_aliasing=True)
        # If grayscale, convert to 3 channels
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        if cls == "normal":
            # For normal, do not apply mask
            masked_img = img
        else:
            if not os.path.exists(mask_path):
                continue
            mask = imread(mask_path)
            mask = resize(mask, (IMG_SIZE, IMG_SIZE), preserve_range=True, anti_aliasing=False)
            mask = (mask > 0).astype(np.uint8)
            if mask.ndim == 2:
                mask3 = mask[..., None]
            else:
                mask3 = mask
            masked_img = img * mask3
        out_img_path = os.path.join(out_class_dir, img_name)
        imsave(out_img_path, masked_img.astype(np.uint8))
        rows.append({"img_path": out_img_path, "label": label_map[cls]})

subset_df = pd.DataFrame(rows)
subset_df = subset_df.sample(frac=1, random_state=42).reset_index(drop=True)
subset_df.to_csv(csv_file, index=False)

print('Class distribution in labels.csv:')
print(subset_df['label'].value_counts())
print('\nFirst 10 rows:')
print(subset_df.head(10))

print(f"Saved masked images to {out_dir}, labels to {csv_file}")