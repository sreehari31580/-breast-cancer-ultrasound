import os
import numpy as np
from skimage import io
from skimage.transform import resize
from registration import multi_atlas_registration
from gabor_features import build_gabor_kernels, extract_gabor_features

def extract_masked_gabor_features(img, mask, kernels):
    masked_img = img * mask
    features = extract_gabor_features(masked_img, kernels)
    return features

def get_atlas_paths(data_dir, class_name="benign", n=3):
    class_dir = os.path.join(data_dir, class_name)
    imgs = [os.path.join(class_dir, x) for x in os.listdir(class_dir) if x.endswith(".png") and '_mask' not in x]
    return imgs[:n]

data_dir = "data"
masks_dir = "tumor_masks"
output_file = "features/gabor_features_masked.npy"
output_labels = "features/labels_masked.npy"

os.makedirs("features", exist_ok=True)

# Choose atlas images
atlas_paths = []
for c in ["benign", "malignant", "normal"]:
    atlas_paths.extend(get_atlas_paths(data_dir, c, n=1))

kernels = build_gabor_kernels()
X = []
y = []
label_map = {"benign": 0, "malignant": 1, "normal": 2}

for c in ["benign", "malignant", "normal"]:
    class_dir = os.path.join(data_dir, c)
    mask_class_dir = os.path.join(masks_dir, c)
    img_files = [f for f in os.listdir(class_dir) if f.endswith(".png") and '_mask' not in f]
    print(f"Processing {len(img_files)} images for class {c}")
    for img_name in img_files:
        img_path = os.path.join(class_dir, img_name)
        mask_name = img_name.replace('.png', '_roi.png')
        mask_path = os.path.join(mask_class_dir, mask_name)
        if not os.path.exists(mask_path):
            print(f"Skipping {img_name} (no ROI mask found)")
            continue  # skip if no mask

        mask = io.imread(mask_path)
        mask = resize(mask, (224, 224), preserve_range=True, anti_aliasing=True)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = (mask > 127).astype(np.uint8)

        # Pass img_path, not img array
        registered_imgs = multi_atlas_registration(img_path, atlas_paths)
        all_feats = []
        for reg_img in registered_imgs:
            reg_img = resize(reg_img, (224, 224), preserve_range=True, anti_aliasing=True).astype(np.uint8)
            feats = extract_masked_gabor_features(reg_img, mask, kernels)
            all_feats.append(feats)
        feature_vector = np.concatenate(all_feats)
        X.append(feature_vector)
        y.append(label_map[c])
        print(f"Processed: {img_name} ({c})")

X = np.array(X)
y = np.array(y)
np.save(output_file, X)
np.save(output_labels, y)
print(f"Masked Gabor features saved to {output_file} and labels to {output_labels}")
