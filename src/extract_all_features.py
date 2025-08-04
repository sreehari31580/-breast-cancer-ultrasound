import os
import numpy as np
from registration import multi_atlas_registration
from gabor_features import build_gabor_kernels, extract_gabor_features
from tqdm import tqdm

def get_atlas_paths(data_dir, class_name="benign", n=3):
    class_dir = os.path.join(data_dir, class_name)
    imgs = [os.path.join(class_dir, x) for x in os.listdir(class_dir) if x.endswith(".png") or x.endswith(".jpg")]
    return imgs[:n]

data_dir = "data"
output_file = "features/gabor_features.npy"
output_labels = "features/labels.npy"

os.makedirs("features", exist_ok=True)

# Choose atlas images from each class
atlas_paths = []
for c in ["benign", "malignant", "normal"]:
    atlas_paths.extend(get_atlas_paths(data_dir, c, n=1))  # 1 per class = 3 total

kernels = build_gabor_kernels()

X = []
y = []
label_map = {"benign":0, "malignant":1, "normal":2}

for c in ["benign", "malignant", "normal"]:
    class_dir = os.path.join(data_dir, c)
    img_files = [f for f in os.listdir(class_dir) if f.endswith(".png") or f.endswith(".jpg")]
    for img_name in tqdm(img_files, desc=f"Processing {c}"):
        img_path = os.path.join(class_dir, img_name)
        registered_imgs = multi_atlas_registration(img_path, atlas_paths)
        all_feats = []
        for reg_img in registered_imgs:
            feats = extract_gabor_features(reg_img, kernels)
            all_feats.append(feats)
        feature_vector = np.concatenate(all_feats)
        X.append(feature_vector)
        y.append(label_map[c])

X = np.array(X)
y = np.array(y)
np.save(output_file, X)
np.save(output_labels, y)
print(f"Saved features to {output_file} and labels to {output_labels}")