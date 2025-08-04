import os
import numpy as np
from registration import multi_atlas_registration
from gabor_features import build_gabor_kernels, extract_gabor_features

def get_atlas_paths(data_dir, class_name="benign", n=3):
    class_dir = os.path.join(data_dir, class_name)
    imgs = [os.path.join(class_dir, x) for x in os.listdir(class_dir) if x.endswith(".png") or x.endswith(".jpg")]
    return imgs[:n]

data_dir = "data"  # <---- fix this line!
atlas_paths = get_atlas_paths(data_dir, "benign", n=3)
kernels = build_gabor_kernels()

img_path = os.path.join(data_dir, "benign", "benign (1).png") # Example
registered_imgs = multi_atlas_registration(img_path, atlas_paths)

all_feats = []
for reg_img in registered_imgs:
    feats = extract_gabor_features(reg_img, kernels)
    all_feats.append(feats)
feature_vector = all_feats[0] if len(all_feats) == 1 else np.concatenate(all_feats)
print("Feature vector shape:", feature_vector.shape)