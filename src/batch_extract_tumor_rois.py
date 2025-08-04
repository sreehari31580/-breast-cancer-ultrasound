import os
import numpy as np
from skimage import io, segmentation, color
from skimage.transform import resize
from tqdm import tqdm

def get_mask_path(img_path):
    base, ext = os.path.splitext(img_path)
    mask_path = base + "_mask" + ext
    return mask_path

def slic_superpixels(img, n_segments=100, compactness=10):
    segments = segmentation.slic(img, n_segments=n_segments, compactness=compactness, start_label=1)
    return segments

def extract_tumor_superpixel_mask(img_path, n_segments=100, compactness=10, min_overlap=0.2):
    img = io.imread(img_path)
    img = resize(img, (224, 224), preserve_range=True, anti_aliasing=True).astype(np.uint8)
    segments = slic_superpixels(img, n_segments, compactness)

    mask_path = get_mask_path(img_path)
    if not os.path.exists(mask_path):
        return None

    mask = io.imread(mask_path)
    mask = resize(mask, (224, 224), preserve_range=True, anti_aliasing=False)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = (mask > 0).astype(np.uint8)

    tumor_superpixels = []
    for sp_idx in np.unique(segments):
        sp_mask = (segments == sp_idx).astype(np.uint8)
        overlap = np.sum(sp_mask * mask) / np.sum(sp_mask)
        if overlap > min_overlap:
            tumor_superpixels.append(sp_idx)

    tumor_roi_mask = np.isin(segments, tumor_superpixels).astype(np.uint8)
    return tumor_roi_mask

def save_mask(mask, out_path):
    mask = (mask * 255).astype(np.uint8)
    io.imsave(out_path, mask)

def process_all(data_root="data", out_root="tumor_masks", n_segments=100, compactness=10, min_overlap=0.2):
    os.makedirs(out_root, exist_ok=True)
    for cls in os.listdir(data_root):
        class_dir = os.path.join(data_root, cls)
        if not os.path.isdir(class_dir): continue
        out_class_dir = os.path.join(out_root, cls)
        os.makedirs(out_class_dir, exist_ok=True)
        img_files = [f for f in os.listdir(class_dir) if f.endswith(".png") and '_mask' not in f]
        for img_name in tqdm(img_files, desc=f"Processing {cls}"):
            img_path = os.path.join(class_dir, img_name)
            mask = extract_tumor_superpixel_mask(img_path, n_segments, compactness, min_overlap)
            if mask is not None:
                out_path = os.path.join(out_class_dir, img_name.replace('.png', '_roi.png'))
                save_mask(mask, out_path)

if __name__ == "__main__":
    process_all()
    print("All tumor ROI masks saved in the tumor_masks/ folder.")