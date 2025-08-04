import os
import numpy as np
from skimage import io, segmentation, color
from skimage.transform import resize
import matplotlib.pyplot as plt

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
        print(f"Mask not found for {img_path}")
        return None, None, None

    mask = io.imread(mask_path)
    mask = resize(mask, (224, 224), preserve_range=True, anti_aliasing=False) # <<< FIXED HERE
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

    tumor_overlay = img.copy()
    tumor_overlay[tumor_roi_mask == 1] = [255, 0, 0] if img.ndim == 3 else 255

    return segments, tumor_roi_mask, tumor_overlay

if __name__ == "__main__":
    data_dir = "data/benign"
    img_name = [f for f in os.listdir(data_dir) if f.endswith('.png') and '_mask' not in f][0]
    img_path = os.path.join(data_dir, img_name)
    segments, tumor_roi_mask, tumor_overlay = extract_tumor_superpixel_mask(img_path)

    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(io.imread(img_path), cmap='gray')
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.title("Tumor ROI mask (from superpixels)")
    plt.imshow(tumor_roi_mask, cmap='gray')
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.title("Tumor Overlay (red)")
    plt.imshow(tumor_overlay, cmap='gray')
    plt.axis('off')
    plt.show()