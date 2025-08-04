import os
from skimage import io, segmentation, color
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt

def slic_superpixels(img_path, n_segments=100, compactness=10):
    img = io.imread(img_path)
    img = resize(img, (224, 224), preserve_range=True, anti_aliasing=True).astype(np.uint8)
    segments = segmentation.slic(img, n_segments=n_segments, compactness=compactness, start_label=1)
    superpixel_img = color.label2rgb(segments, img, kind='avg')
    return segments, superpixel_img

# Example usage
if __name__ == "__main__":
    data_dir = "data/benign"
    img_name = os.listdir(data_dir)[0]
    img_path = os.path.join(data_dir, img_name)
    segments, superpixel_img = slic_superpixels(img_path)
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title("Original")
    plt.imshow(io.imread(img_path), cmap='gray')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.title("Superpixels")
    plt.imshow(superpixel_img.astype(np.uint8))
    plt.axis('off')
    plt.show()