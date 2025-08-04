import os
import shutil
from skimage import io
import numpy as np

src_dir = 'normal'
img_dst = 'cnn_data/normal'
mask_dst = 'cnn_data/normal_masks'
os.makedirs(img_dst, exist_ok=True)
os.makedirs(mask_dst, exist_ok=True)

img_count = 0
mask_count = 0
for fname in os.listdir(src_dir):
    if not fname.lower().endswith('.png'):
        continue
    fpath = os.path.join(src_dir, fname)
    try:
        img = io.imread(fpath)
        # Heuristic: mask if single channel or only 0/255 values
        if (img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1)) or np.isin(img, [0, 255]).all():
            shutil.copy2(fpath, os.path.join(mask_dst, fname))
            mask_count += 1
        else:
            shutil.copy2(fpath, os.path.join(img_dst, fname))
            img_count += 1
    except Exception as e:
        print(f'Error reading {fpath}: {e}')

print(f'Total real images copied to {img_dst}: {img_count}')
print(f'Total masks copied to {mask_dst}: {mask_count}') 