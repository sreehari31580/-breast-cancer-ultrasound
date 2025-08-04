import os
from skimage import io
import numpy as np

normal_dir = 'cnn_data/normal'
blank_images = []

for fname in os.listdir(normal_dir):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        continue
    fpath = os.path.join(normal_dir, fname)
    try:
        img = io.imread(fpath)
        if np.all(img == 0):
            blank_images.append(fpath)
    except Exception as e:
        print(f'Error reading {fpath}: {e}')

print(f'Found {len(blank_images)} blank images in {normal_dir}:')
for f in blank_images:
    print(f)

with open('blank_normal_images.txt', 'w') as f:
    for fname in blank_images:
        f.write(fname + '\n')
print('Saved list to blank_normal_images.txt') 