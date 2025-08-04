import os
from skimage import io

root = 'cnn_data'
expected_shape = (224, 224, 3)
bad_shapes = []

for subdir, _, files in os.walk(root):
    for fname in files:
        if fname.endswith('.png'):
            fpath = os.path.join(subdir, fname)
            try:
                img = io.imread(fpath)
                if img.shape != expected_shape:
                    print(f'BAD SHAPE: {fpath} shape={img.shape}')
                    bad_shapes.append((fpath, img.shape))
            except Exception as e:
                print(f'ERROR: {fpath} could not be read: {e}')

print(f'\nTotal images with bad shape: {len(bad_shapes)}') 