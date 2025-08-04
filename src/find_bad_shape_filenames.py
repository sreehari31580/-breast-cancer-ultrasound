import os
from skimage import io

root = 'cnn_data'
bad_shape = (224, 3, 224)

for subdir, _, files in os.walk(root):
    for fname in files:
        if fname.endswith('.png'):
            fpath = os.path.join(subdir, fname)
            try:
                img = io.imread(fpath)
                if img.shape == bad_shape:
                    print(f'BAD SHAPE: {fpath} shape={img.shape}')
            except Exception as e:
                print(f'ERROR: {fpath} could not be read: {e}') 