import os
from skimage import io

root = 'cnn_data'
allowed_shape = (224, 224, 3)
shapes = {}
problem_files = []

for subdir, _, files in os.walk(root):
    for fname in files:
        if fname.endswith('.png'):
            fpath = os.path.join(subdir, fname)
            try:
                img = io.imread(fpath)
                shape = img.shape
                shapes.setdefault(shape, []).append(fpath)
                if shape != allowed_shape:
                    problem_files.append((fpath, shape))
            except Exception as e:
                print(f'Error reading {fpath}: {e}')

print('Unique shapes found:')
for shape, files in shapes.items():
    print(f'{shape}: {len(files)} images')

print('\nFiles with non-(224, 224, 3) shape:')
for fpath, shape in problem_files:
    print(f'{fpath}: {shape}') 