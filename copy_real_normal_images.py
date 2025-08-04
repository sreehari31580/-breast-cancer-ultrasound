import os
import shutil

src_dir = 'normal'
dst_dir = 'cnn_data/normal'
os.makedirs(dst_dir, exist_ok=True)

count = 0
for fname in os.listdir(src_dir):
    if fname.lower().endswith('.png') and '_mask' not in fname:
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)
        shutil.copy2(src_path, dst_path)
        count += 1
print(f'Copied {count} real normal images to {dst_dir}') 