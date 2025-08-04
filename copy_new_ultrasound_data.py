import os
import shutil

# Define source and destination directories
splits = ['train', 'val']
classes = ['benign', 'malignant']

src_root = 'ultrasound breast classification'
dst_root = 'data'

summary = {}

for split in splits:
    for cls in classes:
        src_dir = os.path.join(src_root, split, cls)
        dst_dir = os.path.join(dst_root, cls)
        if not os.path.exists(src_dir):
            continue
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        copied = 0
        for fname in os.listdir(src_dir):
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(dst_dir, fname)
            # Only copy if not already present
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
                copied += 1
        summary[(split, cls)] = copied
        print(f"Copied {copied} new files from {src_dir} to {dst_dir}")

print("\nSummary:")
for (split, cls), count in summary.items():
    print(f"{split}/{cls}: {count} files copied")

print("\nDone. You can now run: python src/prepare_masked_cnn_dataset.py") 