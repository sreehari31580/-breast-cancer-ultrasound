import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from skimage import io
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from src.cnn_lstm_resnet_train import MaskedTumorDataset

print("=== COMPREHENSIVE DATA DEBUGGING ===")

# 1. Check CSV and splits
print("\n1. Checking CSV and train/test splits...")
df = pd.read_csv("cnn_data/labels.csv")
print(f"Total samples: {len(df)}")
print(f"Class distribution: {df['label'].value_counts().to_dict()}")

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
print(f"Train samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print(f"Train class distribution: {train_df['label'].value_counts().to_dict()}")
print(f"Test class distribution: {test_df['label'].value_counts().to_dict()}")

# 2. Check for data leakage
print("\n2. Checking for data leakage...")
train_paths = set(train_df['img_path'].tolist())
test_paths = set(test_df['img_path'].tolist())
leakage = train_paths.intersection(test_paths)
print(f"Images in both train and test: {len(leakage)}")

# 3. Check if images are actually different
print("\n3. Checking image differences between classes...")
sample_images = {}
for label in [0, 1, 2]:
    class_df = df[df['label'] == label]
    if len(class_df) > 0:
        sample_path = class_df.iloc[0]['img_path']
        try:
            img = io.imread(sample_path)
            sample_images[label] = img
            print(f"Class {label}: Image shape {img.shape}, dtype {img.dtype}")
        except Exception as e:
            print(f"Class {label}: Error reading image - {e}")

# 4. Test DataLoader
print("\n4. Testing DataLoader...")
test_dataset = MaskedTumorDataset("cnn_data/test.csv", augment=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

print("First batch from DataLoader:")
for i, (imgs, labels) in enumerate(test_loader):
    print(f"Batch {i}: Images shape {imgs.shape}, Labels {labels.tolist()}")
    if i >= 2:  # Only check first 3 batches
        break

# 5. Visualize sample images
print("\n5. Visualizing sample images...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, label in enumerate([0, 1, 2]):
    class_df = df[df['label'] == label]
    if len(class_df) > 0:
        sample_path = class_df.iloc[0]['img_path']
        try:
            img = io.imread(sample_path)
            if img.ndim == 3 and img.shape[2] == 3:
                axes[0, i].imshow(img)
            else:
                axes[0, i].imshow(img, cmap='gray')
            axes[0, i].set_title(f'Class {label} (Original)')
            axes[0, i].axis('off')
            
            # Show processed image from DataLoader
            dataset = MaskedTumorDataset("cnn_data/labels.csv", augment=False)
            for j, (img_tensor, label_tensor) in enumerate(dataset):
                if label_tensor == label:
                    img_np = img_tensor.numpy().transpose(1, 2, 0)
                    axes[1, i].imshow(img_np)
                    axes[1, i].set_title(f'Class {label} (Processed)')
                    axes[1, i].axis('off')
                    break
        except Exception as e:
            axes[0, i].text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
            axes[0, i].set_title(f'Class {label} (Error)')

plt.tight_layout()
plt.savefig('debug_images.png')
print("Debug images saved as debug_images.png")

# 6. Check file paths
print("\n6. Checking file paths...")
missing_files = 0
for idx, row in df.iterrows():
    if not os.path.exists(row['img_path']):
        missing_files += 1
        if missing_files <= 5:  # Only print first 5 missing files
            print(f"Missing file: {row['img_path']}")
print(f"Total missing files: {missing_files}")

# 7. Check for duplicate filenames
print("\n7. Checking for duplicate filenames...")
filenames = df['img_path'].apply(os.path.basename).tolist()
duplicates = [x for x in set(filenames) if filenames.count(x) > 1]
print(f"Duplicate filenames: {len(duplicates)}")
if duplicates:
    print("First few duplicates:", duplicates[:5])

print("\n=== DEBUGGING COMPLETE ===")
print("Check the output above and the debug_images.png file.")
print("If all images look the same, that's the problem!")
print("If test set has no benign/normal, that's the problem!")
print("If DataLoader returns wrong labels, that's the problem!") 