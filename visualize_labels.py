import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from skimage import io
import os
from torch.utils.data import DataLoader
from src.cnn_lstm_resnet_train import MaskedTumorDataset

# 1. Load CSV
csv_path = 'cnn_data/labels.csv'
df = pd.read_csv(csv_path)
print('First 10 rows of CSV:')
print(df.head(10))

# 2. Visualize 5 random images from each class
plt.figure(figsize=(15, 9))
for class_idx in [0, 1, 2]:
    class_df = df[df['label'] == class_idx]
    print(f'Class {class_idx} has {len(class_df)} images')
    if len(class_df) == 0:
        continue
    sample_df = class_df.sample(n=min(5, len(class_df)), random_state=42)
    for i, (_, row) in enumerate(sample_df.iterrows()):
        img_path = row['img_path']
        try:
            img = io.imread(img_path)
            plt.subplot(3, 5, class_idx*5 + i + 1)
            if img.ndim == 2:
                plt.imshow(img, cmap='gray')
            else:
                plt.imshow(img)
            plt.title(f'Class {class_idx}\n{os.path.basename(img_path)}')
            plt.axis('off')
        except Exception as e:
            print(f'Error loading {img_path}: {e}')
plt.tight_layout()
plt.savefig('visualize_labels.png')
print('Saved grid as visualize_labels.png')

# 3. Print first batch from DataLoader
print('\nFirst batch from DataLoader:')
dataset = MaskedTumorDataset(csv_path, augment=False)
loader = DataLoader(dataset, batch_size=8, shuffle=False)
for imgs, labels in loader:
    print('Batch labels:', labels.tolist())
    break

# 4. Print file paths and labels for first 10 items in dataset
print('\nFirst 10 items from DataLoader:')
for i in range(10):
    img, label = dataset[i]
    img_path = df.iloc[i]['img_path']
    print(f'Index {i}: Path={img_path}, Label={label}') 