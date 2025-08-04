import pandas as pd
import numpy as np
from skimage import io
from src.cnn_lstm_resnet_train import MaskedTumorDataset
import random

csv_path = 'cnn_data/labels.csv'
df = pd.read_csv(csv_path)
dataset = MaskedTumorDataset(csv_path, augment=False)

print('Pixel value statistics after preprocessing (per class):')
for class_idx in [0, 1, 2]:
    class_df = df[df['label'] == class_idx]
    print(f'\nClass {class_idx} ({len(class_df)} images)')
    if len(class_df) == 0:
        continue
    sample_df = class_df.sample(n=min(3, len(class_df)), random_state=42)
    for i, (row_idx, row) in enumerate(sample_df.iterrows()):
        img_path = row['img_path']
        # Get preprocessed image from dataset
        img_tensor, label = dataset[row_idx]
        img_np = img_tensor.numpy()
        print(f'  Image {i+1}: {img_path}')
        print(f'    Label: {label}')
        print(f'    Shape: {img_np.shape}')
        print(f'    Min: {img_np.min():.4f}, Max: {img_np.max():.4f}, Mean: {img_np.mean():.4f}, Std: {img_np.std():.4f}')
        for c in range(img_np.shape[0]):
            print(f'      Channel {c}: min={img_np[c].min():.4f}, max={img_np[c].max():.4f}, mean={img_np[c].mean():.4f}, std={img_np[c].std():.4f}')
        # Also print stats for raw image
        raw_img = io.imread(img_path)
        print(f'    [Raw] Shape: {raw_img.shape}, dtype: {raw_img.dtype}')
        print(f'    [Raw] Min: {raw_img.min()}, Max: {raw_img.max()}, Mean: {raw_img.mean():.2f}, Std: {raw_img.std():.2f}') 