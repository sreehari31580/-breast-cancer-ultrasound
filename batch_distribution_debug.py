import pandas as pd
from torch.utils.data import DataLoader
from collections import Counter
from src.cnn_lstm_resnet_train import MaskedTumorDataset

csv_path = 'cnn_data/labels.csv'
dataset = MaskedTumorDataset(csv_path, augment=False)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

print('Batch class distributions (first 10 batches):')
summary = []
for batch_idx, (imgs, labels) in enumerate(loader):
    label_counts = Counter(labels.tolist())
    print(f'Batch {batch_idx}:', dict(label_counts))
    summary.append(f'Batch {batch_idx}: {dict(label_counts)}')
    if batch_idx >= 9:
        break

with open('batch_distribution_debug.txt', 'w') as f:
    for line in summary:
        f.write(line + '\n')
print('Saved batch distribution summary to batch_distribution_debug.txt') 