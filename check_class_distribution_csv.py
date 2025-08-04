import pandas as pd

# Load the labels CSV
csv_path = 'cnn_data/labels.csv'
df = pd.read_csv(csv_path)

# Print class distribution
print('Class distribution:')
print('Label 0 (benign):   ', (df['label'] == 0).sum())
print('Label 1 (malignant):', (df['label'] == 1).sum())
print('Label 2 (normal):   ', (df['label'] == 2).sum())
print('\nTotal samples:', len(df)) 