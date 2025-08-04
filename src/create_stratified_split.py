import pandas as pd
from sklearn.model_selection import train_test_split

# Load the full labels CSV
labels_df = pd.read_csv('cnn_data/labels.csv')

# First split: train vs temp (val+test)
train_df, temp_df = train_test_split(
    labels_df, test_size=0.3, stratify=labels_df['label'], random_state=42)

# Second split: val vs test (half-half of temp)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

# Save splits
train_df.to_csv('cnn_data/train.csv', index=False)
val_df.to_csv('cnn_data/val.csv', index=False)
test_df.to_csv('cnn_data/test.csv', index=False)

print('Train set:', train_df['label'].value_counts().to_dict())
print('Val set:', val_df['label'].value_counts().to_dict())
print('Test set:', test_df['label'].value_counts().to_dict()) 