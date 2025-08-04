import pandas as pd

# 1. Load the full CSV
df = pd.read_csv('cnn_data/labels.csv')

# 2. Randomly select 20 images from each class
balanced_df = pd.concat([
    df[df['label'] == 0].sample(n=20, random_state=42),
    df[df['label'] == 1].sample(n=20, random_state=42),
    df[df['label'] == 2].sample(n=20, random_state=42)
], ignore_index=True)

# 3. Shuffle the new DataFrame
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 4. Save to new CSV
balanced_df.to_csv('cnn_data/balanced_subset.csv', index=False)

# 5. Print class distribution and first 10 rows
print('Class distribution in balanced_subset.csv:')
print(balanced_df['label'].value_counts())
print('\nFirst 10 rows:')
print(balanced_df.head(10)) 