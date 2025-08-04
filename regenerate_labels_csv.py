import os
import pandas as pd

rows = []
for label, folder in zip([0, 1, 2], ['benign', 'malignant', 'normal']):
    dir_path = os.path.join('cnn_data', folder)
    for fname in os.listdir(dir_path):
        if fname.lower().endswith('.png'):
            img_path = os.path.join('cnn_data', folder, fname)
            rows.append({'img_path': img_path, 'label': label})
df = pd.DataFrame(rows)
df.to_csv('cnn_data/labels.csv', index=False)
print(f'Regenerated cnn_data/labels.csv with {len(df)} entries.') 