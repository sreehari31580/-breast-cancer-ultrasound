import os
import pandas as pd

# Use only 5 images from each class for a quick overfit test (unmasked images)
root = 'Dataset_BUSI_with_GT'
rows = []
label_map = {'benign': 0, 'malignant': 1, 'normal': 2}

for cls in ['benign', 'malignant', 'normal']:
    class_dir = os.path.join(root, cls)
    img_files = [f for f in os.listdir(class_dir) if f.endswith('.png') and '_mask' not in f]
    # Pick 5 images per class
    for img_name in sorted(img_files)[:5]:
        img_path = os.path.join(class_dir, img_name)
        rows.append({'img_path': img_path, 'label': label_map[cls]})

subset_df = pd.DataFrame(rows)
subset_df = subset_df.sample(frac=1, random_state=42).reset_index(drop=True)
subset_df.to_csv('cnn_data/tiny_unmasked_subset.csv', index=False)

print('Class distribution in tiny_unmasked_subset.csv:')
print(subset_df['label'].value_counts())
print('\nFirst 10 rows:')
print(subset_df.head(10)) 