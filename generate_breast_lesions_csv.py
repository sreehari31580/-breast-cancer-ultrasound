import pandas as pd
import os

# Paths
DATASET_DIR = 'BrEaST-Lesions_USG-images_and_masks'
EXCEL_FILE = os.path.join(DATASET_DIR, 'BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx')
CSV_OUT = os.path.join(DATASET_DIR, 'breast_lesions_labels.csv')

# Read Excel
df = pd.read_excel(EXCEL_FILE)

# Map class labels
def map_label(x):
    if str(x).strip().lower() == 'benign':
        return 0
    elif str(x).strip().lower() == 'malignant':
        return 1
    else:
        return -1  # Unknown/other

data = []
for idx, row in df.iterrows():
    img_file = str(row['Image_filename']).strip()
    mask_file = str(row['Mask_tumor_filename']).strip() if 'Mask_tumor_filename' in row and pd.notna(row['Mask_tumor_filename']) else ''
    label = map_label(row['Classification'])
    if label == -1 or not img_file:
        continue  # skip unknowns
    img_path = os.path.join(DATASET_DIR, img_file)
    mask_path = os.path.join(DATASET_DIR, mask_file) if mask_file else ''
    data.append({'img_path': img_path, 'label': label, 'mask_path': mask_path})

out_df = pd.DataFrame(data)
out_df.to_csv(CSV_OUT, index=False)
print(f"Wrote {len(out_df)} rows to {CSV_OUT}") 