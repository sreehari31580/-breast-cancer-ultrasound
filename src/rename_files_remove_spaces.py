import os

def rename_files_in_folder(folder):
    for fname in os.listdir(folder):
        new_fname = fname.replace(' ', '')
        if new_fname != fname:
            os.rename(os.path.join(folder, fname), os.path.join(folder, new_fname))

folders = [
    "data/benign",
    "data/malignant",
    "tumor_masks/benign",
    "tumor_masks/malignant"
]

for folder in folders:
    if os.path.exists(folder):
        rename_files_in_folder(folder)
        print(f"Renamed files in {folder}")

print("All files renamed to remove spaces.")