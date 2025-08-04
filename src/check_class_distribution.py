import numpy as np

y = np.load("features/labels_masked.npy")
print("Class counts (benign, malignant, normal):", np.bincount(y))