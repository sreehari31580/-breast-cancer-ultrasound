import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load features and labels
X = np.load("features/gabor_features_masked.npy")
y = np.load("features/labels_masked.npy")

# Define label names and all classes
label_names = ["benign", "malignant", "normal"]
all_labels = [0, 1, 2]

# Split the data with stratification to ensure all classes appear in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}\n")

# Confusion Matrix (show all classes even if some are missing in y_test)
cm = confusion_matrix(y_test, y_pred, labels=all_labels)
print("Confusion Matrix:")
print(cm)
print()

# Classification Report (show all classes)
print("Classification Report:")
print(classification_report(
    y_test, y_pred,
    labels=all_labels,
    target_names=label_names,
    zero_division=0  # Avoid warnings for missing classes
))