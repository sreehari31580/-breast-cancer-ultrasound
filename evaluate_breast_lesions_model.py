import torch
import torch.nn.functional as F
from torchvision import models
import pandas as pd
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.metrics import classification_report, confusion_matrix
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 224
MODEL_PATH = 'fixed_best_model.pth'
CLASS_NAMES = ['Benign', 'Malignant']
CSV_PATH = 'BrEaST-Lesions_USG-images_and_masks/breast_lesions_labels.csv'

# Load model
def load_model():
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    model.to(DEVICE)
    model.eval()
    return model

def preprocess_image(image, mask_path=None):
    if image.ndim == 2:
        image = np.stack([image]*3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    if mask_path and os.path.exists(mask_path):
        mask = imread(mask_path)
        mask = resize(mask, (image.shape[0], image.shape[1]), preserve_range=True, anti_aliasing=False)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = (mask > 0).astype(np.uint8)
        mask3 = mask[..., None] if mask.ndim == 2 else mask
        image = image * mask3
    image = resize(image, (IMG_SIZE, IMG_SIZE), preserve_range=True, anti_aliasing=True)
    if image.max() > 1:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)
    image_tensor = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
    return image_tensor.unsqueeze(0).to(DEVICE)

def main():
    df = pd.read_csv(CSV_PATH)
    model = load_model()
    y_true, y_pred = [], []
    for i, row in df.iterrows():
        img_path = row['img_path']
        label = int(row['label'])
        mask_path = row['mask_path'] if 'mask_path' in row and pd.notna(row['mask_path']) and row['mask_path'] else None
        if not os.path.exists(img_path):
            print(f"Missing: {img_path}")
            continue
        image = imread(img_path)
        input_tensor = preprocess_image(image, mask_path)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            pred = torch.argmax(probabilities, dim=1).item()
        y_true.append(label)
        y_pred.append(pred)
        if (i+1) % 25 == 0:
            print(f"Processed {i+1}/{len(df)}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main() 