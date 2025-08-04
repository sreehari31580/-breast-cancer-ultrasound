import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from skimage import io
from skimage.transform import resize
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_CLASSES = 3

MODEL_PATHS = {
    'efficientnet_b0': 'best_model_efficientnet_b0.pth',
    'efficientnet_b3': 'best_model_efficientnet_b3.pth',
    'efficientnet_b4': 'best_model_efficientnet_b4.pth',
    'densenet121': 'best_model_densenet121.pth',
}

def get_model(model_name, num_classes=3):
    if model_name == 'efficientnet_b0':
        effnet = models.efficientnet_b0(weights=None)
        num_features = effnet.classifier[1].in_features
        effnet.classifier[1] = nn.Linear(num_features, num_classes)
        return effnet
    elif model_name == 'efficientnet_b3':
        effnet = models.efficientnet_b3(weights=None)
        num_features = effnet.classifier[1].in_features
        effnet.classifier[1] = nn.Linear(num_features, num_classes)
        return effnet
    elif model_name == 'efficientnet_b4':
        effnet = models.efficientnet_b4(weights=None)
        num_features = effnet.classifier[1].in_features
        effnet.classifier[1] = nn.Linear(num_features, num_classes)
        return effnet
    elif model_name == 'densenet121':
        dnet = models.densenet121(weights=None)
        num_features = dnet.classifier.in_features
        dnet.classifier = nn.Linear(num_features, num_classes)
        return dnet
    else:
        raise ValueError(f"Unknown model: {model_name}")

class MaskedTumorDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row["img_path"]
        label = int(row["label"])
        img = io.imread(img_path)
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        img = resize(img, (IMG_SIZE, IMG_SIZE), preserve_range=True, anti_aliasing=True)
        img = (img / 255.0).astype(np.float32)
        if self.transform:
            img_uint8 = (img * 255).astype(np.uint8)
            img = self.transform(img_uint8).numpy()
        img = torch.tensor(img, dtype=torch.float32)
        return img, label

val_test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

def main():
    test_dataset = MaskedTumorDataset('cnn_data/test.csv', transform=val_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    y_true = test_dataset.data['label'].values
    
    print("=== INDIVIDUAL MODEL PERFORMANCE ===\n")
    
    for model_name in MODEL_PATHS:
        if not os.path.exists(MODEL_PATHS[model_name]):
            print(f"Model weights not found: {MODEL_PATHS[model_name]}")
            continue
            
        print(f"\n--- {model_name.upper()} ---")
        model = get_model(model_name, num_classes=NUM_CLASSES).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATHS[model_name], map_location=DEVICE))
        model.eval()
        
        y_pred = []
        with torch.no_grad():
            for imgs, _ in test_loader:
                imgs = imgs.to(DEVICE)
                outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                y_pred.extend(preds)
        
        y_pred = np.array(y_pred)
        
        # Print confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:")
        print(cm)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=["benign", "malignant", "normal"], zero_division=0))
        
        # Calculate accuracy
        accuracy = (y_pred == y_true).mean()
        print(f"Overall Accuracy: {accuracy:.3f}")
        print("-" * 50)

if __name__ == "__main__":
    main() 