import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
import seaborn as sns

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_CLASSES = 3

def get_model(model_name, num_classes=3):
    if model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
        return model
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
    print("=== TESTING FIXED MODEL ON TEST SET ===\n")
    
    # Load test data
    test_dataset = MaskedTumorDataset('cnn_data/test.csv', transform=val_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Load model
    model = get_model('efficientnet_b0', num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load('fixed_best_model.pth', map_location=DEVICE))
    model.eval()
    
    # Test
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = (all_preds == all_labels).mean()
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"Test Set Accuracy: {accuracy:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)
    
    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_preds, 
                               target_names=["benign", "malignant", "normal"], 
                               zero_division=0))
    
    # ROC-AUC for each class
    print(f"\nROC-AUC Scores:")
    for i, class_name in enumerate(["benign", "malignant", "normal"]):
        try:
            auc = roc_auc_score((all_labels == i).astype(int), all_probs[:, i])
            print(f"  {class_name}: {auc:.4f}")
        except:
            print(f"  {class_name}: Error calculating AUC")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["benign", "malignant", "normal"],
                yticklabels=["benign", "malignant", "normal"])
    plt.title('Test Set Confusion Matrix - Fixed Model')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('test_confusion_matrix.png')
    print(f"\nTest confusion matrix saved as 'test_confusion_matrix.png'")
    
    # Per-class accuracy
    print(f"\nPer-Class Accuracy:")
    for i, class_name in enumerate(["benign", "malignant", "normal"]):
        class_mask = all_labels == i
        if class_mask.sum() > 0:
            class_acc = (all_preds[class_mask] == all_labels[class_mask]).mean()
            print(f"  {class_name}: {class_acc:.4f} ({class_mask.sum()} samples)")
    
    print(f"\n=== TEST COMPLETE ===")

if __name__ == "__main__":
    main() 