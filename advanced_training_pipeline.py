import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from torchvision import models, transforms
from skimage import io
from skimage.transform import resize
import random
import matplotlib.pyplot as plt
import seaborn as sns
import copy

IMG_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 15
NUM_CLASSES = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FREEZE_EPOCHS = 3

# Strong augmentation for all classes
def strong_augment(img):
    if np.random.rand() < 0.5:
        img = np.flip(img, axis=2)
    if np.random.rand() < 0.5:
        img = np.flip(img, axis=1)
    k = np.random.choice([0, 1, 2, 3])
    img = np.rot90(img, k, axes=(1, 2))
    img = img * (0.8 + 0.4 * np.random.rand())
    img = np.clip(img, 0, 1)
    return img

strong_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor()
])

class MaskedTumorDataset(nn.Module):
    def __init__(self, csv_file, augment=False):
        super().__init__()
        self.data = pd.read_csv(csv_file)
        self.augment = augment
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row["img_path"]
        label = int(row["label"])
        try:
            img = io.imread(img_path)
            img = img.astype(np.float32)
            # Handle grayscale
            if img.ndim == 2:
                img = np.stack([img]*3, axis=0)
            elif img.ndim == 3:
                if img.shape == (224, 3, 224):
                    img = img.transpose(1, 0, 2)
                if img.shape[2] == 3:
                    img = img.transpose(2, 0, 1)
                elif img.shape[2] == 4:
                    img = img[..., :3].transpose(2, 0, 1)
                elif img.shape[0] == 3:
                    pass
                elif img.shape[0] == 4:
                    img = img[:3, :, :]
                else:
                    img = img[..., 0]
                    if img.ndim == 2:
                        img = np.stack([img]*3, axis=0)
            # Resize each channel to 224x224
            img = np.stack([resize(img[c], (224, 224), preserve_range=True, anti_aliasing=True) for c in range(3)], axis=0)
            # Final check: force shape
            if img.shape != (3, 224, 224):
                img = np.zeros((3, 224, 224), dtype=np.float32)
            img = img / 255.0
            # --- Augmentation ---
            if self.augment:
                img_uint8 = (img * 255).astype(np.uint8)
                img = strong_transform(img_uint8).numpy()
            return torch.tensor(img, dtype=torch.float32), label
        except Exception as e:
            img = np.zeros((3, 224, 224), dtype=np.float32)
            return torch.tensor(img, dtype=torch.float32), label

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=3, freeze_backbone=False):
        super().__init__()
        effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for param in effnet.parameters():
                param.requires_grad = False
        num_features = effnet.classifier[1].in_features
        effnet.classifier[1] = nn.Linear(num_features, num_classes)
        self.effnet = effnet
    def forward(self, x):
        return self.effnet(x)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
    avg_loss = running_loss / total
    acc = correct / total
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    return avg_loss, acc, all_labels, all_preds

def main():
    print('=== Advanced Training Pipeline Started ===')
    df = pd.read_csv("cnn_data/tiny_balanced_subset.csv")
    print("Class distribution:", df['label'].value_counts().to_dict())
    train_df = df.copy()
    test_df = df.copy()
    train_labels = train_df["label"].values
    results = []
    settings = [
        ("CE, w=1", nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 1.0]).to(DEVICE)), [1.0, 1.0, 1.0]),
        ("CE, w=2", nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 2.0]).to(DEVICE)), [1.0, 2.0, 2.0]),
        ("CE, w=3", nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0, 3.0]).to(DEVICE)), [1.0, 3.0, 3.0]),
        ("Focal, w=2", FocalLoss(alpha=torch.tensor([1.0, 2.0, 2.0]).to(DEVICE)), [1.0, 2.0, 2.0]),
        ("Focal, w=3", FocalLoss(alpha=torch.tensor([1.0, 3.0, 3.0]).to(DEVICE)), [1.0, 3.0, 3.0]),
    ]
    best_macro_f1 = -1
    best_min_recall = -1
    best_model = None
    best_setting = None
    best_metrics = None
    for name, criterion, wts in settings:
        print(f"\n=== Training with {name} ===")
        samples_weight = np.ones_like(train_labels, dtype=np.float32)
        for i, w in enumerate(wts):
            samples_weight[train_labels == i] *= w
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
        model = EfficientNetClassifier(num_classes=NUM_CLASSES, freeze_backbone=True).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3)
        train_loader = DataLoader(MaskedTumorDataset("cnn_data/tiny_balanced_subset.csv", augment=True), batch_size=BATCH_SIZE, sampler=sampler)
        test_loader = DataLoader(MaskedTumorDataset("cnn_data/tiny_balanced_subset.csv", augment=False), batch_size=BATCH_SIZE, shuffle=False)
        for epoch in range(NUM_EPOCHS):
            if epoch == FREEZE_EPOCHS:
                print("Unfreezing EfficientNet backbone for fine-tuning.")
                for param in model.effnet.parameters():
                    param.requires_grad = True
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
            test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion)
            scheduler.step(test_loss)
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}: Train loss {train_loss:.4f}, acc {train_acc:.4f} | Test loss {test_loss:.4f}, acc {test_acc:.4f}")
        _, _, y_true, y_pred = evaluate(model, test_loader, criterion)
        report = classification_report(y_true, y_pred, target_names=["benign", "malignant", "normal"], zero_division=0, output_dict=True)
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        recalls = [report[cls]['recall'] for cls in ['benign', 'malignant', 'normal']]
        min_recall = min(recalls)
        results.append((name, macro_f1, min_recall, recalls, wts))
        if macro_f1 > best_macro_f1 or (macro_f1 == best_macro_f1 and min_recall > best_min_recall):
            best_macro_f1 = macro_f1
            best_min_recall = min_recall
            best_model = copy.deepcopy(model)
            best_setting = name
            best_metrics = (y_true, y_pred, report)
    print(f"\n=== Best model used: {best_setting} ===")
    torch.save(best_model.state_dict(), "model_resnet_advanced.pth")
    print("Model saved as model_resnet_advanced.pth")
    y_true, y_pred, report = best_metrics
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["benign", "malignant", "normal"], zero_division=0))
    y_true_onehot = np.eye(NUM_CLASSES)[y_true]
    y_pred_proba = []
    best_model.eval()
    with torch.no_grad():
        test_loader = DataLoader(MaskedTumorDataset("cnn_data/tiny_balanced_subset.csv", augment=False), batch_size=BATCH_SIZE, shuffle=False)
        for imgs, _ in test_loader:
            imgs = imgs.to(DEVICE)
            outputs = best_model(imgs)
            y_pred_proba.append(torch.softmax(outputs, dim=1).cpu().numpy())
    y_pred_proba = np.concatenate(y_pred_proba, axis=0)
    for i, name in enumerate(["benign", "malignant", "normal"]):
        try:
            auc = roc_auc_score(y_true_onehot[:, i], y_pred_proba[:, i])
            print(f"ROC-AUC for {name}: {auc:.3f}")
        except Exception as e:
            print(f"ROC-AUC for {name}: Error ({e})")
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["benign", "malignant", "normal"], yticklabels=["benign", "malignant", "normal"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix Heatmap')
    plt.tight_layout()
    plt.savefig('confusion_matrix_heatmap_advanced.png')
    print("Confusion matrix heatmap saved as confusion_matrix_heatmap_advanced.png")
    print("\n=== Summary of all settings ===")
    print("Setting\tMacro F1\tMin Recall\tRecalls [benign, malignant, normal]\tWeights")
    for name, macro_f1, min_recall, recalls, wts in results:
        print(f"{name}\t{macro_f1:.3f}\t{min_recall:.3f}\t{[f'{r:.2f}' for r in recalls]}\t{wts}")
    print('=== Training script completed ===')

if __name__ == "__main__":
    main() 