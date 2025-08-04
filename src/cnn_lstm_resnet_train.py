import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
from skimage import io
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torchvision import models, transforms
import random
from torch.utils.data._utils.collate import default_collate
from skimage.transform import resize
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import seaborn as sns
import copy
import argparse

IMG_SIZE = 224  # ResNet expects 224x224
BATCH_SIZE = 16
NUM_EPOCHS = 10
NUM_CLASSES = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === CONFIGURATION TOGGLES ===
USE_FOCAL_LOSS = False  # Toggle for Focal Loss
FREEZE_RESNET = True   # Toggle for transfer learning freeze/unfreeze
FREEZE_EPOCHS = 3      # Number of epochs to keep ResNet frozen
MALIGNANT_WEIGHT = 5.0 # Heavily penalize malignant errors

# Model selection
MODEL_CHOICES = ['efficientnet_b0', 'efficientnet_b3', 'efficientnet_b4', 'densenet121']

def random_augment(img):
    if random.random() > 0.5:
        img = np.flip(img, axis=0)
    if random.random() > 0.5:
        img = np.flip(img, axis=1)
    if random.random() > 0.7:
        k = random.choice([1,2,3])
        img = np.rot90(img, k)
    return img.copy()

def strong_augment(img):
    # Example: random flip, rotation, color jitter
    if np.random.rand() < 0.5:
        img = np.flip(img, axis=2)  # horizontal flip
    if np.random.rand() < 0.5:
        img = np.flip(img, axis=1)  # vertical flip
    # Random rotation (0, 90, 180, 270)
    k = np.random.choice([0, 1, 2, 3])
    img = np.rot90(img, k, axes=(1, 2))
    # Color jitter (brightness)
    img = img * (0.8 + 0.4 * np.random.rand())
    img = np.clip(img, 0, 1)
    return img

def double_strong_augment(img):
    # Apply strong augmentation twice for benign
    img = strong_augment(img)
    img = strong_augment(img)
    return img

def light_augment(img):
    # Only light augmentation (e.g., horizontal flip)
    if np.random.rand() < 0.5:
        img = np.flip(img, axis=2)
    return img

# --- Strong augmentation for malignant ---
malignant_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

# --- Light augmentation for other classes ---
light_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

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
            if self.transform:
                img_uint8 = (img * 255).astype(np.uint8)
                img = self.transform(img_uint8).numpy()
            return torch.tensor(img, dtype=torch.float32), label
        except Exception as e:
            img = np.zeros((3, 224, 224), dtype=np.float32)
        return torch.tensor(img, dtype=torch.float32), label

def custom_collate(batch):
    from torch.utils.data.dataloader import default_collate
    return default_collate(batch)

class EfficientNetLSTM(nn.Module):
    def __init__(self, num_classes=3, lstm_hidden=256, lstm_layers=1, bidirectional=True, H=7, W=7, freeze_backbone=False):
        super(EfficientNetLSTM, self).__init__()
        effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(effnet.features))  # Output: (B, 1280, 7, 7)
        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional
        self.num_classes = num_classes
        self.H = H
        self.W = W
        self.input_size = W * 1280
            self.lstm = nn.LSTM(
            input_size=self.input_size,
                hidden_size=self.lstm_hidden,
                num_layers=self.lstm_layers,
                batch_first=True,
                bidirectional=self.bidirectional
        )
            fc_input_dim = H * self.lstm_hidden * (2 if self.bidirectional else 1)
            self.fc = nn.Sequential(
                nn.Linear(fc_input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(128, self.num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)  # (B, 1280, 7, 7)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)     # (B, 7, 7, 1280)
        x = x.reshape(B, H, W * C)    # (B, 7, 7*1280)
        lstm_out, _ = self.lstm(x)    # (B, 7, hidden*bidirectional)
        lstm_out = lstm_out.reshape(B, -1)
        out = self.fc(lstm_out)
        return out

def compute_class_weights(labels):
    class_sample_count = np.array([np.sum(labels == t) for t in np.unique(labels)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in labels])
    return samples_weight

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

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def get_model(model_name, num_classes=3, freeze_backbone=False):
    if model_name == 'efficientnet_b0':
        effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for param in effnet.parameters():
                param.requires_grad = False
        num_features = effnet.classifier[1].in_features
        effnet.classifier[1] = nn.Linear(num_features, num_classes)
        return effnet
    elif model_name == 'efficientnet_b3':
        effnet = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for param in effnet.parameters():
                param.requires_grad = False
        num_features = effnet.classifier[1].in_features
        effnet.classifier[1] = nn.Linear(num_features, num_classes)
        return effnet
    elif model_name == 'efficientnet_b4':
        effnet = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for param in effnet.parameters():
                param.requires_grad = False
        num_features = effnet.classifier[1].in_features
        effnet.classifier[1] = nn.Linear(num_features, num_classes)
        return effnet
    elif model_name == 'densenet121':
        dnet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for param in dnet.parameters():
                param.requires_grad = False
        num_features = dnet.classifier.in_features
        dnet.classifier = nn.Linear(num_features, num_classes)
        return dnet
    else:
        raise ValueError(f"Unknown model: {model_name}")

# Augmentation transforms
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])
val_test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

# Main training logic
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='efficientnet_b0', choices=MODEL_CHOICES)
    parser.add_argument('--freeze_backbone', action='store_true')
    args = parser.parse_args()

    train_df = pd.read_csv('cnn_data/train.csv')
    val_df = pd.read_csv('cnn_data/val.csv')
    test_df = pd.read_csv('cnn_data/test.csv')

    train_dataset = MaskedTumorDataset('cnn_data/train.csv', transform=train_transform)
    val_dataset = MaskedTumorDataset('cnn_data/val.csv', transform=val_test_transform)
    test_dataset = MaskedTumorDataset('cnn_data/test.csv', transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = get_model(args.model, num_classes=3, freeze_backbone=args.freeze_backbone).to(DEVICE)

    # Example training loop with early stopping and LR scheduling
    PATIENCE = 5  # Early stopping patience
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    num_epochs = NUM_EPOCHS  # or set as desired

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2)

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        train_loss = np.mean(train_losses)
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                val_losses.append(loss.item())
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_loss = np.mean(val_losses)
        val_acc = val_correct / val_total

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}: Train loss {train_loss:.4f}, acc {train_acc:.4f} | Val loss {val_loss:.4f}, acc {val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}. Best epoch was {best_epoch+1}.")
                break

    # Load best model for final evaluation
    torch.save(model.state_dict(), 'final_model.pth')
    model.load_state_dict(torch.load('best_model.pth'))
    print("Best model loaded for test evaluation.")

    # ... test evaluation code ...