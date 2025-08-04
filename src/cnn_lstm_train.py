import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from skimage import io
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from skimage.transform import resize

IMG_SIZE = 128
BATCH_SIZE = 16
NUM_EPOCHS = 10
NUM_CLASSES = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

class SimpleCNNLSTM(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleCNNLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 64x64
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 32x32
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 16x16
        )
        self.lstm = nn.LSTM(input_size=16, hidden_size=32, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(32*16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.cnn(x)  # (B, 64, 16, 16)
        x = x.permute(0, 2, 3, 1)  # (B, 16, 16, 64)
        x = x.reshape(x.size(0), 16, -1)  # (B, 16, 1024)
        x = x[:, :, :16]  # Reduce dim for LSTM input (B, 16, 16)
        lstm_out, _ = self.lstm(x)  # (B, 16, 32)
        lstm_out = lstm_out.contiguous().view(x.size(0), -1)  # (B, 32*16)
        out = self.fc(lstm_out)
        return out

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
    df = pd.read_csv("cnn_data/labels.csv")
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    train_df.to_csv("cnn_data/train.csv", index=False)
    test_df.to_csv("cnn_data/test.csv", index=False)

    train_dataset = MaskedTumorDataset("cnn_data/train.csv")
    test_dataset = MaskedTumorDataset("cnn_data/test.csv")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleCNNLSTM(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}: Train loss {train_loss:.4f}, acc {train_acc:.4f} | Test loss {test_loss:.4f}, acc {test_acc:.4f}")

    # Final evaluation
    _, _, y_true, y_pred = evaluate(model, test_loader, criterion)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=[0,1,2]))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["benign", "malignant", "normal"], zero_division=0))

if __name__ == "__main__":
    main()