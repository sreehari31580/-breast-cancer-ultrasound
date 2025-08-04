import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import os
from sklearn.model_selection import train_test_split

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 224
MODEL_PATH = 'fixed_best_model.pth'
CSV_PATH = 'BrEaST-Lesions_USG-images_and_masks/breast_lesions_labels.csv'
BATCH_SIZE = 8
NUM_EPOCHS = 15
VAL_SPLIT = 0.2

class BreastLesionDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['img_path']
        label = int(row['label'])
        mask_path = row['mask_path'] if 'mask_path' in row and pd.notna(row['mask_path']) and row['mask_path'] else None
        image = imread(img_path)
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
        return image_tensor, label

def get_model():
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    model.to(DEVICE)
    return model

def main():
    dataset = BreastLesionDataset(CSV_PATH)
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = get_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item() * x.size(0)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)
        print(f"Epoch {epoch+1}: Train loss {train_loss:.4f}, Val loss {val_loss:.4f}, Val acc {val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'finetuned_breast_lesions_model.pth')
            print("  Saved new best model!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping.")
                break

if __name__ == "__main__":
    main() 