import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from torchvision import models, transforms
from skimage import io
from skimage.transform import resize
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 20
NUM_CLASSES = 3

# Focal Loss for handling class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

# Strong class weights
CLASS_WEIGHTS = torch.tensor([1.0, 3.0, 2.5]).to(DEVICE)  # Higher weight for malignant and normal

class MaskedTumorDataset(Dataset):
    def __init__(self, csv_file, transform=None, is_train=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.is_train = is_train
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row["img_path"]
        label = int(row["label"])
        
        try:
            img = io.imread(img_path)
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            img = resize(img, (IMG_SIZE, IMG_SIZE), preserve_range=True, anti_aliasing=True)
            img = (img / 255.0).astype(np.float32)
            
            if self.transform:
                img_uint8 = (img * 255).astype(np.uint8)
                img = self.transform(img_uint8).numpy()
            
            return torch.tensor(img, dtype=torch.float32), label
        except Exception as e:
            # Return a blank image if loading fails
            img = np.zeros((3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
            return torch.tensor(img, dtype=torch.float32), label

# Strong augmentation for training
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.ToTensor(),
])

# No augmentation for validation/test
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

def get_model(model_name, num_classes=3):
    if model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
        return model
    elif model_name == 'densenet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)
        return model
    else:
        raise ValueError(f"Unknown model: {model_name}")

def train_epoch(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return total_loss / len(train_loader), correct / total

def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(val_loader), correct / total, all_preds, all_labels

def main():
    print("=== FIXED TRAINING WITH STRONG CLASS BALANCING ===\n")
    
    # Load data
    train_dataset = MaskedTumorDataset('cnn_data/train.csv', transform=train_transform, is_train=True)
    val_dataset = MaskedTumorDataset('cnn_data/val.csv', transform=val_transform)
    
    # Create weighted sampler for training
    train_labels = train_dataset.data['label'].values
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model and training setup
    model = get_model('efficientnet_b0', num_classes=NUM_CLASSES).to(DEVICE)
    
    # Use both weighted cross entropy and focal loss
    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
    focal_criterion = FocalLoss(alpha=1, gamma=2)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    best_model_state = None
    
    print("Class distribution in training set:")
    print(pd.Series(train_labels).value_counts().sort_index())
    print(f"\nClass weights: {CLASS_WEIGHTS.cpu().numpy()}")
    print(f"Training for {NUM_EPOCHS} epochs...\n")
    
    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Print validation confusion matrix every 5 epochs
        if (epoch + 1) % 5 == 0:
            cm = confusion_matrix(val_labels, val_preds)
            print("  Val Confusion Matrix:")
            print(f"    {cm}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model and evaluate
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    print("\n=== FINAL EVALUATION ===")
    val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion)
    
    print(f"Best Validation Accuracy: {val_acc:.4f}")
    print("\nConfusion Matrix:")
    cm = confusion_matrix(val_labels, val_preds)
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=["benign", "malignant", "normal"], zero_division=0))
    
    # Save model
    torch.save(best_model_state, 'fixed_best_model.pth')
    print("\nModel saved as 'fixed_best_model.pth'")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["benign", "malignant", "normal"],
                yticklabels=["benign", "malignant", "normal"])
    plt.title('Confusion Matrix - Fixed Model')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('fixed_confusion_matrix.png')
    print("Confusion matrix saved as 'fixed_confusion_matrix.png'")

if __name__ == "__main__":
    main() 