import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from torchvision import models
from src.cnn_lstm_resnet_train import MaskedTumorDataset
import matplotlib.pyplot as plt
import seaborn as sns

print("=== SIMPLE RESNET18 TEST (BALANCED SUBSET) ===")

class SimpleEfficientNetB0(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleEfficientNetB0, self).__init__()
        self.effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        num_features = self.effnet.classifier[1].in_features
        self.effnet.classifier[1] = nn.Linear(num_features, num_classes)
    def forward(self, x):
        super(SimpleResNet18, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.resnet(x)

def train_simple_model():
    df = pd.read_csv("cnn_data/balanced_subset.csv")
    print(f"Total samples: {len(df)}")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Train class distribution: {train_df['label'].value_counts().to_dict()}")
    print(f"Test class distribution: {test_df['label'].value_counts().to_dict()}")
    # Save splits as CSVs
    train_csv = "cnn_data/balanced_train.csv"
    test_csv = "cnn_data/balanced_test.csv"
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    train_dataset = MaskedTumorDataset(train_csv, augment=True)
    test_dataset = MaskedTumorDataset(test_csv, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = SimpleResNet18(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        test_loss = test_loss / len(test_loader)
        test_acc = test_correct / test_total
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        pred_counts = np.bincount(all_preds, minlength=3)
        print(f"  Test Predictions: Benign={pred_counts[0]}, Malignant={pred_counts[1]}, Normal={pred_counts[2]}")
    print("\n=== FINAL EVALUATION ===")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["benign", "malignant", "normal"]))
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["benign", "malignant", "normal"],
                yticklabels=["benign", "malignant", "normal"])
    plt.title('Confusion Matrix - Simple ResNet18 (Balanced Subset)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('simple_model_confusion_matrix_balanced.png')
    print("Confusion matrix saved as simple_model_confusion_matrix_balanced.png")
    torch.save(model.state_dict(), "simple_resnet18_model_balanced.pth")
    print("Simple model saved as simple_resnet18_model_balanced.pth")
    return model, all_labels, all_preds

if __name__ == "__main__":
    model, labels, preds = train_simple_model() 