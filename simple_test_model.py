import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from torchvision import models, transforms
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
import seaborn as sns

# Import the dataset class
from src.cnn_lstm_resnet_train import MaskedTumorDataset

print("=== SIMPLE RESNET18 TEST ===")

# Simple ResNet18 model without LSTM
class SimpleEfficientNetB0(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleEfficientNetB0, self).__init__()
        self.effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        num_features = self.effnet.classifier[1].in_features
        self.effnet.classifier[1] = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.effnet(x)

def train_simple_model():
    # Load data
    df = pd.read_csv("cnn_data/labels.csv")
    print(f"Total samples: {len(df)}")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    train_df.to_csv("cnn_data/train.csv", index=False)
    test_df.to_csv("cnn_data/test.csv", index=False)
    
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Train class distribution: {train_df['label'].value_counts().to_dict()}")
    print(f"Test class distribution: {test_df['label'].value_counts().to_dict()}")
    
    # Create datasets
    train_dataset = MaskedTumorDataset("cnn_data/train.csv", augment=True)
    test_dataset = MaskedTumorDataset("cnn_data/test.csv", augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SimpleEfficientNetB0(num_classes=3).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 10
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
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
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # Testing
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
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        # Print predictions distribution
        pred_counts = np.bincount(all_preds, minlength=3)
        print(f"  Test Predictions: Benign={pred_counts[0]}, Malignant={pred_counts[1]}, Normal={pred_counts[2]}")
    
    # Final evaluation
    print("\n=== FINAL EVALUATION ===")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["benign", "malignant", "normal"]))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["benign", "malignant", "normal"],
                yticklabels=["benign", "malignant", "normal"])
    plt.title('Confusion Matrix - Simple ResNet18')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('simple_model_confusion_matrix.png')
    print("Confusion matrix saved as simple_model_confusion_matrix.png")
    
    # Save model
    torch.save(model.state_dict(), "simple_resnet18_model.pth")
    print("Simple model saved as simple_resnet18_model.pth")
    
    return model, all_labels, all_preds

if __name__ == "__main__":
    model, labels, preds = train_simple_model() 