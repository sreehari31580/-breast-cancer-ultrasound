import torch
import numpy as np
import pandas as pd
from torchvision import models, transforms
from torch.utils.data import Dataset
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
import os
from utils.gradcam_util import GradCAM

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 224

# Get absolute paths based on script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'fixed_best_model.pth'))
TEST_CSV_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'cnn_data', 'test.csv'))
MODEL_TYPE = 'efficientnet_b0'

def get_model(model_name, num_classes=3):
    if model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(num_features, num_classes)
        return model
    else:
        raise ValueError(f"Unknown model: {model_name}")

class MaskedTumorDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
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
        img = torch.tensor(img.transpose(2,0,1), dtype=torch.float32)
        return img, label, img_path

def overlay_heatmap(img, cam, alpha=0.5):
    import cv2
    cam = (cam * 255).astype(np.uint8)
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    img = (img * 255).astype(np.uint8)
    if img.shape[0] == 3:
        img = img.transpose(1,2,0)
    overlay = cv2.addWeighted(img, 1-alpha, cam, alpha, 0)
    return overlay

def main():
    os.makedirs('gradcam_outputs', exist_ok=True)
    
    print(f"Loading best model from {MODEL_PATH}...")
    model = get_model(MODEL_TYPE, num_classes=3).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # Pick the last conv layer for Grad-CAM (EfficientNet-B0)
    target_layer = model.features[-1]
    gradcam = GradCAM(model, target_layer)
    
    dataset = MaskedTumorDataset(TEST_CSV_PATH)
    
    print("Generating Grad-CAM visualizations...")
    # Visualize first 15 test images (5 from each class)
    class_counts = {0: 0, 1: 0, 2: 0}
    max_per_class = 5
    
    for i in range(len(dataset)):
        if sum(class_counts.values()) >= 15:  # Stop after 15 images
            break
            
        img, label, img_path = dataset[i]
        
        if class_counts[label] >= max_per_class:
            continue
            
        input_tensor = img.unsqueeze(0).to(DEVICE)
        
        # Get model prediction
        with torch.no_grad():
            output = model(input_tensor)
            pred_class = torch.argmax(output, dim=1).item()
            confidence = torch.softmax(output, dim=1).max().item()
        
        # Generate Grad-CAM
        cam = gradcam(input_tensor, class_idx=pred_class)
        
        # Create visualization
        img_np = img.cpu().numpy().transpose(1,2,0)
        overlay = overlay_heatmap(img_np, cam)
        
        # Save with informative filename
        class_names = ["benign", "malignant", "normal"]
        true_class = class_names[label]
        pred_class_name = class_names[pred_class]
        confidence_str = f"{confidence:.2f}"
        
        out_filename = f"gradcam_{true_class}_pred_{pred_class_name}_conf_{confidence_str}_{os.path.basename(img_path)}"
        out_path = os.path.join('gradcam_outputs', out_filename)
        
        plt.imsave(out_path, overlay)
        print(f"Saved: {out_filename}")
        
        class_counts[label] += 1
    
    print(f"\nGrad-CAM visualizations saved in 'gradcam_outputs/' folder")
    print(f"Generated {sum(class_counts.values())} visualizations")

if __name__ == '__main__':
    main() 