import torch
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from PIL import Image
from skimage.transform import resize
from skimage import io
import pandas as pd
import os

# Load model
DEVICE = 'cpu'
MODEL_PATH = 'fixed_best_model.pth'

def load_model():
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_features, 3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def test_training_data_preprocessing(img_path):
    """Test with exact same preprocessing as training"""
    # Load image exactly like training dataset
    img = io.imread(img_path)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    img = resize(img, (224, 224), preserve_range=True, anti_aliasing=True)
    img = (img / 255.0).astype(np.float32)
    
    # Apply validation transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    
    img_uint8 = (img * 255).astype(np.uint8)
    img_tensor = val_transform(img_uint8)
    input_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    
    return input_tensor

def test_cnn_data_images():
    """Test with images from the processed CNN data folder"""
    print("=== TESTING WITH CNN DATA IMAGES ===")
    
    # Load test CSV to get actual test images with labels
    test_df = pd.read_csv('cnn_data/test.csv')
    
    # Load model
    model = load_model()
    
    # Test a few images from each class
    classes_to_test = {0: 'benign', 1: 'malignant', 2: 'normal'}
    
    for class_label, class_name in classes_to_test.items():
        print(f"\n=== TESTING {class_name.upper()} IMAGES ===")
        
        # Get images of this class
        class_images = test_df[test_df['label'] == class_label].head(3)
        
        for _, row in class_images.iterrows():
            img_path = row['img_path']
            true_label = row['label']
            
            print(f"\nTesting: {img_path} (True label: {true_label})")
            
            if not os.path.exists(img_path):
                print(f"  File not found: {img_path}")
                continue
            
            try:
                # Test with training-like preprocessing
                input_tensor = test_training_data_preprocessing(img_path)
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = F.softmax(outputs, dim=1)
                    pred = torch.argmax(probs, dim=1).item()
                    confidence = probs[0][pred].item()
                
                print(f"  Raw outputs: {outputs}")
                print(f"  Probabilities: {probs}")
                print(f"  Predicted: {pred} ({classes_to_test[pred]}) with {confidence*100:.1f}% confidence")
                print(f"  ✅ CORRECT" if pred == true_label else f"  ❌ WRONG (expected {true_label})")
                
            except Exception as e:
                print(f"  Error: {e}")

def check_model_weights():
    """Check if model weights look reasonable"""
    print("\n=== CHECKING MODEL WEIGHTS ===")
    
    model_state = torch.load(MODEL_PATH, map_location='cpu')
    
    # Check classifier weights
    if 'classifier.1.weight' in model_state:
        classifier_weight = model_state['classifier.1.weight']
        classifier_bias = model_state['classifier.1.bias']
        
        print(f"Classifier weight shape: {classifier_weight.shape}")
        print(f"Classifier bias: {classifier_bias}")
        print(f"Weight statistics:")
        print(f"  Mean: {classifier_weight.mean().item():.6f}")
        print(f"  Std: {classifier_weight.std().item():.6f}")
        print(f"  Min: {classifier_weight.min().item():.6f}")
        print(f"  Max: {classifier_weight.max().item():.6f}")
        
        # Check if weights are heavily biased toward class 2 (normal)
        print(f"\nBias values:")
        for i, bias_val in enumerate(classifier_bias):
            print(f"  Class {i} bias: {bias_val.item():.6f}")

if __name__ == "__main__":
    check_model_weights()
    test_cnn_data_images()
