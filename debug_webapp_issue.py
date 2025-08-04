import torch
import torch.nn.functional as F
from torchvision import models
import numpy as np
from PIL import Image
import os
from skimage.transform import resize
import streamlit as st

# Constants - EXACTLY same as webapp
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 224
MODEL_PATH = 'fixed_best_model.pth'
CLASS_NAMES = ['Benign', 'Malignant', 'Normal']

def load_model_exactly_like_webapp():
    """Load model exactly like the web app does"""
    try:
        # Load EfficientNet-B0 model
        model = models.efficientnet_b0(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(num_features, 3)
        
        # Load trained weights
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_image_exactly_like_webapp(image):
    """Preprocess exactly like the web app - FIXED VERSION"""
    # Convert to numpy array if PIL Image
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Handle grayscale images
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    
    # Use EXACTLY the same preprocessing as training
    # 1. Resize using skimage.resize (same as training)
    image = resize(image, (IMG_SIZE, IMG_SIZE), preserve_range=True, anti_aliasing=True)
    
    # 2. Normalize to [0, 1] (same as training)
    if image.max() > 1:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)
    
    # 3. Convert to tensor with correct shape (C, H, W)
    image_tensor = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
    return image_tensor.unsqueeze(0).to(DEVICE)

def test_specific_images():
    """Test specific images that are causing issues"""
    print("Loading model exactly like web app...")
    model = load_model_exactly_like_webapp()
    
    if model is None:
        print("Failed to load model!")
        return
    
    print(f"Model loaded successfully on {DEVICE}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Model file size: {os.path.getsize(MODEL_PATH) / (1024*1024):.1f} MB")
    
    # Test with specific benign images that might be causing issues
    test_images = [
        ("cnn_data/benign/benign (1).png", "Benign"),
        ("cnn_data/benign/benign (9).png", "Benign"),
        ("cnn_data/benign/benign (78).png", "Benign"),
        ("cnn_data/benign/benign (141).png", "Benign"),
        ("cnn_data/benign/benign (200).png", "Benign"),
        ("cnn_data/malignant/malignant (1).png", "Malignant"),
        ("cnn_data/malignant/malignant (10).png", "Malignant"),
        ("cnn_data/normal/normal (1).png", "Normal"),
    ]
    
    print("\n" + "="*70)
    print("DETAILED DEBUG TEST - EXACT WEBAPP PIPELINE")
    print("="*70)
    
    for img_path, expected_class in test_images:
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
            
        print(f"\n{'='*50}")
        print(f"Testing: {img_path}")
        print(f"Expected: {expected_class}")
        
        try:
            # Load image as PIL (exactly like web app)
            pil_image = Image.open(img_path)
            print(f"PIL Image mode: {pil_image.mode}")
            print(f"PIL Image size: {pil_image.size}")
            
            # Convert to numpy and check
            img_array = np.array(pil_image)
            print(f"Numpy array shape: {img_array.shape}")
            print(f"Numpy array dtype: {img_array.dtype}")
            print(f"Numpy array min/max: {img_array.min()}/{img_array.max()}")
            
            # Preprocess exactly like web app
            input_tensor = preprocess_image_exactly_like_webapp(pil_image)
            
            print(f"Preprocessed tensor shape: {input_tensor.shape}")
            print(f"Preprocessed tensor dtype: {input_tensor.dtype}")
            print(f"Preprocessed tensor min/max: {input_tensor.min().item():.4f}/{input_tensor.max().item():.4f}")
            
            # Make prediction
            with torch.no_grad():
                outputs = model(input_tensor)
                print(f"Raw outputs: {outputs}")
                
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Display results
            predicted_name = CLASS_NAMES[predicted_class]
            confidence_percent = confidence * 100
            
            print(f"Predicted: {predicted_name}")
            print(f"Confidence: {confidence_percent:.1f}%")
            
            # Show all class probabilities
            print("All probabilities:")
            for i, (class_name, prob) in enumerate(zip(CLASS_NAMES, probabilities[0])):
                print(f"  {class_name}: {prob.item()*100:.1f}%")
            
            # Check if prediction matches expected
            if predicted_name == expected_class:
                print("✅ CORRECT PREDICTION")
            else:
                print("❌ INCORRECT PREDICTION")
                print(f"   Expected {expected_class} but got {predicted_name}")
                
        except Exception as e:
            print(f"Error processing image: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE CHECK")
    print("="*70)
    print(f"Classifier: {model.classifier}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Check if model weights are loaded correctly
    print(f"\nFirst few classifier weights:")
    print(model.classifier[1].weight.data[:5, :5])

if __name__ == "__main__":
    test_specific_images() 