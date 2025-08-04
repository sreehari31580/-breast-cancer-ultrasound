import torch
import torch.nn.functional as F
from torchvision import models
import numpy as np
from PIL import Image
import cv2
import os
from skimage.transform import resize

# Constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 224
MODEL_PATH = 'fixed_best_model.pth'
CLASS_NAMES = ['Benign', 'Malignant', 'Normal']

def load_model():
    """Load the best trained model"""
    try:
        model = models.efficientnet_b0(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(num_features, 3)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def webapp_preprocess_image(image):
    """Web app preprocessing - FIXED VERSION"""
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

def training_preprocess_image(image_path):
    """Training preprocessing for comparison"""
    from skimage import io
    img = io.imread(image_path)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    img = resize(img, (IMG_SIZE, IMG_SIZE), preserve_range=True, anti_aliasing=True)
    img = (img / 255.0).astype(np.float32)
    img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32)
    return img.unsqueeze(0).to(DEVICE)

def test_preprocessing():
    """Test if web app preprocessing matches training preprocessing"""
    print("Testing preprocessing consistency...")
    
    # Test images
    test_images = [
        ("cnn_data/malignant/malignant (1).png", "Malignant"),
        ("cnn_data/benign/benign (1).png", "Benign"),
        ("cnn_data/normal/normal (1).png", "Normal")
    ]
    
    model = load_model()
    if model is None:
        print("Failed to load model!")
        return
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPARISON TEST")
    print("="*60)
    
    for img_path, expected_class in test_images:
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
            
        print(f"\nTesting: {img_path}")
        print(f"Expected: {expected_class}")
        
        try:
            # Load image as PIL (like web app)
            pil_image = Image.open(img_path)
            
            # Web app preprocessing
            webapp_tensor = webapp_preprocess_image(pil_image)
            
            # Training preprocessing
            training_tensor = training_preprocess_image(img_path)
            
            # Compare tensors
            diff = torch.abs(webapp_tensor - training_tensor).max().item()
            print(f"Max difference between preprocessing methods: {diff:.6f}")
            
            if diff < 1e-6:
                print("✅ Preprocessing methods match!")
            else:
                print("❌ Preprocessing methods differ!")
            
            # Test predictions with web app preprocessing
            with torch.no_grad():
                outputs = model(webapp_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            predicted_name = CLASS_NAMES[predicted_class]
            confidence_percent = confidence * 100
            
            print(f"Web app prediction: {predicted_name} ({confidence_percent:.1f}%)")
            
            # Show all probabilities
            print("All probabilities:")
            for i, (class_name, prob) in enumerate(zip(CLASS_NAMES, probabilities[0])):
                print(f"  {class_name}: {prob.item()*100:.1f}%")
            
            # Check if prediction matches expected
            if predicted_name == expected_class:
                print("✅ CORRECT PREDICTION")
            else:
                print("❌ INCORRECT PREDICTION")
                
        except Exception as e:
            print(f"Error processing image: {e}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("If preprocessing methods match and predictions are correct,")
    print("the web app should now work properly!")

if __name__ == "__main__":
    test_preprocessing() 