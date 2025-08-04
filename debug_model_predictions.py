import torch
import torch.nn.functional as F
from torchvision import models
import numpy as np
from PIL import Image
import cv2
import os

# Constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 224
MODEL_PATH = 'fixed_best_model.pth'
CLASS_NAMES = ['Benign', 'Malignant', 'Normal']

def load_model():
    """Load the best trained model"""
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

def preprocess_image(image_path):
    """Preprocess image for model input"""
    # Load image
    if isinstance(image_path, str):
        image = Image.open(image_path)
        image = np.array(image)
    else:
        image = image_path
    
    # Handle grayscale images
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    
    # Resize to model input size
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    
    # Normalize to [0, 1]
    if image.max() > 1:
        image = image.astype(np.float32) / 255.0
    
    # Convert to tensor and add batch dimension
    image_tensor = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
    return image_tensor.unsqueeze(0).to(DEVICE)

def test_model():
    """Test the model with known images"""
    print("Loading model...")
    model = load_model()
    
    if model is None:
        print("Failed to load model!")
        return
    
    print(f"Model loaded successfully on {DEVICE}")
    
    # Test with some known images from the dataset
    test_images = [
        ("cnn_data/malignant/malignant (1).png", "Malignant"),
        ("cnn_data/benign/benign (1).png", "Benign"),
        ("cnn_data/normal/normal (1).png", "Normal")
    ]
    
    print("\n" + "="*50)
    print("TESTING MODEL PREDICTIONS")
    print("="*50)
    
    for img_path, expected_class in test_images:
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
            
        print(f"\nTesting: {img_path}")
        print(f"Expected: {expected_class}")
        
        try:
            # Preprocess image
            input_tensor = preprocess_image(img_path)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
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
                
        except Exception as e:
            print(f"Error processing image: {e}")
    
    print("\n" + "="*50)
    print("DEBUG INFO")
    print("="*50)
    print(f"Model path: {MODEL_PATH}")
    print(f"Model exists: {os.path.exists(MODEL_PATH)}")
    print(f"Model file size: {os.path.getsize(MODEL_PATH) / (1024*1024):.1f} MB")
    print(f"Device: {DEVICE}")
    
    # Check model architecture
    print(f"\nModel architecture:")
    print(f"Classifier: {model.classifier}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

if __name__ == "__main__":
    test_model() 