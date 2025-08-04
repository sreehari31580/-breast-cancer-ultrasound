import torch
import torch.nn.functional as F
from torchvision import models
import numpy as np
from PIL import Image
import os
from skimage.transform import resize
import hashlib

# Constants - EXACTLY same as webapp
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 224
MODEL_PATH = 'fixed_best_model.pth'
CLASS_NAMES = ['Benign', 'Malignant', 'Normal']

# Simulate Streamlit caching
_cache = {}

def cache_resource(func):
    """Simulate @st.cache_resource decorator"""
    def wrapper(*args, **kwargs):
        # Create a cache key based on function name and arguments
        cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
        
        if cache_key not in _cache:
            _cache[cache_key] = func(*args, **kwargs)
        
        return _cache[cache_key]
    return wrapper

@cache_resource
def load_model():
    """Load the best trained model - EXACTLY like web app"""
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

def preprocess_image(image):
    """Preprocess uploaded image for model input - EXACTLY like web app"""
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

def test_webapp_simulation():
    """Test the exact web app pipeline with caching"""
    print("Testing web app simulation with caching...")
    
    # Test multiple model loads (like web app might do)
    print("\n1. First model load:")
    model1 = load_model()
    print(f"Model 1 loaded: {model1 is not None}")
    
    print("\n2. Second model load (should use cache):")
    model2 = load_model()
    print(f"Model 2 loaded: {model2 is not None}")
    print(f"Models are same object: {model1 is model2}")
    
    # Test with a benign image
    test_image = "cnn_data/benign/benign (1).png"
    
    if not os.path.exists(test_image):
        print(f"Test image not found: {test_image}")
        return
    
    print(f"\n3. Testing with image: {test_image}")
    
    # Load image as PIL (like web app)
    pil_image = Image.open(test_image)
    
    # Preprocess exactly like web app
    input_tensor = preprocess_image(pil_image)
    
    # Make prediction
    with torch.no_grad():
        outputs = model1(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    predicted_name = CLASS_NAMES[predicted_class]
    confidence_percent = confidence * 100
    
    print(f"Predicted: {predicted_name}")
    print(f"Confidence: {confidence_percent:.1f}%")
    
    # Show all probabilities
    print("All probabilities:")
    for i, (class_name, prob) in enumerate(zip(CLASS_NAMES, probabilities[0])):
        print(f"  {class_name}: {prob.item()*100:.1f}%")
    
    # Check model file info
    print(f"\n4. Model file info:")
    print(f"Model path: {MODEL_PATH}")
    print(f"Model exists: {os.path.exists(MODEL_PATH)}")
    print(f"Model file size: {os.path.getsize(MODEL_PATH) / (1024*1024):.1f} MB")
    
    # Check model architecture
    print(f"\n5. Model architecture:")
    print(f"Classifier: {model1.classifier}")
    print(f"Number of parameters: {sum(p.numel() for p in model1.parameters())}")
    
    # Check if this is the correct model by testing a known image
    print(f"\n6. Testing with known images:")
    test_images = [
        ("cnn_data/benign/benign (1).png", "Benign"),
        ("cnn_data/malignant/malignant (1).png", "Malignant"),
        ("cnn_data/normal/normal (1).png", "Normal"),
    ]
    
    for img_path, expected_class in test_images:
        if not os.path.exists(img_path):
            continue
            
        pil_img = Image.open(img_path)
        input_tensor = preprocess_image(pil_img)
        
        with torch.no_grad():
            outputs = model1(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        
        predicted_name = CLASS_NAMES[predicted_class]
        print(f"  {os.path.basename(img_path)}: Expected {expected_class}, Got {predicted_name}")

if __name__ == "__main__":
    test_webapp_simulation() 