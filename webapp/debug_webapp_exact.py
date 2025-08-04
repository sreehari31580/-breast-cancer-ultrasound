import torch
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from PIL import Image
from skimage.transform import resize
from skimage.io import imread
import os

# Constants (same as webapp)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 224
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'fixed_best_model.pth'))

def load_model():
    """Load the best trained model - EXACT COPY FROM WEBAPP"""
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found at: {MODEL_PATH}")
        return None
        
    # Load EfficientNet-B0 model
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_features, 3)
    
    # Load trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    return model

def apply_mask_if_available(image, uploaded_filename):
    """EXACT COPY FROM WEBAPP"""
    filename_lower = uploaded_filename.lower()
    
    # If it's a normal image, don't apply any mask
    if "normal" in filename_lower:
        return image
    
    # For benign and malignant images, try to find corresponding mask
    for cls in ["benign", "malignant"]:
        if cls in filename_lower:
            base_name = os.path.basename(uploaded_filename)
            
            # Skip if this is already a mask file
            if "_mask" in base_name:
                return image
                
            # Try to find corresponding mask
            mask_name = base_name.replace('.png', '_mask.png')
            mask_path = os.path.join("Dataset_BUSI_with_GT", cls, mask_name)
            
            if os.path.exists(mask_path):
                try:
                    # Load and apply mask (same as training data preparation)
                    mask = imread(mask_path)
                    mask = resize(mask, (image.shape[0], image.shape[1]), preserve_range=True, anti_aliasing=False)
                    
                    # Convert mask to binary
                    mask = (mask > 0).astype(np.uint8)
                    
                    # Handle mask dimensions
                    if mask.ndim == 2:
                        mask3 = mask[..., None]  # Add channel dimension
                    else:
                        mask3 = mask
                    
                    # Apply mask (multiply image by mask)
                    masked_img = image * mask3
                    
                    return masked_img.astype(image.dtype)
                    
                except Exception as e:
                    print(f"Could not apply mask: {e}")
                    return image
            else:
                # Mask not found, but this might be a non-BUSI image
                return image
    
    # If no class detected, return original image
    return image

def preprocess_image(image, uploaded_filename=None):
    """EXACT COPY FROM WEBAPP"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Apply masking if it's a BUSI dataset image (and not normal)
    if uploaded_filename is not None:
        image = apply_mask_if_available(image, uploaded_filename)
    
    # Handle grayscale -> RGB conversion
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    
    # Handle RGBA -> RGB conversion  
    if image.shape[-1] == 4:
        image = image[..., :3]
    
    # Resize to 224x224 (same as training)
    image = resize(image, (IMG_SIZE, IMG_SIZE), preserve_range=True, anti_aliasing=True)
    
    # Normalize to 0-1 range (same as training)
    image = (image / 255.0).astype(np.float32)
    
    # Apply the EXACT same transform as training (validation transform)
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # This handles the final normalization and tensor conversion
    ])
    
    # Convert back to uint8 for PIL (same as training)
    img_uint8 = (image * 255).astype(np.uint8)
    
    # Apply transform and add batch dimension
    image_tensor = val_transform(img_uint8)
    
    return image_tensor.unsqueeze(0).to(DEVICE)

def test_webapp_exact_code():
    """Test using exact same code as webapp"""
    print("=== TESTING WITH EXACT WEBAPP CODE ===")
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    print(f"Model loaded from: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    
    # Test images
    test_cases = [
        ("Dataset_BUSI_with_GT/benign/benign (1).png", "benign (1).png", 0),
        ("Dataset_BUSI_with_GT/malignant/malignant (1).png", "malignant (1).png", 1),
        ("Dataset_BUSI_with_GT/normal/normal (1).png", "normal (1).png", 2)
    ]
    
    for img_path, filename, expected_class in test_cases:
        print(f"\n--- Testing: {filename} ---")
        
        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
            continue
        
        # Load and preprocess exactly like webapp
        image = Image.open(img_path)
        print(f"Original shape: {np.array(image).shape}")
        
        # Preprocess
        input_tensor = preprocess_image(image, filename)
        print(f"Input tensor shape: {input_tensor.shape}")
        print(f"Input tensor range: {input_tensor.min().item():.4f} to {input_tensor.max().item():.4f}")
        
        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        print(f"Raw outputs: {outputs}")
        print(f"Probabilities: {probabilities}")
        print(f"Predicted: {predicted_class} ({'Benign' if predicted_class==0 else 'Malignant' if predicted_class==1 else 'Normal'})")
        print(f"Confidence: {confidence*100:.1f}%")
        
        if predicted_class == expected_class:
            print("✅ CORRECT")
        else:
            print(f"❌ WRONG (expected {expected_class})")

if __name__ == "__main__":
    test_webapp_exact_code()
