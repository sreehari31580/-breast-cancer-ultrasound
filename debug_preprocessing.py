import torch
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from PIL import Image
from skimage.transform import resize
import os

# Load model
DEVICE = 'cpu'  # Use CPU for debugging
MODEL_PATH = 'fixed_best_model.pth'

def load_model():
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_features, 3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# Current webapp preprocessing (WRONG)
def webapp_preprocessing(image):
    """Current webapp preprocessing - PROBLEMATIC"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    image = resize(image, (224, 224), preserve_range=True, anti_aliasing=True)
    if image.max() > 1:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)
    image_tensor = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
    return image_tensor.unsqueeze(0).to(DEVICE)

# Correct preprocessing (like training)
def correct_preprocessing(image):
    """Correct preprocessing matching training data"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Handle grayscale
    if image.ndim == 2:
        image = np.stack([image]*3, axis=-1)
    
    # Resize to 224x224
    image = resize(image, (224, 224), preserve_range=True, anti_aliasing=True)
    
    # Normalize to 0-1
    image = (image / 255.0).astype(np.float32)
    
    # Apply the same transform as training (validation transform)
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    
    # Convert back to uint8 for PIL
    img_uint8 = (image * 255).astype(np.uint8)
    
    # Apply transform
    image_tensor = val_transform(img_uint8)
    
    return image_tensor.unsqueeze(0).to(DEVICE)

def test_image(image_path):
    """Test an image with both preprocessing methods"""
    print(f"\n=== TESTING: {image_path} ===")
    
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return
    
    # Load image
    image = Image.open(image_path)
    print(f"Original image shape: {np.array(image).shape}")
    
    # Load model
    model = load_model()
    
    # Test with webapp preprocessing (current - wrong)
    print("\n1. WEBAPP PREPROCESSING (CURRENT - WRONG):")
    input_tensor_wrong = webapp_preprocessing(image)
    print(f"   Input tensor shape: {input_tensor_wrong.shape}")
    print(f"   Input tensor range: {input_tensor_wrong.min().item():.4f} to {input_tensor_wrong.max().item():.4f}")
    
    with torch.no_grad():
        outputs_wrong = model(input_tensor_wrong)
        probs_wrong = F.softmax(outputs_wrong, dim=1)
        pred_wrong = torch.argmax(probs_wrong, dim=1).item()
    
    print(f"   Raw outputs: {outputs_wrong}")
    print(f"   Probabilities: {probs_wrong}")
    print(f"   Prediction: {pred_wrong} ({'Benign' if pred_wrong==0 else 'Malignant' if pred_wrong==1 else 'Normal'})")
    
    # Test with correct preprocessing
    print("\n2. CORRECT PREPROCESSING (LIKE TRAINING):")
    input_tensor_correct = correct_preprocessing(image)
    print(f"   Input tensor shape: {input_tensor_correct.shape}")
    print(f"   Input tensor range: {input_tensor_correct.min().item():.4f} to {input_tensor_correct.max().item():.4f}")
    
    with torch.no_grad():
        outputs_correct = model(input_tensor_correct)
        probs_correct = F.softmax(outputs_correct, dim=1)
        pred_correct = torch.argmax(probs_correct, dim=1).item()
    
    print(f"   Raw outputs: {outputs_correct}")
    print(f"   Probabilities: {probs_correct}")
    print(f"   Prediction: {pred_correct} ({'Benign' if pred_correct==0 else 'Malignant' if pred_correct==1 else 'Normal'})")
    
    return pred_wrong, pred_correct

if __name__ == "__main__":
    print("=== PREPROCESSING DEBUG TEST ===")
    
    # Test with sample images from your dataset
    test_images = [
        "Dataset_BUSI_with_GT/benign/benign (1).png",
        "Dataset_BUSI_with_GT/malignant/malignant (1).png", 
        "Dataset_BUSI_with_GT/normal/normal (1).png"
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            test_image(img_path)
        else:
            print(f"Skipping {img_path} - file not found")
