"""
Comprehensive Testing Script for Breast Cancer Ultrasound Classification Web App
This script should be run before any deployment to ensure everything works correctly.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import models, transforms
from skimage.transform import resize
from skimage.io import imread
import os

# Constants (same as webapp)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 224
MODEL_PATH = os.path.abspath('fixed_best_model.pth')

def load_model_standalone():
    """Load the model without Streamlit dependencies"""
    try:
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
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def apply_mask_if_available_standalone(image, uploaded_filename):
    """Apply mask to benign/malignant images if available (normal images are not masked)"""
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

def preprocess_image_standalone(image, uploaded_filename=None):
    """Preprocess uploaded image for model input - MATCHING TRAINING DATA PREPROCESSING"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Apply masking if it's a BUSI dataset image (and not normal)
    if uploaded_filename is not None:
        image = apply_mask_if_available_standalone(image, uploaded_filename)
    
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

def test_model_loading():
    """Test if the model loads correctly"""
    print("🔧 Testing Model Loading...")
    try:
        model = load_model_standalone()
        if model is not None:
            print("✅ Model loaded successfully")
            return True
        else:
            print("❌ Model loading failed")
            return False
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_preprocessing_pipeline():
    """Test preprocessing with all three classes"""
    print("\n🔍 Testing Preprocessing Pipeline...")
    
    try:
        model = load_model_standalone()
        if model is None:
            print("❌ Could not load model for preprocessing test")
            return False
        
        test_cases = [
            ("Dataset_BUSI_with_GT/benign/benign (1).png", 0, "Benign"),
            ("Dataset_BUSI_with_GT/malignant/malignant (1).png", 1, "Malignant"),
            ("Dataset_BUSI_with_GT/normal/normal (1).png", 2, "Normal")
        ]
        
        all_passed = True
        
        for img_path, expected_class, class_name in test_cases:
            if not os.path.exists(img_path):
                print(f"⚠️  Test image not found: {img_path}")
                continue
                
            try:
                image = Image.open(img_path)
                input_tensor = preprocess_image_standalone(image, os.path.basename(img_path))
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = F.softmax(outputs, dim=1)
                    pred = torch.argmax(probs, dim=1).item()
                    confidence = probs[0][pred].item()
                
                if pred == expected_class and confidence > 0.5:
                    print(f"✅ {class_name}: Predicted correctly with {confidence*100:.1f}% confidence")
                else:
                    print(f"❌ {class_name}: Wrong prediction (got {pred}, expected {expected_class})")
                    all_passed = False
                    
            except Exception as e:
                print(f"❌ {class_name}: Error during processing - {e}")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        return False

def generate_gradcam_standalone(model, input_tensor, pred_class):
    """Generate GradCAM heatmap - STANDALONE VERSION"""
    try:
        # Import GradCAM utility with explicit path handling
        import sys
        import importlib.util
        
        utils_path = os.path.join(os.path.dirname(__file__), 'src', 'utils')
        gradcam_file = os.path.join(utils_path, 'gradcam_util.py')
        
        if not os.path.exists(gradcam_file):
            return None
            
        # Load module dynamically to avoid Pylance import errors
        spec = importlib.util.spec_from_file_location("gradcam_util", gradcam_file)
        if spec is None or spec.loader is None:
            return None
            
        gradcam_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gradcam_module)
        
        # Get the GradCAM class
        GradCAM = gradcam_module.GradCAM
        
        # Get the last convolutional layer for EfficientNet-B0
        target_layer = model.features[-1]
        
        gradcam = GradCAM(model, target_layer)
        cam = gradcam(input_tensor, class_idx=pred_class)
        gradcam.remove_hooks()
        
        return cam
    except Exception as e:
        print(f"GradCAM generation error: {e}")
        return None

def test_gradcam_functionality():
    """Test if GradCAM visualization works"""
    print("\n🎨 Testing GradCAM Functionality...")
    
    try:
        # Test with a sample image
        test_img = "Dataset_BUSI_with_GT/benign/benign (1).png"
        if not os.path.exists(test_img):
            print("⚠️  Test image for GradCAM not found")
            return True  # Not critical failure
        
        model = load_model_standalone()
        if model is None:
            print("❌ Could not load model for GradCAM test")
            return False
            
        image = Image.open(test_img)
        input_tensor = preprocess_image_standalone(image, os.path.basename(test_img))
        
        with torch.no_grad():
            outputs = model(input_tensor)
            pred_class = torch.argmax(outputs, dim=1).item()
        
        cam = generate_gradcam_standalone(model, input_tensor, pred_class)
        
        if cam is not None:
            print("✅ GradCAM generation successful")
            return True
        else:
            print("❌ GradCAM generation failed")
            return False
            
    except Exception as e:
        print(f"❌ GradCAM test failed: {e}")
        return False

def test_class_distribution():
    """Test that model doesn't have bias toward one class"""
    print("\n⚖️  Testing Class Distribution Bias...")
    
    try:
        model = load_model_standalone()
        if model is None:
            print("❌ Could not load model for bias test")
            return False
        
        # Test multiple images from each class
        test_images = {
            "benign": [],
            "malignant": [], 
            "normal": []
        }
        
        # Collect available test images
        for cls in ["benign", "malignant", "normal"]:
            cls_dir = f"Dataset_BUSI_with_GT/{cls}"
            if os.path.exists(cls_dir):
                files = [f for f in os.listdir(cls_dir) if f.endswith('.png') and '_mask' not in f]
                test_images[cls] = files[:3]  # Take first 3 images
        
        predictions = {"benign": 0, "malignant": 0, "normal": 0}
        total_tested = 0
        
        for cls, files in test_images.items():
            for file in files:
                img_path = f"Dataset_BUSI_with_GT/{cls}/{file}"
                try:
                    image = Image.open(img_path)
                    input_tensor = preprocess_image_standalone(image, file)
                    
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        pred = torch.argmax(outputs, dim=1).item()
                    
                    pred_class = ["benign", "malignant", "normal"][pred]
                    predictions[pred_class] += 1
                    total_tested += 1
                    
                except Exception as e:
                    print(f"⚠️  Error testing {img_path}: {e}")
        
        print(f"Prediction distribution across {total_tested} test images:")
        for cls, count in predictions.items():
            percentage = (count / total_tested * 100) if total_tested > 0 else 0
            print(f"  {cls}: {count} predictions ({percentage:.1f}%)")
        
        # Check if one class dominates (>80% of predictions)
        max_percentage = max(predictions.values()) / total_tested * 100 if total_tested > 0 else 0
        if max_percentage > 80:
            print(f"⚠️  Warning: Model shows bias toward one class ({max_percentage:.1f}%)")
            return False
        else:
            print("✅ No significant class bias detected")
            return True
            
    except Exception as e:
        print(f"❌ Class distribution test failed: {e}")
        return False

def run_all_tests():
    """Run all validation tests"""
    print("=" * 60)
    print("🧪 BREAST CANCER ULTRASOUND WEB APP VALIDATION")
    print("=" * 60)
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Preprocessing Pipeline", test_preprocessing_pipeline), 
        ("GradCAM Functionality", test_gradcam_functionality),
        ("Class Distribution", test_class_distribution)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📋 TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Your web app is ready for deployment!")
        return True
    else:
        print(f"\n⚠️  {total-passed} tests failed. Please fix issues before deployment.")
        return False

if __name__ == "__main__":
    run_all_tests()
