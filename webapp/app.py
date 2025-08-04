import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from PIL import Image
import io
import cv2
import os
import sys
from skimage.transform import resize
from skimage.io import imread
import datetime
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)
from utils.gradcam_util import GradCAM  # type: ignore

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Ultrasound Classifier",
    page_icon="🩺",
    layout="wide"
)

# FORCE CACHE CLEAR
if st.sidebar.button("🔄 Clear Cache & Reload"):
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()

# Constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 224
# Get the absolute path to the model file (one directory up from webapp)
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'fixed_best_model.pth'))
CLASS_NAMES = ['Benign', 'Malignant', 'Normal']

@st.cache_resource
def load_model(_version="v2_fixed_preprocessing"):  # Add version parameter to force reload
    """Load the best trained model"""
    try:
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Model file not found at: {MODEL_PATH}")
            return None
            
        # Load EfficientNet-B0 model
        model = models.efficientnet_b0(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(num_features, 3)
        
        # Load trained weights
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        
        # DEBUG: Print model info
        st.write(f"🔍 DEBUG: Model loaded from {MODEL_PATH}")
        st.write(f"🔍 DEBUG: Model file size: {os.path.getsize(MODEL_PATH) / (1024*1024):.1f} MB")
        st.write(f"🔍 DEBUG: Model parameters: {sum(p.numel() for p in model.parameters())}")
        st.write(f"🔍 DEBUG: CACHE REFRESHED: {datetime.datetime.now()}")
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def apply_mask_if_available(image, uploaded_filename):
    """Apply mask to benign/malignant images if available (normal images are not masked)"""
    # Check if this is from the BUSI dataset and determine class
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
                    st.warning(f"Could not apply mask: {e}")
                    return image
            else:
                # Mask not found, but this might be a non-BUSI image
                return image
    
    # If no class detected, return original image
    return image

def preprocess_image(image, uploaded_filename=None):
    """Preprocess uploaded image for model input - MATCHING TRAINING DATA PREPROCESSING"""
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

def generate_gradcam(model, input_tensor, pred_class):
    """Generate Grad-CAM visualization"""
    try:
        target_layer = model.features[-1]
        gradcam = GradCAM(model, target_layer)
        cam = gradcam(input_tensor, class_idx=pred_class)
        return cam
    except Exception as e:
        st.error(f"Error generating Grad-CAM: {e}")
        return None

def overlay_heatmap(img, cam, alpha=0.5):
    """Overlay heatmap on original image"""
    try:
        if img.max() <= 1:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cam = (cam * 255).astype(np.uint8)
        cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cam, alpha, img, 1 - alpha, 0)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        return overlay
    except Exception as e:
        st.error(f"Error creating overlay: {e}")
        return img

def main():
    st.title("🩺 Breast Cancer Ultrasound Classifier")
    st.markdown("---")
    
    # Load model
    with st.spinner("Loading the best trained model..."):
        model = load_model("v2_fixed_preprocessing")
    
    if model is None:
        st.error("Failed to load model. Please check if the model file exists.")
        return
    
    st.success(f"✅ Model loaded successfully! Using: {MODEL_PATH}")
    st.info(f"🔧 Device: {DEVICE}")
    
    # Sidebar for model info
    with st.sidebar:
        st.header("Model Information")
        st.write(f"**Model Type:** EfficientNet-B0")
        st.write(f"**Classes:** {', '.join(CLASS_NAMES)}")
        st.write(f"**Input Size:** {IMG_SIZE}x{IMG_SIZE}")
        
        st.header("Instructions")
        st.write("1. Upload a breast ultrasound image")
        st.write("2. The model will classify it as Benign, Malignant, or Normal")
        st.write("3. Grad-CAM visualization will show the model's focus areas")
        
        st.header("About")
        st.write("This model was trained on the BUSI dataset with:")
        st.write("• **96.6%** test accuracy")
        st.write("• **97% F1-score** (macro average)")
        st.write("• **ROC-AUC: 99.7%** (benign), **99.7%** (malignant), **100%** (normal)")
        st.write("• Balanced class distribution")
        st.write("• Advanced data augmentation")
        
        # --- Grad-CAM Explanation ---
        st.header("What is Grad-CAM?")
        st.write(
            "Grad-CAM (Gradient-weighted Class Activation Mapping) is a technique that helps visualize "
            "which regions of an image were most important for the model's prediction. "
            "It highlights areas the model focused on, making the decision process more interpretable."
        )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a breast ultrasound image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a breast ultrasound image for classification"
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)
            
            # Preprocess image with auto-masking
            input_tensor = preprocess_image(image, uploaded_file.name)
            
            # DEBUG: Show preprocessing info
            st.write(f"🔍 DEBUG: Input tensor shape: {input_tensor.shape}")
            st.write(f"🔍 DEBUG: Input tensor min/max: {input_tensor.min().item():.4f}/{input_tensor.max().item():.4f}")
            
            # Make prediction
            with st.spinner("Analyzing image..."):
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_class].item()
            
            # DEBUG: Show raw outputs
            st.write(f"🔍 DEBUG: Raw outputs: {outputs}")
            st.write(f"🔍 DEBUG: Probabilities: {probabilities}")
            st.write(f"🔍 DEBUG: Predicted class: {predicted_class}")
            st.write(f"🔍 DEBUG: Input filename: {uploaded_file.name}")
            st.write(f"🔍 DEBUG: Detected masking applied: {'Yes' if 'benign' in uploaded_file.name.lower() or 'malignant' in uploaded_file.name.lower() else 'No'}")
            
            # Display results
            st.header("🔍 Analysis Results")
            
            # Prediction with confidence
            class_name = CLASS_NAMES[predicted_class]
            confidence_percent = confidence * 100
            
            # Color-coded prediction
            if predicted_class == 0:  # Benign
                st.success(f"**Prediction: {class_name}**")
                st.metric("Confidence", f"{confidence_percent:.1f}%")
            elif predicted_class == 1:  # Malignant
                st.error(f"**Prediction: {class_name}**")
                st.metric("Confidence", f"{confidence_percent:.1f}%")
            else:  # Normal
                st.info(f"**Prediction: {class_name}**")
                st.metric("Confidence", f"{confidence_percent:.1f}%")
            
            # Confidence bar
            st.progress(confidence)
            
            # All class probabilities
            st.subheader("Class Probabilities")
            prob_dict = {CLASS_NAMES[i]: probabilities[0][i].item() * 100 for i in range(3)}
            
            for cname, prob in prob_dict.items():
                col_a, col_b = st.columns([3, 1])
                col_a.progress(prob/100)
                col_b.write(f"{prob:.1f}%")
                col_b.write(f"**{cname}**")
            
            # --- Download Prediction Result ---
            result_text = (
                f"Breast Cancer Ultrasound Prediction Result\n"
                f"==========================================\n"
                f"File: {uploaded_file.name if uploaded_file else 'N/A'}\n"
                f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Model: EfficientNet-B0 ({MODEL_PATH})\n"
                f"------------------------------------------\n"
                f"Prediction: {class_name}\n"
                f"Confidence: {confidence_percent:.2f}%\n"
                f"\nClass Probabilities:\n"
            )
            for cname, prob in prob_dict.items():
                result_text += f"  {cname}: {prob:.2f}%\n"
            result_text += (
                "\nNote: This result is for educational and research purposes only. "
                "Consult a medical professional for clinical decisions."
            )
            st.download_button(
                label="Download Prediction Result",
                data=result_text,
                file_name="prediction_result.txt",
                mime="text/plain"
            )
    
    with col2:
        if uploaded_file is not None:
            st.header("🎯 Grad-CAM Visualization")
            st.write(
                "Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the regions in the ultrasound image "
                "that were most important for the model's prediction. The colored areas show where the model 'looked' to make its decision."
            )
            
            # Generate Grad-CAM
            with st.spinner("Generating Grad-CAM visualization..."):
                cam = generate_gradcam(model, input_tensor, predicted_class)
                
                if cam is not None:
                    # Convert input tensor back to image for overlay
                    img_np = input_tensor.cpu().numpy()[0].transpose(1, 2, 0)
                    
                    # Create overlay
                    overlay = overlay_heatmap(img_np, cam)
                    
                    if overlay is not None:
                        st.image(overlay, caption="Grad-CAM Overlay", use_column_width=True)
                        
                        # Explanation
                        st.subheader("📊 What Grad-CAM Shows")
                        if predicted_class == 0:  # Benign
                            st.info("🟢 **Benign Features:** The highlighted areas show regions the model identified as characteristic of benign tumors (typically well-defined, smooth borders)")
                        elif predicted_class == 1:  # Malignant
                            st.warning("🔴 **Malignant Features:** The highlighted areas show regions the model identified as characteristic of malignant tumors (typically irregular, spiculated borders)")
                        else:  # Normal
                            st.success("🔵 **Normal Tissue:** The highlighted areas show regions the model identified as normal breast tissue")
                    else:
                        st.error("Failed to create Grad-CAM overlay")
                else:
                    st.error("Failed to generate Grad-CAM visualization")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <p><strong>⚠️ Medical Disclaimer:</strong> This tool is for educational and research purposes only. 
        It should not be used for clinical diagnosis. Always consult with qualified healthcare professionals for medical decisions.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()