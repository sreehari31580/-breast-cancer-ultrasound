import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import torch
import numpy as np
from PIL import Image
from src.utils.gradcam_util import GradCAM, overlay_heatmap, preprocess_image
from src.cnn_lstm_resnet_train import ResNetLSTM

LABEL_MAP = {0: 'benign', 1: 'malignant', 2: 'normal'}
IMG_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_DIR = 'models'
ACTIVE_MODEL_FILE = os.path.join(MODEL_DIR, 'active_model.txt')


def load_active_model():
    if not os.path.exists(ACTIVE_MODEL_FILE):
        raise FileNotFoundError('No active model set.')
    with open(ACTIVE_MODEL_FILE, 'r') as f:
        model_name = f.read().strip()
    model_path = os.path.join(MODEL_DIR, model_name)
    model = ResNetLSTM(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    return model


def preprocess_uploaded_image(uploaded_file):
    img = Image.open(uploaded_file).convert('L')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_np = np.array(img).astype(np.float32) / 255.0
    return img_np


def predict_and_gradcam(img_np):
    model = load_active_model()
    # Preprocess for model
    img_tensor = preprocess_image(img_np, DEVICE)
    # Forward pass
    with torch.no_grad():
        output = model(img_tensor)
        pred_idx = output.argmax(dim=1).item()
        pred_label = LABEL_MAP[pred_idx]
        prob = torch.softmax(output, dim=1)[0, pred_idx].item()
    # Grad-CAM
    target_layer = model.feature_extractor[-1]
    gradcam = GradCAM(model, target_layer)
    cam = gradcam(img_tensor, class_idx=pred_idx)
    gradcam.remove_hooks()
    # Overlay
    overlay = overlay_heatmap((img_np*255).astype(np.uint8), cam)
    return pred_label, prob, overlay 