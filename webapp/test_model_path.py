import os
import torch
from torchvision import models

# Test the model path resolution
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'fixed_best_model.pth'))
print(f"Model path: {MODEL_PATH}")
print(f"Model exists: {os.path.exists(MODEL_PATH)}")
print(f"Model file size: {os.path.getsize(MODEL_PATH) / (1024*1024):.1f} MB")

# Test loading the model
try:
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_features, 3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    print("✅ Model loaded successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
