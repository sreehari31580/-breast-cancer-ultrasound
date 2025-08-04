# API Documentation - Breast Cancer Ultrasound Classification

This document provides comprehensive API documentation for programmatic usage of the breast cancer classification system.

## 🚀 Quick Start

```python
from breast_cancer_classifier import BreastCancerClassifier

# Initialize classifier
classifier = BreastCancerClassifier()

# Make prediction
result = classifier.predict('path/to/ultrasound.png')
print(f"Prediction: {result['class']} ({result['confidence']:.2%})")
```

## 📚 Core API

### BreastCancerClassifier Class

The main interface for breast cancer classification.

#### Constructor

```python
BreastCancerClassifier(
    model_path: str = 'fixed_best_model.pth',
    device: str = 'auto',
    use_gradcam: bool = True
)
```

**Parameters:**
- `model_path` (str): Path to the trained model file
- `device` (str): Device to run inference on ('cpu', 'cuda', 'auto')
- `use_gradcam` (bool): Enable GradCAM visualization

#### Methods

##### predict()

```python
predict(
    image_path: str,
    return_probabilities: bool = False,
    return_gradcam: bool = False
) -> Dict[str, Any]
```

Classify a breast ultrasound image.

**Parameters:**
- `image_path` (str): Path to the ultrasound image
- `return_probabilities` (bool): Include class probabilities in output
- `return_gradcam` (bool): Include GradCAM visualization

**Returns:**
- Dictionary containing prediction results

**Example:**
```python
result = classifier.predict(
    'ultrasound.png',
    return_probabilities=True,
    return_gradcam=True
)

print(result)
# {
#     'class': 'benign',
#     'class_id': 0,
#     'confidence': 0.95,
#     'probabilities': [0.95, 0.03, 0.02],
#     'gradcam': <numpy.ndarray>,
#     'processing_time': 0.234
# }
```

##### predict_batch()

```python
predict_batch(
    image_paths: List[str],
    batch_size: int = 16
) -> List[Dict[str, Any]]
```

Classify multiple images efficiently.

**Parameters:**
- `image_paths` (List[str]): List of image paths
- `batch_size` (int): Number of images to process simultaneously

**Returns:**
- List of prediction dictionaries

##### get_model_info()

```python
get_model_info() -> Dict[str, Any]
```

Get information about the loaded model.

**Returns:**
- Dictionary with model metadata

## 🛠️ Utility Functions

### Image Preprocessing

```python
from breast_cancer_classifier.utils import preprocess_image

def preprocess_image(
    image: Union[str, np.ndarray, Image.Image],
    filename: Optional[str] = None
) -> torch.Tensor
```

Preprocess an image for model input.

**Parameters:**
- `image`: Image to preprocess (path, numpy array, or PIL Image)
- `filename`: Original filename (for masking logic)

**Returns:**
- Preprocessed image tensor

### GradCAM Visualization

```python
from breast_cancer_classifier.utils import generate_gradcam

def generate_gradcam(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_class: int
) -> np.ndarray
```

Generate GradCAM heatmap for model interpretability.

**Parameters:**
- `model`: PyTorch model
- `image_tensor`: Preprocessed image tensor
- `target_class`: Class to generate heatmap for

**Returns:**
- GradCAM heatmap as numpy array

### Model Loading

```python
from breast_cancer_classifier.utils import load_model

def load_model(
    model_path: str = 'fixed_best_model.pth',
    device: str = 'auto'
) -> torch.nn.Module
```

Load a trained model.

**Parameters:**
- `model_path`: Path to model file
- `device`: Device to load model on

**Returns:**
- Loaded PyTorch model

## 📊 Data Structures

### Prediction Result

```python
class PredictionResult:
    class: str              # Class name ('benign', 'malignant', 'normal')
    class_id: int          # Class ID (0, 1, 2)
    confidence: float      # Confidence score (0.0 - 1.0)
    probabilities: List[float]  # All class probabilities
    gradcam: Optional[np.ndarray]  # GradCAM heatmap
    processing_time: float # Time taken for prediction
    metadata: Dict         # Additional information
```

### Model Information

```python
class ModelInfo:
    architecture: str      # Model architecture name
    parameters: int        # Number of parameters
    size_mb: float        # Model size in MB
    accuracy: float       # Test accuracy
    classes: List[str]    # Class names
    version: str          # Model version
```

## 🔧 Configuration

### Environment Variables

```bash
# Model configuration
BREAST_CANCER_MODEL_PATH=/path/to/model.pth
BREAST_CANCER_DEVICE=cuda

# Performance settings
BREAST_CANCER_BATCH_SIZE=16
BREAST_CANCER_NUM_WORKERS=4

# Logging
BREAST_CANCER_LOG_LEVEL=INFO
```

### Configuration File

Create `config.yaml`:

```yaml
model:
  path: "fixed_best_model.pth"
  device: "auto"
  
preprocessing:
  image_size: 224
  normalize: true
  apply_masks: true
  
inference:
  batch_size: 16
  use_gradcam: true
  
logging:
  level: "INFO"
  file: "classifier.log"
```

Load configuration:

```python
from breast_cancer_classifier import BreastCancerClassifier

classifier = BreastCancerClassifier.from_config('config.yaml')
```

## 🧪 Testing API

### Unit Tests

```python
import unittest
from breast_cancer_classifier import BreastCancerClassifier

class TestBreastCancerClassifier(unittest.TestCase):
    
    def setUp(self):
        self.classifier = BreastCancerClassifier()
    
    def test_prediction(self):
        result = self.classifier.predict('test_images/benign_sample.png')
        self.assertIn('class', result)
        self.assertIn('confidence', result)
        self.assertIsInstance(result['confidence'], float)
    
    def test_batch_prediction(self):
        images = ['test1.png', 'test2.png']
        results = self.classifier.predict_batch(images)
        self.assertEqual(len(results), 2)
```

### Performance Testing

```python
import time
from breast_cancer_classifier import BreastCancerClassifier

def test_performance():
    classifier = BreastCancerClassifier()
    
    # Single image performance
    start = time.time()
    result = classifier.predict('test_image.png')
    single_time = time.time() - start
    
    print(f"Single prediction time: {single_time:.3f}s")
    
    # Batch performance
    images = ['test_image.png'] * 10
    start = time.time()
    results = classifier.predict_batch(images)
    batch_time = time.time() - start
    
    print(f"Batch prediction time: {batch_time:.3f}s")
    print(f"Average per image: {batch_time/10:.3f}s")
```

## 🔌 Integration Examples

### Flask Web API

```python
from flask import Flask, request, jsonify
from breast_cancer_classifier import BreastCancerClassifier

app = Flask(__name__)
classifier = BreastCancerClassifier()

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    image_path = f"temp/{image_file.filename}"
    image_file.save(image_path)
    
    result = classifier.predict(image_path)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
```

### FastAPI

```python
from fastapi import FastAPI, UploadFile, File
from breast_cancer_classifier import BreastCancerClassifier

app = FastAPI()
classifier = BreastCancerClassifier()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded file
    with open(f"temp/{file.filename}", "wb") as buffer:
        buffer.write(await file.read())
    
    # Make prediction
    result = classifier.predict(f"temp/{file.filename}")
    return result
```

### Jupyter Notebook

```python
# Install in notebook
!pip install breast-cancer-classifier

# Import and use
from breast_cancer_classifier import BreastCancerClassifier
import matplotlib.pyplot as plt

classifier = BreastCancerClassifier()

# Predict and visualize
result = classifier.predict('sample.png', return_gradcam=True)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(original_image)
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(result['gradcam'], cmap='jet')
plt.title('GradCAM')

plt.subplot(1, 3, 3)
plt.bar(classifier.classes, result['probabilities'])
plt.title('Class Probabilities')
plt.show()
```

## 🚨 Error Handling

### Exception Types

```python
from breast_cancer_classifier.exceptions import (
    ModelLoadError,
    ImageProcessingError,
    PredictionError
)

try:
    classifier = BreastCancerClassifier('invalid_model.pth')
except ModelLoadError as e:
    print(f"Failed to load model: {e}")

try:
    result = classifier.predict('invalid_image.txt')
except ImageProcessingError as e:
    print(f"Image processing failed: {e}")

try:
    result = classifier.predict('corrupted_image.png')
except PredictionError as e:
    print(f"Prediction failed: {e}")
```

### Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| E001 | Model file not found | Check model path |
| E002 | Invalid image format | Use PNG/JPG/JPEG |
| E003 | Image too large | Resize image < 10MB |
| E004 | CUDA out of memory | Use CPU or reduce batch size |
| E005 | Invalid device | Use 'cpu', 'cuda', or 'auto' |

## 📈 Performance Optimization

### GPU Acceleration

```python
# Enable GPU if available
classifier = BreastCancerClassifier(device='cuda')

# Check GPU usage
print(f"Using GPU: {classifier.device}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
```

### Batch Processing

```python
# Process multiple images efficiently
image_paths = ['img1.png', 'img2.png', 'img3.png']
results = classifier.predict_batch(image_paths, batch_size=8)
```

### Memory Management

```python
# Clear GPU cache
torch.cuda.empty_cache()

# Use context manager for temporary predictions
with classifier.temporary_device('cpu'):
    result = classifier.predict('large_image.png')
```

## 📝 Logging

### Basic Logging

```python
import logging
from breast_cancer_classifier import BreastCancerClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

classifier = BreastCancerClassifier()
result = classifier.predict('image.png')
logger.info(f"Prediction: {result['class']} ({result['confidence']:.2%})")
```

### Advanced Logging

```python
# Custom logger configuration
import logging.config

LOGGING_CONFIG = {
    'version': 1,
    'handlers': {
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'classifier.log',
            'formatter': 'detailed',
        },
    },
    'formatters': {
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'loggers': {
        'breast_cancer_classifier': {
            'handlers': ['file'],
            'level': 'INFO',
        },
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
```

## 🔍 Monitoring & Metrics

### Performance Metrics

```python
from breast_cancer_classifier.monitoring import PerformanceMonitor

monitor = PerformanceMonitor()

with monitor.track_prediction():
    result = classifier.predict('image.png')

print(f"Prediction time: {monitor.last_prediction_time:.3f}s")
print(f"Memory usage: {monitor.peak_memory_mb:.1f}MB")
```

### Model Metrics

```python
# Get model performance metrics
metrics = classifier.get_performance_metrics()
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"F1-Score: {metrics['f1_score']:.3f}")
print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
```

This API documentation provides a comprehensive guide for integrating the breast cancer classification system into your applications. For more examples and updates, visit the [GitHub repository](https://github.com/yourusername/breast-cancer-ultrasound).
