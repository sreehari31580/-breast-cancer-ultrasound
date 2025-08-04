# 🔬 Breast Cancer Ultrasound Classification

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

> **AI-powered breast cancer detection using ultrasound images with 96.58% accuracy**

An end-to-end deep learning solution for automated breast cancer classification from ultrasound images. Built with PyTorch and deployed as an interactive web application using Streamlit.

![Demo](docs/images/demo_prediction.png)

## 🎯 Project Overview

This project implements a state-of-the-art deep learning model for classifying breast ultrasound images into three categories:
- **Benign** (0): Non-cancerous lesions
- **Malignant** (1): Cancerous lesions  
- **Normal** (2): Healthy tissue

### 🏆 Key Achievements
- **96.58% Test Accuracy** on BUSI dataset
- **97% F1-Score** across all classes
- **Real-time predictions** with GradCAM visualization
- **Production-ready web application** with user authentication
- **Comprehensive model validation** and testing suite

## 📊 Model Performance

| Metric | Benign | Malignant | Normal | Overall |
|--------|---------|-----------|---------|---------|
| **Precision** | 94% | 100% | 97% | 97% |
| **Recall** | 97% | 94% | 97% | 96% |
| **F1-Score** | 95% | 97% | 97% | 97% |

### Architecture
- **Base Model**: EfficientNet-B0 (pre-trained on ImageNet)
- **Parameters**: 4,011,391 total parameters
- **Model Size**: 15.6 MB
- **Input**: 224×224 RGB images
- **Output**: 3-class classification

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 4GB+ RAM

### Installation

> **⚠️ Important**: The trained model file (`fixed_best_model.pth`) is too large for GitHub. Please download it separately from the [Releases page](https://github.com/sreehari31580/-breast-cancer-ultrasound/releases) or contact the repository owner.

```

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/breast-cancer-ultrasound.git
cd breast-cancer-ultrasound
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download the dataset**
   - Download the [BUSI dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
   - Extract to `Dataset_BUSI_with_GT/` directory

5. **Run the web application**
```bash
streamlit run webapp/app.py
```

6. **Access the application**
   - Open your browser to `http://localhost:8501`
   - Upload ultrasound images for instant classification

## 📁 Project Structure

```
breast-cancer-ultrasound/
├── 📊 Dataset_BUSI_with_GT/          # BUSI dataset
│   ├── benign/                       # Benign ultrasound images
│   ├── malignant/                    # Malignant ultrasound images
│   └── normal/                       # Normal ultrasound images
├── 🧠 fixed_best_model.pth           # Best trained model (96.58% accuracy)
├── 🌐 webapp/                        # Streamlit web application
│   ├── app.py                        # Main application interface
│   ├── auth.py                       # User authentication
│   ├── requirements.txt              # Web app dependencies
│   └── utils/                        # Utility functions
├── 🔬 src/                           # Source code
│   ├── fixed_training.py             # Model training pipeline
│   ├── test_fixed_model.py           # Model evaluation
│   ├── ensemble_predict.py           # Ensemble methods
│   ├── utils/                        # Utility modules
│   │   └── gradcam_util.py           # GradCAM visualization
│   └── check_individual_models.py    # Model comparison
├── 📈 cnn_data/                      # Processed datasets
│   ├── labels.csv                    # Image labels
│   ├── train.csv                     # Training split
│   ├── test.csv                      # Test split
│   └── val.csv                       # Validation split
├── 🧪 test_fixed_webapp.py           # Standalone testing
├── 🔍 webapp_validation_suite.py     # Comprehensive validation
├── 📋 requirements.txt               # Main dependencies
├── 📖 MODEL_SUMMARY.md               # Detailed model documentation
└── 🏗️ advanced_training_pipeline.py  # Advanced training experiments
```

## 🔧 Usage

### Web Application

1. **Start the application**
```bash
streamlit run webapp/app.py
```

2. **Upload an image**
   - Drag & drop or browse for ultrasound images
   - Supports PNG, JPG, JPEG formats

3. **Get predictions**
   - View classification results with confidence scores
   - Explore GradCAM heatmaps showing model attention
   - Download results for further analysis

### Programmatic Usage

```python
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Load the trained model
def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(1280, 3)
    model.load_state_dict(torch.load('fixed_best_model.pth'))
    model.eval()
    return model

# Preprocess image
def preprocess_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

# Make prediction
model = load_model()
image_tensor = preprocess_image('path/to/ultrasound.png')

with torch.no_grad():
    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)
    prediction = torch.argmax(probabilities, dim=1).item()
    
classes = ['Benign', 'Malignant', 'Normal']
print(f"Prediction: {classes[prediction]}")
print(f"Confidence: {probabilities[0][prediction].item():.2%}")
```

## 🧪 Testing & Validation

### Run Comprehensive Tests
```bash
python webapp_validation_suite.py
```

### Individual Component Tests
```bash
# Test model performance
python src/test_fixed_model.py

# Test web app functionality  
python test_fixed_webapp.py

# Compare multiple models
python src/check_individual_models.py
```

### Expected Test Results
```
============================================================
🧪 BREAST CANCER ULTRASOUND WEB APP VALIDATION
============================================================
Model Loading             ✅ PASSED
Preprocessing Pipeline    ✅ PASSED
GradCAM Functionality     ✅ PASSED  
Class Distribution        ✅ PASSED
Overall: 4/4 tests passed
🎉 ALL TESTS PASSED! Your web app is ready for deployment!
```

## 🎨 Model Interpretability

### GradCAM Visualization
The application includes GradCAM (Gradient-weighted Class Activation Mapping) to visualize which regions of the ultrasound image the model focuses on for its predictions.

![GradCAM Example](gradcam_outputs/gradcam_example.png)

### Features:
- **Heat maps** showing model attention
- **Region highlighting** for suspicious areas
- **Confidence scoring** for each prediction
- **Interactive visualization** in the web app

## 📚 Dataset Information

### BUSI Dataset (Breast Ultrasound Images)
- **Total Images**: 780 ultrasound images
- **Classes**: 3 (Benign, Malignant, Normal)
- **Format**: PNG images with corresponding masks
- **Resolution**: Variable (resized to 224×224 for training)

### Data Preprocessing
1. **Mask Application**: Benign and malignant images are masked to focus on lesion areas
2. **Normalization**: Images normalized to [0,1] range
3. **Augmentation**: Random flips, rotations, and color jittering for training
4. **Resizing**: All images resized to 224×224 pixels

## 🏗️ Model Training

### Training Pipeline
```bash
# Train the model from scratch
python src/fixed_training.py

# Advanced training with different architectures
python advanced_training_pipeline.py

# Train ensemble models
python src/ensemble_predict.py
```

### Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Cross-Entropy with class weights
- **Batch Size**: 16
- **Epochs**: 20
- **Scheduler**: ReduceLROnPlateau
- **Early Stopping**: Patience of 5 epochs

### Data Splits
- **Training**: 80% (624 images)
- **Validation**: 20% (156 images)
- **Stratified sampling** to maintain class balance

## 🚀 Deployment

### Local Deployment
```bash
streamlit run webapp/app.py
```

### Docker Deployment
```bash
cd webapp
docker build -t breast-cancer-classifier .
docker run -p 8501:8501 breast-cancer-classifier
```

### Cloud Deployment
The application can be deployed on:
- **Streamlit Cloud** (recommended)
- **Heroku**
- **AWS EC2**
- **Google Cloud Platform**
- **Azure Container Instances**

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Commit your changes: `git commit -m 'Add amazing feature'`
5. Push to the branch: `git push origin feature/amazing-feature`
6. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **BUSI Dataset**: Thanks to the creators of the Breast Ultrasound Images Dataset
- **PyTorch Team**: For the excellent deep learning framework
- **Streamlit**: For the user-friendly web app framework
- **Medical Community**: For providing valuable feedback and validation

## 📞 Contact

- **Author**: Sreehari VS
- **Email**: Sreeharivs31580@gmail.com
- **LinkedIn**: https://www.linkedin.com/in/sreehari-vs/
- **Project Link**: https://github.com/sreehari31580/-breast-cancer-ultrasound.git

## 🔗 Related Work

- [Original BUSI Dataset Paper](https://doi.org/10.1016/j.dib.2019.104863)
- [EfficientNet Architecture](https://arxiv.org/abs/1905.11946)
- [GradCAM Visualization](https://arxiv.org/abs/1610.02391)

---

⭐ If you find this project helpful, please consider giving it a star!

🐛 Found a bug? [Open an issue](https://github.com/yourusername/breast-cancer-ultrasound/issues)

💡 Have a feature request? [Start a discussion](https://github.com/yourusername/breast-cancer-ultrasound/discussions)
