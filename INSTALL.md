# Installation Guide - Breast Cancer Ultrasound Classification

This guide provides detailed installation instructions for different environments and use cases.

## 🎯 Quick Installation (Recommended)

### Prerequisites
- Python 3.8 or higher
- Git
- 4GB+ RAM
- CUDA-compatible GPU (optional but recommended)

### One-Command Setup
```bash
git clone https://github.com/yourusername/breast-cancer-ultrasound.git
cd breast-cancer-ultrasound
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run webapp/app.py
```

## 🔧 Detailed Installation

### 1. System Requirements

#### Minimum Requirements
- **OS**: Windows 10, macOS 10.14, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 - 3.11
- **RAM**: 4GB (8GB recommended)
- **Storage**: 2GB free space
- **Internet**: Required for initial setup

#### Recommended Requirements
- **OS**: Windows 11, macOS 12+, or Linux (Ubuntu 20.04+)
- **Python**: 3.9 or 3.10
- **RAM**: 8GB+ 
- **GPU**: CUDA-compatible GPU with 4GB+ VRAM
- **Storage**: 5GB+ free space

### 2. Python Environment Setup

#### Option A: Using venv (Recommended)
```bash
# Create virtual environment
python -m venv breast-cancer-env

# Activate environment
# On Windows:
breast-cancer-env\Scripts\activate
# On macOS/Linux:
source breast-cancer-env/bin/activate

# Verify Python version
python --version  # Should be 3.8+
```

#### Option B: Using conda
```bash
# Create conda environment
conda create -n breast-cancer python=3.9
conda activate breast-cancer

# Verify installation
python --version
```

#### Option C: Using pyenv (Advanced)
```bash
# Install specific Python version
pyenv install 3.9.16
pyenv local 3.9.16

# Create virtual environment
python -m venv venv
source venv/bin/activate
```

### 3. Clone Repository

```bash
# Clone the repository
git clone https://github.com/yourusername/breast-cancer-ultrasound.git
cd breast-cancer-ultrasound

# Verify repository structure
ls -la  # Should see README.md, requirements.txt, webapp/, etc.
```

### 4. Install Dependencies

#### Core Dependencies
```bash
# Install main requirements
pip install -r requirements.txt

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### Web Application Dependencies
```bash
# Install web app specific requirements
pip install -r webapp/requirements.txt

# Verify Streamlit installation
streamlit --version
```

#### Development Dependencies (Optional)
```bash
# For contributors and developers
pip install pytest black flake8 jupyter

# For advanced model development
pip install tensorboard wandb
```

### 5. GPU Setup (Optional but Recommended)

#### NVIDIA GPU Setup
```bash
# Check CUDA availability
nvidia-smi

# Install CUDA-enabled PyTorch (if not already installed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU support
python -c "import torch; print(f'GPU Count: {torch.cuda.device_count()}')"
```

#### AMD GPU Setup (ROCm)
```bash
# Install ROCm PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2
```

#### Apple Silicon (M1/M2) Setup
```bash
# Install MPS-enabled PyTorch
pip install torch torchvision torchaudio

# Verify MPS support
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### 6. Dataset Setup

#### Download BUSI Dataset
1. **Manual Download** (Recommended)
   - Visit [BUSI Dataset on Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
   - Download and extract to `Dataset_BUSI_with_GT/`

2. **Using Kaggle API** (Advanced)
```bash
# Install Kaggle API
pip install kaggle

# Configure API credentials (requires Kaggle account)
kaggle datasets download -d aryashah2k/breast-ultrasound-images-dataset
unzip breast-ultrasound-images-dataset.zip -d Dataset_BUSI_with_GT/
```

#### Verify Dataset Structure
```bash
# Check dataset structure
ls Dataset_BUSI_with_GT/
# Should show: benign/, malignant/, normal/

# Count images
find Dataset_BUSI_with_GT/ -name "*.png" | wc -l
# Should show ~1560 files (780 images + 780 masks)
```

### 7. Model Setup

#### Download Pre-trained Model
The repository includes `fixed_best_model.pth` (96.58% accuracy). If not present:

```bash
# The model should be included in the repository
ls -la fixed_best_model.pth

# If missing, you can train from scratch
python src/fixed_training.py
```

#### Verify Model Loading
```bash
# Test model loading
python -c "
import torch
from torchvision import models
model = models.efficientnet_b0(weights=None)
model.classifier[1] = torch.nn.Linear(1280, 3)
model.load_state_dict(torch.load('fixed_best_model.pth', map_location='cpu'))
print('Model loaded successfully!')
"
```

### 8. Verification & Testing

#### Quick Verification
```bash
# Run validation suite
python webapp_validation_suite.py

# Expected output:
# ✅ Model Loading: PASSED
# ✅ Preprocessing Pipeline: PASSED  
# ✅ GradCAM Functionality: PASSED
# ✅ Class Distribution: PASSED
```

#### Test Web Application
```bash
# Start the web app
streamlit run webapp/app.py

# Open browser to http://localhost:8501
# Upload a test image and verify predictions
```

#### Test Model Performance
```bash
# Run model evaluation
python src/test_fixed_model.py

# Expected output should show ~96% accuracy
```

## 🐳 Docker Installation

### Using Docker (Alternative Installation)

```bash
# Build Docker image
cd webapp
docker build -t breast-cancer-classifier .

# Run Docker container
docker run -p 8501:8501 breast-cancer-classifier

# Access at http://localhost:8501
```

### Docker Compose
```bash
# Create docker-compose.yml (if not present)
# Then run:
docker-compose up
```

## ☁️ Cloud Installation

### Google Colab
```python
# In a Colab notebook
!git clone https://github.com/yourusername/breast-cancer-ultrasound.git
%cd breast-cancer-ultrasound
!pip install -r requirements.txt

# Run the app with ngrok tunnel
!pip install pyngrok
from pyngrok import ngrok
!streamlit run webapp/app.py &
public_url = ngrok.connect(port='8501')
print(f'Access the app at: {public_url}')
```

### AWS EC2
```bash
# On EC2 instance
sudo apt update
sudo apt install python3-pip git
git clone https://github.com/yourusername/breast-cancer-ultrasound.git
cd breast-cancer-ultrasound
pip3 install -r requirements.txt
streamlit run webapp/app.py --server.port 8501 --server.address 0.0.0.0
```

### Streamlit Cloud
1. Fork the repository
2. Connect your GitHub account to Streamlit Cloud
3. Deploy `webapp/app.py`
4. Set Python version to 3.9

## 🔧 Troubleshooting

### Common Issues

#### Issue: "torch not found"
```bash
# Solution: Install PyTorch
pip install torch torchvision torchaudio
```

#### Issue: "CUDA out of memory"
```bash
# Solution: Reduce batch size or use CPU
export CUDA_VISIBLE_DEVICES=""  # Force CPU usage
```

#### Issue: "ModuleNotFoundError: No module named 'streamlit'"
```bash
# Solution: Install Streamlit
pip install streamlit
```

#### Issue: "Model file not found"
```bash
# Solution: Verify model path
ls -la fixed_best_model.pth
# If missing, check if it's in a different location or train from scratch
```

#### Issue: "Dataset not found"
```bash
# Solution: Verify dataset structure
ls Dataset_BUSI_with_GT/
# Should contain benign/, malignant/, normal/ directories
```

### Performance Issues

#### Slow Loading
- Use SSD storage for dataset
- Ensure sufficient RAM (8GB+)
- Use GPU acceleration if available

#### Memory Issues
- Reduce batch size in training scripts
- Use CPU instead of GPU for inference
- Close other applications

### Platform-Specific Issues

#### Windows
- Use PowerShell or Command Prompt
- Ensure Python is added to PATH
- Use `python` instead of `python3`

#### macOS
- Use Terminal
- Install Xcode Command Line Tools: `xcode-select --install`
- Use `python3` and `pip3`

#### Linux
- Ensure Python development headers: `sudo apt install python3-dev`
- For CUDA: Install NVIDIA drivers and CUDA toolkit

## 📞 Getting Help

If you encounter issues:

1. **Check the troubleshooting section above**
2. **Search existing GitHub issues**
3. **Run the validation suite**: `python webapp_validation_suite.py`
4. **Create a GitHub issue** with:
   - Operating system and version
   - Python version
   - Complete error message
   - Steps to reproduce

## ✅ Verification Checklist

After installation, verify these work:

- [ ] Python environment activated
- [ ] All dependencies installed
- [ ] Dataset downloaded and extracted
- [ ] Model file present and loadable
- [ ] Web application starts successfully
- [ ] Test predictions work correctly
- [ ] Validation suite passes all tests

```bash
# Final verification command
python webapp_validation_suite.py && echo "✅ Installation successful!"
```

Congratulations! You're ready to use the Breast Cancer Ultrasound Classification system! 🎉
