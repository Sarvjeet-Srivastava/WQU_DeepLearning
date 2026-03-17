# WQU_DeepLearning - Deep Learning Projects Guide

Comprehensive deep learning projects demonstrating production-grade neural networks for regression and classification tasks.

---

## 📋 Overview

This guide provides comprehensive instructions for getting started with the WQU Deep Learning projects, including setup, installation, and running training pipelines.

### Projects Included

1. **Project 1: Concrete Strength Prediction** (Regression)
   - Predicts concrete compressive strength from 8 composition features
   - 9 model variations with different architectures and activations
   - Location: `src/p1-concrete_strength_analysis.py`

2. **Project 2: Heart Disease Classification** (Binary Classification)
   - Predicts presence/absence of heart disease from 13 clinical features
   - Feed-forward neural network with ReLU-Sigmoid activations
   - Location: `src/p2-classify_heart_disease.py`

---

## 🚀 System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **RAM**: Minimum 4GB (8GB recommended)
- **GPU**: Optional (CUDA 11.6+) for faster training
- **Git**: Required for cloning repository
- **Disk Space**: ~2GB for data and models

---

## 🌐 GitHub Repository Setup

### Step 1: Create GitHub Repository

#### Option A: Using GitHub Web Interface

1. Navigate to https://github.com/new
2. Enter repository details:
   - **Repository name**: `WQU_DeepLearning`
   - **Description**: `WQU deep learning models`
   - **Visibility**: Public (recommended) or Private
   - **Initialize repository**: **DO NOT** check "Add a README file"
3. Click **"Create repository"**
4. Copy the repository URL shown

#### Option B: Using GitHub CLI

```powershell
# Install GitHub CLI from https://cli.github.com/ (if not already installed)
# Then authenticate
gh auth login

# Create repository
gh repo create WQU_DeepLearning --public --source=. --remote=origin --push
```

### Step 2: Clone or Initialize Repository

#### If Cloning (Already Has Repository)

```bash
# Clone from GitHub
git clone https://github.com/YOUR_USERNAME/WQU_DeepLearning.git
cd WQU_DeepLearning
```

#### If Initializing Local Repository

```powershell
# Navigate to project
cd D:\projects\WQU_DeepLearning

# Initialize git
git init

# Configure user (one time)
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Add files
git add .

# Create initial commit
git commit -m "Initial commit: WQU_DeepLearning-Getting Started Guide"

# Set default branch
git branch -M master

# Add remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/WQU_DeepLearning.git

# Push to GitHub
git push -u origin master
```

### Step 3: Verify GitHub Setup

Visit `https://github.com/YOUR_USERNAME/WQU_DeepLearning` in your browser to confirm files are uploaded.

---

## 🎯 Projects Overview

### Project 1: Concrete Strength Prediction

**File**: `src/p1-concrete_strength_analysis.py` (918 lines)

**Task**: Regression - Predict concrete compressive strength (MPa)

**Dataset**: 1030 samples × 8 features
- Input Features: Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer, Coarse Aggregate, Fine Aggregate, Age
- Target: Concrete Compressive Strength (2.33-82.6 MPa)
- Split: 80% train (824 samples), 20% test (206 samples)

**Models** (9 variations):
1. **CustomManualMLPModel** (3 variations)
   - Hidden sizes: 32, 64, 128 neurons
   - Activation: ReLU (fixed)
   - Architecture: 8→hidden→1

2. **CustomSimpleMLPModel** (3 variations)
   - Hidden size: 32 neurons (fixed)
   - Activations: ReLU, Sigmoid, Tanh
   - Architecture: 8→32→1

3. **CustomDeepMLPModel** (3 variations)
   - Hidden layers: 64, 32 neurons
   - Activations: ReLU, Sigmoid, Tanh
   - Architecture: 8→64→32→1

**Results**:
- Best Model: CustomDeepMLPModel with ReLU
- Test Loss: 0.0750 (12.5% better than baseline)
- All models saved in: `Models/P1/`

**Outputs**:
- `Output/P1/Concrete_Strength_Analysis.png` - Data exploration (target histogram + 8 feature scatter plots)
- `Output/P1/Train Loss Comparison.png` - All 9 model loss curves
- 9 trained model files (.pth format)

**Documentation**: See `Project1 - Concrete Strength.md`

---

### Project 2: Heart Disease Classification

**File**: `src/p2-classify_heart_disease.py` (614 lines)

**Task**: Binary Classification - Predict heart disease presence (0=No, 1=Yes)

**Dataset**: 297 samples × 13 features
- Input Features: Age, Sex, Chest Pain Type, Resting BP, Cholesterol, Fasting Blood Sugar, ECG, Max Heart Rate, Exercise Angina, ST Depression, ST Slope, Vessel Count, Thalassemia
- Target: Heart Disease (Binary 0/1)
- Split: 80% train (237 samples), 20% test (60 samples)

**Model**:
- **CustomManualFNN**: Feed-Forward Neural Network
- Architecture: 13→16→1
- Activations: ReLU (hidden), Sigmoid (output)
- Parameters: 241
- Loss Function: Binary Cross Entropy (BCELoss)

**Results**:
- Train Accuracy: ~87.50%
- Train Loss: 0.3456
- Test Accuracy: ~83.33%
- Test Loss: 0.3821
- Model saved in: `Models/P2/`

**Outputs**:
- `Output/P2/Heart_Disease_Feature_Analysis.png` - Feature pair scatter plots (78 combinations)
- `Output/P2/Train_Loss_Curve.png` - Training loss progression
- `Output/P2/Confusion_Matrix.png` - Classification performance visualization
- 1 trained model file (.pth format)

**Documentation**: See `Project2 - Heart Disease Classification.md`

---

## 📁 Project Structure

```
WQU_DeepLearning/
│
├── 📁 Docs/
│   ├── README.md (this file)
│   ├── Project1 - Concrete Strength.md
│   └── Project2 - Heart Disease Classification.md
│
├── 📁 Data/
│   ├── p1-concrete-data/
│   │   └── Concrete_Data.csv
│   └── p2-heart-disease/
│       └── heart.csv
│
├── 📁 src/
│   ├── p1-concrete_strength_analysis.py
│   └── p2-classify_heart_disease.py
│
├── 📁 utils/
│   ├── logger.py
│   └── check_cuda.py
│
├── 📁 Models/
│   ├── P1/ (9 trained models)
│   └── P2/ (1 trained model)
│
└── 📁 Output/
    ├── P1/ (Concrete visualizations)
    └── P2/ (Heart disease visualizations)
```

---

## 🔧 Installation & Setup

### Step 1: Create Virtual Environment

```bash
# Navigate to project directory
cd D:\projects\WQU_DeepLearning

# Create virtual environment
python -m venv venv

# Activate virtual environment

# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

### Step 2: Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch (choose based on your system)

# Option A: For GPU (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Option B: For CPU only
pip install torch torchvision torchaudio

# Install other required packages
pip install numpy>=1.21.0 pandas>=1.3.0 scikit-learn>=0.24.0 matplotlib>=3.4.0

# Optional: Development tools
pip install pytest black flake8 mypy jupyter
```

### Step 3: Create requirements.txt (Optional)

```bash
# Generate requirements file
pip freeze > requirements.txt

# Later, others can install using:
pip install -r requirements.txt
```

---

## ✅ Verify Installation & Dependencies

### Check Python Version

```bash
python --version  # Should be 3.8 or higher
```

### Check Individual Packages

```bash
# Check PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check NumPy
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"

# Check Pandas
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"

# Check scikit-learn
python -c "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')"

# Check Matplotlib
python -c "import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')"
```

### Check CUDA Availability

```bash
# Using utility script
python utils/check_cuda.py

# Or directly in Python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
```

### Comprehensive Verification

```bash
# Run all checks
python -c "
import sys
print(f'Python: {sys.version}')
import torch
print(f'PyTorch: {torch.__version__}')
import numpy
print(f'NumPy: {numpy.__version__}')
import pandas
print(f'Pandas: {pandas.__version__}')
import sklearn
print(f'Scikit-learn: {sklearn.__version__}')
import matplotlib
print(f'Matplotlib: {matplotlib.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print('✓ All dependencies installed successfully!')
"
```

---

## 📦 Dependency Details

### Core Dependencies

| Package | Version | Purpose | Installation |
|---------|---------|---------|--------------|
| Python | ≥3.8 | Runtime | Pre-installed or download from python.org |
| PyTorch | ≥2.0.0 | Neural networks | `pip install torch` |
| NumPy | ≥1.21.0 | Numerical computing | `pip install numpy` |
| Pandas | ≥1.3.0 | Data manipulation | `pip install pandas` |
| Scikit-learn | ≥0.24.0 | Data preprocessing | `pip install scikit-learn` |
| Matplotlib | ≥3.4.0 | Visualization | `pip install matplotlib` |

### Optional Dependencies (Development)

```bash
# Testing
pip install pytest==7.4.0

# Code formatting
pip install black==23.9.0

# Linting
pip install flake8==6.1.0

# Type checking
pip install mypy==1.5.0

# Jupyter notebooks
pip install jupyter==1.0.0 ipykernel==6.25.0

# Documentation generation
pip install sphinx==7.2.0
```

### GPU Support (Optional)

If you want to use GPU acceleration:

```bash
# Check NVIDIA GPU
nvidia-smi

# Install CUDA Toolkit (from NVIDIA website)
# Download from: https://developer.nvidia.com/cuda-downloads

# Verify CUDA installation
nvcc --version

# Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## 🎯 Running the Project

### Basic Execution

```bash
# Navigate to project root
cd D:\projects\WQU_DeepLearning

# Ensure virtual environment is activated
# venv\Scripts\activate

# Run the training pipeline
python src/p1-concrete_strength_analysis.py
```

### Run with Verbose Output

```bash
python -u src/p1-concrete_strength_analysis.py
```

### Run with Debug Logging

```bash
# Set environment variable for debug level logging
$env:LOGLEVEL = "DEBUG"
python src/p1-concrete_strength_analysis.py
```

---

## 📁 Complete Project Structure

```
WQU_DeepLearning/
│
├── 📁 Docs/
│   ├── README.md (this file)
│   ├── Project1 - Concrete Strength.md
│   └── Project2 - Heart Disease Classification.md
│
├── 📁 Data/
│   ├── p1-concrete-data/
│   │   └── Concrete_Data.csv (1030 samples × 8 features)
│   └── p2-heart-disease/
│       └── heart.csv (297 samples × 13 features)
│
├── 📁 src/
│   ├── p1-concrete_strength_analysis.py (918 lines - Regression)
│   │   ├── CustomManualMLPModel (3 variations: 32/64/128 hidden)
│   │   ├── CustomSimpleMLPModel (3 variations: ReLU/Sigmoid/Tanh)
│   │   ├── CustomDeepMLPModel (3 variations: ReLU/Sigmoid/Tanh)
│   │   └── ConcreteStrengthAnalysis (Pipeline orchestrator)
│   │
│   └── p2-classify_heart_disease.py (614 lines - Binary Classification)
│       ├── CustomManualFNN (13-16-1 architecture)
│       └── HeartDiseaseAnalysis (Pipeline orchestrator)
│
├── 📁 utils/
│   ├── logger.py (Logging configuration)
│   └── check_cuda.py (CUDA availability checker)
│
├── 📁 Models/
│   ├── P1/ (9 trained models for concrete strength)
│   │   ├── CustomDeepMLPModel(8-64-32-1,ReLU).pth ⭐ BEST
│   │   ├── CustomDeepMLPModel(8-64-32-1,Sigmoid).pth
│   │   ├── CustomDeepMLPModel(8-64-32-1,Tanh).pth
│   │   ├── CustomManualMLPModel(8,128,1).pth
│   │   ├── CustomManualMLPModel(8,32,1).pth
│   │   ├── CustomManualMLPModel(8,64,1).pth
│   │   ├── CustomSimpleMLPModel(8,32,1,ReLU).pth
│   │   ├── CustomSimpleMLPModel(8,32,1,Sigmoid).pth
│   │   └── CustomSimpleMLPModel(8,32,1,Tanh).pth
│   │
│   └── P2/ (1 trained model for heart disease)
│       └── CustomManualFNN(13-16-1,ReLU-Sigmoid).pth
│
└── 📁 Output/
    ├── P1/ (Concrete strength visualizations)
    │   ├── Concrete_Strength_Analysis.png (Data exploration)
    │   └── Train Loss Comparison.png (9 models comparison)
    │
    └── P2/ (Heart disease visualizations)
        ├── Heart_Disease_Feature_Analysis.png (78 feature pairs)
        ├── Train_Loss_Curve.png (Training progress)
        └── Confusion_Matrix.png (Classification results)
```

---

## 🎯 Running the Projects

### Project 1: Concrete Strength Prediction

```bash
# Navigate to project root
cd D:\projects\WQU_DeepLearning

# Run pipeline (trains all 9 models)
python src/p1-concrete_strength_analysis.py

# Expected output:
# - Models/P1/: 9 trained models
# - Output/P1/: 2 visualization files
```

**Training Time**: ~5-10 minutes (depends on hardware)

**Expected Results**:
```
Concrete_Strength_Analysis.png: Data exploration
Train Loss Comparison.png: All 9 model comparisons
Best Model Loss: 0.0750 (DeepMLPModel with ReLU)
```

---

### Project 2: Heart Disease Classification

```bash
# Navigate to project root
cd D:\projects\WQU_DeepLearning

# Run pipeline
python src/p2-classify_heart_disease.py

# Expected output:
# - Models/P2/: 1 trained model
# - Output/P2/: 3 visualization files
```

**Training Time**: ~1-2 minutes (depends on hardware)

**Expected Results**:
```
Heart_Disease_Feature_Analysis.png: 78 feature pair plots
Train_Loss_Curve.png: Training progression
Confusion_Matrix.png: Classification results
Train Accuracy: ~87.50%
Test Accuracy: ~83.33%
```

---


## 📚 Additional Resources

### Project Documentation
- **Project1 - Concrete Strength.md**: Complete P1 implementation details
- **Project2 - Heart Disease Classification.md**: Complete P2 implementation details

### Code Documentation
- **src/p1-concrete_strength_analysis.py**: Comprehensive docstrings and type hints
- **src/p2-classify_heart_disease.py**: Comprehensive docstrings and type hints
- **utils/logger.py**: Logging utility documentation
- **utils/check_cuda.py**: CUDA checker utility

### External Resources
- **PyTorch Guide**: https://pytorch.org/tutorials/
- **Scikit-learn Documentation**: https://scikit-learn.org/
- **Pandas Documentation**: https://pandas.pydata.org/
- **Matplotlib Documentation**: https://matplotlib.org/

---

## 💡 Tips & Best Practices

### Activate Virtual Environment Before Working

```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Keep Dependencies Updated

```bash
pip install --upgrade pip
pip install --upgrade torch numpy pandas scikit-learn matplotlib
```

### Use Jupyter for Exploration

```bash
pip install jupyter
jupyter notebook

# Create new notebook and import modules to explore
```

### Monitor Training with Logging

```bash
# Check logs while training
tail -f logs.txt  # macOS/Linux
Get-Content logs.txt -Wait  # Windows
```

---

## ✅ Checklist

### Environment Setup
- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed
- [ ] CUDA checked (optional but recommended for GPU)

### Project 1: Concrete Strength
- [ ] Data file exists at `Data/p1-concrete-data/Concrete_Data.csv`
- [ ] Can run `python src/p1-concrete_strength_analysis.py` without errors
- [ ] Output files generated in `Models/P1/` and `Output/P1/`
- [ ] 9 trained models saved successfully
- [ ] Visualization files created

### Project 2: Heart Disease Classification
- [ ] Data file exists at `Data/p2-heart-disease/heart.csv`
- [ ] Can run `python src/p2-classify_heart_disease.py` without errors
- [ ] Output files generated in `Models/P2/` and `Output/P2/`
- [ ] Trained model saved successfully
- [ ] Visualization files created (3 files)

### Git & Repository
- [ ] Repository initialized/cloned
- [ ] Changes committed to Git
- [ ] Remote configured
- [ ] Push to GitHub successful

---

## 📞 Support

If you encounter issues:

1. Check the **Troubleshooting** section above
2. Review error messages carefully - they provide context
3. Verify dependencies with verification commands
4. Check Python version: `python --version`
5. Read docstrings in source code
6. Review detailed project documentation:
   - `Project1 - Concrete Strength.md`
   - `Project2 - Heart Disease Classification.md`

---

**Status**: ✅ Ready to Start  
**Last Updated**: March 2026  
**Projects**: 2 (Concrete Strength + Heart Disease Classification)

