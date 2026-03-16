# WQU_DeepLearning - Getting Started Guide

A production-grade deep learning project for predicting concrete compressive strength using multiple neural network architectures.

---

## 📋 Overview

This guide provides comprehensive instructions for getting started with the Concrete Strength Prediction project, including GitHub setup, dependency installation, and running the training pipeline.

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

## 📁 Project Structure

```
WQU_DeepLearning/
│
├── 📁 Docs/
│   ├── README.md                    (This file)
│   └── Project1 - Concrete Strength.md
│
├── 📁 src/
│   └── p1-concrete_strength_analysis.py
│       ├── CustomManualMLPModel     (Manual MLP)
│       ├── CustomSimpleMLPModel     (Simple MLP)
│       ├── CustomDeepMLPModel       (Deep MLP)
│       └── ConcreteStrengthAnalysis (Orchestrator)
│
├── 📁 utils/
│   ├── logger.py                    (Logging utility)
│   ├── check_cuda.py               (CUDA checker)
│   └── ...
│
├── 📁 Data/p1-concrete-data/
│   └── Concrete_Data.csv            (1030 samples)
│
├── 📁 Models/P1/                   (Trained models)
│   ├── CustomDeepMLPModel(8-64-32-1,ReLU).pth
│   ├── CustomDeepMLPModel(8-64-32-1,Sigmoid).pth
│   ├── CustomDeepMLPModel(8-64-32-1,Tanh).pth
│   ├── CustomManualMLPModel(8,128,1).pth
│   ├── CustomManualMLPModel(8,32,1).pth
│   ├── CustomManualMLPModel(8,64,1).pth
│   ├── CustomSimpleMLPModel(8,32,1,ReLU).pth
│   ├── CustomSimpleMLPModel(8,32,1,Sigmoid).pth
│   └── CustomSimpleMLPModel(8,32,1,Tanh).pth
│
└── 📁 Images/P1/                   (Outputs)
    ├── Concrete_Strength_Analysis.png
    └── Train Loss Comparison.png
```

---

## 🔍 Troubleshooting

### Issue: Python Not Found

**Error**: `'python' is not recognized as an internal or external command`

**Solution**:
```bash
# Use python3 instead
python3 --version
python3 -m venv venv

# Or add Python to PATH:
# On Windows: Environment Variables → System → Path → Add Python directory
```

### Issue: Module Not Found

**Error**: `ModuleNotFoundError: No module named 'torch'`

**Solution**:
```bash
# Verify virtual environment is activated
# venv\Scripts\activate (Windows)

# Install missing package
pip install torch

# Verify installation
python -c "import torch; print(torch.__version__)"
```

### Issue: Permission Denied (macOS/Linux)

**Error**: `Permission denied` when creating virtual environment

**Solution**:
```bash
# Add execute permission
chmod +x venv/bin/activate

# Or use sudo
sudo python3 -m venv venv
```

### Issue: CUDA Not Available

**Error**: CUDA device not found even though GPU is installed

**Solution**:
```bash
# Check GPU drivers are installed
nvidia-smi

# If CUDA not available, install CPU version and code will still work
pip install torch  # Will use CPU automatically

# Verify fallback works
python -c "import torch; print(f'Using device: {torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")}')"
```

### Issue: Out of Memory

**Error**: `CUDA out of memory` or `MemoryError`

**Solution**:
```bash
# Use CPU instead (slower but works)
# Code automatically falls back to CPU if CUDA fails

# Or reduce training parameters in src/p1-concrete_strength_analysis.py:
# - Reduce num_epochs
# - Reduce batch size (if implementing)
# - Use smaller model
```

### Issue: Data File Not Found

**Error**: `FileNotFoundError: Data file not found`

**Solution**:
```bash
# Ensure data file exists
ls Data/p1-concrete-data/Concrete_Data.csv  # or dir on Windows

# If missing, verify directory structure matches expected layout
```

---

## 📊 Output Files Generated

After running the pipeline, you'll have:

### Generated Models (in Models/P1/)
- 9 trained .pth files (~50KB each)
- Ready for inference
- Can be loaded anytime

### Generated Visualizations (in Images/P1/)
1. **Concrete_Strength_Analysis.png**
   - Data distribution histogram
   - Feature correlations
   - Statistical summaries

2. **Train Loss Comparison.png**
   - Loss curves for all 9 models
   - Performance comparison
   - High-resolution (300 DPI)

---

## 🎯 Next Steps

1. **Clone repository**: https://github.com/YOUR_USERNAME/concrete-strength-prediction
2. **Set up environment**: Create virtual environment and install dependencies
3. **Verify installation**: Run dependency checks
4. **Run pipeline**: Execute `python src/p1-concrete_strength_analysis.py`
5. **Explore results**: Check generated images and models
6. **Use pre-trained models**: Load and use models for predictions
7. **Push improvements**: Commit and push to GitHub

---

## 📚 Additional Resources

- **Project Details**: See `Project1 - Concrete Strength.md`
- **Code Documentation**: See docstrings in `src/p1-concrete_strength_analysis.py`
- **Logging System**: See `utils/logger.py`
- **PyTorch Guide**: https://pytorch.org/tutorials/
- **Scikit-learn Documentation**: https://scikit-learn.org/

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

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed
- [ ] CUDA checked (optional but recommended for GPU)
- [ ] Data file exists at `Data/p1-concrete-data/Concrete_Data.csv`
- [ ] Repository initialized/cloned
- [ ] Can run `python src/p1-concrete_strength_analysis.py` without errors
- [ ] Output files generated in `Models/P1/` and `Images/P1/`
- [ ] Changes committed to Git

---

## 📞 Support

If you encounter issues:

1. Check the **Troubleshooting** section above
2. Review error messages carefully - they provide context
3. Verify dependencies with verification commands
4. Check Python version: `python --version`
5. Read docstrings in source code: `src/p1-concrete_strength_analysis.py`
6. Review detailed project documentation: `Project1 - Concrete Strength.md`

---

**Status**: ✅ Ready to Start  
**Last Updated**: March 2026

