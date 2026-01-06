# 🎓 Model Training Guide

Complete guide for training Age Classification and Emotion Detection models.

## 📦 Required Datasets

### 1. UTKFace (Age Classification)
- **Size**: ~3 GB
- **Images**: 20,000+
- **URL**: https://susanqq.github.io/UTKFace/
- **Format**: `[age]_[gender]_[race]_[timestamp].jpg`

### 2. FER2013 (Emotion Detection)
- **Size**: ~300 MB
- **Images**: 35,887
- **URL**: https://www.kaggle.com/datasets/msambare/fer2013
- **Classes**: 7 emotions (angry, disgust, fear, happy, sad, surprise, neutral)

## 🚀 Quick Start

### Step 1: Download Datasets
```bash
python scripts/download_datasets.py
```

This will show instructions for downloading both datasets.

### Step 2: Verify Structure
```
data/datasets/
├── UTKFace/
│   ├── 1_0_0_20161219140623618.jpg
│   ├── 2_0_1_20161219203650988.jpg
│   └── ...
└── FER2013/
    ├── fer2013.csv
    OR
    ├── train/
    │   ├── angry/
    │   ├── happy/
    │   └── ...
    └── test/
```

### Step 3: Train Models
```bash
# Train Age Classifier (6-8 hours on CPU)
python training/train_age_model.py

# Train Emotion Detector (6-8 hours on CPU)
python training/train_emotion_model.py
```

## 📊 Training Details

### Age Classification Model

**Architecture**: ResNet18
**Framework**: PyTorch
**Input Size**: 224x224x3
**Output Classes**: 5 age groups
- 0-12 (Child)
- 13-19 (Teen)
- 20-35 (Young Adult)
- 36-60 (Adult)
- 60+ (Senior)

**Training Configuration**:
```python
{
    'batch_size': 32,
    'num_epochs': 20,
    'learning_rate': 0.001,
    'optimizer': 'Adam',
    'scheduler': 'ReduceLROnPlateau',
    'device': 'cuda' or 'cpu'
}
```

**Data Augmentation**:
- Random horizontal flip
- Random rotation (±10°)
- Color jitter (brightness ±20%, contrast ±20%)
- Normalization (ImageNet stats)

**Expected Results**:
- Training Accuracy: ~85-90%
- Validation Accuracy: >85%
- Training Time: 6-8 hours (CPU), 1-2 hours (GPU)

### Emotion Detection Model

**Architecture**: Custom CNN
**Framework**: TensorFlow/Keras
**Input Size**: 48x48x1 (grayscale)
**Output Classes**: 7 emotions

**Model Layers**:
- 4 Convolutional blocks (64, 128, 256, 512 filters)
- Batch Normalization after each conv layer
- MaxPooling after each block
- Dropout (0.25 after pooling, 0.5 in FC layers)
- 2 Fully connected layers (512, 256 units)
- Softmax output layer

**Training Configuration**:
```python
{
    'batch_size': 64,
    'num_epochs': 50,
    'learning_rate': 0.001,
    'optimizer': 'Adam',
    'callbacks': ['ModelCheckpoint', 'EarlyStopping', 'ReduceLROnPlateau']
}
```

**Data Augmentation**:
- Rotation range: ±20°
- Width/Height shift: ±20%
- Horizontal flip: Yes
- Zoom range: ±20%
- Shear range: 0.2

**Expected Results**:
- Training Accuracy: ~85-90%
- Validation Accuracy: >80%
- Training Time: 6-8 hours (CPU), 2-3 hours (GPU)

## 💻 Hardware Requirements

### Minimum (CPU Training)
- CPU: Intel i5 or AMD Ryzen 5
- RAM: 8 GB
- Storage: 10 GB free space
- Time: 12-16 hours total

### Recommended (GPU Training)
- GPU: NVIDIA GTX 1060 or better (4GB VRAM)
- RAM: 16 GB
- Storage: 10 GB free space
- Time: 3-5 hours total

### Cloud Options
- **Google Colab**: Free GPU (limited hours)
- **Kaggle Notebooks**: Free GPU (30 hours/week)
- **AWS SageMaker**: Paid GPU instances

## 🐍 Python Environment

### Using Python 3.11 (Recommended)
```bash
# Create virtual environment
python3.11 -m venv venv_train

# Activate
# Windows:
venv_train\Scripts\activate
# Linux/Mac:
source venv_train/bin/activate

# Install dependencies
pip install torch torchvision tensorflow opencv-python pillow pandas numpy
```

### Python 3.14 Limitations
- ❌ TensorFlow not available yet
- ✅ PyTorch works fine
- ⚠️ Use Python 3.11 for both models

## 📈 Monitoring Training

### Age Model Output
```
Epoch 1/20
----------------------------------------------------------
Train Loss: 0.8123, Train Acc: 62.45%
Val Loss: 0.7234, Val Acc: 65.23%
✓ Best model saved (Val Acc: 65.23%)

Epoch 2/20
----------------------------------------------------------
Train Loss: 0.6543, Train Acc: 70.12%
Val Loss: 0.6789, Val Acc: 68.45%
...
```

### Emotion Model Output
```
Epoch 1/50: Loss: 1.8234, Acc: 32.45%, Val Acc: 35.12%
Epoch 2/50: Loss: 1.6543, Acc: 38.67%, Val Acc: 40.23%
...
Epoch 50/50: Loss: 0.5234, Acc: 89.12%, Val Acc: 82.34%

Training complete!
Final validation accuracy: 82.34%
Model saved to: models/emotion/emotion_model.h5
```

## 💾 Model Output

After training, models are saved to:
```
models/
├── age/
│   └── age_classifier.pth        # Age model (ResNet18)
└── emotion/
    ├── emotion_model.h5          # Emotion model (CNN)
    └── emotion_model_history.npy # Training history
```

## 🔄 Using Trained Models

### Load Age Model
```python
import torch
from torchvision import models

model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 5)
checkpoint = torch.load('models/age/age_classifier.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Load Emotion Model
```python
from tensorflow import keras

model = keras.models.load_model('models/emotion/emotion_model.h5')
```

## 🐛 Troubleshooting

### Out of Memory
- Reduce batch size: `batch_size = 16`
- Use smaller model
- Close other applications

### Training Too Slow
- Use GPU if available
- Reduce number of epochs
- Use smaller dataset subset for testing

### Low Accuracy
- Train for more epochs
- Adjust learning rate
- Try different data augmentation
- Check dataset quality

### CUDA Errors
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Use CPU instead
# Edit training script: device = 'cpu'
```

## 📊 Hyperparameter Tuning

Try these variations for better accuracy:

### Learning Rate
```python
learning_rate = 0.0001  # Lower for fine-tuning
learning_rate = 0.01    # Higher for faster initial training
```

### Batch Size
```python
batch_size = 16   # Less memory, slower
batch_size = 64   # More memory, faster
batch_size = 128  # Large batches for GPU
```

### Epochs
```python
num_epochs = 30   # Age model
num_epochs = 100  # Emotion model (with early stopping)
```

## 🎯 Expected Timeline

| Task | Time (CPU) | Time (GPU) |
|------|------------|------------|
| Download datasets | 30 min | 30 min |
| Setup environment | 15 min | 15 min |
| Train age model | 6-8 hours | 1-2 hours |
| Train emotion model | 6-8 hours | 2-3 hours |
| **Total** | **13-17 hours** | **4-6 hours** |

## 🚀 Advanced: Google Colab

### Upload Scripts to Colab
1. Visit: https://colab.research.google.com
2. Upload `train_age_model.py` or `train_emotion_model.py`
3. Upload dataset or mount Google Drive
4. Run cells with GPU enabled

### Enable GPU
Runtime → Change runtime type → GPU (T4)

## 📝 Model Evaluation

After training, evaluate models:
```bash
# Create evaluation script
python scripts/evaluate_models.py
```

Check metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## 🔗 Related Files

- Training Scripts: `training/`
- Dataset Helper: `scripts/download_datasets.py`
- Model Integration: `src/detection/age_classifier.py`, `src/detection/emotion_detector.py`

---

**Status**: ✅ Complete Training Pipeline  
**Last Updated**: January 6, 2026
