# Quick Start Guide - Child Safety System

## \ud83d\ude80 5-Minute Setup

### 1. Install Python Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### 2. Download Pre-trained Models

**YOLOv8 Model** (auto-downloads on first run):
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Downloads automatically
```

**For other models**, you'll need to train them (see Training section) or use dummy models.

### 3. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your credentials (optional for testing):
# - TWILIO_ACCOUNT_SID
# - TWILIO_AUTH_TOKEN
# - TWILIO_PHONE
# - FIREBASE_CREDENTIALS_PATH
```

### 4. Initialize Database

```bash
python -c "from src.api.database import init_database; init_database()"
```

### 5. Run the System!

```bash
# Basic run with webcam
python src/main_detector.py

# Multiple cameras
python src/main_detector.py --cameras 0 1 2

# Test mode (single frame)
python src/main_detector.py --test
```

## \ud83d\udcf9 Testing Individual Modules

### Test Person Detection
```bash
python src/detection/person_detector.py
```

### Test Age Classification
```bash
python src/detection/age_classifier.py
```

### Test Emotion Detection
```bash
python src/detection/emotion_detector.py
```

### Test Pose Analysis
```bash
python src/detection/pose_analyzer.py
```

### Test Face Recognition
```bash
python src/detection/face_recognizer.py
```

### Test Multi-Object Tracking
```bash
python src/tracking/deep_sort.py
```

### Test Alert System
```bash
python src/alerts/alert_manager.py
```

## \ud83d\udda5\ufe0f Run API Server

```bash
python src/api/app.py
```

API will be available at: `http://localhost:5000`

Test endpoints:
- `http://localhost:5000/` - Health check
- `http://localhost:5000/api/alerts/recent` - Get recent alerts
- `http://localhost:5000/api/alerts/stats` - Get statistics

## \ud83e\uddea Run Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src

# Specific test file
pytest tests/test_detection.py -v
```

## \ud83c\udf93 Training Models (Optional)

### Age Classifier
```python
from src.detection.age_classifier import train_age_classifier

# Download UTKFace dataset first
train_age_classifier(
    data_dir='data/datasets/utkface',
    epochs=20,
    batch_size=32
)
```

### Emotion Detector
```python
from src.detection.emotion_detector import train_emotion_model

# Download FER2013 dataset first
train_emotion_model(
    data_dir='data/datasets/fer2013',
    epochs=50,
    batch_size=64
)
```

## \ud83c\udf0d Raspberry Pi Deployment

### 1. Setup Raspberry Pi OS
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-opencv
```

### 2. Clone and Install
```bash
git clone https://github.com/YOUR_USERNAME/child-safety-system.git
cd child-safety-system
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Enable GPIO
```bash
# Edit .env
ENABLE_GPIO=True

# Add user to GPIO group
sudo usermod -a -G gpio $USER
```

### 4. Configure Cameras
```python
# Edit config/settings.py
CAMERAS = {
    'camera_1': {'device_id': 0, ...},
    'camera_2': {'device_id': 2, ...},
    'camera_3': {'device_id': 4, ...}
}
```

### 5. Auto-start on Boot
```bash
sudo nano /etc/systemd/system/child-safety.service

# Add service configuration (see README)
sudo systemctl enable child-safety.service
sudo systemctl start child-safety.service
```

## \ud83d\udc1b Common Issues

**Issue: YOLOv8 model not found**
```bash
# Solution: It will auto-download on first run
# Or manually download:
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
mv yolov8n.pt models/yolo/
```

**Issue: Camera not detected**
```bash
# Check available cameras
ls /dev/video*

# Test camera
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
```

**Issue: Out of memory on Raspberry Pi**
```python
# Solution: Reduce resolution in config/settings.py
DETECTION = {
    'detection_resolution': (416, 416),  # Lower resolution
    'frame_skip': 2  # Process every 2nd frame
}
```

**Issue: Slow FPS**
```bash
# Use TensorFlow Lite for models
# Reduce camera resolution
# Enable frame skipping
```

## \ud83d\udce6 Project Structure After Setup

```
child_safety_system/
\u251c\u2500\u2500 config/
\u2502   \u2514\u2500\u2500 settings.py          \u2705 Configured
\u251c\u2500\u2500 src/
\u2502   \u251c\u2500\u2500 detection/            \u2705 All modules ready
\u2502   \u251c\u2500\u2500 tracking/             \u2705 Tracker ready
\u2502   \u251c\u2500\u2500 alerts/              \u2705 Alert system ready
\u2502   \u251c\u2500\u2500 api/                 \u2705 API server ready
\u2502   \u2514\u2500\u2500 main_detector.py     \u2705 Main system ready
\u251c\u2500\u2500 models/                  \u26a0\ufe0f Download/train models
\u251c\u2500\u2500 data/                    \u2705 Database initialized
\u251c\u2500\u2500 tests/                   \u2705 All tests ready
\u251c\u2500\u2500 .env                     \u26a0\ufe0f Configure credentials
\u2514\u2500\u2500 README.md                \u2705 Documentation complete
```

## \ud83c\udfaf Next Steps

1. **Test the system** with your webcam
2. **Configure API credentials** (Twilio, Firebase) for alerts
3. **Train models** on your specific dataset (optional)
4. **Deploy to Raspberry Pi** with cameras
5. **Build mobile app** using the API
6. **Customize detection parameters** in config/settings.py

## \ud83d\udcde Support

- Check README.md for detailed documentation
- Run tests to verify everything works
- Check logs/ directory for debugging
- Contact team for support

## \u2728 Features Status

- \u2705 Person Detection (YOLOv8)
- \u2705 Age Classification (ResNet18)
- \u2705 Emotion Detection (CNN)
- \u2705 Pose Analysis (MediaPipe)
- \u2705 Face Recognition (FaceNet)
- \u2705 Multi-Object Tracking (DeepSORT)
- \u2705 Multi-Channel Alerts
- \u2705 RESTful API
- \u2705 Database Logging
- \u2705 Comprehensive Tests

**System is ready for testing and deployment!** \ud83c\udf89
