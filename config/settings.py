"""
Configuration settings for Child Safety System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
LOG_DIR = BASE_DIR / "logs"

# Ensure directories exist
for directory in [DATA_DIR, MODEL_DIR, OUTPUT_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Camera settings
CAMERAS = {
    'camera_1': {
        'device_id': 0,
        'name': 'Gate Camera',
        'location': 'Main Entrance',
        'fps': 30,
        'resolution': (640, 480)
    },
    'camera_2': {
        'device_id': 2,
        'name': 'Playground Camera',
        'location': 'Central Area',
        'fps': 30,
        'resolution': (640, 480)
    },
    'camera_3': {
        'device_id': 4,
        'name': 'Exit Camera',
        'location': 'Exit Gate',
        'fps': 30,
        'resolution': (640, 480)
    }
}

# Detection settings
DETECTION = {
    'confidence_threshold': 0.5,
    'nms_threshold': 0.4,
    'min_detection_size': (30, 30),
    'target_fps': 30,
}

# Age classification
AGE_THRESHOLD = 12  # Below = child, above = adult
AGE_CONFIDENCE_THRESHOLD = 0.7

# Emotion detection
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
DISTRESS_EMOTIONS = ['Fear', 'Sad', 'Angry']
EMOTION_CONFIDENCE_THRESHOLD = 0.6

# Alert thresholds
ALERTS = {
    'unattended_child_time': 600,  # 10 minutes in seconds
    'suspicious_confidence': 0.75,
    'high_priority_threshold': 0.9,
    'medium_priority_threshold': 0.7,
    'buzzer_duration': 2,
    'alert_cooldown': 60,  # seconds between duplicate alerts
}

# GPIO pins (Raspberry Pi)
GPIO_PINS = {
    'buzzer': 17,
    'led_red': 27,
    'led_green': 22,
    'emergency_button': 23
}

# Enable GPIO (set to True on Raspberry Pi)
ENABLE_GPIO = os.getenv('ENABLE_GPIO', 'False').lower() == 'true'

# API settings
API = {
    'host': os.getenv('API_HOST', '0.0.0.0'),
    'port': int(os.getenv('API_PORT', 5000)),
    'debug': os.getenv('API_DEBUG', 'True').lower() == 'true'
}

# Twilio (SMS)
TWILIO = {
    'account_sid': os.getenv('TWILIO_ACCOUNT_SID'),
    'auth_token': os.getenv('TWILIO_AUTH_TOKEN'),
    'phone_number': os.getenv('TWILIO_PHONE'),
    'recipient_numbers': os.getenv('RECIPIENT_NUMBERS', '').split(',')
}

# Firebase (Push notifications)
FIREBASE = {
    'credentials_path': os.getenv('FIREBASE_CREDENTIALS_PATH', 'config/firebase-credentials.json'),
    'topic': 'security_guards'
}

# Database
DATABASE = {
    'path': os.getenv('DATABASE_PATH', str(DATA_DIR / 'alerts.db'))
}

# Logging
LOGGING = {
    'level': os.getenv('LOG_LEVEL', 'INFO'),
    'file': LOG_DIR / 'system.log',
    'max_bytes': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5,
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}

# Model paths
MODEL_PATHS = {
    'yolo': MODEL_DIR / 'yolo' / 'yolov8n.pt',
    'age': MODEL_DIR / 'age' / 'age_classifier.pth',
    'emotion': MODEL_DIR / 'emotion' / 'emotion_model.h5',
    'face': MODEL_DIR / 'face' / 'facenet_model.pth',
}

# Tracking settings
TRACKING = {
    'max_age': 30,  # Maximum frames to keep track without detection
    'n_init': 3,  # Minimum detections before track is confirmed
    'max_iou_distance': 0.7,  # Maximum IOU distance for matching
    'max_cosine_distance': 0.3,  # Maximum cosine distance for appearance matching
}

# Performance optimization
OPTIMIZATION = {
    'use_tensorrt': False,  # Enable TensorRT optimization (NVIDIA only)
    'use_tflite': False,  # Use TensorFlow Lite models
    'frame_skip': 1,  # Process every Nth frame (1 = process all)
    'resize_before_detection': True,
    'detection_resolution': (640, 480),
}
