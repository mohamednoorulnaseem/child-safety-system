"""
Emotion Detection Module
Detects facial emotions to identify distressed children
Uses CNN trained on FER2013 dataset
"""
import cv2
import numpy as np
from typing import Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import (MODEL_PATHS, EMOTION_LABELS, DISTRESS_EMOTIONS, 
                             EMOTION_CONFIDENCE_THRESHOLD)
from src.utils.logger import setup_logger

logger = setup_logger('EmotionDetector')


class EmotionDetector:
    """
    CNN-based emotion detector for facial expression recognition.
    Identifies 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize emotion detection model.
        
        Args:
            model_path: Path to trained Keras model (.h5 file)
        """
        self.model_path = model_path or str(MODEL_PATHS['emotion'])
        self.emotion_labels = EMOTION_LABELS
        self.distress_emotions = DISTRESS_EMOTIONS
        self.confidence_threshold = EMOTION_CONFIDENCE_THRESHOLD
        
        logger.info(f"Initializing EmotionDetector with model: {self.model_path}")
        
        # Initialize face detection cascade
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        try:
            # Try to load TensorFlow/Keras model
            import tensorflow as tf
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info("Emotion model loaded successfully")
            self.model_loaded = True
        except FileNotFoundError:
            logger.warning(f"Model file not found: {self.model_path}")
            logger.info("Creating dummy model - needs training")
            self.model_loaded = False
            self._create_dummy_model()
        except Exception as e:
            logger.error(f"Failed to load emotion model: {e}")
            self.model_loaded = False
            self._create_dummy_model()
    
    def _create_dummy_model(self):
        """Create dummy model for testing (returns random predictions)."""
        import tensorflow as tf
        from tensorflow.keras import layers, models
        
        # Create simple CNN architecture
        self.model = models.Sequential([
            layers.Input(shape=(48, 48, 1)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(7, activation='softmax')
        ])
        
        logger.info("Created untrained emotion model")
    
    def detect_faces(self, image: np.ndarray) -> list:
        """
        Detect faces in image using Haar Cascade.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            List of face bounding boxes [(x, y, w, h), ...]
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces
    
    def detect_emotion(self, face_image: np.ndarray) -> Tuple[str, float]:
        """
        Detect emotion from face image.
        
        Args:
            face_image: BGR image of face (numpy array)
            
        Returns:
            Tuple of (emotion, confidence) where:
                emotion: One of EMOTION_LABELS
                confidence: float between 0 and 1
        """
        if face_image is None or face_image.size == 0:
            logger.warning("Empty face image received")
            return 'Neutral', 0.0
        
        try:
            # Convert to grayscale
            if len(face_image.shape) == 3:
                gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray_face = face_image
            
            # Resize to model input size (48x48)
            resized_face = cv2.resize(gray_face, (48, 48))
            
            # Normalize pixel values
            normalized_face = resized_face / 255.0
            
            # Reshape for model input
            input_face = normalized_face.reshape(1, 48, 48, 1)
            
            # Predict emotion
            predictions = self.model.predict(input_face, verbose=0)
            emotion_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][emotion_idx])
            
            emotion = self.emotion_labels[emotion_idx]
            
            logger.debug(f"Detected emotion: {emotion} (confidence: {confidence:.2f})")
            
            return emotion, confidence
            
        except Exception as e:
            logger.error(f"Emotion detection failed: {e}")
            return 'Neutral', 0.0
    
    def is_distressed(self, emotion: str, confidence: float) -> bool:
        """
        Check if emotion indicates distress.
        
        Args:
            emotion: Detected emotion label
            confidence: Confidence score
            
        Returns:
            True if person appears distressed, False otherwise
        """
        return (emotion in self.distress_emotions and 
                confidence >= self.confidence_threshold)
    
    def analyze_frame(self, frame: np.ndarray) -> list:
        """
        Detect all faces and their emotions in frame.
        
        Args:
            frame: Input image (BGR)
            
        Returns:
            List of dicts containing:
                - bbox: (x, y, w, h)
                - emotion: str
                - confidence: float
                - is_distressed: bool
        """
        results = []
        
        # Detect faces
        faces = self.detect_faces(frame)
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = frame[y:y+h, x:x+w]
            
            # Detect emotion
            emotion, confidence = self.detect_emotion(face_roi)
            
            # Check if distressed
            distressed = self.is_distressed(emotion, confidence)
            
            result = {
                'bbox': (x, y, w, h),
                'emotion': emotion,
                'confidence': confidence,
                'is_distressed': distressed
            }
            
            results.append(result)
        
        return results
    
    def draw_emotions(self, frame: np.ndarray, results: list) -> np.ndarray:
        """
        Draw emotion labels on frame.
        
        Args:
            frame: Input image
            results: List of emotion detection results
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        for result in results:
            x, y, w, h = result['bbox']
            emotion = result['emotion']
            confidence = result['confidence']
            is_distressed = result['is_distressed']
            
            # Choose color based on distress
            color = (0, 0, 255) if is_distressed else (0, 255, 0)
            
            # Draw rectangle around face
            cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw emotion label
            label = f"{emotion} ({confidence:.2f})"
            cv2.putText(annotated_frame, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add warning if distressed
            if is_distressed:
                cv2.putText(annotated_frame, "DISTRESSED!", (x, y+h+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return annotated_frame


# Training function
def train_emotion_model(data_dir: str, epochs: int = 50, batch_size: int = 64):
    """
    Train emotion detection model on FER2013 dataset.
    
    Args:
        data_dir: Directory containing FER2013 data
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    logger.info("Starting emotion model training...")
    
    # Data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load data
    train_generator = train_datagen.flow_from_directory(
        f"{data_dir}/train",
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical'
    )
    
    val_generator = val_datagen.flow_from_directory(
        f"{data_dir}/validation",
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical'
    )
    
    # Create model
    model = models.Sequential([
        layers.Input(shape=(48, 48, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(7, activation='softmax')
    ])
    
    # Compile
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATHS['emotion'],
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[checkpoint, early_stop]
    )
    
    logger.info("Training complete")
    
    return history


# Test code
if __name__ == '__main__':
    print("Testing EmotionDetector...")
    
    # Initialize detector
    detector = EmotionDetector()
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break
        
        # Analyze emotions
        results = detector.analyze_frame(frame)
        
        # Draw results
        annotated_frame = detector.draw_emotions(frame, results)
        
        # Display statistics
        distressed_count = sum(1 for r in results if r['is_distressed'])
        cv2.putText(annotated_frame, f"Faces: {len(results)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Distressed: {distressed_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Emotion Detection', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
