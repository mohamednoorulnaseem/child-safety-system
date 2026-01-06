"""
Unit tests for detection modules
"""
import pytest
import numpy as np
import cv2
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.detection import (
    PersonDetector, AgeClassifier, EmotionDetector,
    PoseAnalyzer, FaceRecognizer
)


@pytest.fixture
def sample_image():
    """Create sample test image."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def person_detector():
    """Create PersonDetector instance."""
    return PersonDetector()


@pytest.fixture
def age_classifier():
    """Create AgeClassifier instance."""
    return AgeClassifier()


@pytest.fixture
def emotion_detector():
    """Create EmotionDetector instance."""
    return EmotionDetector()


@pytest.fixture
def pose_analyzer():
    """Create PoseAnalyzer instance."""
    return PoseAnalyzer()


def test_person_detector_initialization(person_detector):
    """Test PersonDetector initializes correctly."""
    assert person_detector is not None
    assert person_detector.model is not None
    assert person_detector.confidence_threshold > 0


def test_person_detection(person_detector, sample_image):
    """Test person detection returns valid results."""
    detections = person_detector.detect(sample_image)
    
    assert isinstance(detections, list)
    for det in detections:
        assert 'bbox' in det
        assert 'confidence' in det
        assert 'class_id' in det
        assert det['class_id'] == 0  # Person class


def test_age_classifier_initialization(age_classifier):
    """Test AgeClassifier initializes correctly."""
    assert age_classifier is not None
    assert age_classifier.model is not None


def test_age_classification(age_classifier, sample_image):
    """Test age classification returns valid results."""
    age_class, confidence = age_classifier.classify(sample_image)
    
    assert age_class in ['child', 'adult', 'unknown']
    assert 0 <= confidence <= 1


def test_emotion_detector_initialization(emotion_detector):
    """Test EmotionDetector initializes correctly."""
    assert emotion_detector is not None
    assert len(emotion_detector.emotion_labels) == 7


def test_emotion_detection(emotion_detector, sample_image):
    """Test emotion detection returns valid results."""
    emotion, confidence = emotion_detector.detect_emotion(sample_image)
    
    assert emotion in emotion_detector.emotion_labels
    assert 0 <= confidence <= 1


def test_emotion_distress_detection(emotion_detector):
    """Test distress emotion detection."""
    assert emotion_detector.is_distressed('Fear', 0.8) == True
    assert emotion_detector.is_distressed('Happy', 0.9) == False
    assert emotion_detector.is_distressed('Angry', 0.5) == False  # Low confidence


def test_pose_analyzer_initialization(pose_analyzer):
    """Test PoseAnalyzer initializes correctly."""
    assert pose_analyzer is not None
    assert pose_analyzer.pose is not None


def test_pose_analysis(pose_analyzer, sample_image):
    """Test pose analysis returns valid results."""
    landmarks, analysis = pose_analyzer.analyze_pose(sample_image)
    
    # Even if no pose detected, should return empty dict
    assert isinstance(analysis, dict)


def test_pose_suspicious_detection(pose_analyzer):
    """Test suspicious pose detection logic."""
    analysis = {
        'struggling': True,
        'being_dragged': False,
        'distress_posture': False,
        'confidence': 0.7
    }
    
    assert pose_analyzer.is_suspicious(analysis, confidence_threshold=0.6) == True
    assert pose_analyzer.is_suspicious(analysis, confidence_threshold=0.9) == False


def test_empty_image_handling(person_detector, age_classifier, emotion_detector):
    """Test modules handle empty images gracefully."""
    empty_image = np.array([])
    
    assert person_detector.detect(empty_image) == []
    assert age_classifier.classify(empty_image) == ('unknown', 0.0)
    assert emotion_detector.detect_emotion(empty_image) == ('Neutral', 0.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
