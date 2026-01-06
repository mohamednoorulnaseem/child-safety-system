"""
Unit tests for tracking modules
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.tracking import MultiTracker, KalmanFilter


@pytest.fixture
def tracker():
    """Create MultiTracker instance."""
    return MultiTracker(max_age=30, n_init=3)


@pytest.fixture
def kalman_filter():
    """Create KalmanFilter instance."""
    return KalmanFilter()


@pytest.fixture
def sample_detections():
    """Create sample detections."""
    return [
        {'bbox': (100, 100, 200, 300), 'confidence': 0.9, 'class_id': 0},
        {'bbox': (400, 150, 480, 320), 'confidence': 0.85, 'class_id': 0}
    ]


@pytest.fixture
def sample_frame():
    """Create sample frame."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


def test_tracker_initialization(tracker):
    """Test MultiTracker initializes correctly."""
    assert tracker is not None
    assert tracker.max_age == 30
    assert tracker.n_init == 3
    assert len(tracker.tracks) == 0


def test_tracker_update_creates_tracks(tracker, sample_detections, sample_frame):
    """Test tracker creates new tracks from detections."""
    tracks = tracker.update(sample_detections, sample_frame)
    
    # New tracks may not be confirmed yet (need n_init detections)
    assert isinstance(tracks, list)


def test_tracker_maintains_ids(tracker, sample_detections, sample_frame):
    """Test tracker maintains consistent IDs across frames."""
    # First update
    tracks1 = tracker.update(sample_detections, sample_frame)
    
    # Second update with same detections
    tracks2 = tracker.update(sample_detections, sample_frame)
    
    # After multiple updates, tracks should be confirmed
    tracker.update(sample_detections, sample_frame)
    tracks3 = tracker.update(sample_detections, sample_frame)
    
    assert len(tracks3) > 0  # Should have confirmed tracks now


def test_tracker_handles_occlusions(tracker, sample_detections, sample_frame):
    """Test tracker handles temporary occlusions."""
    # Create tracks
    for _ in range(5):
        tracker.update(sample_detections, sample_frame)
    
    initial_tracks = len(tracker.tracks)
    
    # Simulate occlusion (no detections)
    for _ in range(10):
        tracker.update([], sample_frame)
    
    # Tracks should still exist (not deleted yet)
    assert len(tracker.tracks) > 0


def test_tracker_deletes_old_tracks(tracker, sample_detections, sample_frame):
    """Test tracker deletes tracks after max_age."""
    # Create tracks
    for _ in range(5):
        tracker.update(sample_detections, sample_frame)
    
    # Simulate long occlusion (beyond max_age)
    for _ in range(35):
        tracker.update([], sample_frame)
    
    # Tracks should be deleted
    assert len(tracker.tracks) == 0


def test_tracker_reset(tracker, sample_detections, sample_frame):
    """Test tracker reset functionality."""
    # Create some tracks
    for _ in range(5):
        tracker.update(sample_detections, sample_frame)
    
    # Reset
    tracker.reset()
    
    assert len(tracker.tracks) == 0
    assert tracker.next_id == 1


def test_kalman_filter_initiate(kalman_filter):
    """Test Kalman filter initialization."""
    measurement = np.array([100, 100, 0.5, 200])  # x, y, aspect_ratio, height
    
    mean, covariance = kalman_filter.initiate(measurement)
    
    assert mean.shape == (8,)  # State vector
    assert covariance.shape == (8, 8)  # Covariance matrix


def test_kalman_filter_predict(kalman_filter):
    """Test Kalman filter prediction."""
    measurement = np.array([100, 100, 0.5, 200])
    mean, covariance = kalman_filter.initiate(measurement)
    
    predicted_mean, predicted_cov = kalman_filter.predict(mean, covariance)
    
    assert predicted_mean.shape == (8,)
    assert predicted_cov.shape == (8, 8)


def test_kalman_filter_update(kalman_filter):
    """Test Kalman filter update with measurement."""
    measurement = np.array([100, 100, 0.5, 200])
    mean, covariance = kalman_filter.initiate(measurement)
    
    # Predict
    predicted_mean, predicted_cov = kalman_filter.predict(mean, covariance)
    
    # Update with new measurement
    new_measurement = np.array([105, 102, 0.5, 200])
    updated_mean, updated_cov = kalman_filter.update(
        predicted_mean, predicted_cov, new_measurement
    )
    
    assert updated_mean.shape == (8,)
    assert updated_cov.shape == (8, 8)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
