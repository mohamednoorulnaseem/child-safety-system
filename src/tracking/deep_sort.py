"""
DeepSORT Multi-Object Tracker
Maintains consistent IDs for persons across frames
Combines appearance features and motion prediction
"""
import numpy as np
import cv2
from typing import List, Dict, Tuple
from scipy.optimize import linear_sum_assignment
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import TRACKING
from src.utils.logger import setup_logger
from src.utils.helpers import calculate_iou
from .kalman_filter import KalmanFilter

logger = setup_logger('MultiTracker')


class Track:
    """
    Represents a single tracked object.
    """
    
    def __init__(self, track_id: int, bbox: Tuple[int, int, int, int],
                 feature: np.ndarray = None):
        """
        Initialize track.
        
        Args:
            track_id: Unique track ID
            bbox: Bounding box (x1, y1, x2, y2)
            feature: Appearance feature vector
        """
        self.track_id = track_id
        self.bbox = bbox
        self.age = 0  # Frames since last detection
        self.hits = 1  # Number of detections
        self.time_since_update = 0
        
        # Convert bbox to [x, y, a, h] format for Kalman filter
        # where (x,y) is center, a is aspect ratio, h is height
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2
        y = y1 + h / 2
        a = w / h if h > 0 else 1.0
        
        self.kf = KalmanFilter()
        self.mean, self.covariance = self.kf.initiate(np.array([x, y, a, h]))
        
        # Appearance features
        self.features = []
        if feature is not None:
            self.features.append(feature)
    
    def predict(self):
        """Predict next position using Kalman filter."""
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
    
    def update(self, bbox: Tuple[int, int, int, int], feature: np.ndarray = None):
        """
        Update track with new detection.
        
        Args:
            bbox: New bounding box
            feature: New appearance feature
        """
        # Convert bbox to measurement format
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2
        y = y1 + h / 2
        a = w / h if h > 0 else 1.0
        
        measurement = np.array([x, y, a, h])
        self.mean, self.covariance = self.kf.update(
            self.mean, self.covariance, measurement
        )
        
        self.bbox = bbox
        self.hits += 1
        self.time_since_update = 0
        
        # Update appearance features
        if feature is not None:
            self.features.append(feature)
            # Keep only recent features (last 100)
            if len(self.features) > 100:
                self.features = self.features[-100:]
    
    def get_current_bbox(self) -> Tuple[int, int, int, int]:
        """
        Get current bounding box from Kalman filter state.
        
        Returns:
            Bounding box (x1, y1, x2, y2)
        """
        x, y, a, h = self.mean[:4]
        w = a * h
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
        
        return (x1, y1, x2, y2)
    
    def is_confirmed(self, n_init: int = 3) -> bool:
        """Check if track is confirmed (has enough detections)."""
        return self.hits >= n_init
    
    def is_deleted(self, max_age: int = 30) -> bool:
        """Check if track should be deleted (no updates for too long)."""
        return self.time_since_update > max_age


class MultiTracker:
    """
    Multi-object tracker using DeepSORT algorithm.
    Maintains consistent IDs across frames using appearance + motion.
    """
    
    def __init__(self, max_age: int = None, n_init: int = None):
        """
        Initialize tracker.
        
        Args:
            max_age: Maximum frames to keep track without detection
            n_init: Minimum detections before track is confirmed
        """
        self.max_age = max_age or TRACKING['max_age']
        self.n_init = n_init or TRACKING['n_init']
        self.max_iou_distance = TRACKING['max_iou_distance']
        self.max_cosine_distance = TRACKING['max_cosine_distance']
        
        self.tracks = []
        self.next_id = 1
        
        logger.info(f"MultiTracker initialized (max_age={self.max_age}, "
                   f"n_init={self.n_init})")
    
    def _extract_features(self, frame: np.ndarray, 
                         bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract appearance feature from bbox region.
        
        Args:
            frame: Full frame
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Feature vector
        """
        x1, y1, x2, y2 = bbox
        
        # Ensure coordinates are valid
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        # Extract region
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return np.zeros(128)
        
        # Simple feature: histogram (in production, use CNN features)
        roi_resized = cv2.resize(roi, (64, 128))
        hist = cv2.calcHist([roi_resized], [0, 1, 2], None, 
                           [8, 8, 8], [0, 256, 0, 256, 0, 256])
        feature = hist.flatten()
        feature = feature / (np.linalg.norm(feature) + 1e-6)  # Normalize
        
        return feature
    
    def _cosine_distance(self, features1: List[np.ndarray], 
                        features2: np.ndarray) -> float:
        """
        Calculate cosine distance between track features and detection feature.
        
        Args:
            features1: List of track feature vectors
            features2: Single detection feature vector
            
        Returns:
            Minimum cosine distance (0 = similar, 1 = dissimilar)
        """
        if not features1:
            return 1.0
        
        # Compare with recent features (last 10)
        recent_features = features1[-10:]
        
        distances = []
        for f1 in recent_features:
            # Cosine distance = 1 - cosine similarity
            similarity = np.dot(f1, features2) / (
                np.linalg.norm(f1) * np.linalg.norm(features2) + 1e-6
            )
            distance = 1.0 - similarity
            distances.append(distance)
        
        return min(distances)
    
    def _match_tracks_to_detections(self, detections: List[Dict],
                                   frame: np.ndarray) -> Tuple[List, List, List]:
        """
        Match tracks to detections using Hungarian algorithm.
        
        Args:
            detections: List of detection dictionaries
            frame: Current frame for feature extraction
            
        Returns:
            Tuple of (matches, unmatched_tracks, unmatched_detections)
        """
        if len(self.tracks) == 0:
            return [], list(range(len(self.tracks))), list(range(len(detections)))
        
        if len(detections) == 0:
            return [], list(range(len(self.tracks))), []
        
        # Build cost matrix (tracks x detections)
        cost_matrix = np.zeros((len(self.tracks), len(detections)))
        
        for t_idx, track in enumerate(self.tracks):
            for d_idx, detection in enumerate(detections):
                # IoU distance
                track_bbox = track.get_current_bbox()
                det_bbox = detection['bbox']
                iou = calculate_iou(track_bbox, det_bbox)
                iou_distance = 1.0 - iou
                
                # Appearance distance
                det_feature = self._extract_features(frame, det_bbox)
                appearance_distance = self._cosine_distance(track.features, det_feature)
                
                # Combined cost
                cost = 0.5 * iou_distance + 0.5 * appearance_distance
                cost_matrix[t_idx, d_idx] = cost
        
        # Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Filter matches with high cost
        matches = []
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_detections = list(range(len(detections)))
        
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < 0.5:  # Threshold
                matches.append((row, col))
                unmatched_tracks.remove(row)
                unmatched_detections.remove(col)
        
        return matches, unmatched_tracks, unmatched_detections
    
    def update(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of detection dicts from PersonDetector
            frame: Current frame for appearance features
            
        Returns:
            List of active tracks with:
                - id: Track ID
                - bbox: Bounding box
                - age: Frames since last detection
                - confidence: Detection confidence
        """
        # Predict all tracks
        for track in self.tracks:
            track.predict()
        
        # Match tracks to detections
        matches, unmatched_tracks, unmatched_detections = \
            self._match_tracks_to_detections(detections, frame)
        
        # Update matched tracks
        for track_idx, det_idx in matches:
            detection = detections[det_idx]
            feature = self._extract_features(frame, detection['bbox'])
            self.tracks[track_idx].update(detection['bbox'], feature)
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            feature = self._extract_features(frame, detection['bbox'])
            new_track = Track(self.next_id, detection['bbox'], feature)
            self.tracks.append(new_track)
            self.next_id += 1
            logger.debug(f"Created new track: {new_track.track_id}")
        
        # Delete old tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted(self.max_age)]
        
        # Return confirmed tracks
        active_tracks = []
        for track in self.tracks:
            if track.is_confirmed(self.n_init):
                active_tracks.append({
                    'id': track.track_id,
                    'bbox': track.get_current_bbox(),
                    'age': track.age,
                    'hits': track.hits,
                    'time_since_update': track.time_since_update
                })
        
        logger.debug(f"Active tracks: {len(active_tracks)}")
        return active_tracks
    
    def reset(self):
        """Reset tracker (clear all tracks)."""
        self.tracks = []
        self.next_id = 1
        logger.info("Tracker reset")


# Test code
if __name__ == '__main__':
    import time
    
    print("Testing MultiTracker...")
    
    # Initialize tracker
    tracker = MultiTracker()
    
    # Dummy person detector (replace with actual PersonDetector)
    class DummyDetector:
        def detect(self, frame):
            # Simulate some detections
            return [
                {'bbox': (100, 100, 200, 300), 'confidence': 0.9, 'class_id': 0},
                {'bbox': (400, 150, 480, 320), 'confidence': 0.85, 'class_id': 0}
            ]
    
    detector = DummyDetector()
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    print("Press 'q' to quit")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Detect
        detections = detector.detect(frame)
        
        # Track
        tracks = tracker.update(detections, frame)
        
        # Draw tracks
        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            track_id = track['id']
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Calculate FPS
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Tracks: {len(tracks)}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Multi-Object Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nAverage FPS: {fps:.2f}")
