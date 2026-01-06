"""
Main Detection System
Orchestrates all detection modules and manages the complete pipeline
"""
import cv2
import numpy as np
import time
import sys
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent))

from config.settings import CAMERAS, ALERTS
from src.detection import (
    PersonDetector, AgeClassifier, EmotionDetector, 
    PoseAnalyzer, FaceRecognizer
)
from src.tracking import MultiTracker
from src.alerts import AlertManager
from src.utils.logger import setup_logger
from src.utils.helpers import get_timestamp

logger = setup_logger('ChildSafetySystem')


class ChildSafetySystem:
    """
    Main child safety detection and monitoring system.
    Integrates person detection, age classification, emotion detection,
    pose analysis, tracking, and multi-channel alerting.
    """
    
    def __init__(self):
        """Initialize all detection modules and alert system."""
        logger.info("=" * 60)
        logger.info("Initializing Child Safety System")
        logger.info("=" * 60)
        
        # Initialize detection modules
        logger.info("Loading detection modules...")
        self.person_detector = PersonDetector()
        self.age_classifier = AgeClassifier()
        self.emotion_detector = EmotionDetector()
        self.pose_analyzer = PoseAnalyzer()
        self.face_recognizer = FaceRecognizer()
        
        # Initialize tracker
        logger.info("Initializing tracker...")
        self.tracker = MultiTracker()
        
        # Initialize alert system
        logger.info("Initializing alert system...")
        self.alert_manager = AlertManager()
        
        # State tracking
        self.unattended_children = {}  # {track_id: start_time}
        self.alert_history = defaultdict(list)
        
        logger.info("Child Safety System initialized successfully")
        logger.info("=" * 60)
    
    def process_frame(self, frame: np.ndarray, camera_id: int) -> Dict:
        """
        Main detection pipeline for single frame.
        
        Flow:
        1. Detect persons (YOLO)
        2. Track persons (DeepSORT)
        3. For each tracked person:
           a. Extract person crop
           b. Classify age
           c. If child:
              - Detect emotion
              - Analyze pose
              - Check if unattended
              - Generate alerts if needed
        4. Annotate frame
        5. Return results
        
        Args:
            frame: Input image (BGR numpy array)
            camera_id: Camera identifier
            
        Returns:
            Dictionary with:
                - detections: List of all detections
                - children: List of detected children
                - alerts: List of triggered alerts
                - annotated_frame: Frame with visualizations
                - statistics: Frame processing statistics
        """
        start_time = time.time()
        
        # Step 1: Detect persons
        detections = self.person_detector.detect(frame)
        
        # Step 2: Track persons
        tracks = self.tracker.update(detections, frame)
        
        # Step 3: Analyze each tracked person
        children = []
        alerts = []
        
        for track in tracks:
            track_id = track['id']
            bbox = track['bbox']
            
            # Extract person crop
            x1, y1, x2, y2 = bbox
            person_crop = frame[y1:y2, x1:x2]
            
            if person_crop.size == 0:
                continue
            
            # Classify age
            age_class, age_confidence = self.age_classifier.classify(person_crop)
            
            # If child detected, perform detailed analysis
            if age_class == 'child' and age_confidence > 0.7:
                child_analysis = self.analyze_child(
                    person_crop, track_id, bbox, frame, camera_id
                )
                
                children.append(child_analysis)
                
                # Check for alerts
                child_alerts = self.check_for_alerts(
                    child_analysis, track_id, camera_id
                )
                
                alerts.extend(child_alerts)
        
        # Step 4: Annotate frame
        annotated_frame = self.annotate_frame(
            frame, tracks, children, camera_id
        )
        
        # Calculate statistics
        processing_time = time.time() - start_time
        fps = 1.0 / processing_time if processing_time > 0 else 0
        
        statistics = {
            'fps': fps,
            'processing_time': processing_time,
            'detections': len(detections),
            'tracks': len(tracks),
            'children': len(children),
            'alerts': len(alerts)
        }
        
        return {
            'detections': detections,
            'tracks': tracks,
            'children': children,
            'alerts': alerts,
            'annotated_frame': annotated_frame,
            'statistics': statistics
        }
    
    def analyze_child(self, child_crop: np.ndarray, track_id: int,
                     bbox: tuple, full_frame: np.ndarray,
                     camera_id: int) -> Dict:
        """
        Perform detailed analysis on detected child.
        
        Args:
            child_crop: Cropped child image
            track_id: Track ID
            bbox: Bounding box
            full_frame: Full frame for context analysis
            camera_id: Camera ID
            
        Returns:
            Dictionary with complete child analysis
        """
        analysis = {
            'track_id': track_id,
            'bbox': bbox,
            'camera_id': camera_id,
            'timestamp': get_timestamp(),
            'age_class': 'child',
            'emotion': None,
            'emotion_confidence': 0.0,
            'is_distressed': False,
            'pose_analysis': {},
            'is_suspicious': False,
            'is_unattended': False,
            'unattended_duration': 0
        }
        
        try:
            # Emotion detection
            emotion, emotion_confidence = self.emotion_detector.detect_emotion(child_crop)
            analysis['emotion'] = emotion
            analysis['emotion_confidence'] = emotion_confidence
            analysis['is_distressed'] = self.emotion_detector.is_distressed(
                emotion, emotion_confidence
            )
            
            # Pose analysis
            landmarks, pose_analysis = self.pose_analyzer.analyze_pose(child_crop)
            analysis['pose_analysis'] = pose_analysis
            analysis['is_suspicious'] = self.pose_analyzer.is_suspicious(pose_analysis)
            
            # Check if unattended
            unattended_info = self.check_unattended(track_id, bbox, full_frame)
            analysis['is_unattended'] = unattended_info['is_unattended']
            analysis['unattended_duration'] = unattended_info['duration']
            
        except Exception as e:
            logger.error(f\"Child analysis failed for track {track_id}: {e}\")\n        
        return analysis
    
    def check_unattended(self, track_id: int, child_bbox: tuple,
                        frame: np.ndarray) -> Dict:
        """
        Check if child appears unattended (no adults nearby).
        
        Args:
            track_id: Child track ID
            child_bbox: Child bounding box
            frame: Full frame
            
        Returns:
            Dictionary with unattended status and duration
        """
        # Detect all persons in frame
        all_detections = self.person_detector.detect(frame)
        
        # Check for adults nearby (within certain distance)
        adults_nearby = False
        proximity_threshold = 200  # pixels
        
        cx1, cy1, cx2, cy2 = child_bbox
        child_center = ((cx1 + cx2) / 2, (cy1 + cy2) / 2)
        
        for det in all_detections:
            dx1, dy1, dx2, dy2 = det['bbox']
            det_crop = frame[dy1:dy2, dx1:dx2]
            
            if det_crop.size == 0:
                continue
            
            # Classify age
            age_class, confidence = self.age_classifier.classify(det_crop)
            
            if age_class == 'adult' and confidence > 0.7:
                # Calculate distance
                adult_center = ((dx1 + dx2) / 2, (dy1 + dy2) / 2)
                distance = np.sqrt(
                    (child_center[0] - adult_center[0])**2 + 
                    (child_center[1] - adult_center[1])**2
                )
                
                if distance < proximity_threshold:
                    adults_nearby = True
                    break
        
        # Track unattended duration
        current_time = time.time()
        
        if not adults_nearby:
            if track_id not in self.unattended_children:
                self.unattended_children[track_id] = current_time
            
            duration = current_time - self.unattended_children[track_id]
            is_unattended = duration > 60  # Unattended if >1 minute
        else:
            if track_id in self.unattended_children:
                del self.unattended_children[track_id]
            duration = 0
            is_unattended = False
        
        return {
            'is_unattended': is_unattended,
            'duration': duration,
            'adults_nearby': adults_nearby
        }
    
    def check_for_alerts(self, child_info: Dict, track_id: int,
                        camera_id: int) -> List[Dict]:
        """
        Check all alert conditions and generate alerts.
        
        Args:
            child_info: Child analysis dictionary
            track_id: Track ID
            camera_id: Camera ID
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        # HIGH PRIORITY: Distressed + suspicious behavior
        if (child_info['is_distressed'] and child_info['is_suspicious']):
            alert_data = self.alert_manager.create_alert_data(
                alert_type='CHILD_DISTRESS',
                track_id=track_id,
                camera_id=camera_id,
                confidence=0.95,
                details={
                    'distressed': True,
                    'struggling': child_info['pose_analysis'].get('struggling', False),
                    'being_dragged': child_info['pose_analysis'].get('being_dragged', False),
                    'emotion': child_info['emotion']
                }
            )
            
            if self.alert_manager.trigger_alert(alert_data):
                alerts.append(alert_data)
        
        # MEDIUM PRIORITY: Unattended child
        elif child_info['is_unattended'] and child_info['unattended_duration'] > ALERTS['unattended_child_time']:
            alert_data = self.alert_manager.create_alert_data(
                alert_type='UNATTENDED_CHILD',
                track_id=track_id,
                camera_id=camera_id,
                confidence=0.80,
                details={
                    'duration': child_info['unattended_duration'],
                    'location': f\"Camera {camera_id}\"
                }
            )
            
            if self.alert_manager.trigger_alert(alert_data):
                alerts.append(alert_data)
        
        # MEDIUM PRIORITY: Suspicious behavior alone
        elif child_info['is_suspicious']:
            alert_data = self.alert_manager.create_alert_data(
                alert_type='SUSPICIOUS_BEHAVIOR',
                track_id=track_id,
                camera_id=camera_id,
                confidence=0.75,
                details=child_info['pose_analysis']
            )
            
            if self.alert_manager.trigger_alert(alert_data):
                alerts.append(alert_data)
        
        return alerts
    
    def annotate_frame(self, frame: np.ndarray, tracks: List[Dict],
                      children: List[Dict], camera_id: int) -> np.ndarray:
        """
        Annotate frame with detection results.
        
        Args:
            frame: Input frame
            tracks: List of tracked objects
            children: List of detected children
            camera_id: Camera ID
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Draw all tracks
        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            track_id = track['id']
            
            # Default color (green for adults/unknown)
            color = (0, 255, 0)
            label = f\"ID: {track_id}\"
            
            # Check if this is a child
            child_info = next((c for c in children if c['track_id'] == track_id), None)
            
            if child_info:
                # Red for distressed/suspicious children
                if child_info['is_distressed'] or child_info['is_suspicious']:
                    color = (0, 0, 255)
                    label = f\"CHILD-{track_id} ALERT\"
                # Yellow for unattended
                elif child_info['is_unattended']:
                    color = (0, 255, 255)
                    label = f\"CHILD-{track_id} UNATTENDED\"
                # Blue for normal children
                else:
                    color = (255, 0, 0)
                    label = f\"CHILD-{track_id}\"
            
            # Draw bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            cv2.putText(annotated, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw header info
        header_y = 30
        cv2.putText(annotated, f\"Camera {camera_id}\", (10, header_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(annotated, f\"Tracks: {len(tracks)}\", (10, header_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(annotated, f\"Children: {len(children)}\", (10, header_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(annotated, get_timestamp(), (10, annotated.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated
    
    def run(self, camera_ids: List[int] = None):
        """
        Start detection on specified cameras.
        
        Args:
            camera_ids: List of camera device IDs (default: [0])
        """
        if camera_ids is None:
            camera_ids = [0]  # Default webcam
        
        logger.info(f\"Starting detection on cameras: {camera_ids}\")
        
        # Open cameras
        caps = []
        for cam_id in camera_ids:
            cap = cv2.VideoCapture(cam_id)
            if cap.isOpened():
                caps.append((cam_id, cap))
                logger.info(f\"Camera {cam_id} opened successfully\")
            else:
                logger.error(f\"Failed to open camera {cam_id}\")
        
        if not caps:
            logger.error(\"No cameras available\")
            return
        
        logger.info(\"Press 'q' to quit\")
        logger.info(\"System running...\")
        
        try:
            while True:
                frames_data = []
                
                # Capture and process frames from all cameras
                for cam_id, cap in caps:
                    ret, frame = cap.read()
                    
                    if not ret:
                        logger.warning(f\"Failed to read from camera {cam_id}\")
                        continue
                    
                    # Process frame
                    result = self.process_frame(frame, cam_id)
                    frames_data.append(result)
                    
                    # Display
                    cv2.imshow(f\"Camera {cam_id} - Child Safety System\", 
                              result['annotated_frame'])
                    
                    # Log statistics
                    stats = result['statistics']
                    if stats['alerts'] > 0:
                        logger.warning(f\"Camera {cam_id}: {stats['alerts']} alerts triggered\")
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info(\"Quit signal received\")
                    break
        
        except KeyboardInterrupt:
            logger.info(\"Interrupted by user\")
        
        finally:
            # Cleanup
            logger.info(\"Shutting down...\")
            for _, cap in caps:
                cap.release()
            cv2.destroyAllWindows()
            self.alert_manager.cleanup()
            logger.info(\"System shutdown complete\")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Child Safety Detection System')
    parser.add_argument('--cameras', type=int, nargs='+', default=[0],
                       help='Camera device IDs (default: 0)')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode (single frame)')
    
    args = parser.parse_args()
    
    # Initialize system
    system = ChildSafetySystem()
    
    if args.test:
        logger.info(\"Running in test mode\")
        cap = cv2.VideoCapture(args.cameras[0])
        ret, frame = cap.read()
        if ret:
            result = system.process_frame(frame, args.cameras[0])
            logger.info(f\"Test result: {result['statistics']}\")
        cap.release()
    else:
        # Run system
        system.run(camera_ids=args.cameras)


if __name__ == '__main__':
    main()
