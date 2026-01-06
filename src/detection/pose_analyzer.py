"""
Pose Analysis Module using MediaPipe
Analyzes body language to detect suspicious interactions and distress
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Tuple, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import setup_logger

logger = setup_logger('PoseAnalyzer')


class PoseAnalyzer:
    """
    MediaPipe-based pose analyzer for detecting suspicious behavior.
    Analyzes body language patterns to identify distress, struggling, or being dragged.
    """
    
    def __init__(self, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe Pose detector.
        
        Args:
            min_detection_confidence: Minimum confidence for person detection
            min_tracking_confidence: Minimum confidence for pose tracking
        """
        logger.info("Initializing PoseAnalyzer...")
        
        try:
            # Initialize MediaPipe Pose
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                model_complexity=0  # 0=Lite, 1=Full, 2=Heavy (use Lite for Raspberry Pi)
            )
            self.mediapipe_available = True
        except AttributeError:
            # MediaPipe API changed or not fully installed
            logger.warning("MediaPipe pose solutions not available")
            logger.info("Pose analysis will use placeholder mode")
            self.mediapipe_available = False
            self.pose = None
        
        # Pose landmark indices (MediaPipe has 33 landmarks)
        self.LANDMARKS = {
            'NOSE': 0,
            'LEFT_SHOULDER': 11,
            'RIGHT_SHOULDER': 12,
            'LEFT_ELBOW': 13,
            'RIGHT_ELBOW': 14,
            'LEFT_WRIST': 15,
            'RIGHT_WRIST': 16,
            'LEFT_HIP': 23,
            'RIGHT_HIP': 24,
            'LEFT_KNEE': 25,
            'RIGHT_KNEE': 26,
            'LEFT_ANKLE': 27,
            'RIGHT_ANKLE': 28
        }
        
        logger.info("PoseAnalyzer initialized successfully")
    
    def analyze_pose(self, image: np.ndarray) -> Tuple[Optional[object], Dict]:
        """
        Analyze body pose and detect suspicious behavior.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Tuple of (landmarks, analysis_dict) where:
                landmarks: MediaPipe pose landmarks object
                analysis_dict: Dictionary containing behavior flags
        """
        if image is None or image.size == 0:
            logger.warning("Empty image received")
            return None, {}
        
        if not self.mediapipe_available or self.pose is None:
            # Return placeholder analysis when MediaPipe not available
            return None, {
                'struggling': False,
                'being_dragged': False,
                'distress_posture': False,
                'is_suspicious': False
            }
        
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process pose
            results = self.pose.process(image_rgb)
            
            if not results.pose_landmarks:
                logger.debug("No pose detected")
                return None, {}
            
            # Analyze pose for suspicious patterns
            analysis = self.detect_suspicious_pose(results.pose_landmarks)
            
            return results.pose_landmarks, analysis
            
        except Exception as e:
            logger.error(f"Pose analysis failed: {e}")
            return None, {}
    
    def detect_suspicious_pose(self, landmarks: object) -> Dict:
        """
        Detect suspicious body language patterns.
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            Dictionary with behavior flags:
                - struggling: bool
                - being_dragged: bool
                - distress_posture: bool
                - confidence: float
        """
        analysis = {
            'struggling': False,
            'being_dragged': False,
            'distress_posture': False,
            'confidence': 0.0
        }
        
        try:
            lm = landmarks.landmark
            
            # Get key landmark positions
            left_wrist = lm[self.LANDMARKS['LEFT_WRIST']]
            right_wrist = lm[self.LANDMARKS['RIGHT_WRIST']]
            left_shoulder = lm[self.LANDMARKS['LEFT_SHOULDER']]
            right_shoulder = lm[self.LANDMARKS['RIGHT_SHOULDER']]
            left_hip = lm[self.LANDMARKS['LEFT_HIP']]
            right_hip = lm[self.LANDMARKS['RIGHT_HIP']]
            nose = lm[self.LANDMARKS['NOSE']]
            
            # Calculate shoulder center
            shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            
            # 1. Check for struggling (both arms raised above shoulders)
            if (left_wrist.y < shoulder_y and right_wrist.y < shoulder_y and
                left_wrist.visibility > 0.5 and right_wrist.visibility > 0.5):
                analysis['struggling'] = True
                logger.debug("Detected struggling pose")
            
            # 2. Check for being dragged (body tilted unnaturally)
            # Calculate body tilt angle
            hip_center_y = (left_hip.y + right_hip.y) / 2
            body_tilt = abs(nose.y - hip_center_y)
            
            # Also check if person is at unusual angle
            shoulder_hip_dist = abs(shoulder_y - hip_center_y)
            if shoulder_hip_dist < 0.1:  # Body compressed/bent unnaturally
                analysis['being_dragged'] = True
                logger.debug("Detected dragging pose")
            
            # 3. Check for distress posture (crouched, defensive)
            # If shoulders are significantly lower or body is compressed
            if shoulder_y > 0.6 and nose.y > 0.5:  # Lower in frame = crouched
                analysis['distress_posture'] = True
                logger.debug("Detected distress posture")
            
            # Calculate overall confidence based on landmark visibility
            avg_visibility = np.mean([
                left_wrist.visibility, right_wrist.visibility,
                left_shoulder.visibility, right_shoulder.visibility,
                left_hip.visibility, right_hip.visibility
            ])
            
            analysis['confidence'] = float(avg_visibility)
            
        except Exception as e:
            logger.error(f"Failed to analyze pose: {e}")
        
        return analysis
    
    def is_suspicious(self, analysis: Dict, confidence_threshold: float = 0.6) -> bool:
        """
        Check if pose indicates suspicious activity.
        
        Args:
            analysis: Analysis dictionary from detect_suspicious_pose
            confidence_threshold: Minimum confidence to flag as suspicious
            
        Returns:
            True if any suspicious behavior detected with sufficient confidence
        """
        if analysis.get('confidence', 0) < confidence_threshold:
            return False
        
        return (analysis.get('struggling', False) or 
                analysis.get('being_dragged', False) or
                analysis.get('distress_posture', False))
    
    def draw_pose(self, image: np.ndarray, landmarks: object, 
                  analysis: Dict = None) -> np.ndarray:
        """
        Draw pose landmarks on image.
        
        Args:
            image: Input image
            landmarks: MediaPipe pose landmarks
            analysis: Optional analysis results to display
            
        Returns:
            Annotated image
        """
        if landmarks is None:
            return image
        
        annotated_image = image.copy()
        
        # Draw pose landmarks
        self.mp_drawing.draw_landmarks(
            annotated_image,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        # Draw analysis results if provided
        if analysis:
            y_offset = 30
            
            if analysis.get('struggling'):
                cv2.putText(annotated_image, "STRUGGLING DETECTED!", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                y_offset += 30
            
            if analysis.get('being_dragged'):
                cv2.putText(annotated_image, "DRAGGING DETECTED!", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                y_offset += 30
            
            if analysis.get('distress_posture'):
                cv2.putText(annotated_image, "DISTRESS POSTURE!", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                y_offset += 30
            
            # Show confidence
            confidence = analysis.get('confidence', 0)
            cv2.putText(annotated_image, f"Confidence: {confidence:.2f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_image
    
    def __del__(self):
        """Cleanup MediaPipe resources."""
        if hasattr(self, 'pose'):
            self.pose.close()


# Test code
if __name__ == '__main__':
    import time
    
    print("Testing PoseAnalyzer...")
    print("Raise both arms above shoulders to simulate struggling")
    print("Crouch down to simulate distress posture")
    print("Press 'q' to quit")
    
    # Initialize analyzer
    analyzer = PoseAnalyzer()
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break
        
        # Analyze pose
        landmarks, analysis = analyzer.analyze_pose(frame)
        
        # Draw results
        annotated_frame = analyzer.draw_pose(frame, landmarks, analysis)
        
        # Calculate FPS
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, annotated_frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Check if suspicious
        if analyzer.is_suspicious(analysis):
            cv2.putText(annotated_frame, "ALERT: SUSPICIOUS ACTIVITY!", 
                       (annotated_frame.shape[1]//2 - 200, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        cv2.imshow('Pose Analysis', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nAverage FPS: {fps:.2f}")
