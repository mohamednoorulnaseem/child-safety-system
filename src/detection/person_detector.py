"""
Person Detection Module using YOLOv8
Detects all persons in camera frame with high accuracy
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import MODEL_PATHS, DETECTION
from src.utils.logger import setup_logger
from src.utils.helpers import is_bbox_valid, draw_bbox_with_label

logger = setup_logger('PersonDetector')


class PersonDetector:
    """
    YOLOv8-based person detector for real-time surveillance.
    Optimized for Raspberry Pi with nano model variant.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize YOLOv8 person detector.
        
        Args:
            model_path: Path to YOLOv8 model file (default: yolov8n.pt)
        """
        self.model_path = model_path or str(MODEL_PATHS['yolo'])
        self.confidence_threshold = DETECTION['confidence_threshold']
        self.nms_threshold = DETECTION['nms_threshold']
        self.min_size = DETECTION['min_detection_size']
        
        logger.info(f"Initializing PersonDetector with model: {self.model_path}")
        
        try:
            # Import ultralytics YOLO
            from ultralytics import YOLO
            
            # Load model
            self.model = YOLO(self.model_path)
            self.model.conf = self.confidence_threshold
            self.model.iou = self.nms_threshold
            
            # Set to detect only person class (class_id = 0 in COCO dataset)
            self.person_class_id = 0
            
            logger.info("PersonDetector initialized successfully")
            
        except FileNotFoundError:
            logger.warning(f"Model file not found at {self.model_path}")
            logger.info("The model will be downloaded automatically on first use")
            from ultralytics import YOLO
            self.model = YOLO('yolov8n.pt')  # Will auto-download
            self.model.conf = self.confidence_threshold
            self.model.iou = self.nms_threshold
            self.person_class_id = 0
            
        except Exception as e:
            logger.error(f"Failed to initialize PersonDetector: {e}")
            raise
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect persons in frame using YOLOv8.
        
        Args:
            frame: Input image (BGR format, numpy array)
            
        Returns:
            List of detections, each containing:
                - bbox: (x1, y1, x2, y2)
                - confidence: float (0-1)
                - class_id: int (always 0 for person)
                - class_name: str ('person')
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame received")
            return []
        
        try:
            # Run YOLOv8 inference
            results = self.model(frame, verbose=False)
            
            detections = []
            
            # Process results
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    # Get class ID
                    class_id = int(box.cls[0])
                    
                    # Filter for person class only
                    if class_id != self.person_class_id:
                        continue
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    bbox = (int(x1), int(y1), int(x2), int(y2))
                    
                    # Check if bbox is valid (not too small)
                    if not is_bbox_valid(bbox, self.min_size):
                        continue
                    
                    # Get confidence
                    confidence = float(box.conf[0])
                    
                    detection = {
                        'bbox': bbox,
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': 'person'
                    }
                    
                    detections.append(detection)
            
            logger.debug(f"Detected {len(detections)} persons")
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    def draw_detections(self, frame: np.ndarray, 
                       detections: List[Dict],
                       color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Draw bounding boxes on frame for visualization.
        
        Args:
            frame: Input image
            detections: List of detection dictionaries
            color: BGR color for bounding boxes
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        for det in detections:
            bbox = det['bbox']
            confidence = det['confidence']
            
            # Create label
            label = f"Person {confidence:.2f}"
            
            # Draw bbox with label
            annotated_frame = draw_bbox_with_label(
                annotated_frame, bbox, label, color
            )
        
        return annotated_frame
    
    def get_person_crops(self, frame: np.ndarray, 
                        detections: List[Dict]) -> List[Tuple[np.ndarray, Dict]]:
        """
        Extract cropped person images from frame.
        
        Args:
            frame: Input image
            detections: List of detection dictionaries
            
        Returns:
            List of tuples (cropped_image, detection_dict)
        """
        crops = []
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Ensure coordinates are within frame bounds
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            # Crop person from frame
            person_crop = frame[y1:y2, x1:x2]
            
            if person_crop.size > 0:
                crops.append((person_crop, det))
        
        return crops


# Test code
if __name__ == '__main__':
    import time
    
    print("Testing PersonDetector...")
    
    # Initialize detector
    detector = PersonDetector()
    
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
            print("Failed to grab frame")
            break
        
        # Detect persons
        detections = detector.detect(frame)
        
        # Draw detections
        annotated_frame = detector.draw_detections(frame, detections)
        
        # Calculate FPS
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        # Display FPS
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Persons: {len(detections)}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Person Detection', annotated_frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nAverage FPS: {fps:.2f}")
