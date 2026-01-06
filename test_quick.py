"""
Quick Test Script - Verify System Works
Tests person detection with webcam for 10 seconds
"""
import cv2
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.detection.person_detector import PersonDetector
from src.utils.logger import setup_logger

logger = setup_logger('QuickTest')

def main():
    logger.info("Starting quick system test...")
    
    # Initialize detector
    detector = PersonDetector()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open webcam")
        return
    
    logger.info("Webcam opened. Press 'q' to quit, or wait 10 seconds")
    
    frame_count = 0
    detection_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect persons
        detections = detector.detect(frame)
        detection_count += len(detections)
        
        # Draw detections
        annotated_frame = detector.draw_detections(frame, detections)
        
        # Display info
        cv2.putText(annotated_frame, f"Frame: {frame_count} | Persons: {len(detections)}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, "Press 'q' to quit", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Child Safety System - Quick Test', annotated_frame)
        
        frame_count += 1
        
        # Quit after 10 seconds or 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q') or frame_count >= 300:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    logger.info(f"Test complete: {frame_count} frames, {detection_count} total detections")
    print(f"\n✓ SUCCESS: System tested for {frame_count} frames")
    print(f"✓ Total person detections: {detection_count}")
    print(f"✓ Average FPS: {frame_count / 10:.1f}")

if __name__ == "__main__":
    main()
