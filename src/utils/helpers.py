"""
Helper functions for Child Safety System
"""
import cv2
import numpy as np
from typing import Tuple, List, Dict, Any
from datetime import datetime
import json


def crop_bbox(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop image using bounding box coordinates.
    
    Args:
        image: Input image (numpy array)
        bbox: Bounding box (x1, y1, x2, y2)
        
    Returns:
        Cropped image region
    """
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    
    # Ensure coordinates are within image bounds
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    
    return image[y1:y2, x1:x2]


def calculate_iou(bbox1: Tuple[int, int, int, int], 
                  bbox2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: First bounding box (x1, y1, x2, y2)
        bbox2: Second bounding box (x1, y1, x2, y2)
        
    Returns:
        IoU score (0 to 1)
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection area
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = bbox1_area + bbox2_area - intersection_area
    
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area


def resize_with_aspect_ratio(image: np.ndarray, 
                             target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image maintaining aspect ratio.
    
    Args:
        image: Input image
        target_size: Target size (width, height)
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    return resized


def draw_bbox_with_label(image: np.ndarray, 
                         bbox: Tuple[int, int, int, int],
                         label: str,
                         color: Tuple[int, int, int] = (0, 255, 0),
                         thickness: int = 2) -> np.ndarray:
    """
    Draw bounding box with label on image.
    
    Args:
        image: Input image
        bbox: Bounding box (x1, y1, x2, y2)
        label: Text label
        color: BGR color tuple
        thickness: Line thickness
        
    Returns:
        Annotated image
    """
    x1, y1, x2, y2 = bbox
    
    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label background
    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
    
    # Draw label text
    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return image


def get_timestamp() -> str:
    """
    Get current timestamp as string.
    
    Returns:
        Timestamp string (YYYY-MM-DD HH:MM:SS)
    """
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def save_frame(frame: np.ndarray, filename: str, output_dir: str = 'output') -> str:
    """
    Save frame to disk.
    
    Args:
        frame: Image to save
        filename: Output filename
        output_dir: Output directory
        
    Returns:
        Full path to saved file
    """
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    full_path = output_path / filename
    cv2.imwrite(str(full_path), frame)
    
    return str(full_path)


def format_alert_message(alert_data: Dict[str, Any]) -> str:
    """
    Format alert data into human-readable message.
    
    Args:
        alert_data: Alert information dictionary
        
    Returns:
        Formatted message string
    """
    priority = alert_data.get('priority', 'UNKNOWN')
    alert_type = alert_data.get('type', 'UNKNOWN')
    camera = alert_data.get('camera_id', 'N/A')
    timestamp = alert_data.get('timestamp', get_timestamp())
    
    message = f"🚨 [{priority}] {alert_type}\n"
    message += f"Camera: {camera}\n"
    message += f"Time: {timestamp}\n"
    
    details = alert_data.get('details', {})
    if details:
        message += f"Details: {json.dumps(details, indent=2)}"
    
    return message


def calculate_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: First point (x, y)
        point2: Second point (x, y)
        
    Returns:
        Distance
    """
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def is_bbox_valid(bbox: Tuple[int, int, int, int], 
                  min_size: Tuple[int, int] = (30, 30)) -> bool:
    """
    Check if bounding box is valid (not too small).
    
    Args:
        bbox: Bounding box (x1, y1, x2, y2)
        min_size: Minimum size (width, height)
        
    Returns:
        True if valid, False otherwise
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    
    return w >= min_size[0] and h >= min_size[1]
