"""
Face Recognition Module
Search for missing children in video footage using face embeddings
Uses FaceNet for generating face embeddings
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import MODEL_PATHS
from src.utils.logger import setup_logger

logger = setup_logger('FaceRecognizer')


class FaceRecognizer:
    """
    Face recognition system for matching missing children.
    Uses FaceNet to generate 128-dimensional face embeddings.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize face recognition model.
        
        Args:
            model_path: Path to FaceNet model
        """
        self.model_path = model_path or str(MODEL_PATHS['face'])
        self.similarity_threshold = 0.7  # Cosine similarity threshold
        
        logger.info("Initializing FaceRecognizer...")
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # For now, use a simple face embedding approach
        # In production, use actual FaceNet or similar model
        logger.warning("Using dummy face embeddings - replace with trained FaceNet model")
        self.model_loaded = False
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in image.
        
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
    
    def encode_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        Generate 128-D embedding for face.
        
        Args:
            face_image: Face image (BGR numpy array)
            
        Returns:
            128-dimensional embedding vector
        """
        if face_image is None or face_image.size == 0:
            logger.warning("Empty face image")
            return np.zeros(128)
        
        try:
            # Resize face to standard size
            face_resized = cv2.resize(face_image, (160, 160))
            
            # For demonstration, use histogram-based simple embedding
            # In production, replace with actual FaceNet model
            gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            
            # Create simple feature vector (THIS IS A PLACEHOLDER)
            # Replace with actual FaceNet embedding
            hist = cv2.calcHist([gray], [0], None, [128], [0, 256])
            embedding = hist.flatten() / hist.sum()  # Normalize
            
            return embedding
            
        except Exception as e:
            logger.error(f"Face encoding failed: {e}")
            return np.zeros(128)
    
    def compare_faces(self, embedding1: np.ndarray, 
                     embedding2: np.ndarray) -> float:
        """
        Compare two face embeddings using cosine similarity.
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            Similarity score (0 to 1, higher = more similar)
        """
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        # Convert to 0-1 range
        similarity = (similarity + 1) / 2
        
        return float(similarity)
    
    def search_missing_child(self, query_image: np.ndarray, 
                           video_path: str) -> List[Dict]:
        """
        Search for child in video footage.
        
        Args:
            query_image: Photo of missing child (BGR)
            video_path: Path to video file to search
            
        Returns:
            List of matches with:
                - timestamp: str
                - frame_num: int
                - confidence: float
                - bbox: tuple
                - frame: numpy array (optional)
        """
        logger.info(f"Searching for child in video: {video_path}")
        
        # Encode query face
        faces = self.detect_faces(query_image)
        if len(faces) == 0:
            logger.error("No face detected in query image")
            return []
        
        # Use the largest face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        query_face = query_image[y:y+h, x:x+w]
        query_embedding = self.encode_face(query_face)
        
        # Search video
        matches = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_num = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_num += 1
                
                # Process every 10th frame for speed
                if frame_num % 10 != 0:
                    continue
                
                # Detect faces in frame
                faces_in_frame = self.detect_faces(frame)
                
                for (fx, fy, fw, fh) in faces_in_frame:
                    face_crop = frame[fy:fy+fh, fx:fx+fw]
                    
                    # Encode and compare
                    face_embedding = self.encode_face(face_crop)
                    similarity = self.compare_faces(query_embedding, face_embedding)
                    
                    # If similarity is above threshold, add to matches
                    if similarity >= self.similarity_threshold:
                        timestamp = frame_num / fps if fps > 0 else 0
                        
                        match = {
                            'timestamp': f"{int(timestamp // 60)}:{int(timestamp % 60):02d}",
                            'frame_num': frame_num,
                            'confidence': similarity,
                            'bbox': (fx, fy, fw, fh)
                        }
                        
                        matches.append(match)
                        logger.info(f"Match found at frame {frame_num} "
                                  f"(confidence: {similarity:.2f})")
        
        finally:
            cap.release()
        
        logger.info(f"Search complete. Found {len(matches)} matches")
        return matches
    
    def save_database_face(self, name: str, image: np.ndarray, 
                          relationship: str = 'unknown') -> Dict:
        """
        Save face to trusted persons database.
        
        Args:
            name: Person's name
            image: Face image
            relationship: Relationship (parent, teacher, staff, etc.)
            
        Returns:
            Dictionary with person info and embedding
        """
        faces = self.detect_faces(image)
        
        if len(faces) == 0:
            logger.error("No face detected in image")
            return {}
        
        # Use largest face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        face_crop = image[y:y+h, x:x+w]
        
        # Generate embedding
        embedding = self.encode_face(face_crop)
        
        person_data = {
            'name': name,
            'relationship': relationship,
            'embedding': embedding.tobytes(),  # Convert to bytes for storage
            'bbox': (x, y, w, h)
        }
        
        logger.info(f"Saved face for {name} ({relationship})")
        return person_data
    
    def match_against_database(self, face_image: np.ndarray, 
                              database: List[Dict]) -> Tuple[str, float]:
        """
        Match face against database of known persons.
        
        Args:
            face_image: Face image to match
            database: List of person dictionaries with embeddings
            
        Returns:
            Tuple of (matched_name, confidence)
        """
        if not database:
            return 'Unknown', 0.0
        
        # Encode input face
        query_embedding = self.encode_face(face_image)
        
        # Compare against all database entries
        best_match = None
        best_similarity = 0.0
        
        for person in database:
            # Convert bytes back to numpy array
            db_embedding = np.frombuffer(person['embedding'], dtype=np.float64)
            
            # Compare
            similarity = self.compare_faces(query_embedding, db_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = person['name']
        
        # Only return match if above threshold
        if best_similarity >= self.similarity_threshold:
            return best_match, best_similarity
        else:
            return 'Unknown', best_similarity


# Test code
if __name__ == '__main__':
    print("Testing FaceRecognizer...")
    
    recognizer = FaceRecognizer()
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    print("Press 's' to save your face to database")
    print("Press 'q' to quit")
    
    database = []
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Detect faces
        faces = recognizer.detect_faces(frame)
        
        # Draw faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Try to match against database
            if database:
                face_crop = frame[y:y+h, x:x+w]
                name, confidence = recognizer.match_against_database(face_crop, database)
                
                label = f"{name} ({confidence:.2f})"
                cv2.putText(frame, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Database size: {len(database)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Face Recognition', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s') and len(faces) > 0:
            # Save face to database
            name = input("Enter name: ")
            person_data = recognizer.save_database_face(name, frame)
            if person_data:
                database.append(person_data)
                print(f"Saved {name} to database")
    
    cap.release()
    cv2.destroyAllWindows()
