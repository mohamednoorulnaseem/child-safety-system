"""
Age Classification Module
Classifies detected persons as Child (<12 years) or Adult (≥12 years)
Uses ResNet18 fine-tuned on UTKFace dataset
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from typing import Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import MODEL_PATHS, AGE_THRESHOLD, AGE_CONFIDENCE_THRESHOLD
from src.utils.logger import setup_logger

logger = setup_logger('AgeClassifier')


class AgeClassifier:
    """
    Binary age classifier: Child vs Adult
    Based on ResNet18 architecture
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize age classification model.
        
        Args:
            model_path: Path to trained model weights
        """
        self.model_path = model_path or str(MODEL_PATHS['age'])
        self.age_threshold = AGE_THRESHOLD
        self.confidence_threshold = AGE_CONFIDENCE_THRESHOLD
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize model
        try:
            self._load_model()
            logger.info("AgeClassifier initialized successfully")
        except Exception as e:
            logger.warning(f"Could not load trained model: {e}")
            logger.info("Using pretrained ResNet18 (needs fine-tuning)")
            self._create_base_model()
    
    def _create_base_model(self):
        """Create base ResNet18 model for age classification."""
        # Load pretrained ResNet18
        self.model = models.resnet18(pretrained=True)
        
        # Modify final layer for binary classification
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 2)  # 2 classes: child, adult
        )
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def _load_model(self):
        """Load trained model from file."""
        # Create model architecture
        self.model = models.resnet18(pretrained=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 2)
        )
        
        # Load weights
        if Path(self.model_path).exists():
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded trained model from {self.model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def classify(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Classify age group from person image.
        
        Args:
            image: BGR image of person (numpy array)
            
        Returns:
            Tuple of (age_class, confidence) where:
                age_class: 'child' or 'adult'
                confidence: float between 0 and 1
        """
        if image is None or image.size == 0:
            logger.warning("Empty image received")
            return 'unknown', 0.0
        
        try:
            # Convert BGR to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                logger.warning("Image is not in BGR format")
                return 'unknown', 0.0
            
            # Transform image
            input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # Convert to class name
            class_idx = predicted.item()
            confidence_score = confidence.item()
            
            age_class = 'child' if class_idx == 0 else 'adult'
            
            logger.debug(f"Age classification: {age_class} (confidence: {confidence_score:.2f})")
            
            return age_class, confidence_score
            
        except Exception as e:
            logger.error(f"Age classification failed: {e}")
            return 'unknown', 0.0
    
    def is_child(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Check if person in image is a child.
        
        Args:
            image: BGR image of person
            
        Returns:
            Tuple of (is_child, confidence)
        """
        age_class, confidence = self.classify(image)
        
        # Only return True if confidence is above threshold
        if age_class == 'child' and confidence >= self.confidence_threshold:
            return True, confidence
        elif age_class == 'adult':
            return False, confidence
        else:
            return False, 0.0


def train_age_classifier(data_dir: str, epochs: int = 20, batch_size: int = 32):
    """
    Train age classification model on UTKFace dataset.
    
    Args:
        data_dir: Directory containing training data
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torchvision import datasets
    
    logger.info("Starting age classifier training...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = datasets.ImageFolder(
        root=f"{data_dir}/train",
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        root=f"{data_dir}/val",
        transform=val_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 2)
    )
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Training loop
    best_acc = 0.0
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        logger.info(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, MODEL_PATHS['age'])
            logger.info(f"Saved best model with accuracy: {best_acc:.2f}%")
        
        scheduler.step()
    
    logger.info(f"Training complete. Best validation accuracy: {best_acc:.2f}%")


# Test code
if __name__ == '__main__':
    print("Testing AgeClassifier...")
    
    # Initialize classifier
    classifier = AgeClassifier()
    
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
        
        # Classify age
        age_class, confidence = classifier.classify(frame)
        
        # Display result
        text = f"{age_class.upper()} ({confidence:.2f})"
        color = (0, 255, 0) if age_class == 'child' else (0, 0, 255)
        
        cv2.putText(frame, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow('Age Classification', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
