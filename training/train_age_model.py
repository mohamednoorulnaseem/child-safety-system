"""
Age Classification Model Training Script
Train ResNet18 model on UTKFace dataset

Dataset: https://susanqq.github.io/UTKFace/
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import glob
from pathlib import Path

# Configuration
CONFIG = {
    'dataset_path': 'data/datasets/UTKFace',
    'batch_size': 32,
    'num_epochs': 20,
    'learning_rate': 0.001,
    'num_workers': 4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_path': 'models/age/age_classifier.pth',
    'age_groups': 5,  # 0-12, 13-19, 20-35, 36-60, 60+
}


class UTKFaceDataset(Dataset):
    """
    UTKFace dataset loader
    Filename format: [age]_[gender]_[race]_[date&time].jpg
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load all images
        for img_path in glob.glob(os.path.join(root_dir, '*.jpg')):
            try:
                # Extract age from filename
                filename = os.path.basename(img_path)
                age = int(filename.split('_')[0])
                
                # Convert age to age group
                age_group = self._age_to_group(age)
                
                self.images.append(img_path)
                self.labels.append(age_group)
            except (ValueError, IndexError):
                # Skip invalid filenames
                continue
    
    def _age_to_group(self, age):
        """Convert age to age group index"""
        if age <= 12:
            return 0  # Child
        elif age <= 19:
            return 1  # Teen
        elif age <= 35:
            return 2  # Young Adult
        elif age <= 60:
            return 3  # Adult
        else:
            return 4  # Senior
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_age_classifier(num_classes=5, pretrained=True):
    """
    Create ResNet18 model for age classification
    """
    model = models.resnet18(pretrained=pretrained)
    
    # Replace final layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def main():
    print("=" * 60)
    print("Age Classification Model Training")
    print("=" * 60)
    print(f"Device: {CONFIG['device']}")
    print(f"Dataset: {CONFIG['dataset_path']}")
    print(f"Batch Size: {CONFIG['batch_size']}")
    print(f"Epochs: {CONFIG['num_epochs']}")
    print("=" * 60)
    
    # Check if dataset exists
    if not os.path.exists(CONFIG['dataset_path']):
        print(f"ERROR: Dataset not found at {CONFIG['dataset_path']}")
        print("\nPlease download the UTKFace dataset:")
        print("1. Visit: https://susanqq.github.io/UTKFace/")
        print("2. Download: UTKFace.tar.gz")
        print(f"3. Extract to: {CONFIG['dataset_path']}")
        return
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    print("Loading dataset...")
    full_dataset = UTKFaceDataset(CONFIG['dataset_path'], transform=train_transform)
    
    # Split into train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Apply validation transform
    val_dataset.dataset.transform = val_transform
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers']
    )
    
    # Create model
    print("\nCreating model...")
    device = torch.device(CONFIG['device'])
    model = create_age_classifier(num_classes=CONFIG['age_groups']).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=3, verbose=True
    )
    
    # Training loop
    print("\nStarting training...")
    best_val_acc = 0.0
    
    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{CONFIG['num_epochs']}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(CONFIG['save_path']), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, CONFIG['save_path'])
            print(f"✓ Best model saved (Val Acc: {val_acc:.2f}%)")
    
    print("\n" + "=" * 60)
    print(f"Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {CONFIG['save_path']}")
    print("=" * 60)


if __name__ == '__main__':
    main()
