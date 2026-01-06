"""
Emotion Detection Model Training Script
Train CNN model on FER2013 dataset

Dataset: https://www.kaggle.com/datasets/msambare/fer2013
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import pandas as pd
import os
from pathlib import Path

# Configuration
CONFIG = {
    'dataset_path': 'data/datasets/FER2013',
    'batch_size': 64,
    'num_epochs': 50,
    'learning_rate': 0.001,
    'image_size': (48, 48),
    'save_path': 'models/emotion/emotion_model.h5',
    'emotions': ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'],
    'num_classes': 7,
}


def create_emotion_model(input_shape=(48, 48, 1), num_classes=7):
    """
    Create CNN model for emotion detection
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Block 4
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Fully connected layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def load_fer2013_data(dataset_path):
    """
    Load FER2013 dataset
    Supports both CSV format and folder structure
    """
    csv_path = os.path.join(dataset_path, 'fer2013.csv')
    
    if os.path.exists(csv_path):
        # Load from CSV
        print("Loading dataset from CSV...")
        df = pd.read_csv(csv_path)
        
        # Extract images and labels
        images = []
        labels = []
        
        for idx, row in df.iterrows():
            pixels = np.array(row['pixels'].split(), dtype=np.uint8)
            image = pixels.reshape(48, 48, 1)
            images.append(image)
            labels.append(row['emotion'])
        
        images = np.array(images) / 255.0
        labels = keras.utils.to_categorical(labels, CONFIG['num_classes'])
        
        # Split train/val
        split_idx = int(0.8 * len(images))
        return (
            images[:split_idx], labels[:split_idx],
            images[split_idx:], labels[split_idx:]
        )
    else:
        # Use folder structure with ImageDataGenerator
        return None


def main():
    print("=" * 60)
    print("Emotion Detection Model Training")
    print("=" * 60)
    print(f"Dataset: {CONFIG['dataset_path']}")
    print(f"Batch Size: {CONFIG['batch_size']}")
    print(f"Epochs: {CONFIG['num_epochs']}")
    print(f"Emotions: {', '.join(CONFIG['emotions'])}")
    print("=" * 60)
    
    # Check if dataset exists
    if not os.path.exists(CONFIG['dataset_path']):
        print(f"ERROR: Dataset not found at {CONFIG['dataset_path']}")
        print("\nPlease download the FER2013 dataset:")
        print("1. Visit: https://www.kaggle.com/datasets/msambare/fer2013")
        print("2. Download the dataset (requires Kaggle account)")
        print(f"3. Extract to: {CONFIG['dataset_path']}")
        print("\nFile structure should be:")
        print(f"{CONFIG['dataset_path']}/")
        print("  ├── fer2013.csv")
        print("  OR")
        print("  ├── train/")
        print("  │   ├── angry/")
        print("  │   ├── happy/")
        print("  │   └── ...")
        print("  └── test/")
        return
    
    # Try loading from CSV first
    data = load_fer2013_data(CONFIG['dataset_path'])
    
    if data is not None:
        # CSV format
        X_train, y_train, X_val, y_val = data
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Data augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        # Create generators
        train_generator = train_datagen.flow(
            X_train, y_train,
            batch_size=CONFIG['batch_size']
        )
        
        val_generator = ImageDataGenerator().flow(
            X_val, y_val,
            batch_size=CONFIG['batch_size']
        )
        
        steps_per_epoch = len(X_train) // CONFIG['batch_size']
        validation_steps = len(X_val) // CONFIG['batch_size']
    else:
        # Folder structure
        print("Loading dataset from folders...")
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            os.path.join(CONFIG['dataset_path'], 'train'),
            target_size=CONFIG['image_size'],
            batch_size=CONFIG['batch_size'],
            color_mode='grayscale',
            class_mode='categorical'
        )
        
        val_generator = val_datagen.flow_from_directory(
            os.path.join(CONFIG['dataset_path'], 'test'),
            target_size=CONFIG['image_size'],
            batch_size=CONFIG['batch_size'],
            color_mode='grayscale',
            class_mode='categorical'
        )
        
        steps_per_epoch = train_generator.samples // CONFIG['batch_size']
        validation_steps = val_generator.samples // CONFIG['batch_size']
    
    # Create model
    print("\nCreating model...")
    model = create_emotion_model(
        input_shape=(*CONFIG['image_size'], 1),
        num_classes=CONFIG['num_classes']
    )
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Model summary
    print("\nModel Summary:")
    model.summary()
    
    # Callbacks
    os.makedirs(os.path.dirname(CONFIG['save_path']), exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            CONFIG['save_path'],
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    print("\nStarting training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=CONFIG['num_epochs'],
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    # Final results
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    best_val_acc = max(history.history['val_accuracy'])
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final training accuracy: {final_train_acc * 100:.2f}%")
    print(f"Final validation accuracy: {final_val_acc * 100:.2f}%")
    print(f"Best validation accuracy: {best_val_acc * 100:.2f}%")
    print(f"Model saved to: {CONFIG['save_path']}")
    print("=" * 60)
    
    # Save training history
    history_path = CONFIG['save_path'].replace('.h5', '_history.npy')
    np.save(history_path, history.history)
    print(f"Training history saved to: {history_path}")


if __name__ == '__main__':
    # Set GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU available: {len(gpus)} device(s)")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU available, training on CPU")
    
    main()
