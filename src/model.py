"""
CNN Model Architecture for Certificate Recognition System

This module defines the neural network architecture using ResNet50
with transfer learning for certificate type classification.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import ResNet50, VGG16, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from typing import Tuple, Dict, Optional, List
import os
import json

class CertificateClassifier:
    """CNN-based certificate type classifier using transfer learning."""
    
    def __init__(self, 
                 num_classes: int = 5,
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 base_model_name: str = 'resnet50',
                 dropout_rate: float = 0.5,
                 learning_rate: float = 0.001):
        """
        Initialize the certificate classifier.
        
        Args:
            num_classes: Number of certificate types to classify
            input_shape: Input image shape (height, width, channels)
            base_model_name: Name of the pre-trained model to use
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimization
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.base_model_name = base_model_name
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        
        # Define class names
        self.class_names = [
            'degree_certificates',
            'training_certificates',
            'participation_certificates', 
            'award_certificates',
            'work_experience_certificates'
        ]
        
        # Build the model
        self._build_model()
    
    def _get_base_model(self):
        """
        Get the pre-trained base model.
        
        Returns:
            Pre-trained model without top layers
        """
        if self.base_model_name.lower() == 'resnet50':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.base_model_name.lower() == 'vgg16':
            base_model = VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.base_model_name.lower() == 'efficientnet':
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        else:
            raise ValueError(f"Unsupported base model: {self.base_model_name}")
        
        return base_model
    
    def _build_model(self):
        """Build the complete model architecture."""
        # Get base model
        base_model = self._get_base_model()
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Create the complete model
        inputs = layers.Input(shape=self.input_shape)
        
        # Preprocessing layer
        if self.base_model_name.lower() == 'resnet50':
            # ResNet50 preprocessing
            x = tf.keras.applications.resnet50.preprocess_input(inputs)
        elif self.base_model_name.lower() == 'vgg16':
            # VGG16 preprocessing
            x = tf.keras.applications.vgg16.preprocess_input(inputs)
        elif self.base_model_name.lower() == 'efficientnet':
            # EfficientNet preprocessing
            x = tf.keras.applications.efficientnet.preprocess_input(inputs)
        
        # Pass through base model
        x = base_model(x, training=False)
        
        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dropout for regularization
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Dense layer for classification
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        self.model = models.Model(inputs, outputs)
        
        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_2_accuracy']
        )
        
        print(f"Model built successfully using {self.base_model_name}")
        print(f"Total parameters: {self.model.count_params():,}")
        print(f"Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights]):,}")
    
    def unfreeze_base_model(self, unfreeze_layers: int = 30):
        """
        Unfreeze the last few layers of the base model for fine-tuning.
        
        Args:
            unfreeze_layers: Number of layers to unfreeze from the end
        """
        base_model = self.model.layers[2]  # Base model is typically the 3rd layer
        
        # Unfreeze the last few layers
        base_model.trainable = True
        
        # Freeze all layers except the last 'unfreeze_layers'
        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False
        
        # Recompile with lower learning rate for fine-tuning
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate / 10),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_2_accuracy']
        )
        
        print(f"Unfroze last {unfreeze_layers} layers of base model")
        print(f"Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights]):,}")
    
    def get_callbacks(self, 
                     model_save_path: str = 'models/best_model.h5',
                     patience: int = 10,
                     monitor: str = 'val_accuracy') -> List[callbacks.Callback]:
        """
        Get training callbacks.
        
        Args:
            model_save_path: Path to save the best model
            patience: Number of epochs to wait before early stopping
            monitor: Metric to monitor for early stopping and model saving
            
        Returns:
            List of callbacks
        """
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        callbacks_list = [
            # Model checkpoint
            callbacks.ModelCheckpoint(
                filepath=model_save_path,
                monitor=monitor,
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            ),
            
            # Early stopping
            callbacks.EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=0.5,
                patience=patience // 2,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard logging
            callbacks.TensorBoard(
                log_dir='logs',
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
        ]
        
        return callbacks_list
    
    def train(self,
              train_generator: ImageDataGenerator,
              validation_generator: ImageDataGenerator,
              epochs: int = 50,
              steps_per_epoch: Optional[int] = None,
              validation_steps: Optional[int] = None,
              callbacks: Optional[List[callbacks.Callback]] = None,
              verbose: int = 1) -> Dict:
        """
        Train the model.
        
        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator
            epochs: Number of training epochs
            steps_per_epoch: Steps per epoch (if None, calculated automatically)
            validation_steps: Validation steps (if None, calculated automatically)
            callbacks: List of callbacks
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        if callbacks is None:
            callbacks = self.get_callbacks()
        
        print("Starting model training...")
        print(f"Training for {epochs} epochs")
        
        # Train the model
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=verbose
        )
        
        print("Training completed!")
        return self.history.history
    
    def evaluate(self, test_generator: ImageDataGenerator) -> Dict:
        """
        Evaluate the model on test data.
        
        Args:
            test_generator: Test data generator
            
        Returns:
            Evaluation metrics
        """
        print("Evaluating model on test data...")
        
        # Evaluate the model
        test_loss, test_accuracy, test_top2_accuracy = self.model.evaluate(
            test_generator,
            verbose=1
        )
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_top2_accuracy': test_top2_accuracy
        }
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Top-2 Accuracy: {test_top2_accuracy:.4f}")
        
        return results
    
    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on a single image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        # Ensure image has correct shape
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Make prediction
        predictions = self.model.predict(image, verbose=0)
        
        # Get predicted class and confidence
        predicted_class = np.argmax(predictions, axis=1)
        confidence_scores = np.max(predictions, axis=1)
        
        return predicted_class, confidence_scores
    
    def predict_batch(self, images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on a batch of images.
        
        Args:
            images: Batch of images as numpy array
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        # Make predictions
        predictions = self.model.predict(images, verbose=0)
        
        # Get predicted classes and confidence scores
        predicted_classes = np.argmax(predictions, axis=1)
        confidence_scores = np.max(predictions, axis=1)
        
        return predicted_classes, confidence_scores
    
    def save_model(self, model_path: str = 'models/certificate_classifier.h5'):
        """
        Save the trained model.
        
        Args:
            model_path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save the model
        self.model.save(model_path)
        
        # Save model configuration
        config = {
            'num_classes': self.num_classes,
            'input_shape': self.input_shape,
            'base_model_name': self.base_model_name,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'class_names': self.class_names
        }
        
        config_path = model_path.replace('.h5', '_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Model saved to {model_path}")
        print(f"Configuration saved to {config_path}")
    
    def load_model(self, model_path: str = 'models/certificate_classifier.h5'):
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
        """
        # Load the model
        self.model = models.load_model(model_path)
        
        # Load configuration
        config_path = model_path.replace('.h5', '_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            self.num_classes = config['num_classes']
            self.input_shape = tuple(config['input_shape'])
            self.base_model_name = config['base_model_name']
            self.dropout_rate = config['dropout_rate']
            self.learning_rate = config['learning_rate']
            self.class_names = config['class_names']
        
        print(f"Model loaded from {model_path}")
    
    def get_model_summary(self) -> str:
        """
        Get a string representation of the model summary.
        
        Returns:
            Model summary as string
        """
        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        return '\n'.join(summary_list)
    
    def plot_model_architecture(self, save_path: str = 'models/model_architecture.png'):
        """
        Plot and save the model architecture.
        
        Args:
            save_path: Path to save the architecture plot
        """
        try:
            from tensorflow.keras.utils import plot_model
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Plot model
            plot_model(
                self.model,
                to_file=save_path,
                show_shapes=True,
                show_layer_names=True,
                rankdir='TB'
            )
            
            print(f"Model architecture saved to {save_path}")
            
        except ImportError:
            print("graphviz not installed. Skipping model architecture plot.")
        except Exception as e:
            print(f"Error plotting model architecture: {str(e)}")

def create_data_generators(train_dir: str,
                          val_dir: str,
                          test_dir: str,
                          batch_size: int = 32,
                          target_size: Tuple[int, int] = (224, 224)) -> Tuple[ImageDataGenerator, ImageDataGenerator, ImageDataGenerator]:
    """
    Create data generators for training, validation, and testing.
    
    Args:
        train_dir: Training data directory
        val_dir: Validation data directory
        test_dir: Test data directory
        batch_size: Batch size for training
        target_size: Target image size
        
    Returns:
        Tuple of (train_generator, val_generator, test_generator)
    """
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Validation and test data generators (no augmentation)
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator

if __name__ == "__main__":
    # Example usage
    classifier = CertificateClassifier(
        num_classes=5,
        input_shape=(224, 224, 3),
        base_model_name='resnet50',
        dropout_rate=0.5,
        learning_rate=0.001
    )
    
    print("Model Summary:")
    print(classifier.get_model_summary())


