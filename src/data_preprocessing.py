"""
Data Preprocessing Module for Certificate Recognition System

This module handles:
- Image loading and preprocessing
- Data augmentation
- Train/validation/test splits
- Dataset preparation
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import albumentations as A
from tqdm import tqdm
import json
import random
from typing import Tuple, List, Dict, Optional

class CertificateDataPreprocessor:
    """Handles all data preprocessing operations for certificate images."""
    
    def __init__(self, 
                 input_size: Tuple[int, int] = (224, 224),
                 augment: bool = True,
                 normalize: bool = True):
        """
        Initialize the data preprocessor.
        
        Args:
            input_size: Target size for images (height, width)
            augment: Whether to apply data augmentation
            normalize: Whether to normalize pixel values
        """
        self.input_size = input_size
        self.augment = augment
        self.normalize = normalize
        self.label_encoder = LabelEncoder()
        
        # Define augmentation pipeline
        self.augmentation_pipeline = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.Transpose(p=0.5),
            A.OneOf([
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                A.IAAPiecewiseAffine(p=0.3),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.IAASharpen(),
                A.IAAEmboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.HueSaturationValue(p=0.3),
        ])
        
        # Define validation pipeline (no augmentation)
        self.validation_pipeline = A.Compose([
            A.Resize(height=input_size[0], width=input_size[1]),
        ])
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image
            image = cv2.resize(image, (self.input_size[1], self.input_size[0]))
            
            # Normalize if required
            if self.normalize:
                image = image.astype(np.float32) / 255.0
            
            return image
            
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return None
    
    def apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to an image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Augmented image
        """
        if not self.augment:
            return image
        
        # Convert to uint8 for augmentation
        if self.normalize:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
        
        # Apply augmentation
        augmented = self.augmentation_pipeline(image=image_uint8)
        augmented_image = augmented['image']
        
        # Convert back to float if normalized
        if self.normalize:
            augmented_image = augmented_image.astype(np.float32) / 255.0
        
        return augmented_image
    
    def create_dataset(self, 
                      data_dir: str,
                      output_dir: str,
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15,
                      test_ratio: float = 0.15,
                      augment_factor: int = 3) -> Dict:
        """
        Create a complete dataset with train/validation/test splits.
        
        Args:
            data_dir: Directory containing certificate images organized by class
            output_dir: Directory to save processed data
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            test_ratio: Ratio of test data
            augment_factor: Number of augmented versions per original image
            
        Returns:
            Dictionary containing dataset information
        """
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
        
        # Get class names
        class_names = [d for d in os.listdir(data_dir) 
                      if os.path.isdir(os.path.join(data_dir, d))]
        class_names.sort()
        
        print(f"Found {len(class_names)} classes: {class_names}")
        
        dataset_info = {
            'classes': class_names,
            'train_samples': 0,
            'val_samples': 0,
            'test_samples': 0,
            'total_samples': 0
        }
        
        # Process each class
        for class_name in tqdm(class_names, desc="Processing classes"):
            class_dir = os.path.join(data_dir, class_name)
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            print(f"Processing {len(image_files)} images for class '{class_name}'")
            
            # Split images into train/val/test
            train_files, temp_files = train_test_split(
                image_files, 
                train_size=train_ratio, 
                random_state=42
            )
            
            val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
            val_files, test_files = train_test_split(
                temp_files,
                train_size=val_ratio_adjusted,
                random_state=42
            )
            
            # Process training data with augmentation
            self._process_split(
                image_files=train_files,
                class_dir=class_dir,
                output_dir=os.path.join(output_dir, 'train'),
                class_name=class_name,
                augment=True,
                augment_factor=augment_factor
            )
            dataset_info['train_samples'] += len(train_files) * augment_factor
            
            # Process validation data
            self._process_split(
                image_files=val_files,
                class_dir=class_dir,
                output_dir=os.path.join(output_dir, 'val'),
                class_name=class_name,
                augment=False
            )
            dataset_info['val_samples'] += len(val_files)
            
            # Process test data
            self._process_split(
                image_files=test_files,
                class_dir=class_dir,
                output_dir=os.path.join(output_dir, 'test'),
                class_name=class_name,
                augment=False
            )
            dataset_info['test_samples'] += len(test_files)
            
            dataset_info['total_samples'] += len(image_files)
        
        # Save dataset info
        with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"\nDataset created successfully!")
        print(f"Total samples: {dataset_info['total_samples']}")
        print(f"Training samples: {dataset_info['train_samples']}")
        print(f"Validation samples: {dataset_info['val_samples']}")
        print(f"Test samples: {dataset_info['test_samples']}")
        
        return dataset_info
    
    def _process_split(self, 
                      image_files: List[str],
                      class_dir: str,
                      output_dir: str,
                      class_name: str,
                      augment: bool = False,
                      augment_factor: int = 1):
        """
        Process a specific data split (train/val/test).
        
        Args:
            image_files: List of image filenames
            class_dir: Directory containing the images
            output_dir: Output directory for processed images
            class_name: Name of the class
            augment: Whether to apply augmentation
            augment_factor: Number of augmented versions per image
        """
        class_output_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)
        
        for i, filename in enumerate(image_files):
            image_path = os.path.join(class_dir, filename)
            image = self.load_image(image_path)
            
            if image is None:
                continue
            
            # Save original image
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(class_output_dir, f"{base_name}_original.jpg")
            self._save_image(image, output_path)
            
            # Create augmented versions if required
            if augment and augment_factor > 1:
                for j in range(augment_factor - 1):
                    augmented_image = self.apply_augmentation(image)
                    aug_output_path = os.path.join(class_output_dir, f"{base_name}_aug_{j+1}.jpg")
                    self._save_image(augmented_image, aug_output_path)
    
    def _save_image(self, image: np.ndarray, output_path: str):
        """
        Save an image to disk.
        
        Args:
            image: Image as numpy array
            output_path: Path to save the image
        """
        try:
            # Convert to uint8 for saving
            if self.normalize:
                save_image = (image * 255).astype(np.uint8)
            else:
                save_image = image.astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            save_image = cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, save_image)
            
        except Exception as e:
            print(f"Error saving image {output_path}: {str(e)}")
    
    def create_sample_dataset(self, output_dir: str = "data/raw") -> Dict:
        """
        Create a sample dataset structure for demonstration.
        This function creates placeholder directories and sample images.
        
        Args:
            output_dir: Directory to create the sample dataset
            
        Returns:
            Dictionary with dataset information
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Define certificate types
        certificate_types = [
            "degree_certificates",
            "training_certificates", 
            "participation_certificates",
            "award_certificates",
            "work_experience_certificates"
        ]
        
        # Create directories for each type
        for cert_type in certificate_types:
            os.makedirs(os.path.join(output_dir, cert_type), exist_ok=True)
        
        print(f"Sample dataset structure created in {output_dir}")
        print("Please add your certificate images to the respective directories:")
        for cert_type in certificate_types:
            print(f"  - {output_dir}/{cert_type}/")
        
        return {
            'classes': certificate_types,
            'total_samples': 0,
            'message': 'Please add your certificate images to the created directories'
        }

def main():
    """Main function to demonstrate data preprocessing."""
    # Initialize preprocessor
    preprocessor = CertificateDataPreprocessor(
        input_size=(224, 224),
        augment=True,
        normalize=True
    )
    
    # Create sample dataset structure
    print("Creating sample dataset structure...")
    preprocessor.create_sample_dataset("data/raw")
    
    # If you have actual data, uncomment the following lines:
    # print("Processing actual dataset...")
    # dataset_info = preprocessor.create_dataset(
    #     data_dir="data/raw",
    #     output_dir="data/processed",
    #     train_ratio=0.7,
    #     val_ratio=0.15,
    #     test_ratio=0.15,
    #     augment_factor=3
    # )

if __name__ == "__main__":
    main()


