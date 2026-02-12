"""
CNN Feature Extraction Script

This script extracts features from pre-trained CNN models as described in the paper:
"Comprehensive Analysis of Manual and CNN-based Feature Extraction for 
Deepfake Detection on the Celeb-DF Dataset"

Models used (all with ImageNet pre-trained weights):
- ResNet-50
- ResNet-152
- ResNeXt-101
- ViT-B/16 (Vision Transformer)

Features are extracted from the penultimate layer (before classification).
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import timm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")


class CNNFeatureExtractor:
    """Extract features from pre-trained CNN models"""
    
    def __init__(self, model_name: str):
        """
        Initialize feature extractor with specified model.
        
        Args:
            model_name: One of 'resnet50', 'resnet152', 'resnext101', 'vit_b_16'
        """
        self.model_name = model_name
        self.model = self._load_model(model_name)
        self.model.eval()
        self.model.to(device)
        
        # Define image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        logger.info(f"Loaded {model_name} model")
    
    def _load_model(self, model_name: str) -> nn.Module:
        """Load pre-trained model and remove classification head"""
        
        if model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            # Remove final FC layer
            model = nn.Sequential(*list(model.children())[:-1])
            
        elif model_name == 'resnet152':
            model = models.resnet152(pretrained=True)
            model = nn.Sequential(*list(model.children())[:-1])
            
        elif model_name == 'resnext101':
            model = models.resnext101_32x8d(pretrained=True)
            model = nn.Sequential(*list(model.children())[:-1])
            
        elif model_name == 'vit_b_16':
            # Use timm for Vision Transformer
            model = timm.create_model('vit_base_patch16_224', pretrained=True)
            # Remove classification head
            model.head = nn.Identity()
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        return model
    
    def extract_features(self, image_path: Path) -> np.ndarray:
        """
        Extract features from a single image.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Feature vector as numpy array
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(image_tensor)
            
            # Convert to numpy and flatten
            features = features.cpu().numpy().flatten()
            
            return features
        
        except Exception as e:
            logger.error(f"Error extracting features from {image_path}: {e}")
            return None
    
    def extract_video_features(self, frames_dir: Path) -> pd.DataFrame:
        """
        Extract features from all frames of a video.
        
        Args:
            frames_dir: Directory containing video frames
        
        Returns:
            DataFrame with features for each frame
        """
        # Get all frame files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        frame_files = sorted([
            f for f in frames_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ])
        
        if not frame_files:
            logger.warning(f"No frames found in {frames_dir}")
            return pd.DataFrame()
        
        # Extract features from each frame
        features_list = []
        
        for frame_file in frame_files:
            features = self.extract_features(frame_file)
            
            if features is not None:
                features_dict = {f'feature_{i}': val for i, val in enumerate(features)}
                features_dict['frame_name'] = frame_file.name
                features_list.append(features_dict)
        
        if not features_list:
            return pd.DataFrame()
        
        return pd.DataFrame(features_list)


def aggregate_video_features(frame_features: pd.DataFrame) -> dict:
    """
    Aggregate frame-level features to video-level features.
    
    Args:
        frame_features: DataFrame with features for each frame
    
    Returns:
        Dictionary with aggregated video-level features
    """
    if frame_features.empty:
        return None
    
    # Remove non-numeric columns
    numeric_features = frame_features.select_dtypes(include=[np.number])
    
    # Calculate mean and std
    video_features = {}
    
    for col in numeric_features.columns:
        video_features[f"{col}_mean"] = numeric_features[col].mean()
        video_features[f"{col}_std"] = numeric_features[col].std()
    
    # Add number of frames processed
    video_features['num_frames'] = len(frame_features)
    
    return video_features


def process_dataset(
    model_name: str,
    frames_root: Path,
    output_file: Path,
    label: str
):
    """
    Process entire dataset with specified CNN model.
    
    Args:
        model_name: Name of CNN model to use
        frames_root: Root directory containing video frame directories
        output_file: Path to save features CSV
        label: 'real' or 'fake'
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Extracting {model_name} features for {label.upper()} videos")
    logger.info(f"Frames root: {frames_root}")
    logger.info(f"{'='*60}\n")
    
    # Initialize feature extractor
    extractor = CNNFeatureExtractor(model_name)
    
    # Get all video directories
    video_dirs = sorted([d for d in frames_root.iterdir() if d.is_dir()])
    
    logger.info(f"Found {len(video_dirs)} video directories")
    
    # Process each video
    all_features = []
    
    for i, video_dir in enumerate(tqdm(video_dirs, desc=f"Processing {label} videos")):
        video_id = f"{label}_{i:04d}"
        
        try:
            # Extract frame-level features
            frame_features = extractor.extract_video_features(video_dir)
            
            if frame_features.empty:
                logger.warning(f"No features extracted from {video_dir.name}")
                continue
            
            # Aggregate to video-level
            video_features = aggregate_video_features(frame_features)
            
            if video_features is not None:
                video_features['video_id'] = video_id
                video_features['video_name'] = video_dir.name
                video_features['label'] = label
                all_features.append(video_features)
        
        except Exception as e:
            logger.error(f"Error processing {video_dir.name}: {e}")
            continue
    
    # Convert to DataFrame
    if not all_features:
        logger.error(f"No features extracted for {label} videos!")
        return
    
    df = pd.DataFrame(all_features)
    
    # Save to CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Saved {len(df)} {label} video features to {output_file}")
    logger.info(f"Feature dimensions: {len([c for c in df.columns if c.startswith('feature')])}")
    logger.info(f"{'='*60}\n")


def main():
    """Main execution function"""
    
    # Models to process
    models_to_process = ['resnet50', 'resnet152', 'resnext101', 'vit_b_16']
    
    # Define paths
    # TODO: Update these paths based on your dataset structure
    frames_root_real = Path("path/to/real/frames")  # Update this
    frames_root_fake = Path("path/to/fake/frames")  # Update this
    
    output_dir = Path("results/features")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each model
    for model_name in models_to_process:
        logger.info(f"\n{'#'*60}")
        logger.info(f"Processing model: {model_name.upper()}")
        logger.info(f"{'#'*60}\n")
        
        # Process real videos
        if frames_root_real.exists():
            process_dataset(
                model_name,
                frames_root_real,
                output_dir / f"{model_name}_features_real.csv",
                "real"
            )
        
        # Process fake videos
        if frames_root_fake.exists():
            process_dataset(
                model_name,
                frames_root_fake,
                output_dir / f"{model_name}_features_fake.csv",
                "fake"
            )
        
        # Combine real and fake
        real_csv = output_dir / f"{model_name}_features_real.csv"
        fake_csv = output_dir / f"{model_name}_features_fake.csv"
        
        if real_csv.exists() and fake_csv.exists():
            df_real = pd.read_csv(real_csv)
            df_fake = pd.read_csv(fake_csv)
            
            df_combined = pd.concat([df_real, df_fake], ignore_index=True)
            
            combined_csv = output_dir / f"{model_name}_features_all.csv"
            df_combined.to_csv(combined_csv, index=False)
            
            logger.info(f"Combined {model_name} features: {len(df_combined)} videos")


if __name__ == "__main__":
    main()
