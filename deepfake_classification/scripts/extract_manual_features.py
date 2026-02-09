"""
Manual Feature Extraction Script

This script extracts all 30 manual features from the Celeb-DF dataset
as described in the paper:
"Comprehensive Analysis of Manual and CNN-based Feature Extraction for 
Deepfake Detection on the Celeb-DF Dataset"

Features extracted:
- 7 Landmark features
- 6 Illumination features  
- 11 Color features
- 6 Compression features

The script processes all videos, extracts features from each frame,
and aggregates to video-level features.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from manualfeatures import extract_features_from_image
from config import (
    DATASET_PATH,
    REAL_VIDEOS_PATH,
    FAKE_VIDEOS_PATH,
    FEATURES_OUTPUT_PATH
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_video_features(frames_dir: Path, masks_dir: Path) -> pd.DataFrame:
    """
    Extract features from all frames of a video.
    
    Args:
        frames_dir: Directory containing video frames
        masks_dir: Directory containing facial masks
    
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
        # Check if corresponding mask exists
        mask_file = masks_dir / f"{frame_file.stem}_mask{frame_file.suffix}"
        
        if not mask_file.exists():
            logger.warning(f"Mask not found for {frame_file.name}, skipping")
            continue
        
        # Extract features
        features = extract_features_from_image(str(frame_file))
        
        if features is not None:
            features['frame_name'] = frame_file.name
            features_list.append(features)
    
    if not features_list:
        logger.warning(f"No features extracted from {frames_dir}")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(features_list)
    
    return df


def aggregate_video_features(frame_features: pd.DataFrame) -> dict:
    """
    Aggregate frame-level features to video-level features.
    
    Uses mean and standard deviation across all frames.
    
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


def process_video_directory(
    video_dir: Path,
    masks_root: Path,
    video_id: str
) -> dict:
    """
    Process a single video directory.
    
    Args:
        video_dir: Directory containing video frames
        masks_root: Root directory containing masks
        video_id: Video identifier
    
    Returns:
        Dictionary with video-level features
    """
    # Get corresponding masks directory
    masks_dir = masks_root / video_dir.name
    
    if not masks_dir.exists():
        logger.warning(f"Masks directory not found: {masks_dir}")
        return None
    
    # Extract frame-level features
    frame_features = extract_video_features(video_dir, masks_dir)
    
    if frame_features.empty:
        return None
    
    # Aggregate to video-level
    video_features = aggregate_video_features(frame_features)
    
    if video_features is not None:
        video_features['video_id'] = video_id
        video_features['video_name'] = video_dir.name
    
    return video_features


def process_dataset(
    frames_root: Path,
    masks_root: Path,
    output_file: Path,
    label: str
):
    """
    Process entire dataset (real or fake videos).
    
    Args:
        frames_root: Root directory containing video frame directories
        masks_root: Root directory containing mask directories
        output_file: Path to save features CSV
        label: 'real' or 'fake'
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {label.upper()} videos")
    logger.info(f"Frames root: {frames_root}")
    logger.info(f"Masks root: {masks_root}")
    logger.info(f"{'='*60}\n")
    
    # Get all video directories
    video_dirs = sorted([d for d in frames_root.iterdir() if d.is_dir()])
    
    logger.info(f"Found {len(video_dirs)} video directories")
    
    # Process each video
    all_features = []
    
    for i, video_dir in enumerate(tqdm(video_dirs, desc=f"Processing {label} videos")):
        video_id = f"{label}_{i:04d}"
        
        try:
            video_features = process_video_directory(video_dir, masks_root, video_id)
            
            if video_features is not None:
                video_features['label'] = label
                all_features.append(video_features)
            else:
                logger.warning(f"Failed to extract features from {video_dir.name}")
        
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
    logger.info(f"Feature columns: {len(df.columns)}")
    logger.info(f"{'='*60}\n")


def main():
    """Main execution function"""
    
    # Define paths
    # TODO: Update these paths based on your dataset structure
    frames_root_real = Path("path/to/real/frames")  # Update this
    frames_root_fake = Path("path/to/fake/frames")  # Update this
    masks_root_real = Path("path/to/real/masks")    # Update this
    masks_root_fake = Path("path/to/fake/masks")    # Update this
    
    output_dir = Path("results/features")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process real videos
    logger.info("Starting manual feature extraction for Celeb-DF dataset")
    
    if frames_root_real.exists():
        process_dataset(
            frames_root_real,
            masks_root_real,
            output_dir / "manual_features_real.csv",
            "real"
        )
    else:
        logger.warning(f"Real frames directory not found: {frames_root_real}")
    
    # Process fake videos
    if frames_root_fake.exists():
        process_dataset(
            frames_root_fake,
            masks_root_fake,
            output_dir / "manual_features_fake.csv",
            "fake"
        )
    else:
        logger.warning(f"Fake frames directory not found: {frames_root_fake}")
    
    # Combine real and fake features
    real_csv = output_dir / "manual_features_real.csv"
    fake_csv = output_dir / "manual_features_fake.csv"
    
    if real_csv.exists() and fake_csv.exists():
        df_real = pd.read_csv(real_csv)
        df_fake = pd.read_csv(fake_csv)
        
        df_combined = pd.concat([df_real, df_fake], ignore_index=True)
        
        combined_csv = output_dir / "manual_features_all.csv"
        df_combined.to_csv(combined_csv, index=False)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Combined features saved to {combined_csv}")
        logger.info(f"Total videos: {len(df_combined)}")
        logger.info(f"Real: {len(df_real)}, Fake: {len(df_fake)}")
        logger.info(f"{'='*60}\n")
        
        # Print summary statistics
        print("\nFeature Summary:")
        print(df_combined.describe())


if __name__ == "__main__":
    main()
