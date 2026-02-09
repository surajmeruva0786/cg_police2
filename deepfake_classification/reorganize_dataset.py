"""
Dataset Reorganization Script

This script reorganizes the Celeb-DF dataset from DF1 and DF2 folders
into the required format for the pipeline.

Current structure:
DF1/
  ├── Celeb-real/ (158 videos)
  ├── Celeb-synthesis/ (795 videos)
  └── YouTube-real/ (250 videos)
DF2/
  ├── Celeb-real/ (590 folders - frames)
  ├── Celeb-synthesis/ (5639 folders - frames)
  └── YouTube-real/ (300 folders - frames)

Target structure:
dataset/
  ├── real/
  │   ├── video_001/
  │   │   ├── frame_0000.jpg
  │   │   └── ...
  │   └── ...
  └── fake/
      ├── video_001/
      └── ...
"""

import os
import shutil
from pathlib import Path
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_if_frames_or_videos(directory: Path) -> str:
    """
    Check if directory contains videos or frame folders.
    
    Returns:
        'videos' if contains .mp4 files
        'frames' if contains subdirectories with images
    """
    items = list(directory.iterdir())[:5]  # Check first 5 items
    
    for item in items:
        if item.is_file() and item.suffix.lower() in ['.mp4', '.avi', '.mov']:
            return 'videos'
        elif item.is_dir():
            # Check if subdirectory contains images
            sub_items = list(item.iterdir())[:5]
            for sub_item in sub_items:
                if sub_item.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    return 'frames'
    
    return 'unknown'


def copy_frame_folders(source_dir: Path, target_dir: Path, prefix: str = ""):
    """
    Copy frame folders from source to target.
    
    Args:
        source_dir: Source directory containing frame folders
        target_dir: Target directory to copy to
        prefix: Prefix for folder names
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    
    folders = sorted([d for d in source_dir.iterdir() if d.is_dir()])
    logger.info(f"Copying {len(folders)} frame folders from {source_dir.name}")
    
    for i, folder in enumerate(tqdm(folders, desc=f"Copying {source_dir.name}")):
        target_folder = target_dir / f"{prefix}{folder.name}"
        
        if target_folder.exists():
            logger.warning(f"Folder {target_folder.name} already exists, skipping")
            continue
        
        # Copy entire folder
        shutil.copytree(folder, target_folder)


def extract_frames_from_videos(source_dir: Path, target_dir: Path, prefix: str = ""):
    """
    Extract frames from videos and organize into folders.
    
    Args:
        source_dir: Source directory containing video files
        target_dir: Target directory to save frames
        prefix: Prefix for folder names
    """
    import cv2
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    videos = sorted([f for f in source_dir.iterdir() if f.suffix.lower() in ['.mp4', '.avi', '.mov']])
    logger.info(f"Extracting frames from {len(videos)} videos in {source_dir.name}")
    
    for i, video_file in enumerate(tqdm(videos, desc=f"Processing {source_dir.name}")):
        # Create folder for this video
        video_name = video_file.stem
        output_folder = target_dir / f"{prefix}{video_name}"
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Extract frames
        cap = cv2.VideoCapture(str(video_file))
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save every 10th frame to reduce data size
            if frame_count % 10 == 0:
                frame_filename = output_folder / f"frame_{saved_count:04d}.jpg"
                cv2.imwrite(str(frame_filename), frame)
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
        
        if saved_count == 0:
            logger.warning(f"No frames extracted from {video_file.name}")


def reorganize_dataset():
    """Main function to reorganize the dataset"""
    
    # Define paths
    base_dir = Path("d:/github_projects/cg_police2/deepfake_classification")
    df1_dir = base_dir / "DF1"
    df2_dir = base_dir / "DF2"
    output_dir = base_dir / "dataset"
    
    # Create output directories
    real_dir = output_dir / "real"
    fake_dir = output_dir / "fake"
    
    logger.info("="*60)
    logger.info("DATASET REORGANIZATION STARTED")
    logger.info("="*60)
    
    # Check what type of data we have
    df1_real_type = check_if_frames_or_videos(df1_dir / "Celeb-real")
    df1_fake_type = check_if_frames_or_videos(df1_dir / "Celeb-synthesis")
    df2_real_type = check_if_frames_or_videos(df2_dir / "Celeb-real")
    df2_fake_type = check_if_frames_or_videos(df2_dir / "Celeb-synthesis")
    
    logger.info(f"\nDF1 Celeb-real: {df1_real_type}")
    logger.info(f"DF1 Celeb-synthesis: {df1_fake_type}")
    logger.info(f"DF2 Celeb-real: {df2_real_type}")
    logger.info(f"DF2 Celeb-synthesis: {df2_fake_type}")
    
    # Process DF1 Real videos
    logger.info("\n" + "="*60)
    logger.info("Processing DF1 Real Videos")
    logger.info("="*60)
    
    if df1_real_type == 'videos':
        extract_frames_from_videos(
            df1_dir / "Celeb-real",
            real_dir,
            prefix="df1_real_"
        )
    elif df1_real_type == 'frames':
        copy_frame_folders(
            df1_dir / "Celeb-real",
            real_dir,
            prefix="df1_real_"
        )
    
    # Process DF1 Fake videos
    logger.info("\n" + "="*60)
    logger.info("Processing DF1 Fake Videos")
    logger.info("="*60)
    
    if df1_fake_type == 'videos':
        extract_frames_from_videos(
            df1_dir / "Celeb-synthesis",
            fake_dir,
            prefix="df1_fake_"
        )
    elif df1_fake_type == 'frames':
        copy_frame_folders(
            df1_dir / "Celeb-synthesis",
            fake_dir,
            prefix="df1_fake_"
        )
    
    # Process DF2 Real videos
    logger.info("\n" + "="*60)
    logger.info("Processing DF2 Real Videos")
    logger.info("="*60)
    
    if df2_real_type == 'videos':
        extract_frames_from_videos(
            df2_dir / "Celeb-real",
            real_dir,
            prefix="df2_real_"
        )
    elif df2_real_type == 'frames':
        copy_frame_folders(
            df2_dir / "Celeb-real",
            real_dir,
            prefix="df2_real_"
        )
    
    # Process DF2 Fake videos
    logger.info("\n" + "="*60)
    logger.info("Processing DF2 Fake Videos")
    logger.info("="*60)
    
    if df2_fake_type == 'videos':
        extract_frames_from_videos(
            df2_dir / "Celeb-synthesis",
            fake_dir,
            prefix="df2_fake_"
        )
    elif df2_fake_type == 'frames':
        copy_frame_folders(
            df2_dir / "Celeb-synthesis",
            fake_dir,
            prefix="df2_fake_"
        )
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("REORGANIZATION COMPLETE")
    logger.info("="*60)
    
    real_count = len(list(real_dir.iterdir())) if real_dir.exists() else 0
    fake_count = len(list(fake_dir.iterdir())) if fake_dir.exists() else 0
    
    logger.info(f"\nReal video folders: {real_count}")
    logger.info(f"Fake video folders: {fake_count}")
    logger.info(f"Total: {real_count + fake_count}")
    logger.info(f"\nOutput directory: {output_dir}")
    logger.info(f"  - Real: {real_dir}")
    logger.info(f"  - Fake: {fake_dir}")


if __name__ == "__main__":
    reorganize_dataset()
