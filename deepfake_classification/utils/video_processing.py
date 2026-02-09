"""
Video Processing Utilities
Extract representative frames from videos for analysis
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, List, Tuple, Optional
from tqdm import tqdm


def extract_frames_from_video(
    video_path: Union[str, Path],
    num_frames: int = 1,
    method: str = "uniform",
    output_dir: Optional[Union[str, Path]] = None,
    save_frames: bool = True
) -> List[np.ndarray]:
    """
    Extract representative frames from a video.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        method: Extraction method - "uniform", "middle", or "random"
        output_dir: Directory to save extracted frames (if save_frames=True)
        save_frames: Whether to save frames to disk
        
    Returns:
        List of extracted frames as numpy arrays (H, W, C)
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames == 0:
        raise ValueError(f"Video has no frames: {video_path}")
    
    # Determine frame indices to extract
    if method == "uniform":
        # Uniformly sample frames across the video
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    elif method == "middle":
        # Extract frames from the middle of the video
        middle_idx = total_frames // 2
        half_range = num_frames // 2
        start_idx = max(0, middle_idx - half_range)
        frame_indices = np.arange(start_idx, start_idx + num_frames)
        frame_indices = np.clip(frame_indices, 0, total_frames - 1)
    elif method == "random":
        # Randomly sample frames
        frame_indices = np.random.choice(total_frames, num_frames, replace=False)
        frame_indices = np.sort(frame_indices)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Extract frames
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
            # Save frame if requested
            if save_frames and output_dir is not None:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                frame_filename = f"{video_path.stem}_frame_{idx:06d}.jpg"
                frame_path = output_dir / frame_filename
                
                # Save as BGR (OpenCV format)
                cv2.imwrite(str(frame_path), frame)
    
    cap.release()
    
    return frames


def extract_frames_from_dataset(
    video_dir: Union[str, Path],
    output_dir: Union[str, Path],
    num_frames: int = 1,
    method: str = "middle",
    video_extensions: List[str] = [".mp4", ".avi", ".mov", ".mkv"]
) -> dict:
    """
    Extract frames from all videos in a directory.
    
    Args:
        video_dir: Directory containing videos
        output_dir: Directory to save extracted frames
        num_frames: Number of frames to extract per video
        method: Extraction method
        video_extensions: List of valid video file extensions
        
    Returns:
        Dictionary mapping video paths to extracted frame paths
    """
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all video files
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_dir.rglob(f"*{ext}"))
    
    print(f"Found {len(video_files)} videos in {video_dir}")
    
    # Extract frames from each video
    frame_mapping = {}
    failed_videos = []
    
    for video_path in tqdm(video_files, desc="Extracting frames"):
        try:
            # Create subdirectory for this video's frames
            video_output_dir = output_dir / video_path.stem
            
            frames = extract_frames_from_video(
                video_path,
                num_frames=num_frames,
                method=method,
                output_dir=video_output_dir,
                save_frames=True
            )
            
            # Get saved frame paths
            frame_paths = sorted(video_output_dir.glob("*.jpg"))
            frame_mapping[str(video_path)] = [str(p) for p in frame_paths]
            
        except Exception as e:
            print(f"Failed to process {video_path}: {e}")
            failed_videos.append(str(video_path))
    
    print(f"\nSuccessfully processed {len(frame_mapping)} videos")
    if failed_videos:
        print(f"Failed to process {len(failed_videos)} videos")
    
    return frame_mapping


def resize_frame(
    frame: np.ndarray,
    target_size: Tuple[int, int],
    maintain_aspect: bool = False
) -> np.ndarray:
    """
    Resize a frame to target size.
    
    Args:
        frame: Input frame (H, W, C)
        target_size: Target size as (width, height)
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized frame
    """
    if maintain_aspect:
        # Resize maintaining aspect ratio
        h, w = frame.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to target size
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        resized = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
    else:
        # Direct resize
        resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
    
    return resized


def get_video_info(video_path: Union[str, Path]) -> dict:
    """
    Get information about a video file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video information
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    info = {
        "path": str(video_path),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration_sec": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    
    return info
