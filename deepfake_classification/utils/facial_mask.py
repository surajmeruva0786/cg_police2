"""
Facial Mask Extraction Pipeline

This module implements facial mask extraction as described in the paper:
"Comprehensive Analysis of Manual and CNN-based Feature Extraction for 
Deepfake Detection on the Celeb-DF Dataset"

The pipeline:
1. Detects faces using dlib's face detector
2. Extracts 68 facial landmarks
3. Creates a binary mask from the convex hull of landmarks
4. Processes all frames in a video directory
"""

import cv2
import dlib
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FacialMaskExtractor:
    """Extract facial masks from images using dlib landmarks"""
    
    def __init__(self, predictor_path: Optional[str] = None):
        """
        Initialize the facial mask extractor.
        
        Args:
            predictor_path: Path to dlib's shape predictor model.
                          If None, assumes it's in the default location.
        """
        # Initialize dlib's face detector
        self.detector = dlib.get_frontal_face_detector()
        
        # Initialize the shape predictor for 68 landmarks
        if predictor_path is None:
            # Default path - user should download shape_predictor_68_face_landmarks.dat
            predictor_path = "shape_predictor_68_face_landmarks.dat"
        
        try:
            self.predictor = dlib.shape_predictor(predictor_path)
            logger.info(f"Loaded shape predictor from {predictor_path}")
        except RuntimeError as e:
            logger.error(f"Failed to load shape predictor: {e}")
            logger.error("Please download shape_predictor_68_face_landmarks.dat from:")
            logger.error("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            raise
    
    def detect_face_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect face and extract 68 facial landmarks.
        
        Args:
            image: Input image (BGR format from cv2)
        
        Returns:
            Array of shape (68, 2) containing landmark coordinates,
            or None if no face detected
        """
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Ensure 8-bit image (dlib requirement)
        if gray.dtype != np.uint8:
            logger.warning(f"Image not uint8 ({gray.dtype}), converting...")
            gray = gray.astype(np.uint8)
            
        # Ensure C-contiguous memory layout (dlib requirement)
        if not gray.flags['C_CONTIGUOUS']:
            gray = np.ascontiguousarray(gray)
        
        # Detect faces
        faces = self.detector(gray, 1)
        
        if len(faces) == 0:
            logger.warning("No face detected in image")
            return None
        
        if len(faces) > 1:
            logger.warning(f"Multiple faces detected ({len(faces)}), using the first one")
        
        # Get landmarks for the first face
        face = faces[0]
        landmarks = self.predictor(gray, face)
        
        # Convert to numpy array
        landmarks_array = np.array([[p.x, p.y] for p in landmarks.parts()])
        
        return landmarks_array
    
    def create_facial_mask(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Create binary mask from facial landmarks using convex hull.
        
        Args:
            image: Input image (for dimensions)
            landmarks: Array of shape (68, 2) containing landmark coordinates
        
        Returns:
            Binary mask of same height and width as input image
        """
        # Create empty mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Get convex hull of landmarks
        hull = cv2.convexHull(landmarks.astype(np.int32))
        
        # Fill the convex hull to create mask
        cv2.fillConvexPoly(mask, hull, 255)
        
        return mask
    
    def extract_mask_from_image(self, image_path: Union[str, Path]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract facial mask from a single image.
        
        Args:
            image_path: Path to the input image
        
        Returns:
            Tuple of (mask, landmarks) or (None, None) if face not detected
        """
        # Load image
        image = cv2.imread(str(image_path))
        
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None, None
        
        # Detect landmarks
        landmarks = self.detect_face_landmarks(image)
        
        if landmarks is None:
            return None, None
        
        # Create mask
        mask = self.create_facial_mask(image, landmarks)
        
        return mask, landmarks
    
    def process_video_frames(
        self,
        frames_dir: Union[str, Path],
        output_dir: Union[str, Path],
        save_landmarks: bool = False
    ) -> int:
        """
        Process all frames in a directory and generate masks.
        
        Args:
            frames_dir: Directory containing video frames
            output_dir: Directory to save masks
            save_landmarks: Whether to save landmark coordinates as .npy files
        
        Returns:
            Number of successfully processed frames
        """
        frames_dir = Path(frames_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [
            f for f in frames_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            logger.warning(f"No image files found in {frames_dir}")
            return 0
        
        logger.info(f"Processing {len(image_files)} frames from {frames_dir}")
        
        successful = 0
        failed = 0
        
        for image_file in sorted(image_files):
            # Extract mask
            mask, landmarks = self.extract_mask_from_image(image_file)
            
            if mask is None:
                logger.warning(f"Failed to extract mask from {image_file.name}")
                failed += 1
                continue
            
            # Save mask
            mask_path = output_dir / f"{image_file.stem}_mask{image_file.suffix}"
            cv2.imwrite(str(mask_path), mask)
            
            # Optionally save landmarks
            if save_landmarks and landmarks is not None:
                landmarks_path = output_dir / f"{image_file.stem}_landmarks.npy"
                np.save(str(landmarks_path), landmarks)
            
            successful += 1
            
            if successful % 10 == 0:
                logger.info(f"Processed {successful}/{len(image_files)} frames")
        
        logger.info(f"Completed: {successful} successful, {failed} failed")
        return successful


def process_dataset_masks(
    dataset_root: Union[str, Path],
    output_root: Union[str, Path],
    predictor_path: Optional[str] = None
):
    """
    Process entire Celeb-DF dataset to generate masks.
    
    Args:
        dataset_root: Root directory containing video frame directories
        output_root: Root directory to save masks
        predictor_path: Path to dlib shape predictor model
    """
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    
    # Initialize extractor
    extractor = FacialMaskExtractor(predictor_path)
    
    # Process all video directories
    video_dirs = [d for d in dataset_root.iterdir() if d.is_dir()]
    
    logger.info(f"Found {len(video_dirs)} video directories")
    
    for video_dir in video_dirs:
        logger.info(f"\nProcessing video: {video_dir.name}")
        
        # Create output directory for this video
        output_dir = output_root / video_dir.name
        
        # Process frames
        extractor.process_video_frames(video_dir, output_dir)


if __name__ == "__main__":
    """
    Example usage:
    
    python utils/facial_mask.py
    """
    import sys
    
    # Example: Process a single video's frames
    if len(sys.argv) > 1:
        frames_dir = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else frames_dir + "_masks"
        
        extractor = FacialMaskExtractor()
        extractor.process_video_frames(frames_dir, output_dir, save_landmarks=True)
    else:
        print("Usage: python facial_mask.py <frames_dir> [output_dir]")
        print("\nOr modify the script to process your entire dataset")
