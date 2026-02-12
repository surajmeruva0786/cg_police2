"""
Configuration file for Deepfake Detection Research Reproduction
Reproducing: "Comprehensive Analysis of Manual and CNN-based Feature 
Extraction for Deepfake Detection on the Celeb-DF Dataset"
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Data subdirectories
RAW_VIDEO_DIR = DATA_DIR / "raw"
FRAMES_DIR = DATA_DIR / "frames"
MASKED_DIR = DATA_DIR / "masked"
FEATURES_DIR = DATA_DIR / "features"

# Output subdirectories
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
MODELS_DIR = OUTPUT_DIR / "models"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_VIDEO_DIR, FRAMES_DIR, MASKED_DIR, 
                  FEATURES_DIR, OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

# Celeb-DF dataset structure
REAL_VIDEO_COUNT = 158
FAKE_VIDEO_COUNT = 795
TOTAL_VIDEOS = REAL_VIDEO_COUNT + FAKE_VIDEO_COUNT  # 953

# Video processing
FRAMES_PER_VIDEO = 1  # Extract representative frame(s) per video
TARGET_FRAME_SIZE = (224, 224)  # For CNN input

# ============================================================================
# FACIAL MASK PIPELINE
# ============================================================================

# Face detection (dlib)
FACE_DETECTOR_MODEL = "hog"  # Options: "hog" or "cnn"
FACE_LANDMARKS_MODEL = "68"  # 68-point facial landmarks

# Landmark indices for facial regions
LANDMARKS = {
    "jaw": list(range(0, 17)),
    "right_eyebrow": list(range(17, 22)),
    "left_eyebrow": list(range(22, 27)),
    "nose_bridge": list(range(27, 31)),
    "nose_tip": list(range(31, 36)),
    "right_eye": list(range(36, 42)),
    "left_eye": list(range(42, 48)),
    "outer_mouth": list(range(48, 60)),
    "inner_mouth": list(range(60, 68))
}

# Mask generation parameters
MASK_PADDING = 15  # Pixels to expand mask region
MASK_BLUR_KERNEL = (21, 21)  # Gaussian blur kernel size
MASK_BLUR_SIGMA = 10  # Gaussian blur sigma
MASK_ALPHA = 0.8  # Alpha blending factor (0=original, 1=full mask)

# Include forehead in mask
INCLUDE_FOREHEAD = True
FOREHEAD_ARC_RADIUS_MULTIPLIER = 1.3  # Multiplier for eyebrow-based arc

# ============================================================================
# MANUAL FEATURE EXTRACTION (30 FEATURES)
# ============================================================================

MANUAL_FEATURES = {
    # Geometric features (8)
    "geometric": [
        "eye_ratio",
        "eye_width_ratio",
        "height_width_ratio",
        "eye_mouth_ratio",
        "eye_distance",
        "face_width",
        "face_height",
        "eye_mouth_distance"
    ],
    
    # Illumination features (4)
    "illumination": [
        "light_direction",
        "direction_consistency",
        "highlight_shadow_ratio",
        "illumination_uniformity"
    ],
    
    # Texture & Edge features (2)
    "texture_edge": [
        "gradient_magnitude",
        "local_contrast"
    ],
    
    # HSV Color Statistics (7)
    "hsv_stats": [
        "hue_mean",
        "hue_std",
        "saturation_mean",
        "saturation_std",
        "value_mean",
        "value_std",
        "color_contrast"
    ],
    
    # Color Channel Correlations (3)
    "color_correlations": [
        "hs_correlation",
        "hv_correlation",
        "sv_correlation"
    ],
    
    # Skin Consistency (1)
    "skin": [
        "skin_pixel_ratio"
    ],
    
    # DCT Frequency features (4)
    "dct": [
        "dct_mean",
        "dct_std",
        "dct_range",
        "high_low_freq_ratio"
    ],
    
    # Pixel Variance features (2)
    "variance": [
        "variance_mean",
        "variance_std"
    ]
}

# Total manual features
TOTAL_MANUAL_FEATURES = sum(len(v) for v in MANUAL_FEATURES.values())  # Should be 30

# Feature extraction parameters
SOBEL_KERNEL_SIZE = 3
DCT_BLOCK_SIZE = 8
VARIANCE_BLOCK_SIZE = 16

# Skin detection (HSV ranges)
SKIN_HSV_LOWER = (0, 20, 70)
SKIN_HSV_UPPER = (20, 255, 255)

# ============================================================================
# CNN FEATURE EXTRACTION
# ============================================================================

CNN_MODELS = {
    "resnet50": {
        "name": "resnet50",
        "pretrained": True,
        "feature_dim": 2048,
        "layer": "avgpool"
    },
    "resnet152": {
        "name": "resnet152",
        "pretrained": True,
        "feature_dim": 2048,
        "layer": "avgpool"
    },
    "resnext101_32x8d": {
        "name": "resnext101_32x8d",
        "pretrained": True,
        "feature_dim": 2048,
        "layer": "avgpool"
    },
    "vit_b_16": {
        "name": "vit_base_patch16_224",  # timm model name
        "pretrained": True,
        "feature_dim": 768,  # CLS token dimension
        "layer": "cls_token"
    }
}

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Batch processing
CNN_BATCH_SIZE = 32

# ============================================================================
# DIMENSIONALITY REDUCTION
# ============================================================================

# PCA parameters
PCA_N_COMPONENTS = 3
PCA_RANDOM_STATE = 42

# t-SNE parameters (as specified in paper)
TSNE_N_COMPONENTS = 3
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000
TSNE_INIT = "pca"
TSNE_RANDOM_STATE = 42
TSNE_LEARNING_RATE = "auto"

# ============================================================================
# VISUALIZATION
# ============================================================================

# Plot settings
FIGURE_DPI = 300
FIGURE_FORMAT = "png"
PLOT_STYLE = "seaborn-v0_8-darkgrid"

# Color scheme
COLOR_REAL = "#3498db"  # Blue
COLOR_FAKE = "#e74c3c"  # Red
COLOR_PALETTE = "Set2"

# 3D scatter plot settings
SCATTER_POINT_SIZE = 50
SCATTER_ALPHA = 0.6

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

# Significance testing
ALPHA_LEVEL = 0.05  # For p-value threshold
CONFIDENCE_INTERVAL = 0.95

# Feature importance methods
FEATURE_IMPORTANCE_METHODS = ["t_test", "ks_test", "mutual_info"]

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

RANDOM_SEED = 42

# Set random seeds for reproducibility
import numpy as np
import random

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# PyTorch seeds (will be set when torch is imported)
try:
    import torch
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
except ImportError:
    pass

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
except ImportError:
    DEVICE = "cpu"
