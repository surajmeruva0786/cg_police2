"""
Dimensionality Reduction Script

This script applies PCA and t-SNE dimensionality reduction as described in the paper:
"Comprehensive Analysis of Manual and CNN-based Feature Extraction for 
Deepfake Detection on the Celeb-DF Dataset"

PCA Configuration:
- n_components = 3

t-SNE Configuration:
- n_components = 3
- perplexity = 30
- n_iter = 1000
- init = 'pca'
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import joblib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_features(features_file: Path) -> tuple:
    """
    Load features from CSV file.
    
    Args:
        features_file: Path to features CSV
    
    Returns:
        Tuple of (features_array, labels, metadata_df)
    """
    df = pd.read_csv(features_file)
    
    # Separate features from metadata
    metadata_cols = ['video_id', 'video_name', 'label', 'num_frames']
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    
    features = df[feature_cols].values
    labels = df['label'].values if 'label' in df.columns else None
    metadata = df[metadata_cols] if all(c in df.columns for c in metadata_cols) else df[['video_id', 'label']]
    
    logger.info(f"Loaded {len(features)} samples with {features.shape[1]} features")
    
    return features, labels, metadata


def apply_pca(features: np.ndarray, n_components: int = 3) -> tuple:
    """
    Apply PCA dimensionality reduction.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        n_components: Number of components to keep
    
    Returns:
        Tuple of (reduced_features, pca_model, explained_variance_ratio)
    """
    logger.info(f"Applying PCA with {n_components} components")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    features_reduced = pca.fit_transform(features_scaled)
    
    explained_variance = pca.explained_variance_ratio_
    total_variance = explained_variance.sum()
    
    logger.info(f"Explained variance by component: {explained_variance}")
    logger.info(f"Total explained variance: {total_variance:.4f}")
    
    return features_reduced, pca, explained_variance


def apply_tsne(
    features: np.ndarray,
    n_components: int = 3,
    perplexity: int = 30,
    n_iter: int = 1000,
    init: str = 'pca'
) -> tuple:
    """
    Apply t-SNE dimensionality reduction.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        n_components: Number of components to keep
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations
        init: Initialization method ('pca' or 'random')
    
    Returns:
        Tuple of (reduced_features, tsne_model)
    """
    logger.info(f"Applying t-SNE with {n_components} components, perplexity={perplexity}")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply t-SNE
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        n_iter=n_iter,
        init=init,
        random_state=42,
        verbose=1
    )
    features_reduced = tsne.fit_transform(features_scaled)
    
    logger.info("t-SNE completed")
    
    return features_reduced, tsne


def save_reduced_features(
    features_reduced: np.ndarray,
    metadata: pd.DataFrame,
    output_file: Path,
    method: str
):
    """
    Save reduced features to CSV.
    
    Args:
        features_reduced: Reduced feature matrix
        metadata: Metadata DataFrame
        output_file: Path to save CSV
        method: 'pca' or 'tsne'
    """
    # Create DataFrame
    component_cols = [f'{method}_component_{i+1}' for i in range(features_reduced.shape[1])]
    df_reduced = pd.DataFrame(features_reduced, columns=component_cols)
    
    # Add metadata
    df_combined = pd.concat([metadata.reset_index(drop=True), df_reduced], axis=1)
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(output_file, index=False)
    
    logger.info(f"Saved reduced features to {output_file}")


def process_feature_file(
    features_file: Path,
    output_dir: Path,
    feature_type: str
):
    """
    Process a single feature file with PCA and t-SNE.
    
    Args:
        features_file: Path to features CSV
        output_dir: Directory to save reduced features
        feature_type: Type of features (e.g., 'manual', 'resnet50')
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {feature_type} features")
    logger.info(f"Input: {features_file}")
    logger.info(f"{'='*60}\n")
    
    # Load features
    features, labels, metadata = load_features(features_file)
    
    # Apply PCA
    features_pca, pca_model, explained_variance = apply_pca(features, n_components=3)
    
    # Save PCA results
    pca_output = output_dir / f"{feature_type}_pca.csv"
    save_reduced_features(features_pca, metadata, pca_output, 'pca')
    
    # Save PCA model and explained variance
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca_model, model_dir / f"{feature_type}_pca_model.pkl")
    np.save(model_dir / f"{feature_type}_pca_explained_variance.npy", explained_variance)
    
    # Apply t-SNE
    features_tsne, tsne_model = apply_tsne(
        features,
        n_components=3,
        perplexity=30,
        n_iter=1000,
        init='pca'
    )
    
    # Save t-SNE results
    tsne_output = output_dir / f"{feature_type}_tsne.csv"
    save_reduced_features(features_tsne, metadata, tsne_output, 'tsne')
    
    logger.info(f"Completed {feature_type} dimensionality reduction\n")


def main():
    """Main execution function"""
    
    # Define paths
    features_dir = Path("results/features")
    output_dir = Path("results/reduced")
    
    # Feature files to process
    feature_files = {
        'manual': features_dir / "manual_features_all.csv",
        'resnet50': features_dir / "resnet50_features_all.csv",
        'resnet152': features_dir / "resnet152_features_all.csv",
        'resnext101': features_dir / "resnext101_features_all.csv",
        'vit_b_16': features_dir / "vit_b_16_features_all.csv"
    }
    
    logger.info("Starting dimensionality reduction")
    logger.info(f"Output directory: {output_dir}")
    
    # Process each feature file
    for feature_type, feature_file in feature_files.items():
        if feature_file.exists():
            process_feature_file(feature_file, output_dir, feature_type)
        else:
            logger.warning(f"Feature file not found: {feature_file}")
    
    logger.info("\n" + "="*60)
    logger.info("Dimensionality reduction completed!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
