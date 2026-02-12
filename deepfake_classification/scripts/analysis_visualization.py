"""
Analysis and Visualization Script

This script generates all plots and tables from the paper:
"Comprehensive Analysis of Manual and CNN-based Feature Extraction for 
Deepfake Detection on the Celeb-DF Dataset"

Visualizations:
- 3D PCA scatter plots (Real vs Fake)
- 3D t-SNE scatter plots (Real vs Fake)
- Feature comparison tables
- Feature importance visualizations
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_3d_scatter(
    df: pd.DataFrame,
    component_prefix: str,
    title: str,
    output_file: Path
):
    """
    Create 3D scatter plot for dimensionality-reduced features.
    
    Args:
        df: DataFrame with reduced features and labels
        component_prefix: Prefix for component columns ('pca' or 'tsne')
        title: Plot title
        output_file: Path to save plot
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Separate real and fake
    df_real = df[df['label'] == 'real']
    df_fake = df[df['label'] == 'fake']
    
    # Plot real samples
    ax.scatter(
        df_real[f'{component_prefix}_component_1'],
        df_real[f'{component_prefix}_component_2'],
        df_real[f'{component_prefix}_component_3'],
        c='blue',
        marker='o',
        label='Real',
        alpha=0.6,
        s=50
    )
    
    # Plot fake samples
    ax.scatter(
        df_fake[f'{component_prefix}_component_1'],
        df_fake[f'{component_prefix}_component_2'],
        df_fake[f'{component_prefix}_component_3'],
        c='red',
        marker='^',
        label='Fake',
        alpha=0.6,
        s=50
    )
    
    # Labels and title
    ax.set_xlabel(f'{component_prefix.upper()} Component 1', fontsize=12)
    ax.set_ylabel(f'{component_prefix.upper()} Component 2', fontsize=12)
    ax.set_zlabel(f'{component_prefix.upper()} Component 3', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved 3D scatter plot to {output_file}")


def create_comparison_table(
    features_df: pd.DataFrame,
    output_file: Path
):
    """
    Create comparison table of feature statistics for Real vs Fake.
    
    Args:
        features_df: DataFrame with features and labels
        output_file: Path to save table
    """
    # Separate real and fake
    df_real = features_df[features_df['label'] == 'real']
    df_fake = features_df[features_df['label'] == 'fake']
    
    # Get feature columns (exclude metadata)
    metadata_cols = ['video_id', 'video_name', 'label', 'num_frames']
    feature_cols = [c for c in features_df.columns if c not in metadata_cols]
    
    # Calculate statistics
    comparison_data = []
    
    for col in feature_cols:
        comparison_data.append({
            'Feature': col,
            'Real_Mean': df_real[col].mean(),
            'Real_Std': df_real[col].std(),
            'Fake_Mean': df_fake[col].mean(),
            'Fake_Std': df_fake[col].std(),
            'Difference': abs(df_real[col].mean() - df_fake[col].mean())
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Sort by difference
    df_comparison = df_comparison.sort_values('Difference', ascending=False)
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_comparison.to_csv(output_file, index=False)
    
    logger.info(f"Saved comparison table to {output_file}")
    
    return df_comparison


def plot_feature_importance(
    comparison_df: pd.DataFrame,
    title: str,
    output_file: Path,
    top_n: int = 20
):
    """
    Plot feature importance based on mean difference between Real and Fake.
    
    Args:
        comparison_df: DataFrame with feature comparison statistics
        title: Plot title
        output_file: Path to save plot
        top_n: Number of top features to display
    """
    # Get top N features
    top_features = comparison_df.head(top_n)
    
    # Create bar plot
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_features)), top_features['Difference'], color='steelblue')
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Mean Absolute Difference (Real vs Fake)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved feature importance plot to {output_file}")


def plot_explained_variance(
    explained_variance: np.ndarray,
    feature_type: str,
    output_file: Path
):
    """
    Plot PCA explained variance.
    
    Args:
        explained_variance: Array of explained variance ratios
        feature_type: Type of features
        output_file: Path to save plot
    """
    plt.figure(figsize=(10, 6))
    
    components = range(1, len(explained_variance) + 1)
    cumulative_variance = np.cumsum(explained_variance)
    
    plt.bar(components, explained_variance, alpha=0.7, label='Individual')
    plt.plot(components, cumulative_variance, 'ro-', label='Cumulative')
    
    plt.xlabel('Principal Component', fontsize=12)
    plt.ylabel('Explained Variance Ratio', fontsize=12)
    plt.title(f'PCA Explained Variance - {feature_type}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved explained variance plot to {output_file}")


def generate_all_visualizations():
    """Generate all visualizations for the paper"""
    
    # Define paths
    reduced_dir = Path("results/reduced")
    features_dir = Path("results/features")
    viz_dir = Path("results/visualizations")
    tables_dir = Path("results/tables")
    models_dir = reduced_dir / "models"
    
    viz_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Feature types to process
    feature_types = ['manual', 'resnet50', 'resnet152', 'resnext101', 'vit_b_16']
    
    logger.info("Generating visualizations and tables")
    
    for feature_type in feature_types:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {feature_type} features")
        logger.info(f"{'='*60}\n")
        
        # Load reduced features
        pca_file = reduced_dir / f"{feature_type}_pca.csv"
        tsne_file = reduced_dir / f"{feature_type}_tsne.csv"
        features_file = features_dir / f"{feature_type}_features_all.csv"
        
        if not pca_file.exists() or not tsne_file.exists():
            logger.warning(f"Reduced features not found for {feature_type}")
            continue
        
        # Load data
        df_pca = pd.read_csv(pca_file)
        df_tsne = pd.read_csv(tsne_file)
        
        # Generate PCA 3D scatter plot
        plot_3d_scatter(
            df_pca,
            'pca',
            f'PCA 3D Visualization - {feature_type.upper()}',
            viz_dir / f"{feature_type}_pca_3d.png"
        )
        
        # Generate t-SNE 3D scatter plot
        plot_3d_scatter(
            df_tsne,
            'tsne',
            f't-SNE 3D Visualization - {feature_type.upper()}',
            viz_dir / f"{feature_type}_tsne_3d.png"
        )
        
        # Generate comparison table and feature importance
        if features_file.exists():
            df_features = pd.read_csv(features_file)
            
            comparison_df = create_comparison_table(
                df_features,
                tables_dir / f"{feature_type}_comparison.csv"
            )
            
            plot_feature_importance(
                comparison_df,
                f'Top 20 Discriminative Features - {feature_type.upper()}',
                viz_dir / f"{feature_type}_feature_importance.png"
            )
        
        # Plot PCA explained variance
        variance_file = models_dir / f"{feature_type}_pca_explained_variance.npy"
        if variance_file.exists():
            explained_variance = np.load(variance_file)
            plot_explained_variance(
                explained_variance,
                feature_type.upper(),
                viz_dir / f"{feature_type}_pca_variance.png"
            )
    
    logger.info("\n" + "="*60)
    logger.info("All visualizations generated!")
    logger.info(f"Visualizations saved to: {viz_dir}")
    logger.info(f"Tables saved to: {tables_dir}")
    logger.info("="*60)


def main():
    """Main execution function"""
    generate_all_visualizations()


if __name__ == "__main__":
    main()
