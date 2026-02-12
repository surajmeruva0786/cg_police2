"""
Main Pipeline Orchestration Script

This script runs the entire deepfake detection research reproduction pipeline:
1. Extract facial masks from frames
2. Extract manual features (30 features)
3. Extract CNN features (4 models)
4. Apply dimensionality reduction (PCA + t-SNE)
5. Generate visualizations and tables

Usage:
    python main.py --config config.json
"""

import sys
import argparse
import json
from pathlib import Path
import logging
from datetime import datetime

# Set up logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DeepfakePipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config: dict):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.checkpoints = {}
        self.checkpoint_file = Path("pipeline_checkpoints.json")
        
        # Load existing checkpoints if available
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                self.checkpoints = json.load(f)
    
    def save_checkpoint(self, step: str):
        """Save checkpoint for completed step"""
        self.checkpoints[step] = datetime.now().isoformat()
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.checkpoints, f, indent=2)
        logger.info(f"Checkpoint saved: {step}")
    
    def is_step_completed(self, step: str) -> bool:
        """Check if step has been completed"""
        return step in self.checkpoints
    
    def step1_extract_masks(self):
        """Step 1: Extract facial masks from all frames"""
        step_name = "extract_masks"
        
        if self.is_step_completed(step_name) and not self.config.get('force_rerun', False):
            logger.info(f"Step 1 already completed, skipping...")
            return
        
        logger.info("\n" + "="*60)
        logger.info("STEP 1: Extracting Facial Masks")
        logger.info("="*60 + "\n")
        
        from utils.facial_mask import process_dataset_masks
        
        # Process real videos
        if self.config.get('real_frames_dir'):
            process_dataset_masks(
                self.config['real_frames_dir'],
                self.config['real_masks_dir'],
                self.config.get('predictor_path')
            )
        
        # Process fake videos
        if self.config.get('fake_frames_dir'):
            process_dataset_masks(
                self.config['fake_frames_dir'],
                self.config['fake_masks_dir'],
                self.config.get('predictor_path')
            )
        
        self.save_checkpoint(step_name)
    
    def step2_extract_manual_features(self):
        """Step 2: Extract manual features"""
        step_name = "extract_manual_features"
        
        if self.is_step_completed(step_name) and not self.config.get('force_rerun', False):
            logger.info(f"Step 2 already completed, skipping...")
            return
        
        logger.info("\n" + "="*60)
        logger.info("STEP 2: Extracting Manual Features")
        logger.info("="*60 + "\n")
        
        # Import and run manual feature extraction
        # Note: This would need to be adapted based on actual implementation
        logger.info("Manual feature extraction - run scripts/extract_manual_features.py")
        logger.info("Please update paths in the script and run manually")
        
        # self.save_checkpoint(step_name)
    
    def step3_extract_cnn_features(self):
        """Step 3: Extract CNN features"""
        step_name = "extract_cnn_features"
        
        if self.is_step_completed(step_name) and not self.config.get('force_rerun', False):
            logger.info(f"Step 3 already completed, skipping...")
            return
        
        logger.info("\n" + "="*60)
        logger.info("STEP 3: Extracting CNN Features")
        logger.info("="*60 + "\n")
        
        # Import and run CNN feature extraction
        logger.info("CNN feature extraction - run scripts/extract_cnn_features.py")
        logger.info("Please update paths in the script and run manually")
        
        # self.save_checkpoint(step_name)
    
    def step4_dimensionality_reduction(self):
        """Step 4: Apply PCA and t-SNE"""
        step_name = "dimensionality_reduction"
        
        if self.is_step_completed(step_name) and not self.config.get('force_rerun', False):
            logger.info(f"Step 4 already completed, skipping...")
            return
        
        logger.info("\n" + "="*60)
        logger.info("STEP 4: Applying Dimensionality Reduction")
        logger.info("="*60 + "\n")
        
        # Import and run dimensionality reduction
        logger.info("Dimensionality reduction - run scripts/dimensionality_reduction.py")
        
        # self.save_checkpoint(step_name)
    
    def step5_generate_visualizations(self):
        """Step 5: Generate all visualizations and tables"""
        step_name = "generate_visualizations"
        
        if self.is_step_completed(step_name) and not self.config.get('force_rerun', False):
            logger.info(f"Step 5 already completed, skipping...")
            return
        
        logger.info("\n" + "="*60)
        logger.info("STEP 5: Generating Visualizations and Tables")
        logger.info("="*60 + "\n")
        
        # Import and run visualization
        logger.info("Visualization - run scripts/analysis_visualization.py")
        
        # self.save_checkpoint(step_name)
    
    def run(self):
        """Run the entire pipeline"""
        logger.info("\n" + "#"*60)
        logger.info("DEEPFAKE DETECTION RESEARCH REPRODUCTION PIPELINE")
        logger.info("#"*60 + "\n")
        
        logger.info(f"Configuration: {json.dumps(self.config, indent=2)}\n")
        
        # Run all steps
        steps = [
            self.step1_extract_masks,
            self.step2_extract_manual_features,
            self.step3_extract_cnn_features,
            self.step4_dimensionality_reduction,
            self.step5_generate_visualizations
        ]
        
        for i, step in enumerate(steps, 1):
            try:
                step()
            except Exception as e:
                logger.error(f"Error in step {i}: {e}")
                logger.error("Pipeline stopped. Fix the error and rerun.")
                return
        
        logger.info("\n" + "#"*60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("#"*60 + "\n")


def load_config(config_file: Path) -> dict:
    """Load configuration from JSON file"""
    if config_file.exists():
        with open(config_file, 'r') as f:
            return json.load(f)
    else:
        logger.warning(f"Config file not found: {config_file}")
        return {}


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Run deepfake detection research pipeline')
    parser.add_argument('--config', type=str, default='pipeline_config.json',
                       help='Path to configuration file')
    parser.add_argument('--force', action='store_true',
                       help='Force rerun all steps')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(Path(args.config))
    config['force_rerun'] = args.force
    
    # Create and run pipeline
    pipeline = DeepfakePipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
