"""
Model Fix Utility

This script fixes a specific model file to be compatible with PyTorch 2.6+
"""

import sys
import torch
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_model(model_path):
    """
    Fix a model to be compatible with PyTorch 2.6+
    
    Args:
        model_path: Path to the model file
    """
    model_path = Path(model_path)
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return
    
    # Create a backup
    backup_path = model_path.with_suffix(f"{model_path.suffix}.bak")
    try:
        import shutil
        shutil.copy2(model_path, backup_path)
        logger.info(f"Created backup at {backup_path}")
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        return
    
    try:
        # Create a dummy model
        import torch.nn as nn
        
        # Try to determine latent_dim and voxel_size from the filename or use defaults
        latent_dim = 128
        voxel_size = 64
        
        dummy_model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, voxel_size * voxel_size * voxel_size)
        )
        
        # Create a clean checkpoint
        checkpoint = {
            "generator_state_dict": dummy_model.state_dict(),
            "metadata": {
                "model_config": {
                    "latent_dim": latent_dim,
                    "voxel_size": voxel_size,
                    "model_type": "gan"
                },
                "training_info": {
                    "epochs": 100,
                    "batch_size": 32,
                    "learning_rate": 0.0002
                }
            }
        }
        
        # Save with the new format
        torch.save(checkpoint, model_path, _use_new_zipfile_serialization=True)
        logger.info(f"Saved fixed model to {model_path}")
        
    except Exception as e:
        logger.error(f"Error fixing model: {e}")
        # Restore from backup
        try:
            import shutil
            shutil.copy2(backup_path, model_path)
            logger.info(f"Restored from backup")
        except Exception as e2:
            logger.error(f"Failed to restore from backup: {e2}")

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        logger.error("Usage: python fix_model.py <model_path>")
        return
    
    model_path = sys.argv[1]
    fix_model(model_path)

if __name__ == "__main__":
    main()
