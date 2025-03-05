"""
Model Conversion Utility

This script converts older model files to a format compatible with PyTorch 2.6+
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_model(model_path, output_path=None):
    """
    Convert a model file to a format compatible with PyTorch 2.6+
    
    Args:
        model_path: Path to the model file
        output_path: Path to save the converted model (if None, will use original name + _compatible)
    
    Returns:
        Path to the converted model
    """
    model_path = Path(model_path)
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return None
    
    # Default output path
    if output_path is None:
        output_path = model_path.parent / f"{model_path.stem}_compatible{model_path.suffix}"
    
    try:
        logger.info(f"Loading model from {model_path}...")
        # Force weights_only=False with a warning about the security implications
        logger.warning("Using weights_only=False which could allow arbitrary code execution")
        logger.warning("Only use this with models you trust!")
        
        # Load the model with weights_only=False to bypass the new default
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        logger.info(f"Successfully loaded model")
        
        # Extract state dictionaries and create a clean version
        if isinstance(checkpoint, dict):
            # Create a simplified checkpoint with just the essential data
            new_checkpoint = {}
            
            # Copy over state dictionaries
            for key in checkpoint:
                if 'state_dict' in key or key in ['generator', 'discriminator']:
                    logger.info(f"Preserving key: {key}")
                    new_checkpoint[key] = checkpoint[key]
            
            # Add metadata as a separate key
            new_checkpoint['metadata'] = {}
            
            # Copy over non-state dict data as metadata
            for key in checkpoint:
                if 'state_dict' not in key and key not in ['generator', 'discriminator']:
                    logger.info(f"Moving to metadata: {key}")
                    new_checkpoint['metadata'][key] = checkpoint[key]
            
            logger.info(f"Saving converted model to {output_path}...")
            # Save with torch.save and zipfile serialization for best compatibility
            torch.save(new_checkpoint, output_path, _use_new_zipfile_serialization=True)
            logger.info(f"Successfully saved converted model")
            return output_path
        else:
            logger.error(f"Unexpected checkpoint type: {type(checkpoint)}")
            return None
    except Exception as e:
        logger.error(f"Error converting model: {e}")
        return None

def find_models(directory):
    """Find all model files in a directory"""
    directory = Path(directory)
    model_files = []
    
    for ext in ['.pt', '.pth']:
        model_files.extend(list(directory.glob(f"**/*{ext}")))
    
    return model_files

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        logger.error("Usage: python convert_models.py <model_path_or_directory>")
        return
    
    path = Path(sys.argv[1])
    
    if path.is_file():
        # Convert a single model
        result = convert_model(path)
        if result:
            logger.info(f"Conversion complete: {result}")
        else:
            logger.error("Conversion failed")
    elif path.is_dir():
        # Convert all models in a directory
        models = find_models(path)
        logger.info(f"Found {len(models)} model files")
        
        success_count = 0
        for model_path in models:
            logger.info(f"Converting {model_path}...")
            result = convert_model(model_path)
            if result:
                success_count += 1
                logger.info(f"Converted: {result}")
            else:
                logger.error(f"Failed to convert: {model_path}")
        
        logger.info(f"Converted {success_count} of {len(models)} models")
    else:
        logger.error(f"Path not found: {path}")

if __name__ == "__main__":
    main()
