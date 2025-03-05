"""
Direct test script for the TextTo3DManager class

This script tests the TextTo3DManager class directly, bypassing the API layer
to validate core functionality.
"""

import os
import sys
import logging
import torch
import numpy as np
import trimesh
from pathlib import Path
from django.conf import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("text_to_3d_manager_tester")

# Setup Django settings if not already configured
if not settings.configured:
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ai_3d_generator.settings')
    import django
    django.setup()

from generator.text_to_3d_manager import TextTo3DManager

def test_manager(prompt="A simple blue cube", detail_level=3):
    """
    Test the TextTo3DManager directly
    
    Args:
        prompt: Text description of the 3D model to generate
        detail_level: Level of detail for the model (1-5)
    """
    logger.info(f"Testing TextTo3DManager with prompt: '{prompt}'")
    
    # Get model paths from settings
    from django.conf import settings
    gan_model_path = getattr(settings, 'GAN_MODEL_PATH', None)
    llm_model_name = getattr(settings, 'TEXT_TO_3D_LLM_MODEL', "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    logger.info(f"Using GAN model: {gan_model_path}")
    logger.info(f"Using LLM model: {llm_model_name}")
    
    # Create output directory
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize the TextTo3DManager
    try:
        logger.info("Initializing TextTo3DManager...")
        manager = TextTo3DManager(
            llm_model_name=llm_model_name,
            gan_model_path=gan_model_path,
            latent_dim=128,
            voxel_size=64
        )
        logger.info("TextTo3DManager initialized successfully")
        
        # Generate 3D model from text
        logger.info(f"Generating model from prompt: '{prompt}'")
        result = manager.generate_from_text(
            prompt=prompt,
            output_formats=["voxel", "mesh", "point_cloud"],
            detail_level=detail_level
        )
        
        # Log results
        logger.info("Model generated successfully")
        logger.info(f"Shape information: {result['metadata']['shape_info']}")
        logger.info(f"Generation time: {result['metadata']['generation_time']:.2f} seconds")
        
        # Save mesh to file
        if 'mesh' in result:
            mesh_path = output_dir / f"{prompt.replace(' ', '_')}.obj"
            logger.info(f"Saving mesh to {mesh_path}")
            result['mesh'].export(str(mesh_path))
            
        # Visualize voxel grid
        if 'voxel_grid' in result:
            logger.info("Voxel grid statistics:")
            voxel_grid = result['voxel_grid']
            logger.info(f"  Shape: {voxel_grid.shape}")
            logger.info(f"  Occupied voxels: {np.sum(voxel_grid > 0.5)}")
            logger.info(f"  Min value: {np.min(voxel_grid)}")
            logger.info(f"  Max value: {np.max(voxel_grid)}")
            
            # Save voxel visualization as mesh
            try:
                from skimage import measure
                logger.info("Converting voxels to visualization mesh...")
                vertices, faces, normals, _ = measure.marching_cubes(voxel_grid, level=0.5)
                voxel_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, normals=normals)
                voxel_path = output_dir / f"{prompt.replace(' ', '_')}_voxels.obj"
                voxel_mesh.export(str(voxel_path))
                logger.info(f"Voxel visualization saved to {voxel_path}")
            except Exception as e:
                logger.error(f"Error creating voxel visualization: {e}")
            
        return result
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the TextTo3DManager class directly")
    parser.add_argument("--prompt", type=str, default="A blue cube", 
                      help="Text description of the desired 3D model")
    parser.add_argument("--detail", type=int, default=3, choices=[1, 2, 3, 4, 5],
                      help="Detail level (1-5)")
    
    args = parser.parse_args()
    
    test_manager(args.prompt, args.detail)
