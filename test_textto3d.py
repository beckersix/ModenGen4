"""
Test script for the TextTo3D manager to verify the model loading adaptation works.
"""

import os
import sys
import torch
import logging
import numpy as np
from pathlib import Path
from generator.text_to_3d_manager import TextTo3DManager

# Set the OpenMP environment variable to suppress warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def main():
    print("Starting TextTo3D test script...")
    
    # Find a GAN model file if it exists
    model_dir = Path("models")
    if not model_dir.exists():
        model_dir = Path("generator/models")
    
    # Look for .pth or .pt files
    model_files = list(model_dir.glob("*.pth")) + list(model_dir.glob("*.pt"))
    
    gan_model_path = None
    if model_files:
        gan_model_path = str(model_files[0])
        print(f"Found model file: {gan_model_path}")
    else:
        print("No model file found. Will use untrained generator.")
    
    # Initialize the TextTo3D manager with debug mode
    print("Initializing TextTo3D manager...")
    manager = TextTo3DManager(
        gan_model_path=gan_model_path,
        debug_mode=True
    )
    
    # Test with a simple prompt
    prompt = "A simple cube with rounded edges"
    print(f"Generating 3D model from prompt: '{prompt}'")
    
    try:
        print("Starting generation process...")
        result = manager.generate_from_text(prompt, debug=True)
        print(f"Generation complete!")
        print(f"Result keys: {list(result.keys())}")
        
        # Check if we have a mesh file
        if "mesh_file" in result and os.path.exists(result["mesh_file"]):
            print(f"Mesh file created at: {result['mesh_file']}")
        else:
            print("No mesh file was created.")
        
        # Check if we have a voxel array
        if "voxel_grid" in result:
            voxel_grid = result["voxel_grid"]
            print(f"Voxel grid shape: {voxel_grid.shape}")
            print(f"Voxel grid stats: min={voxel_grid.min()}, max={voxel_grid.max()}, mean={voxel_grid.mean()}")
            
            # Check if any debug visualizations were saved
            debug_dir = Path("debug_output")
            if debug_dir.exists():
                vis_files = list(debug_dir.glob("*.png"))
                if vis_files:
                    print("Debug visualizations were saved:")
                    for vis_file in vis_files:
                        print(f"  - {vis_file}")
        else:
            print("No voxel grid in result.")
        
        # Check if there was an error
        if "error" in result:
            print(f"Error occurred during generation: {result['error']}")
        
        # Print the complete result for debugging
        print("\nComplete result dictionary:")
        import pprint
        pprint.pprint(result)
    
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
    
    print("Test script completed.")

if __name__ == "__main__":
    main()
