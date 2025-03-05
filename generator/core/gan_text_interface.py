"""
GAN Text Interface: Small Language Model to 3D Model Generator Interface

This module creates an interface between a small language model (SLM) and the 
GAN-based 3D model generator. It allows users to describe 3D models in natural
language, which are then converted to parameters for the generator.
"""

import torch
import numpy as np
import logging
import json
import os
import re
import time
from pathlib import Path
from sklearn.manifold import TSNE
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..models.gan_models import Generator3D
import torch_geometric
from torch_geometric.data import Data
from tqdm import tqdm

logger = logging.getLogger(__name__)

class GANTextInterface:
    """Interface between language models and GAN-based 3D generators."""
    
    def __init__(self, 
                 llm_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                 gan_model_path=None,
                 gan_latent_dim=128,
                 gan_output_size=64,
                 device=None):
        """
        Initialize the interface between text and GAN-based 3D generation.
        
        Args:
            llm_model_name: Name of the language model to use (default: TinyLlama-1.1B-Chat)
            gan_model_path: Path to the pretrained GAN generator model
            gan_latent_dim: Latent dimension size for the GAN generator
            gan_output_size: Output size for the GAN generator (voxel resolution)
            device: PyTorch device to use (cuda/cpu)
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"GANTextInterface using device: {self.device}")
        
        # Initialize latent dimension and output size
        self.gan_latent_dim = gan_latent_dim
        self.gan_output_size = gan_output_size
        
        # Load language model
        logger.info(f"Loading language model: {llm_model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                llm_model_name, 
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
            ).to(self.device)
        except Exception as e:
            logger.error(f"Failed to load language model: {e}")
            raise
        
        # Load GAN generator
        logger.info("Initializing GAN generator")
        self.generator = Generator3D(latent_dim=gan_latent_dim, output_size=gan_output_size).to(self.device)
        
        # Load pretrained GAN model if provided
        if gan_model_path:
            logger.info(f"Loading pretrained GAN model from: {gan_model_path}")
            try:
                self.generator.load_state_dict(torch.load(gan_model_path, map_location=self.device))
            except Exception as e:
                logger.error(f"Failed to load GAN model: {e}")
                raise
        
        # Set to evaluation mode
        self.generator.eval()
        
        # Load or create attribute mappings
        self.attributes_file = Path(__file__).parent / "shape_attributes.json"
        if not self.attributes_file.exists():
            self._create_default_attributes_file()
        
        with open(self.attributes_file, 'r') as f:
            self.shape_attributes = json.load(f)
        
        logger.info("GANTextInterface initialized successfully")
    
    def generate_from_text(self, prompt, num_points=2048, detail_level=3):
        """
        Generate a 3D model from a text prompt.
        
        Args:
            prompt: Text description of the 3D model
            num_points: Number of points in the point cloud (if applicable)
            detail_level: Level of detail (1-5)
            
        Returns:
            Tuple of (voxel_grid, point_cloud, metadata)
        """
        start_time = time.time()
        logger.info(f"Generating 3D model from prompt: '{prompt}'")
        
        # Step 1: Process the text with the language model
        shape_info = self._process_text_with_llm(prompt)
        logger.info(f"Extracted shape information: {shape_info}")
        
        # Step 2: Generate latent vector for the GAN
        z_vector = self._generate_latent_vector(shape_info)
        
        # Step 3: Generate 3D model with the GAN
        voxel_grid = self._generate_3d_model(z_vector)
        
        # Step 4: Convert to point cloud if needed
        point_cloud = self._voxel_to_point_cloud(voxel_grid, num_points) if num_points > 0 else None
        
        # Create metadata about the generation
        generation_time = time.time() - start_time
        metadata = {
            "prompt": prompt,
            "shape_info": shape_info,
            "detail_level": detail_level,
            "voxel_resolution": voxel_grid.shape[0],
            "point_count": num_points if point_cloud is not None else 0,
            "generation_time": generation_time
        }
        
        logger.info(f"3D model generated in {generation_time:.2f} seconds")
        
        return voxel_grid, point_cloud, metadata
    
    def _process_text_with_llm(self, prompt):
        """
        Process the text prompt with the language model to extract shape information.
        
        Args:
            prompt: Text prompt
            
        Returns:
            Dictionary with shape information
        """
        # Create a structured prompt for the language model
        system_prompt = "You are a 3D model interpreter. Extract key features from the text to help generate a 3D model."
        instruction = f"""Extract the following information from this text description of a 3D object:

1. Basic shape (e.g., sphere, cube, cylinder, cone, torus)
2. Size (small, medium, large)
3. Main colors
4. Texture (smooth, rough, bumpy)
5. Any distinctive features or parts

Text description: {prompt}

Output as JSON with these fields only: "basic_shape", "size", "colors", "texture", "features"."""

        # Format prompt based on model
        if "TinyLlama" in self.tokenizer.name_or_path:
            formatted_prompt = f"<|system|>\n{system_prompt}\n{instruction}\n{prompt}"
        else:
            formatted_prompt = f"{system_prompt}\n{instruction}\n{prompt}"

        # Tokenize the prompt
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

        # Generate response from the language model
        response = self.llm_model.generate(**inputs, max_length=256)

        # Decode the response
        response_text = self.tokenizer.decode(response[0], skip_special_tokens=True)

        # Parse the response as JSON
        try:
            shape_info = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse response as JSON: {e}")
            raise

        return shape_info
