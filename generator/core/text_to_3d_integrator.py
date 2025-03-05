"""
Text-to-3D Integrator

This module integrates a small language model (SLM) with the GAN-based 3D generator,
allowing users to describe 3D models in natural language that will then be
converted to parameters for the generator.
"""

import torch
import numpy as np
import logging
import json
import os
from pathlib import Path
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..models.gan_models import Generator3D
from django.conf import settings
from ..models.django_models import TrainedModel

logger = logging.getLogger(__name__)

class TextTo3DIntegrator:
    """
    Integrates a small language model with the GAN-based 3D generator.
    This class handles the conversion from text descriptions to 3D models.
    """
    
    def __init__(self, 
                 llm_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                 gan_model_path=None,
                 gan_latent_dim=128,
                 gan_output_size=64,
                 device=None,
                 trained_model_id=None):
        """
        Initialize the Text-to-3D integrator.
        
        Args:
            llm_model_name: Name of the language model to use (default: TinyLlama-1.1B-Chat)
            gan_model_path: Path to the pretrained GAN generator model
            gan_latent_dim: Latent dimension size for the GAN generator
            gan_output_size: Output size for the GAN generator
            device: PyTorch device to use (cuda/cpu)
            trained_model_id: UUID of a trained model to use from the database
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"TextTo3DIntegrator using device: {self.device}")
        
        # Check if we should use a trained model from the database
        if trained_model_id:
            logger.info(f"Loading trained model with ID: {trained_model_id}")
            try:
                trained_model = TrainedModel.objects.get(id=trained_model_id)
                if trained_model.status != 'completed':
                    logger.warning(f"Trained model {trained_model_id} is not in 'completed' status")
                    raise ValueError(f"Cannot use incomplete model: {trained_model.status}")
                
                if trained_model.model_file:
                    gan_model_path = os.path.join(settings.MEDIA_ROOT, trained_model.model_file.name)
                    gan_latent_dim = trained_model.latent_dim
                    gan_output_size = trained_model.voxel_size
                    logger.info(f"Using trained model parameters: latent_dim={gan_latent_dim}, output_size={gan_output_size}")
                else:
                    logger.warning(f"Trained model {trained_model_id} has no model file")
            except TrainedModel.DoesNotExist:
                logger.error(f"Trained model with ID {trained_model_id} not found")
                raise
        else:
            # Try to get the default model if no specific model was requested
            try:
                default_model = TrainedModel.objects.filter(status='completed', is_default=True).first()
                if default_model and default_model.model_file:
                    logger.info(f"Using default trained model: {default_model.name}")
                    gan_model_path = os.path.join(settings.MEDIA_ROOT, default_model.model_file.name)
                    gan_latent_dim = default_model.latent_dim
                    gan_output_size = default_model.voxel_size
            except Exception as e:
                logger.warning(f"Could not load default model: {e}")
        
        # Initialize latent dimension
        self.gan_latent_dim = gan_latent_dim
        
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
                self.generator.eval()
            except Exception as e:
                logger.error(f"Failed to load GAN model: {e}")
                raise
        
        # Load or create attribute mappings
        self.attributes_file = Path(__file__).parent / "shape_attributes.json"
        if not self.attributes_file.exists():
            self._create_default_attributes_file()
        
        with open(self.attributes_file, 'r') as f:
            self.shape_attributes = json.load(f)
        
        logger.info("TextTo3DIntegrator initialized successfully")
    
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
        metadata = {
            "prompt": prompt,
            "shape_info": shape_info,
            "detail_level": detail_level,
            "voxel_resolution": voxel_grid.shape[0],
            "point_count": num_points if point_cloud is not None else 0
        }
        
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

Output as JSON with these fields."""

        chat_format = f"<|system|>\n{system_prompt}\n<|assistant|>\n{instruction}\n<|end_of_text|>"
        
        # Tokenize the chat format
        inputs = self.tokenizer(chat_format, return_tensors="pt").to(self.device)
        
        # Generate response from the language model
        response = self.llm_model.generate(**inputs, max_new_tokens=256)
        
        # Convert response to text
        response_text = self.tokenizer.decode(response[0], skip_special_tokens=True)
        
        # Parse JSON response
        try:
            shape_info = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            raise
        
        return shape_info
