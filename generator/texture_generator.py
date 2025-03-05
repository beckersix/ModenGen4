"""
Texture generator module using Stable Diffusion for generating textures for 3D models.
"""

import torch
import numpy as np
from PIL import Image
import logging
import time
import os
from diffusers import StableDiffusionPipeline
# This import is not available in your version of diffusers
# from diffusers.utils import fix_glibc

logger = logging.getLogger(__name__)

class TextureGenerator:
    """Generate textures for 3D models using Stable Diffusion."""
    
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device=None):
        """
        Initialize the texture generator.
        
        Args:
            model_id: ID of the Stable Diffusion model to use
            device: PyTorch device to use (cuda/cpu)
        """
        # To avoid GLIBC version issues on some platforms
        # fix_glibc()
        
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        try:
            # Load Stable Diffusion pipeline
            logger.info(f"Loading Stable Diffusion model: {model_id}")
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                safety_checker=None,  # Disable safety checker for efficiency
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
            ).to(self.device)
            
            # Optimize for memory usage
            self.pipeline.enable_attention_slicing()
            if self.device.type == 'cuda':
                try:
                    # Try using xformers for even better performance
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("Using xformers for memory efficient attention")
                except Exception as e:
                    logger.warning(f"Could not enable xformers: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to load Stable Diffusion model: {e}")
            raise
            
        logger.info("TextureGenerator initialized successfully")
    
    def generate_texture(self, prompt, width=1024, height=1024, num_inference_steps=50, guidance_scale=7.5):
        """
        Generate a texture based on the prompt.
        
        Args:
            prompt: Text description for the texture
            width: Width of the texture
            height: Height of the texture
            num_inference_steps: Number of denoising steps for Stable Diffusion
            guidance_scale: How much the image generation follows the prompt (higher = more faithful)
            
        Returns:
            PIL.Image object containing the generated texture
        """
        start_time = time.time()
        
        # Enhance prompt for texture generation
        texture_prompt = self._enhance_prompt_for_texture(prompt)
        logger.info(f"Generating texture for prompt: '{texture_prompt}'")
        
        try:
            # Run Stable Diffusion
            with torch.no_grad():
                image = self.pipeline(
                    prompt=texture_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale
                ).images[0]
            
            generation_time = time.time() - start_time
            logger.info(f"Texture generation completed in {generation_time:.2f} seconds")
            
            return image, generation_time
            
        except Exception as e:
            logger.error(f"Failed to generate texture: {e}")
            # Return a default texture in case of error
            return self._generate_default_texture(width, height), 0.0
    
    def _enhance_prompt_for_texture(self, prompt):
        """Enhance a prompt to make it more suitable for texture generation."""
        # Add texture-specific keywords to improve quality
        texture_keywords = [
            "seamless texture", "detailed material", "high resolution",
            "physically based rendering", "surface material", "tileable"
        ]
        
        # Check if the prompt already contains texture-specific words
        contains_texture_words = any(keyword in prompt.lower() for keyword in 
                                 ["texture", "material", "surface", "seamless", "tileable"])
        
        # Add texture keywords if not already present
        if not contains_texture_words:
            enhanced_prompt = f"{prompt}, {texture_keywords[0]}, {texture_keywords[2]}"
        else:
            enhanced_prompt = prompt
            
        return enhanced_prompt
    
    def _generate_default_texture(self, width, height):
        """Generate a default checkerboard texture in case of errors."""
        # Create a checkerboard pattern
        checker_size = 16
        img = Image.new('RGB', (width, height), color='white')
        pixels = img.load()
        
        for i in range(width):
            for j in range(height):
                if ((i // checker_size) % 2) == ((j // checker_size) % 2):
                    pixels[i, j] = (200, 200, 200)
                else:
                    pixels[i, j] = (100, 100, 100)
        
        return img
    
    def generate_multiple_textures(self, prompt, count=4, width=512, height=512):
        """Generate multiple texture variations for the same prompt."""
        textures = []
        for i in range(count):
            # Add a slight variation to each prompt
            variation_prompt = f"{prompt}, variation {i+1}"
            texture, _ = self.generate_texture(variation_prompt, width, height)
            textures.append(texture)
        
        return textures
    
    def save_texture(self, texture, file_path):
        """Save texture to a file."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save image
        texture.save(file_path)
        logger.info(f"Texture saved to {file_path}")
        
        return file_path
