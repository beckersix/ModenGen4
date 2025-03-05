"""
Text-to-3D Interpreter Module

This module uses a small language model to interpret user text prompts and
convert them into a feature vector that can be used by the GAN generator
to create 3D models.
"""

import torch
import numpy as np
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
from pathlib import Path
import re

logger = logging.getLogger(__name__)

class TextInterpreter:
    """Interpret text prompts and convert them to feature vectors for the GAN generator."""
    
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device=None):
        """
        Initialize the text interpreter with a small language model.
        
        Args:
            model_name: Name of the language model to use
            device: PyTorch device to use (cuda/cpu)
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"TextInterpreter using device: {self.device}")
        
        # Load small language model
        logger.info(f"Loading language model: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
            ).to(self.device)
        except Exception as e:
            logger.error(f"Failed to load language model: {e}")
            raise
        
        # Load shape attributes mapping
        self.attributes_file = Path(__file__).parent / "shape_attributes.json"
        if not self.attributes_file.exists():
            self._create_default_attributes_file()
        
        with open(self.attributes_file, 'r') as f:
            self.shape_attributes = json.load(f)
        
        logger.info("TextInterpreter initialized successfully")
    
    def interpret_prompt(self, prompt, z_dim=128):
        """
        Interpret a text prompt and convert it to a feature vector for the GAN.
        
        Args:
            prompt: Text prompt describing the 3D model
            z_dim: Dimension of the latent vector for the GAN
            
        Returns:
            Dictionary containing interpretation results and feature vector
        """
        logger.info(f"Interpreting prompt: '{prompt}'")
        
        # Process prompt with language model
        shape_info = self._process_with_slm(prompt)
        
        # Convert shape info to feature vector
        z_vector = self._generate_feature_vector(shape_info, z_dim)
        
        # Add feature vector to shape info
        shape_info["z_vector"] = z_vector.tolist()
        
        return shape_info
    
    def _process_with_slm(self, prompt):
        """
        Process the prompt with the small language model to extract shape information.
        
        Args:
            prompt: Text prompt
            
        Returns:
            Dictionary with shape information
        """
        # Create a structured prompt for the language model
        system_prompt = "You are a 3D model interpreter. Extract key features from the text to help generate a 3D model."
        input_prompt = f"Extract the following information from this text description of a 3D object:\n\n\
1. Basic shape (e.g., sphere, cube, cylinder, etc.)\n\
2. Size (small, medium, large)\n\
3. Main colors\n\
4. Texture (smooth, rough, bumpy, etc.)\n\
5. Any distinctive features or parts\n\
\nText description: {prompt}\n\nOutput as JSON with these fields."
        
        full_prompt = f"<|system|>\n{system_prompt}\n{input_prompt}"
        
        # Generate response from language model
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract JSON data from response
        shape_info = self._extract_json_from_response(response)
        
        # If no valid JSON found, create a basic shape info dictionary
        if not shape_info:
            logger.warning("Could not extract valid JSON from language model response")
            shape_info = {
                "basic_shape": "cube",
                "size": "medium",
                "main_colors": ["gray"],
                "texture": "smooth",
                "distinctive_features": []
            }
        
        return shape_info
    
    def _extract_json_from_response(self, response):
        """Extract JSON data from the language model response."""
        # Try to find JSON in response using regex
        json_pattern = r'\{(?:[^{}]|"[^"]*"|\{(?:[^{}]|"[^"]*")*\})*\}'
        match = re.search(json_pattern, response)
        
        if match:
            try:
                data = json.loads(match.group(0))
                return data
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON: {e}")
        
        # Try manual parsing as a fallback
        try:
            lines = response.split('\n')
            json_lines = []
            in_json = False
            
            for line in lines:
                if line.strip().startswith('{'):
                    in_json = True
                
                if in_json:
                    json_lines.append(line)
                
                if line.strip().endswith('}'):
                    break
            
            if json_lines:
                json_text = ''.join(json_lines)
                return json.loads(json_text)
        except Exception as e:
            logger.warning(f"Manual JSON parsing failed: {e}")
        
        return None
    
    def _generate_feature_vector(self, shape_info, z_dim):
        """
        Convert shape information to a feature vector for the GAN.
        
        Args:
            shape_info: Dictionary with shape information
            z_dim: Dimension of the latent vector
            
        Returns:
            Numpy array with feature vector
        """
        # Initialize random feature vector
        np.random.seed(hash(str(shape_info)) % 2**32)
        z_vector = np.random.normal(0, 1, z_dim).astype(np.float32)
        
        # Modify vector based on shape attributes
        if 'basic_shape' in shape_info and shape_info['basic_shape'] in self.shape_attributes:
            shape_vector = np.array(self.shape_attributes[shape_info['basic_shape']])
            # Apply shape vector to a portion of the feature vector
            dim = min(len(shape_vector), z_dim // 4)
            z_vector[:dim] = shape_vector[:dim]
        
        # TODO: Modify vector further based on other attributes
        
        return torch.tensor(z_vector)
    
    def _create_default_attributes_file(self):
        """Create a default shape attributes file if none exists."""
        default_attributes = {
            "cube": [1.0, 0.0, 0.0, 0.0, 0.0],
            "sphere": [0.0, 1.0, 0.0, 0.0, 0.0],
            "cylinder": [0.0, 0.0, 1.0, 0.0, 0.0],
            "cone": [0.0, 0.0, 0.0, 1.0, 0.0],
            "torus": [0.0, 0.0, 0.0, 0.0, 1.0]
        }
        
        os.makedirs(os.path.dirname(self.attributes_file), exist_ok=True)
        with open(self.attributes_file, 'w') as f:
            json.dump(default_attributes, f, indent=2)
        
        logger.info(f"Created default shape attributes file at {self.attributes_file}")
