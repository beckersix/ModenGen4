"""
Sample data for testing and training the text-to-3D pipeline.
"""

import os
import json
import logging

logger = logging.getLogger(__name__)

class SampleDataGenerator:
    """Generate sample data for testing and training the text-to-3D system."""
    
    def __init__(self):
        """Initialize the sample data generator."""
        # Basic 3D objects with modifiers for sample generation
        self.basic_objects = [
            'cube', 'sphere', 'cylinder', 'cone', 'torus', 'pyramid', 'plane'
        ]
        
        self.modifiers = [
            'large', 'small', 'wide', 'tall', 'thin', 'flat', 'round',
            'smooth', 'rough', 'sharp', 'stretched', 'compressed', 'twisted'
        ]
        
        self.materials = [
            'wooden', 'metal', 'plastic', 'glass', 'stone', 'ceramic',
            'concrete', 'marble', 'leather', 'rubber', 'fabric'
        ]
        
        self.colors = [
            'red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink',
            'black', 'white', 'gray', 'brown', 'teal'
        ]
        
        self.objects = [
            # Furniture
            'chair', 'table', 'desk', 'shelf', 'lamp', 'sofa', 'bed', 'stool',
            # Household items
            'bottle', 'cup', 'plate', 'bowl', 'vase', 'clock', 'camera', 'phone',
            # Vehicles
            'car', 'truck', 'bicycle', 'motorcycle', 'boat', 'airplane',
            # Buildings
            'house', 'tower', 'bridge', 'castle', 'skyscraper', 'temple',
            # Nature
            'tree', 'mountain', 'rock', 'flower', 'cloud', 'star', 'planet',
            # Animals
            'cat', 'dog', 'bird', 'fish', 'horse', 'elephant', 'penguin'
        ]
        
        # Complex prompts
        self.complex_prompts = [
            "A futuristic skyscraper with curved glass facades",
            "A medieval castle with tall stone towers and a moat",
            "An ancient temple with ornate pillars and detailed carvings",
            "A modern minimalist chair with clean lines and smooth surfaces",
            "A vintage car with rounded edges and chrome details",
            "A fantasy tree house built into a massive ancient oak",
            "A steampunk-inspired mechanical device with gears and pipes",
            "A spaceship with sleek aerodynamic design and engine thrusters",
            "A smartphone with rounded corners and a large touch screen",
            "A robot with articulated limbs and a cylindrical head",
            "A mountain landscape with jagged peaks and a winding river",
            "A tropical island with palm trees, sandy beaches, and clear water",
            "A cozy cottage with a thatched roof and a stone chimney",
            "A grand piano with a polished black surface and elegant curves",
            "A sailing ship with tall masts and billowing sails",
            "A dragon with scales, wings, and a long tail",
            "A chess set with intricately carved pieces",
            "A Victorian-style street lamp with ornate metalwork",
            "A telescope with a long cylindrical tube on a tripod stand",
            "A backpack with multiple compartments and straps"
        ]
    
    def generate_simple_prompts(self, count=100):
        """Generate simple object prompts with modifiers."""
        import random
        
        prompts = []
        for _ in range(count):
            # Randomly decide what elements to include
            use_modifier = random.random() > 0.3
            use_material = random.random() > 0.5
            use_color = random.random() > 0.5
            
            # Randomly select elements
            if random.random() > 0.7:
                obj = random.choice(self.basic_objects)
            else:
                obj = random.choice(self.objects)
                
            modifier = random.choice(self.modifiers) if use_modifier else None
            material = random.choice(self.materials) if use_material else None
            color = random.choice(self.colors) if use_color else None
            
            # Construct prompt
            prompt_parts = []
            if modifier:
                prompt_parts.append(modifier)
            if material:
                prompt_parts.append(material)
            if color:
                prompt_parts.append(color)
            prompt_parts.append(obj)
            
            prompt = " ".join(prompt_parts)
            prompts.append(prompt)
        
        return prompts
    
    def generate_test_dataset(self, simple_count=30, complex_count=10):
        """Generate a test dataset with a mix of simple and complex prompts."""
        import random
        
        # Generate simple prompts
        simple_prompts = self.generate_simple_prompts(simple_count)
        
        # Sample complex prompts
        selected_complex = random.sample(self.complex_prompts, min(complex_count, len(self.complex_prompts)))
        
        # Combine datasets
        dataset = simple_prompts + selected_complex
        random.shuffle(dataset)
        
        return dataset
    
    def save_dataset(self, dataset, file_path):
        """Save a dataset to a JSON file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        logger.info(f"Dataset with {len(dataset)} items saved to {file_path}")
        return file_path
    
    def load_dataset(self, file_path):
        """Load a dataset from a JSON file."""
        try:
            with open(file_path, 'r') as f:
                dataset = json.load(f)
            logger.info(f"Loaded dataset with {len(dataset)} items from {file_path}")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load dataset from {file_path}: {e}")
            raise
