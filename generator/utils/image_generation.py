# generator/utils/image_generation.py

import os
import torch
from diffusers import StableDiffusionPipeline
from pathlib import Path

class SDImageGenerator:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device=None):
        """Initialize the Stable Diffusion image generator"""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Stable Diffusion model on {self.device}...")
        
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.pipe = self.pipe.to(self.device)
        
        # Enable attention slicing for lower memory usage
        self.pipe.enable_attention_slicing()
        
    def generate_multiview_images(self, prompt, num_views=8, output_dir=None, seed=None):
        """Generate multiple views of the same object for 3D reconstruction"""
        # Create output directory if provided
        if output_dir:
            output_dir = Path(output_dir)
            os.makedirs(output_dir, exist_ok=True)
        
        # Set random seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Create prompts for different view angles
        view_prompts = []
        for i in range(num_views):
            angle = i * (360 // num_views)
            view_prompt = f"{prompt}, from {angle} degree angle view, white background, centered, product photography"
            view_prompts.append((view_prompt, angle))
        
        # Generate images
        images = []
        for i, (view_prompt, angle) in enumerate(view_prompts):
            print(f"Generating view {i+1}/{num_views}: {angle} degrees")
            
            image = self.pipe(
                view_prompt, 
                generator=generator,
                num_inference_steps=50,
                guidance_scale=7.5
            ).images[0]
            
            # Save image if output directory is provided
            if output_dir:
                image_path = output_dir / f"view_{i:03d}_{angle:03d}deg.png"
                image.save(image_path)
                print(f"Saved image to {image_path}")
            
            images.append(image)
        
        return images

# Simple test function
def test_generator():
    generator = SDImageGenerator()
    prompt = "A red office chair with armrests"
    output_dir = Path("media/generated/test_multiview")
    
    images = generator.generate_multiview_images(
        prompt=prompt,
        num_views=4,  # Start with fewer views for testing
        output_dir=output_dir,
        seed=42  # Fixed seed for reproducibility
    )
    
    print(f"Generated {len(images)} images")
    
if __name__ == "__main__":
    test_generator()