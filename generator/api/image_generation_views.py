# generator/api/image_generation_views.py

from django.http import JsonResponse
from rest_framework.decorators import api_view
from ..utils.image_generation import SDImageGenerator
import os
import uuid
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Initialize the image generator
sd_generator = None

def get_generator():
    global sd_generator
    if sd_generator is None:
        sd_generator = SDImageGenerator()
    return sd_generator

@api_view(['POST'])
def generate_multiview_images(request):
    """Generate multiple views of an object based on a text prompt"""
    try:
        # Get parameters from request
        prompt = request.data.get('prompt')
        num_views = int(request.data.get('num_views', 8))
        seed = request.data.get('seed')
        
        if seed is not None:
            seed = int(seed)
        
        if not prompt:
            return JsonResponse({"error": "Prompt is required"}, status=400)
        
        if num_views < 1 or num_views > 16:
            return JsonResponse({"error": "Number of views must be between 1 and 16"}, status=400)
            
        # Create a unique output directory
        session_id = request.session.session_key or str(uuid.uuid4())
        output_dir = Path(f"media/generated/multiview/{session_id}_{prompt[:20].replace(' ', '_')}")
        
        # Generate images
        generator = get_generator()
        images = generator.generate_multiview_images(
            prompt=prompt,
            num_views=num_views,
            output_dir=output_dir,
            seed=seed
        )
        
        # Prepare response with image paths
        image_paths = [str(output_dir / f"view_{i:03d}_{(i * (360 // num_views)):03d}deg.png") for i in range(num_views)]
        
        return JsonResponse({
            "status": "success",
            "prompt": prompt,
            "num_views": num_views,
            "seed": seed,
            "output_dir": str(output_dir),
            "image_paths": image_paths
        })
    
    except Exception as e:
        logger.exception("Error generating multiview images")
        return JsonResponse({"error": str(e)}, status=500)