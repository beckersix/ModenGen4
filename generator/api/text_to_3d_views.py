"""
Text-to-3D API Views

This module provides REST API views for text-to-3D model generation,
integrating the text-to-3D functionality with the Django web application.
"""

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status

import os
import json
import logging
import threading
import uuid
from pathlib import Path
import numpy as np
import trimesh

# Import Django models directly from django_models.py
from ..models.django_models import GeneratedModel, TrainedModel
from ..core.text_to_3d_manager import TextTo3DManager
from .serializers import TrainedModelSerializer

logger = logging.getLogger(__name__)

# Dictionary to cache Text-to-3D managers for different models
text_to_3d_managers = {}

def get_text_to_3d_manager(trained_model_id=None):
    """
    Get or initialize the Text-to-3D manager.
    
    Args:
        trained_model_id: Optional UUID of a trained model to use
    
    Returns:
        TextTo3DManager: The initialized manager
    """
    global text_to_3d_managers
    
    # Create a cache key
    cache_key = str(trained_model_id) if trained_model_id else 'default'
    
    # If we have a cached manager for this model, return it
    if cache_key in text_to_3d_managers:
        return text_to_3d_managers[cache_key]
    
    # Get model paths from settings
    gan_model_path = getattr(settings, 'GAN_MODEL_PATH', None)
    llm_model_name = getattr(settings, 'TEXT_TO_3D_LLM_MODEL', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    
    # If a specific trained model is requested, use its parameters
    if trained_model_id:
        try:
            trained_model = TrainedModel.objects.get(id=trained_model_id)
            if trained_model.status == 'completed' and trained_model.model_file:
                gan_model_path = os.path.join(settings.MEDIA_ROOT, trained_model.model_file.name)
                latent_dim = trained_model.latent_dim
                voxel_size = trained_model.voxel_size
                logger.info(f"Using trained model: {trained_model.name}")
            else:
                logger.warning(f"Requested trained model {trained_model_id} is not available, using default")
        except TrainedModel.DoesNotExist:
            logger.warning(f"Trained model {trained_model_id} not found, using default")
    else:
        # Try to find the default model
        try:
            default_model = TrainedModel.objects.filter(status='completed', is_default=True).first()
            if default_model and default_model.model_file:
                gan_model_path = os.path.join(settings.MEDIA_ROOT, default_model.model_file.name)
                latent_dim = default_model.latent_dim
                voxel_size = default_model.voxel_size
                logger.info(f"Using default trained model: {default_model.name}")
            else:
                latent_dim = 128
                voxel_size = 64
        except Exception as e:
            logger.warning(f"Could not load default model: {e}")
            latent_dim = 128
            voxel_size = 64
    
    try:
        text_to_3d_managers[cache_key] = TextTo3DManager(
            llm_model_name=llm_model_name,
            gan_model_path=gan_model_path,
            latent_dim=latent_dim if 'latent_dim' in locals() else 128,
            voxel_size=voxel_size if 'voxel_size' in locals() else 64,
            trained_model_id=trained_model_id
        )
        logger.info(f"Text-to-3D manager initialized successfully for {cache_key}")
    except Exception as e:
        logger.error(f"Failed to initialize Text-to-3D manager: {e}")
        raise
    
    return text_to_3d_managers[cache_key]

@api_view(['POST'])
def generate_model_from_text(request):
    """
    API view to generate a 3D model from a text prompt.
    
    Request format:
    {
        "prompt": "A small blue cube with rounded corners",
        "detail_level": 3,
        "output_formats": ["mesh", "point_cloud"],
        "trained_model_id": "optional-uuid-of-trained-model"
    }
    
    Returns:
        JSON response with the model ID and status
    """
    # Validate request data
    if not request.data or 'prompt' not in request.data:
        return Response(
            {"error": "Missing required parameter: prompt"}, 
            status=status.HTTP_400_BAD_REQUEST
        )
    
    prompt = request.data['prompt']
    detail_level = int(request.data.get('detail_level', 3))
    output_formats = request.data.get('output_formats', ['mesh', 'point_cloud'])
    trained_model_id = request.data.get('trained_model_id')
    
    # Create a new model entry
    model = GeneratedModel(
        prompt=prompt,
        status='pending',
        detail_level=detail_level
    )
    model.save()
    
    # Start generation in a background thread
    threading.Thread(
        target=_generate_model_async,
        args=(model.id, prompt, detail_level, output_formats, trained_model_id)
    ).start()
    
    return Response({
        "id": str(model.id),
        "status": "pending",
        "message": "Model generation started"
    })

def _generate_model_async(model_id, prompt, detail_level, output_formats, trained_model_id=None):
    """
    Generate the 3D model asynchronously.
    
    Args:
        model_id: ID of the model in the database
        prompt: Text prompt for generation
        detail_level: Level of detail (1-5)
        output_formats: List of output formats
        trained_model_id: Optional ID of a trained model to use
    """
    try:
        # Get the model from the database
        model = GeneratedModel.objects.get(id=model_id)
        
        # Update status to processing
        model.status = 'processing'
        model.save()
        
        # Get the Text-to-3D manager
        manager = get_text_to_3d_manager(trained_model_id)
        
        # Generate the model
        results = manager.generate_from_text(
            prompt=prompt,
            output_formats=output_formats,
            detail_level=detail_level
        )
        
        # Save generated files
        base_filename = model.filename()
        media_dir = Path(settings.MEDIA_ROOT)
        
        # Save mesh if generated
        if 'mesh' in results:
            mesh_dir = media_dir / 'models' / 'meshes'
            mesh_dir.mkdir(parents=True, exist_ok=True)
            mesh_path = mesh_dir / f"{base_filename}.obj"
            
            # Check if mesh is a dictionary (new format) or an object (old format)
            if isinstance(results['mesh'], dict):
                # Convert dictionary to Trimesh object for export
                vertices = np.array(results['mesh']['vertices'])
                faces = np.array(results['mesh']['faces'])
                mesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces)
                mesh_obj.export(str(mesh_path), file_type='obj')
                
                # Save mesh statistics from the dictionary
                model.vertex_count = len(results['mesh']['vertices'])
                model.face_count = len(results['mesh']['faces'])
            else:
                # Original format: mesh is an object with attributes
                results['mesh'].export(str(mesh_path), file_type='obj')
                model.vertex_count = len(results['mesh'].vertices)
                model.face_count = len(results['mesh'].faces)
            
            model.mesh_file = f"models/meshes/{base_filename}.obj"
        
        # Save point cloud if generated
        if 'point_cloud' in results:
            pc_dir = media_dir / 'models' / 'point_clouds'
            pc_dir.mkdir(parents=True, exist_ok=True)
            pc_path = pc_dir / f"{base_filename}.ply"
            
            # Convert to trimesh point cloud and save
            points = results['point_cloud'].pos.numpy()
            pc_trimesh = trimesh.PointCloud(points)
            pc_trimesh.export(str(pc_path), file_type='ply')
            
            model.point_cloud_file = f"models/point_clouds/{base_filename}.ply"
        
        # Update model metadata
        model.generation_time = results['metadata']['generation_time']
        model.status = 'completed'
        model.save()
        
        logger.info(f"Successfully generated model {model_id} from text prompt")
        
    except Exception as e:
        logger.error(f"Error generating model {model_id}: {e}")
        try:
            model = GeneratedModel.objects.get(id=model_id)
            model.status = 'failed'
            model.save()
        except:
            pass

@api_view(['GET'])
def model_generation_status(request, model_id):
    """
    Get the status of a model generation.
    
    Args:
        model_id: ID of the model
        
    Returns:
        JSON response with the model status
    """
    try:
        model = GeneratedModel.objects.get(id=model_id)
        
        response_data = {
            "id": str(model.id),
            "status": model.status,
            "created_at": model.created_at,
            "updated_at": model.updated_at,
        }
        
        # Add file URLs if available
        if model.status == 'completed':
            if model.mesh_file:
                response_data["mesh_file"] = request.build_absolute_uri(model.mesh_file.url)
                response_data["mesh_url"] = request.build_absolute_uri(model.mesh_file.url)  # Keep for backward compatibility
            
            if model.point_cloud_file:
                response_data["point_cloud_file"] = request.build_absolute_uri(model.point_cloud_file.url)
                response_data["point_cloud_url"] = request.build_absolute_uri(model.point_cloud_file.url)  # Keep for backward compatibility
            
            # Add model stats
            response_data.update({
                "vertex_count": model.vertex_count,
                "face_count": model.face_count,
                "generation_time": model.generation_time
            })
        
        return Response(response_data)
    
    except GeneratedModel.DoesNotExist:
        return Response(
            {"error": f"Model with ID {model_id} not found"}, 
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        return Response(
            {"error": str(e)}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
def list_available_models(request):
    """
    List all available trained models for generation.
    
    Returns:
        JSON response with the list of available models
    """
    try:
        models = TrainedModel.objects.filter(status='completed').order_by('-is_default', '-created_at')
        serializer = TrainedModelSerializer(models, many=True)
        return Response(serializer.data)
    except Exception as e:
        return Response(
            {"error": str(e)}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
def get_model_details(request, model_id):
    """
    API view to get details of a generated 3D model.
    
    Args:
        request: HTTP request object
        model_id: ID of the model to retrieve
        
    Returns:
        JSON response with the model details
    """
    try:
        model = GeneratedModel.objects.get(id=model_id)
        
        # Prepare response data
        response_data = {
            'id': str(model.id),
            'prompt': model.prompt,
            'status': model.status,
            'created_at': model.created_at.isoformat(),
            'updated_at': model.updated_at.isoformat(),
            'detail_level': model.detail_level
        }
        
        # Add file URLs if available
        if model.status == 'completed':
            if model.mesh_file:
                response_data['mesh_url'] = request.build_absolute_uri(model.mesh_file.url)
            
            if model.point_cloud_file:
                response_data['point_cloud_url'] = request.build_absolute_uri(model.point_cloud_file.url)
            
            # Add model stats
            response_data.update({
                'vertex_count': model.vertex_count,
                'face_count': model.face_count,
                'generation_time': model.generation_time
            })
        
        return Response(response_data)
    
    except GeneratedModel.DoesNotExist:
        return Response(
            {"error": f"Model with ID {model_id} not found"}, 
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        logger.error(f"Error retrieving model details: {e}")
        return Response(
            {"error": f"Error retrieving model details: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

from django.shortcuts import render, get_object_or_404

def model_detail(request, model_id):
    """
    View for displaying a single generated model's details.
    
    Args:
        request: HTTP request object
        model_id: ID of the model to display
        
    Returns:
        Rendered HTML page with model details
    """
    model = get_object_or_404(GeneratedModel, id=model_id)
    
    context = {
        'model': model,
        'model_id': str(model.id),
        'prompt': model.prompt,
        'status': model.status,
        'created_at': model.created_at,
        'updated_at': model.updated_at,
        'page_title': f'3D Model: {model.prompt}'
    }
    
    # Add file URLs if available
    if model.status == 'completed':
        if model.mesh_file:
            context['mesh_url'] = model.mesh_file.url
        
        if model.point_cloud_file:
            context['point_cloud_url'] = model.point_cloud_file.url
        
        # Add model stats
        context.update({
            'vertex_count': model.vertex_count,
            'face_count': model.face_count,
            'generation_time': model.generation_time
        })
    
    return render(request, 'generator/model_viewer.html', context)
