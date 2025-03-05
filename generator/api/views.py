from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.generic import ListView, DetailView, TemplateView
from rest_framework.decorators import api_view, action
from rest_framework.response import Response
from rest_framework import status, viewsets
import json
import os
import uuid
import numpy as np

from ..models.django_models import GeneratedModel, GenerationSetting
from .serializers import GeneratedModelSerializer, GenerationSettingSerializer
import logging
import threading
import time
from ..visualization.point_cloud_generator import PointCloudGenerator
from ..utils.mesh_processor import MeshProcessor
from ..texture_generator import TextureGenerator
import torch

logger = logging.getLogger(__name__)

# Web views
class HomeView(TemplateView):
    """Home page view."""
    template_name = 'generator/home.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['recent_models'] = GeneratedModel.objects.filter(status='completed').order_by('-created_at')[:5]
        return context

class ModelListView(ListView):
    """View for listing all generated models."""
    model = GeneratedModel
    template_name = 'generator/model_list.html'
    context_object_name = 'models'
    paginate_by = 12
    
    def get_queryset(self):
        return GeneratedModel.objects.all().order_by('-created_at')

class ModelDetailView(DetailView):
    """View for viewing a specific generated model."""
    model = GeneratedModel
    template_name = 'generator/model_viewer.html'
    context_object_name = 'model'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Add necessary context for the template
        context['model_id'] = str(self.object.id)
        context['prompt'] = self.object.prompt
        context['status'] = self.object.status
        context['created_at'] = self.object.created_at
        context['updated_at'] = self.object.updated_at
        context['page_title'] = f'3D Model: {self.object.prompt}'
        
        # Add file URLs if available
        if self.object.status == 'completed':
            if self.object.mesh_file:
                context['mesh_url'] = self.object.mesh_file.url
            
            if self.object.point_cloud_file:
                context['point_cloud_url'] = self.object.point_cloud_file.url
            
            # Add model stats
            context.update({
                'vertex_count': self.object.vertex_count,
                'face_count': self.object.face_count,
                'generation_time': self.object.generation_time
            })
            
        return context

class TextTo3DView(TemplateView):
    """View for the text-to-3D generation interface."""
    template_name = 'generator/text_to_3d.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['recent_text_models'] = GeneratedModel.objects.filter(
            status='completed', 
            prompt__isnull=False
        ).order_by('-created_at')[:5]
        return context
    
class TrainView(TemplateView):
    """View for the model training interface."""
    template_name = 'generator/train.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Add any context data needed for the training page
        return context

# API views
@api_view(['POST'])
def generate_model(request):
    """API endpoint for generating a new 3D model from text."""
    if request.method == 'POST':
        serializer = GeneratedModelSerializer(data=request.data)
        if serializer.is_valid():
            model = serializer.save()
            # Start asynchronous generation process
            thread = threading.Thread(target=generate_model_async, args=(model.id,))
            thread.daemon = True
            thread.start()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

def generate_model_async(model_id):
    """Asynchronous function to generate a 3D model."""
    # Get the model object
    try:
        model = GeneratedModel.objects.get(id=model_id)
    except GeneratedModel.DoesNotExist:
        logger.error(f"Model with id {model_id} does not exist")
        return
    
    # Update status
    model.status = 'processing'
    model.save()
    
    try:
        start_time = time.time()
        
        # Use the TextTo3DManager instead of the older generation pipeline
        from .text_to_3d_views import get_text_to_3d_manager
        
        # Get the Text-to-3D manager with default model
        manager = get_text_to_3d_manager()
        
        # Generate the model using our improved text-to-3d pipeline
        output_formats = ['mesh']  # Default format is mesh only
        results = manager.generate_from_text(
            prompt=model.prompt,
            output_formats=output_formats,
            detail_level=model.detail_level
        )
        
        # Save generated files
        base_filename = f"{model.prompt.lower().replace(' ', '_')}_{model.id}"
        media_dir = os.path.join(settings.MEDIA_ROOT)
        
        # Save mesh if generated
        if 'mesh' in results:
            mesh_dir = os.path.join(media_dir, 'models', 'meshes')
            os.makedirs(mesh_dir, exist_ok=True)
            mesh_path = os.path.join(mesh_dir, f"{base_filename}.obj")
            
            # Check if mesh is a dictionary (new format) or an object (old format)
            if isinstance(results['mesh'], dict):
                # Convert dictionary to Trimesh object for export
                vertices = np.array(results['mesh']['vertices'])
                faces = np.array(results['mesh']['faces'])
                mesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces)
                mesh_obj.export(mesh_path, file_type='obj')
                
                # Save mesh statistics from the dictionary
                model.vertex_count = len(results['mesh']['vertices'])
                model.face_count = len(results['mesh']['faces'])
            else:
                # Original format: mesh is an object with attributes
                results['mesh'].export(mesh_path, file_type='obj')
                model.vertex_count = len(results['mesh'].vertices)
                model.face_count = len(results['mesh'].faces)
            
            model.mesh_file = f"models/meshes/{base_filename}.obj"
        
        # Save point cloud if generated
        if 'point_cloud' in results:
            pc_dir = os.path.join(media_dir, 'models', 'point_clouds')
            os.makedirs(pc_dir, exist_ok=True)
            pc_path = os.path.join(pc_dir, f"{base_filename}.ply")
            
            # Convert to trimesh point cloud and save
            points = results['point_cloud'].pos.numpy()
            pc_trimesh = trimesh.PointCloud(points)
            pc_trimesh.export(pc_path, file_type='ply')
            
            model.point_cloud_file = f"models/point_clouds/{base_filename}.ply"

        # Update model with completion info
        model.status = 'completed'
        model.generation_time = time.time() - start_time
        model.save()
        
        logger.info(f"Model generation completed for {model_id} in {model.generation_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error generating model {model_id}: {str(e)}")
        logger.exception(e)
        
        # Update model with failure info
        try:
            model = GeneratedModel.objects.get(id=model_id)
            model.status = 'failed'
            model.save()
        except:
            pass

class GeneratedModelViewSet(viewsets.ModelViewSet):
    """API viewset for generated models."""
    queryset = GeneratedModel.objects.all().order_by('-created_at')
    serializer_class = GeneratedModelSerializer
    
    @action(detail=True, methods=['get'])
    def status(self, request, pk=None):
        """Get the status of a model generation."""
        model = self.get_object()
        return Response({
            'status': model.status,
            'created_at': model.created_at,
            'updated_at': model.updated_at,
            'generation_time': model.generation_time,
        })

class GenerationSettingViewSet(viewsets.ModelViewSet):
    """API viewset for generation settings."""
    queryset = GenerationSetting.objects.all()
    serializer_class = GenerationSettingSerializer
