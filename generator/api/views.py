from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.generic import ListView, DetailView, TemplateView
from django.conf import settings
from rest_framework.decorators import api_view, action
from rest_framework.response import Response
from rest_framework import status, viewsets
import json
import os
import uuid
import numpy as np
from pathlib import Path
import logging
import threading
import time
from ..visualization.point_cloud_generator import PointCloudGenerator
from ..utils.mesh_processor import MeshProcessor
from ..texture_generator import TextureGenerator
import torch

from ..models.django_models import GeneratedModel, GenerationSetting
from .serializers import GeneratedModelSerializer, GenerationSettingSerializer
import logging
import threading
import time
from ..utils.dataset_manager import get_dataset_managers
from django.core.cache import cache

logger = logging.getLogger(__name__)

# Global dataset managers
MEDIA_ROOT = Path(settings.MEDIA_ROOT)
dataset_base_dir = MEDIA_ROOT / "datasets"
shapenet, objectnet3d, custom_datasets, combined_datasets = get_dataset_managers(dataset_base_dir)

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
        ).order_by('-created_at')
        return context
    
class TrainView(TemplateView):
    """View for the model training interface."""
    template_name = 'generator/train.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Add any context data needed for the training page
        return context

def datasets_view(request):
    """
    View for managing datasets
    """
    return render(request, 'generator/datasets.html')

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

# ShapeNet endpoints
@api_view(['GET'])
def get_shapenet_categories(request):
    """
    Get available ShapeNet categories with download status.
    """
    try:
        categories = shapenet.get_available_categories()
        return Response(categories)
    except Exception as e:
        logger.error(f"Error getting ShapeNet categories: {str(e)}")
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def download_shapenet_category(request):
    """
    Download a specific ShapeNet category.
    """
    category_id = request.data.get('category_id')
    if not category_id:
        return Response({"error": "Category ID is required"}, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        result = shapenet.download_category(category_id)
        return Response({"path": result, "success": True})
    except Exception as e:
        logger.error(f"Error downloading ShapeNet category: {str(e)}")
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def get_shapenet_download_status(request):
    """
    Get status of background ShapeNet downloads.
    """
    status_key = request.query_params.get('status_key')
    if not status_key:
        return Response({"error": "Status key is required"}, status=status.HTTP_400_BAD_REQUEST)
    
    status_data = cache.get(status_key, {})
    return Response(status_data)

@api_view(['POST'])
def download_all_shapenet(request):
    """
    Download all ShapeNet categories in background.
    """
    # Ensure the datasets directory exists
    import os
    from concurrent.futures import ThreadPoolExecutor
    import threading
    
    os.makedirs('media/datasets', exist_ok=True)
    
    try:
        # Import here to avoid circular imports
        from generator.utils.dataset_manager import get_dataset_managers
        shapenet_downloader, _, _ = get_dataset_managers('media/datasets')
        
        # Get all categories
        categories = shapenet_downloader.get_available_categories()
        category_ids = [cat['id'] for cat in categories if not cat.get('downloaded', False)]
        
        if not category_ids:
            return Response({
                'status': 'success', 
                'message': 'All ShapeNet categories are already downloaded.'
            })
        
        # Function to download in background
        def download_all_categories():
            for cat_id in category_ids:
                try:
                    shapenet_downloader.download_category(cat_id)
                except Exception as e:
                    print(f"Error downloading category {cat_id}: {str(e)}")
        
        # Start download in a separate thread
        download_thread = threading.Thread(target=download_all_categories)
        download_thread.daemon = True
        download_thread.start()
        
        return Response({
            'status': 'success',
            'message': f'Started downloading {len(category_ids)} ShapeNet categories in the background.',
            'total_categories': len(category_ids)
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return Response({'status': 'error', 'message': str(e)}, status=500)

@api_view(['GET'])
def get_custom_datasets(request):
    """
    Get available custom datasets
    """
    # Implementation to return custom datasets
    return Response({'datasets': []})

@api_view(['GET'])
def get_combined_datasets(request):
    """
    Get available combined datasets
    """
    # Implementation to return combined datasets
    return Response({'datasets': []})

@api_view(['GET'])
def get_objectnet3d_categories(request):
    """
    Get available ObjectNet3D categories with download status.
    """
    try:
        categories = objectnet3d.get_available_categories()
        return Response(categories)
    except Exception as e:
        logger.error(f"Error getting ObjectNet3D categories: {str(e)}")
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def download_objectnet3d_category(request):
    """
    Download a specific ObjectNet3D category.
    """
    category_id = request.data.get('category_id')
    if not category_id:
        return Response({"error": "Category ID is required"}, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        result = objectnet3d.download_category(category_id)
        
        # Check if the result indicates procedurally generated models
        if isinstance(result, dict) and result.get('is_procedural', False):
            return Response({
                "path": result.get('path'),
                "success": True,
                "is_procedural": True,
                "model_count": result.get('model_count', 0),
                "message": "Could not download actual ObjectNet3D models. Using procedurally generated models as fallback.",
                "note": result.get('note', '')
            })
        elif isinstance(result, str):
            # Handle case where the function returns just a string path
            return Response({
                "path": result,
                "success": True,
                "message": "Successfully retrieved models"
            })
        else:
            # New response format
            return Response({
                "path": result.get('path'),
                "success": True,
                "is_procedural": False,
                "model_count": result.get('model_count', 0),
                "message": "Successfully downloaded ObjectNet3D models"
            })
    except Exception as e:
        logger.error(f"Error downloading ObjectNet3D category: {str(e)}")
        return Response({
            "error": str(e),
            "message": "Failed to download category. Check server logs for details."
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def download_objectnet3d_toolbox(request):
    """
    Download the ObjectNet3D toolbox.
    """
    try:
        result = objectnet3d.download_toolbox()
        return Response({"path": result, "success": True})
    except Exception as e:
        logger.error(f"Error downloading ObjectNet3D toolbox: {str(e)}")
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def download_all_objectnet3d(request):
    """
    Download all ObjectNet3D categories in background.
    """
    try:
        # Start background task to download all components
        status_key = f"objectnet3d_batch_download_{int(time.time())}"
        
        def download_task():
            progress = {}
            components = ['cad_models', 'images', 'annotations', 'splits', 'toolbox']
            total = len(components)
            
            for i, component in enumerate(components):
                try:
                    if component == 'cad_models':
                        # This will download and extract all CAD models
                        objectnet3d._download_cad_models()
                    elif component == 'toolbox':
                        objectnet3d.download_toolbox()
                    else:
                        objectnet3d.download_dataset_components(component)
                    
                    progress[component] = {"status": "completed"}
                except Exception as e:
                    progress[component] = {"status": "error", "message": str(e)}
                
                # Update global progress
                cache.set(status_key, {
                    "progress": progress,
                    "completed": i + 1,
                    "total": total,
                    "percent": int(100 * (i + 1) / total)
                }, timeout=3600)
            
            # Mark as completed
            cache_data = cache.get(status_key, {})
            cache_data["status"] = "completed"
            cache.set(status_key, cache_data, timeout=3600)
        
        # Start in background
        threading.Thread(target=download_task).start()
        
        return Response({
            "status": "started",
            "status_key": status_key,
            "message": "Downloading ObjectNet3D dataset in background. This includes 3D models, images, annotations, and the toolbox."
        })
        
    except Exception as e:
        logger.error(f"Error initiating ObjectNet3D batch download: {str(e)}")
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def get_objectnet3d_download_status(request):
    """
    Get status of background ObjectNet3D downloads.
    """
    status_key = request.query_params.get('status_key')
    if not status_key:
        return Response({"error": "Status key is required"}, status=status.HTTP_400_BAD_REQUEST)
    
    status_data = cache.get(status_key, {})
    return Response(status_data)
