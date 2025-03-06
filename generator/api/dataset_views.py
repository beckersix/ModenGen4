"""
Dataset management API views.

This module provides REST API views for dataset management,
allowing users to browse, download, and combine datasets for training.
"""

import os
import json
import logging
from pathlib import Path
from django.conf import settings
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from django.views.generic import TemplateView

from ..utils.dataset_manager import get_dataset_managers

logger = logging.getLogger(__name__)

# Dataset base directory
DATASET_BASE_DIR = Path(settings.MEDIA_ROOT) / "datasets"
DATASET_BASE_DIR.mkdir(parents=True, exist_ok=True)

# Initialize dataset managers
shapenet_downloader, custom_dataset_manager, combined_dataset_manager = get_dataset_managers(DATASET_BASE_DIR)


class DatasetView(TemplateView):
    """View for the dataset management page."""
    template_name = 'generator/datasets.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Get available datasets
        shapenet_categories = shapenet_downloader.get_available_categories()
        custom_datasets = custom_dataset_manager.get_available_datasets()
        combined_datasets = combined_dataset_manager.get_available_combined_datasets()
        
        # Count downloaded categories
        downloaded_count = sum(1 for cat in shapenet_categories if cat.get('downloaded', False))
        
        context.update({
            'shapenet_categories': shapenet_categories,
            'shapenet_downloaded_count': downloaded_count,
            'shapenet_total_count': len(shapenet_categories),
            'custom_datasets': custom_datasets,
            'custom_datasets_count': len(custom_datasets),
            'combined_datasets': combined_datasets,
            'combined_datasets_count': len(combined_datasets),
            'alternative_sources': shapenet_downloader.dataset_sources
        })
        
        return context


@api_view(['GET'])
def list_shapenet_categories(request):
    """
    List all available ShapeNet categories.
    
    Returns:
        JSON response with the list of categories
    """
    categories = shapenet_downloader.get_available_categories()
    return Response(categories)


@api_view(['POST'])
def download_shapenet_category(request):
    """
    Download a ShapeNet category.
    
    Request format:
    {
        "category_id": "03001627"
    }
    
    Returns:
        JSON response with the result
    """
    if not request.data or 'category_id' not in request.data:
        return Response(
            {"error": "Missing category_id parameter"},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    category_id = request.data['category_id']
    
    try:
        category_path = shapenet_downloader.download_category(category_id)
        return Response({
            "status": "success",
            "message": f"Category {category_id} downloaded successfully",
            "path": category_path
        })
    except Exception as e:
        logger.exception(f"Error downloading category {category_id}: {e}")
        return Response(
            {"error": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
def download_alternative_dataset(request):
    """
    Download an alternative dataset.
    
    Request format:
    {
        "dataset_name": "thingi10k",
        "sample_only": true
    }
    
    Returns:
        JSON response with the result
    """
    if not request.data or 'dataset_name' not in request.data:
        return Response(
            {"error": "Missing dataset_name parameter"},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    dataset_name = request.data['dataset_name']
    sample_only = request.data.get('sample_only', True)
    
    try:
        dataset_path = shapenet_downloader.download_alternative_dataset(
            dataset_name, sample_only=sample_only
        )
        return Response({
            "status": "success",
            "message": f"Dataset {dataset_name} downloaded successfully",
            "path": dataset_path
        })
    except Exception as e:
        logger.exception(f"Error downloading dataset {dataset_name}: {e}")
        return Response(
            {"error": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def list_custom_datasets(request):
    """
    List all available custom datasets.
    
    Returns:
        JSON response with the list of datasets
    """
    datasets = custom_dataset_manager.get_available_datasets()
    return Response(datasets)


@api_view(['POST'])
def create_custom_dataset(request):
    """
    Create a new custom dataset.
    
    Request format:
    {
        "name": "My Custom Dataset",
        "description": "A dataset of my custom models"
    }
    
    Returns:
        JSON response with the result
    """
    if not request.data or 'name' not in request.data:
        return Response(
            {"error": "Missing name parameter"},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    name = request.data['name']
    description = request.data.get('description', '')
    
    try:
        dataset_path = custom_dataset_manager.create_dataset(name, description)
        return Response({
            "status": "success",
            "message": f"Custom dataset {name} created successfully",
            "path": dataset_path
        })
    except Exception as e:
        logger.exception(f"Error creating custom dataset {name}: {e}")
        return Response(
            {"error": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
def upload_model_to_dataset(request):
    """
    Upload a model file to a custom dataset.
    
    This endpoint expects a multipart/form-data request with the following fields:
    - dataset_name: Name of the dataset
    - model_file: The model file to upload
    
    Returns:
        JSON response with the result
    """
    if 'dataset_name' not in request.data or 'model_file' not in request.FILES:
        return Response(
            {"error": "Missing dataset_name or model_file parameters"},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    dataset_name = request.data['dataset_name']
    model_file = request.FILES['model_file']
    
    try:
        # Save the uploaded file temporarily
        temp_dir = Path(settings.MEDIA_ROOT) / "temp"
        temp_dir.mkdir(exist_ok=True)
        
        temp_file = temp_dir / model_file.name
        with open(temp_file, 'wb+') as destination:
            for chunk in model_file.chunks():
                destination.write(chunk)
        
        # Add the model to the dataset
        model_path = custom_dataset_manager.add_model_to_dataset(dataset_name, temp_file)
        
        # Delete the temporary file
        if temp_file.exists():
            os.remove(temp_file)
        
        return Response({
            "status": "success",
            "message": f"Model {model_file.name} uploaded to dataset {dataset_name} successfully",
            "path": model_path
        })
    except Exception as e:
        logger.exception(f"Error uploading model to dataset {dataset_name}: {e}")
        return Response(
            {"error": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
def upload_zip_to_dataset(request):
    """
    Upload and extract a zip file to a custom dataset.
    
    This endpoint expects a multipart/form-data request with the following fields:
    - dataset_name: Name of the dataset
    - zip_file: The zip file to upload and extract
    
    Returns:
        JSON response with the result
    """
    if 'dataset_name' not in request.data or 'zip_file' not in request.FILES:
        return Response(
            {"error": "Missing dataset_name or zip_file parameters"},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    dataset_name = request.data['dataset_name']
    zip_file = request.FILES['zip_file']
    
    try:
        # Save the uploaded file temporarily
        temp_dir = Path(settings.MEDIA_ROOT) / "temp"
        temp_dir.mkdir(exist_ok=True)
        
        temp_file = temp_dir / zip_file.name
        with open(temp_file, 'wb+') as destination:
            for chunk in zip_file.chunks():
                destination.write(chunk)
        
        # Extract the zip file to the dataset
        model_count = custom_dataset_manager.extract_zip_to_dataset(dataset_name, temp_file)
        
        # Delete the temporary file
        if temp_file.exists():
            os.remove(temp_file)
        
        return Response({
            "status": "success",
            "message": f"Extracted {model_count} models from {zip_file.name} to dataset {dataset_name} successfully"
        })
    except Exception as e:
        logger.exception(f"Error extracting zip to dataset {dataset_name}: {e}")
        return Response(
            {"error": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['DELETE'])
def delete_custom_dataset(request, dataset_name):
    """
    Delete a custom dataset.
    
    Args:
        dataset_name: Name of the dataset to delete
        
    Returns:
        JSON response with the result
    """
    try:
        custom_dataset_manager.delete_dataset(dataset_name)
        return Response({
            "status": "success",
            "message": f"Dataset {dataset_name} deleted successfully"
        })
    except Exception as e:
        logger.exception(f"Error deleting dataset {dataset_name}: {e}")
        return Response(
            {"error": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def list_combined_datasets(request):
    """
    List all available combined datasets.
    
    Returns:
        JSON response with the list of datasets
    """
    datasets = combined_dataset_manager.get_available_combined_datasets()
    return Response(datasets)


@api_view(['POST'])
def create_combined_dataset(request):
    """
    Create a new combined dataset from multiple source datasets.
    
    Request format:
    {
        "name": "My Combined Dataset",
        "description": "A combination of several datasets",
        "source_datasets": [
            "/path/to/dataset1",
            "/path/to/dataset2"
        ]
    }
    
    Returns:
        JSON response with the result
    """
    if not request.data or 'name' not in request.data or 'source_datasets' not in request.data:
        return Response(
            {"error": "Missing name or source_datasets parameters"},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    name = request.data['name']
    description = request.data.get('description', '')
    source_datasets = request.data['source_datasets']
    
    if not isinstance(source_datasets, list) or len(source_datasets) == 0:
        return Response(
            {"error": "source_datasets must be a non-empty list"},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    try:
        dataset_path = combined_dataset_manager.create_combined_dataset(
            name, source_datasets, description
        )
        return Response({
            "status": "success",
            "message": f"Combined dataset {name} created successfully",
            "path": dataset_path
        })
    except Exception as e:
        logger.exception(f"Error creating combined dataset {name}: {e}")
        return Response(
            {"error": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['DELETE'])
def delete_combined_dataset(request, dataset_name):
    """
    Delete a combined dataset.
    
    Args:
        dataset_name: Name of the dataset to delete
        
    Returns:
        JSON response with the result
    """
    try:
        combined_dataset_manager.delete_combined_dataset(dataset_name)
        return Response({
            "status": "success",
            "message": f"Combined dataset {dataset_name} deleted successfully"
        })
    except Exception as e:
        logger.exception(f"Error deleting combined dataset {dataset_name}: {e}")
        return Response(
            {"error": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
