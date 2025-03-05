"""
Training API Views

This module provides REST API views for model training,
integrating the training functionality with the Django web application.
"""

from django.conf import settings
from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status

import os
import json
import logging
import threading
import time
import uuid
from pathlib import Path
import numpy as np
import torch
import json

# Import Django models directly from django_models.py
from ..models.django_models import TrainedModel
from .serializers import TrainedModelSerializer

logger = logging.getLogger(__name__)

# Dictionary to keep track of training processes
active_training_processes = {}

@api_view(['POST'])
def start_training(request):
    """
    API view to start training a 3D generation model.
    
    Request format:
    {
        "name": "my_trained_model",
        "model_type": "gan",
        "voxel_size": 64,
        "latent_dim": 128,
        "batch_size": 32,
        "learning_rate": 0.0002,
        "dataset": "shapenet",
        "total_epochs": 100,
        "continue_training": false,
        "base_model_id": null
    }
    
    Returns:
        JSON response with the model ID and status
    """
    # Validate request data
    if not request.data or 'name' not in request.data:
        return Response(
            {"error": "Missing required parameter: name"}, 
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Check if we're continuing training from an existing model
    continue_training = request.data.get('continue_training', False)
    base_model_id = request.data.get('base_model_id')
    
    if continue_training and base_model_id:
        try:
            base_model = TrainedModel.objects.get(id=base_model_id)
            # Create a new model based on the existing one
            trained_model = TrainedModel(
                name=request.data['name'],
                model_type=base_model.model_type,
                voxel_size=base_model.voxel_size,
                latent_dim=base_model.latent_dim,
                batch_size=request.data.get('batch_size', base_model.batch_size),
                learning_rate=request.data.get('learning_rate', base_model.learning_rate),
                dataset=request.data.get('dataset', base_model.dataset),
                total_epochs=request.data.get('total_epochs', 100),
                status='pending'
            )
            trained_model.save()
        except TrainedModel.DoesNotExist:
            return Response(
                {"error": f"Base model with ID {base_model_id} not found"}, 
                status=status.HTTP_404_NOT_FOUND
            )
    else:
        # Create a new training model entry
        serializer = TrainedModelSerializer(data=request.data)
        if serializer.is_valid():
            trained_model = serializer.save(status='pending')
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    # Start training in a background thread
    threading.Thread(
        target=_train_model_async,
        args=(trained_model.id, continue_training, base_model_id)
    ).start()
    
    return Response({
        "id": str(trained_model.id),
        "status": "pending",
        "message": "Model training started"
    })

def _train_model_async(model_id, continue_training=False, base_model_id=None):
    """
    Train the model asynchronously.
    
    Args:
        model_id: ID of the model in the database
        continue_training: Whether to continue training from a base model
        base_model_id: ID of the base model to continue training from
    """
    try:
        # Get the model from the database
        trained_model = TrainedModel.objects.get(id=model_id)
        
        # Update status to training
        trained_model.status = 'training'
        trained_model.save()
        
        # Record the training process
        active_training_processes[str(model_id)] = {
            'pid': os.getpid(),
            'start_time': time.time(),
            'should_stop': False
        }
        
        # Prepare to track losses
        generator_losses = []
        discriminator_losses = []
        start_time = time.time()
        
        # Initialize training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Starting model training for {model_id} on {device}")
        
        # Get model parameters
        voxel_size = trained_model.voxel_size
        latent_dim = trained_model.latent_dim
        batch_size = trained_model.batch_size
        learning_rate = trained_model.learning_rate
        epochs = trained_model.total_epochs
        model_type = trained_model.model_type
        dataset_name = trained_model.dataset
        
        # Load base model if continuing training
        if continue_training and base_model_id:
            base_model = TrainedModel.objects.get(id=base_model_id)
            if base_model.model_file:
                logger.info(f"Loading base model from {base_model.model_file.path}")
                # In a real scenario, you would load the model weights here
        
        # Here you would implement the actual model training
        # For this example, we'll simulate the training process
        for epoch in range(epochs):
            # Check if we should stop training
            if active_training_processes[str(model_id)]['should_stop']:
                logger.info(f"Training for model {model_id} was stopped")
                break
            
            # Simulate training for one epoch
            time.sleep(0.5)  # Simulating training time
            
            # Generate fake losses
            gen_loss = 1.0 - (epoch / epochs) + np.random.normal(0, 0.1)
            disc_loss = 0.5 - (epoch / (2 * epochs)) + np.random.normal(0, 0.05)
            
            # Track losses
            generator_losses.append({"epoch": epoch + 1, "loss": gen_loss})
            discriminator_losses.append({"epoch": epoch + 1, "loss": disc_loss})
            
            # Update model progress
            trained_model.current_epoch = epoch + 1
            trained_model.generator_loss = generator_losses
            trained_model.discriminator_loss = discriminator_losses
            trained_model.save()
            
            # Log progress
            logger.info(f"Model {model_id} - Epoch {epoch+1}/{epochs} - G_loss: {gen_loss:.4f}, D_loss: {disc_loss:.4f}")
        
        # Training completed
        training_time = time.time() - start_time
        
        # Save the model (in a real scenario)
        media_dir = Path(settings.MEDIA_ROOT)
        trained_models_dir = media_dir / 'models' / 'trained'
        trained_models_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model weights (simulated)
        model_path = trained_models_dir / f"{trained_model.name.replace(' ', '_').lower()}_{trained_model.id}.pth"
        
        # Create a simple model structure that can be properly loaded
        import torch.nn as nn
        dummy_model = nn.Sequential(
            nn.Linear(trained_model.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, trained_model.voxel_size * trained_model.voxel_size * trained_model.voxel_size)
        )
        
        # Create a checkpoint dictionary with proper structure
        checkpoint = {
            "generator_state_dict": dummy_model.state_dict(),
            "metadata": {
                "model_config": {
                    "latent_dim": trained_model.latent_dim,
                    "voxel_size": trained_model.voxel_size,
                    "model_type": trained_model.model_type
                },
                "training_info": {
                    "epochs": trained_model.total_epochs,
                    "batch_size": trained_model.batch_size,
                    "learning_rate": trained_model.learning_rate
                }
            }
        }
        
        # Save the model with proper PyTorch serialization
        torch.save(checkpoint, model_path, _use_new_zipfile_serialization=True)
        logger.info(f"Saved PyTorch 2.6 compatible model to {model_path}")
        
        # Save model configuration
        config_dir = media_dir / 'models' / 'configs'
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / f"{trained_model.name.replace(' ', '_').lower()}_{trained_model.id}.json"
        
        config = {
            'model_type': model_type,
            'voxel_size': voxel_size,
            'latent_dim': latent_dim,
            'dataset': dataset_name,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'training_time': training_time,
            'created_at': str(trained_model.created_at),
            'completed_at': str(timezone.now())
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Update model record
        trained_model.model_file = f"models/trained/{model_path.name}"
        trained_model.config_file = f"models/configs/{config_path.name}"
        trained_model.status = 'completed'
        trained_model.training_time = training_time
        trained_model.save()
        
        # Remove from active processes
        if str(model_id) in active_training_processes:
            del active_training_processes[str(model_id)]
        
        logger.info(f"Training completed for model {model_id} in {training_time:.2f} seconds")
    
    except Exception as e:
        logger.error(f"Error training model {model_id}: {e}")
        try:
            trained_model = TrainedModel.objects.get(id=model_id)
            trained_model.status = 'failed'
            trained_model.save()
            
            # Remove from active processes
            if str(model_id) in active_training_processes:
                del active_training_processes[str(model_id)]
        except:
            pass

@api_view(['GET'])
def training_status(request, model_id):
    """
    Get the status of a model training.
    
    Args:
        model_id: ID of the model
        
    Returns:
        JSON response with the model training status
    """
    try:
        trained_model = TrainedModel.objects.get(id=model_id)
        
        response_data = {
            "id": str(trained_model.id),
            "name": trained_model.name,
            "status": trained_model.status,
            "current_epoch": trained_model.current_epoch,
            "total_epochs": trained_model.total_epochs,
            "progress": trained_model.training_progress(),
            "created_at": trained_model.created_at,
            "updated_at": trained_model.updated_at,
        }
        
        # Add losses if available
        if trained_model.generator_loss:
            response_data["generator_loss"] = trained_model.generator_loss[-10:]  # Last 10 entries
        
        if trained_model.discriminator_loss:
            response_data["discriminator_loss"] = trained_model.discriminator_loss[-10:]  # Last 10 entries
        
        # Add model file URL if available
        if trained_model.model_file:
            response_data["model_file"] = request.build_absolute_uri(trained_model.model_file.url)
        
        if trained_model.config_file:
            response_data["config_file"] = request.build_absolute_uri(trained_model.config_file.url)
        
        return Response(response_data)
    
    except TrainedModel.DoesNotExist:
        return Response(
            {"error": f"Model with ID {model_id} not found"}, 
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        return Response(
            {"error": str(e)}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['POST'])
def stop_training(request, model_id):
    """
    Stop an ongoing model training.
    
    Args:
        model_id: ID of the model to stop training
        
    Returns:
        JSON response with the result
    """
    try:
        trained_model = TrainedModel.objects.get(id=model_id)
        
        # Check if the model is currently training
        if trained_model.status != 'training':
            return Response(
                {"error": f"Model {model_id} is not currently training"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Signal the training process to stop
        if str(model_id) in active_training_processes:
            active_training_processes[str(model_id)]['should_stop'] = True
            return Response({"message": f"Training for model {model_id} is being stopped"})
        else:
            # If the process is not in our dictionary, update the status directly
            trained_model.status = 'failed'
            trained_model.save()
            return Response({"message": f"Training for model {model_id} was stopped"})
    
    except TrainedModel.DoesNotExist:
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
def list_trained_models(request):
    """
    List all trained models.
    
    Returns:
        JSON response with the list of models
    """
    try:
        models = TrainedModel.objects.all().order_by('-created_at')
        serializer = TrainedModelSerializer(models, many=True)
        return Response(serializer.data)
    except Exception as e:
        return Response(
            {"error": str(e)}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['POST'])
def set_default_model(request, model_id):
    """
    Set a model as the default model for generation.
    
    Args:
        model_id: ID of the model to set as default
        
    Returns:
        JSON response with the result
    """
    try:
        trained_model = TrainedModel.objects.get(id=model_id)
        
        # Check if the model is completed
        if trained_model.status != 'completed':
            return Response(
                {"error": f"Cannot set incomplete model as default"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Set as default
        trained_model.is_default = True
        trained_model.save()  # This will un-set any other defaults
        
        return Response({"message": f"Model {trained_model.name} set as default"})
    
    except TrainedModel.DoesNotExist:
        return Response(
            {"error": f"Model with ID {model_id} not found"}, 
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        return Response(
            {"error": str(e)}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
