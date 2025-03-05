"""
Serializers for the generator app.
"""

from rest_framework import serializers
from ..models.django_models import GeneratedModel, GenerationSetting, TrainedModel

class GeneratedModelSerializer(serializers.ModelSerializer):
    """Serializer for GeneratedModel model."""
    
    class Meta:
        model = GeneratedModel
        fields = [
            'id', 'prompt', 'status', 'created_at', 'updated_at', 
            'point_cloud_file', 'mesh_file', 'texture_file',
            'detail_level', 'refine_iterations',
            'vertex_count', 'face_count', 'generation_time'
        ]
        read_only_fields = [
            'id', 'created_at', 'updated_at', 'status',
            'point_cloud_file', 'mesh_file', 'texture_file',
            'vertex_count', 'face_count', 'generation_time'
        ]
        
    def validate_prompt(self, value):
        """Validate the prompt."""
        if not value or len(value.strip()) == 0:
            raise serializers.ValidationError("Prompt cannot be empty")
        return value
    
    def validate_detail_level(self, value):
        """Validate the detail level."""
        if value < 1 or value > 5:
            raise serializers.ValidationError("Detail level must be between 1 and 5")
        return value
    
    def validate_refine_iterations(self, value):
        """Validate the refinement iterations."""
        if value < 1 or value > 10:
            raise serializers.ValidationError("Refinement iterations must be between 1 and 10")
        return value

class GenerationSettingSerializer(serializers.ModelSerializer):
    """Serializer for GenerationSetting model."""
    
    class Meta:
        model = GenerationSetting
        fields = ['id', 'name', 'value', 'description']

class TrainedModelSerializer(serializers.ModelSerializer):
    """Serializer for TrainedModel model."""
    
    training_progress = serializers.SerializerMethodField()
    
    class Meta:
        model = TrainedModel
        fields = [
            'id', 'name', 'model_type', 'status', 'created_at', 'updated_at',
            'voxel_size', 'latent_dim', 'batch_size', 'learning_rate', 
            'dataset', 'total_epochs', 'current_epoch', 
            'model_file', 'config_file', 'generator_loss', 'discriminator_loss',
            'training_time', 'is_default', 'training_progress'
        ]
        read_only_fields = [
            'id', 'created_at', 'updated_at', 'status',
            'current_epoch', 'model_file', 'config_file', 
            'generator_loss', 'discriminator_loss', 'training_time'
        ]
    
    def get_training_progress(self, obj):
        """Get the training progress percentage."""
        return obj.training_progress()
    
    def validate_name(self, value):
        """Validate the model name."""
        if not value or len(value.strip()) == 0:
            raise serializers.ValidationError("Model name cannot be empty")
        return value
    
    def validate_voxel_size(self, value):
        """Validate the voxel size."""
        valid_sizes = [32, 64, 128]
        if value not in valid_sizes:
            raise serializers.ValidationError(f"Voxel size must be one of {valid_sizes}")
        return value
    
    def validate_latent_dim(self, value):
        """Validate the latent dimension."""
        if value < 16 or value > 512:
            raise serializers.ValidationError("Latent dimension must be between 16 and 512")
        return value
    
    def validate_batch_size(self, value):
        """Validate the batch size."""
        if value < 1 or value > 128:
            raise serializers.ValidationError("Batch size must be between 1 and 128")
        return value
    
    def validate_learning_rate(self, value):
        """Validate the learning rate."""
        if value <= 0 or value > 0.1:
            raise serializers.ValidationError("Learning rate must be between 0 and 0.1")
        return value
