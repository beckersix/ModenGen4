from django.db import models
from django.utils import timezone
import uuid
import os

# Create your models here.

class GeneratedModel(models.Model):
    """Model representing a generated 3D model from text prompt"""
    
    STATUS_CHOICES = (
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    )
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    prompt = models.TextField(help_text="Text prompt used to generate the model")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Files
    point_cloud_file = models.FileField(upload_to='models/point_clouds/', null=True, blank=True)
    mesh_file = models.FileField(upload_to='models/meshes/', null=True, blank=True)
    texture_file = models.ImageField(upload_to='models/textures/', null=True, blank=True)
    
    # Parameters
    detail_level = models.IntegerField(default=1, help_text="Level of detail (1-5)")
    refine_iterations = models.IntegerField(default=3, help_text="Number of refinement iterations")
    
    # Metadata
    vertex_count = models.IntegerField(default=0)
    face_count = models.IntegerField(default=0)
    generation_time = models.FloatField(default=0.0, help_text="Time taken to generate the model (seconds)")
    
    def __str__(self):
        return f"Model {self.id}: {self.prompt[:30]}{'...' if len(self.prompt) > 30 else ''}"
    
    def get_absolute_url(self):
        from django.urls import reverse
        return reverse('model_detail', args=[str(self.id)])
    
    def filename(self):
        """Generate a filename based on the prompt"""
        base_name = self.prompt.lower().replace(' ', '_')[:30]
        return f"{base_name}_{self.id}"
    
    class Meta:
        ordering = ['-created_at']


class GenerationSetting(models.Model):
    """Settings for the model generation process"""
    
    name = models.CharField(max_length=100)
    value = models.TextField()
    description = models.TextField(blank=True)
    
    def __str__(self):
        return self.name


class TrainedModel(models.Model):
    """Model representing a trained 3D generation model"""
    
    STATUS_CHOICES = (
        ('pending', 'Pending'),
        ('training', 'Training'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    )
    
    MODEL_TYPE_CHOICES = (
        ('gan', 'GAN'),
        ('diffusion', 'Diffusion'),
    )
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100, help_text="Name of the trained model")
    model_type = models.CharField(max_length=20, choices=MODEL_TYPE_CHOICES, default='gan')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Training parameters
    voxel_size = models.IntegerField(default=64, help_text="Voxel resolution (32, 64, 128)")
    latent_dim = models.IntegerField(default=128, help_text="Latent space dimension")
    batch_size = models.IntegerField(default=32, help_text="Batch size used during training")
    learning_rate = models.FloatField(default=0.0002, help_text="Learning rate used during training")
    
    # Training data and progress
    dataset = models.CharField(max_length=255, help_text="Dataset used for training")
    total_epochs = models.IntegerField(default=100, help_text="Total epochs to train")
    current_epoch = models.IntegerField(default=0, help_text="Current training epoch")
    
    # Save paths for model weights and configurations
    model_file = models.FileField(upload_to='models/trained/', null=True, blank=True)
    config_file = models.FileField(upload_to='models/configs/', null=True, blank=True)
    
    # Metrics
    generator_loss = models.JSONField(null=True, blank=True, help_text="Generator loss history")
    discriminator_loss = models.JSONField(null=True, blank=True, help_text="Discriminator loss history")
    training_time = models.FloatField(default=0.0, help_text="Total training time in seconds")
    
    # If this model is the default one to use
    is_default = models.BooleanField(default=False)
    
    def __str__(self):
        return f"{self.name} ({self.model_type})"
    
    def save(self, *args, **kwargs):
        # If this model is being set as default, un-set any other defaults
        if self.is_default:
            TrainedModel.objects.filter(is_default=True).exclude(id=self.id).update(is_default=False)
        super().save(*args, **kwargs)
    
    def training_progress(self):
        """Calculate training progress percentage"""
        if self.total_epochs > 0:
            return (self.current_epoch / self.total_epochs) * 100
        return 0
