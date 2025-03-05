from django.contrib import admin
from django.urls import path
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.template.response import TemplateResponse
from django.contrib import messages
import subprocess
import os
import threading
import json

from .models.django_models import GeneratedModel, GenerationSetting

# Register models
@admin.register(GeneratedModel)
class GeneratedModelAdmin(admin.ModelAdmin):
    list_display = ('id', 'prompt', 'status', 'created_at', 'updated_at')
    list_filter = ('status', 'created_at')
    search_fields = ('prompt',)
    readonly_fields = ('id', 'created_at', 'updated_at')
    
    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('train_gan/', self.admin_site.admin_view(self.train_gan_view), name='train_gan'),
            path('train_gan_status/', self.admin_site.admin_view(self.train_gan_status), name='train_gan_status'),
        ]
        return custom_urls + urls
    
    def train_gan_view(self, request):
        """View for GAN training configuration and execution."""
        # Check if the request is a POST
        if request.method == 'POST':
            # Get form data
            model_type = request.POST.get('model_type', 'voxel')
            num_epochs = int(request.POST.get('num_epochs', 100))
            batch_size = int(request.POST.get('batch_size', 16))
            generate_samples = 'generate_samples' in request.POST
            use_wandb = 'use_wandb' in request.POST
            
            # Prepare config
            config = {
                'model_type': model_type,
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'use_wandb': use_wandb,
                'output_dir': 'media/gan_output',
                'data_dir': 'media/sample_data'
            }
            
            # Save config to file
            with open('gan_config_current.json', 'w') as f:
                json.dump(config, f, indent=2)
            
            # Start training in a separate thread
            self.train_gan_thread = threading.Thread(
                target=self._run_gan_training,
                args=(config, generate_samples)
            )
            self.train_gan_thread.daemon = True
            self.train_gan_thread.start()
            
            messages.success(request, "GAN training started in the background")
            return redirect('..')
        
        # Otherwise, render the form
        context = {
            'title': 'Train 3D GAN Model',
            'app_label': 'generator',
            'opts': self.model._meta,
            'has_change_permission': self.has_change_permission(request),
        }
        return TemplateResponse(request, 'admin/generator/train_gan.html', context)
    
    def train_gan_status(self, request):
        """AJAX endpoint to check the status of GAN training."""
        if hasattr(self, 'train_gan_thread') and self.train_gan_thread.is_alive():
            return JsonResponse({'status': 'running'})
        else:
            return JsonResponse({'status': 'idle'})
    
    def _run_gan_training(self, config, generate_samples):
        """Run the GAN training process."""
        try:
            # Change to the project directory
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            os.chdir(base_dir)
            
            # Make sure output directories exist
            os.makedirs('media/gan_output', exist_ok=True)
            os.makedirs('media/sample_data', exist_ok=True)
            
            # Build command
            cmd = ['python', 'manage.py', 'runscript', 'train_gan_script', 
                   '--script-args', f'config_file=gan_config_current.json']
            
            if generate_samples:
                cmd.append('generate_samples=True')
            
            # Execute command
            subprocess.run(cmd, check=True)
            
        except Exception as e:
            # Log the error
            with open('gan_training_error.log', 'a') as f:
                f.write(f"Error running GAN training: {str(e)}\n")


@admin.register(GenerationSetting)
class GenerationSettingAdmin(admin.ModelAdmin):
    list_display = ('name', 'value', 'description')
    search_fields = ('name', 'description')
