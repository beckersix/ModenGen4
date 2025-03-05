# Generator Module

This module contains the core functionality for the AI 3D Generator application, including text-to-3D generation, GAN model training, and 3D model visualization.

## Module Structure

The generator module has been refactored into the following subpackages:

### `/models`
Contains model definitions:
- `django_models.py` - Django database models (GeneratedModel, TrainedModel, etc.)
- `gan_models.py` - GAN neural network definitions (Generator3D, Discriminator3D, etc.)

### `/api`
Contains API-related functionality:
- `text_to_3d_views.py` - REST API views for text-to-3D generation
- `train_views.py` - REST API views for model training
- `serializers.py` - Django REST Framework serializers
- `urls.py` - URL routing
- `views.py` - General web and API views

### `/core`
Contains core application functionality:
- `text_to_3d_manager.py` - Main manager for the text-to-3D generation pipeline
- `llm_interpreter.py`, `text_interpreter.py` - Text processing and interpretation
- `text_to_3d.py`, `text_to_3d_integrator.py` - Core generation interfaces
- `complete_gan_text_interface.py`, `gan_text_interface.py` - GAN interfaces

### `/training`
Contains training functionality:
- `gan_trainer.py` - GAN training implementations
- `training_pipeline.py` - Model training pipeline
- `train.py`, `train_gan.py` - Training scripts and utilities

### `/visualization`
Contains visualization utilities:
- `voxel_visualizer.py` - Voxel grid visualization
- `point_cloud_generator.py` - Point cloud generation and visualization

### `/utils`
Contains utility functions:
- `mesh_dataset.py` - Dataset loaders and processors
- `mesh_processor.py` - Mesh conversion and processing
- `sample_data.py` - Sample data generation

## Import Structure

All subpackages have an `__init__.py` file that exports the relevant classes and functions, so you can import them directly:

```python
# Import from subpackages
from generator.models import Generator3D, Discriminator3D
from generator.core import TextTo3DManager
from generator.visualization import visualize_voxel_grid

# Or use the main package imports for frequently used components
from generator import GeneratedModel, TrainedModel, TextTo3DManager
```
