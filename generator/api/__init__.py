"""
Generator API package

This package contains:
1. REST API views for text-to-3D model generation
2. API views for model training
3. API serializers
4. URL routing
"""

# Import API serializers
from .serializers import (
    GeneratedModelSerializer,
    TrainedModelSerializer,
    GenerationSettingSerializer
)

# Re-export views, but don't import them here to avoid circular imports
# Views are accessible via their respective modules:
# - api.text_to_3d_views
# - api.train_views
# - api.views

__all__ = [
    'GeneratedModelSerializer',
    'TrainedModelSerializer',
    'GenerationSettingSerializer'
]