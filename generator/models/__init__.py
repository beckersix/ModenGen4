"""
Generator models package

This package contains:
1. Django database models
2. GAN model definitions for 3D generation
"""

# Import GAN models which don't depend on Django
from .gan_models import (
    Generator3D,
    Discriminator3D,
    PointCloudGenerator,
    PointCloudDiscriminator
)

# Don't import Django models at the module level
# These should be imported directly from .django_models when needed
# to avoid AppRegistryNotReady errors

__all__ = [
    'Generator3D',
    'Discriminator3D',
    'PointCloudGenerator',
    'PointCloudDiscriminator'
]