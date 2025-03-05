"""
3D Model Generator Package

This package provides functionality for generating 3D models from text descriptions,
training GAN models, and visualizing 3D content.

The package is organized into several submodules:
1. models - Database and GAN model definitions
2. api - REST API views and serializers
3. core - Core text-to-3D functionality
4. training - Model training components
5. visualization - 3D visualization utilities
6. utils - Utility functions for mesh processing and datasets
"""

# Django app configuration
default_app_config = 'generator.apps.GeneratorConfig'

# Don't import Django models at the module level to avoid AppRegistryNotReady error
# These will be imported when needed in the relevant modules