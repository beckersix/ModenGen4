"""
Generator training package

This package contains:
1. GAN trainer implementations
2. Training pipelines
3. Training utilities and scripts
"""

# Import trainer components
from .gan_trainer import (
    GANTrainer,
    VoxelGANTrainer,
    PointCloudGANTrainer
)

# Import training pipelines
from .training_pipeline import (
    TrainingPipeline,
    ModelTrainingConfig
)

# Import training utilities (but don't expose implementation details)
# These are typically run as scripts

__all__ = [
    'GANTrainer',
    'VoxelGANTrainer',
    'PointCloudGANTrainer',
    'TrainingPipeline',
    'ModelTrainingConfig'
]