"""
GAN models for 3D mesh generation.
This module implements the Generator and Discriminator networks for 3D GAN architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Generator3D(nn.Module):
    """
    Generator network for 3D mesh generation.
    Transforms a random latent vector into a 3D voxel grid or point cloud representation.
    """
    
    def __init__(self, latent_dim=100, output_size=64, feat_dims=[512, 256, 128, 64, 32], voxel_dim=None):
        """
        Initialize the 3D Generator network.
        
        Args:
            latent_dim: Dimensionality of the latent space
            output_size: Dimension of the output voxel grid (e.g., 64 = 64x64x64)
            feat_dims: Feature dimensions for each transposed convolution layer
            voxel_dim: Backward compatibility for voxel_dim parameter (uses output_size if provided)
        """
        super(Generator3D, self).__init__()
        
        self.latent_dim = latent_dim
        # Use voxel_dim for backward compatibility, but prefer output_size if provided
        self.voxel_dim = output_size if voxel_dim is None else voxel_dim
        
        # Calculate how many upsampling layers we need to reach the desired output size
        # Starting from 4x4x4, each layer doubles the resolution
        start_size = 4
        num_upsample_layers = int(np.log2(self.voxel_dim / start_size))
        
        # Ensure we have enough feature dimensions for the desired number of layers
        if len(feat_dims) < num_upsample_layers:
            raise ValueError(f"Not enough feature dimensions for output size {self.voxel_dim}. Need at least {num_upsample_layers} layers.")
        
        # Use only the number of feature dimensions we need
        feat_dims = feat_dims[:num_upsample_layers]
        
        # Initial dense layer to project latent vector
        self.projection = nn.Linear(latent_dim, feat_dims[0] * 4 * 4 * 4)
        
        # Create transposed convolution layers
        layers = []
        
        # Initial shape after projection: [batch, feat_dims[0], 4, 4, 4]
        in_dim = feat_dims[0]
        
        for out_dim in feat_dims[1:]:
            layers.append(nn.Sequential(
                nn.ConvTranspose3d(in_dim, out_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(out_dim),
                nn.ReLU(inplace=True)
            ))
            in_dim = out_dim
        
        # Final layer to produce 1-channel output
        layers.append(nn.Sequential(
            nn.ConvTranspose3d(in_dim, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Values between 0 and 1
        ))
        
        self.layers = nn.Sequential(*layers)
        
        logger.info(f"Initialized 3D Generator with latent dimension {latent_dim} and output dimension {self.voxel_dim}")
    
    def forward(self, z):
        """
        Forward pass through the generator.
        
        Args:
            z: Latent vector of shape [batch_size, latent_dim]
            
        Returns:
            Voxel grid of shape [batch_size, 1, voxel_dim, voxel_dim, voxel_dim]
        """
        batch_size = z.size(0)
        
        # Project and reshape
        x = self.projection(z)
        x = x.view(batch_size, -1, 4, 4, 4)
        
        # Apply transposed convolution layers
        x = self.layers(x)
        
        return x
    
    def generate_sample(self, batch_size=1, device=None):
        """
        Generate random samples from the generator.
        
        Args:
            batch_size: Number of samples to generate
            device: PyTorch device
            
        Returns:
            Generated voxel grids
        """
        if device is None:
            device = next(self.parameters()).device
            
        z = torch.randn(batch_size, self.latent_dim, device=device)
        with torch.no_grad():
            samples = self.forward(z)
        
        return samples


class Discriminator3D(nn.Module):
    """
    Discriminator network for 3D GAN.
    Determines if a 3D voxel grid is real or generated.
    """
    
    def __init__(self, input_size=64, feat_dims=[32, 64, 128, 256, 512], voxel_dim=None):
        """
        Initialize the 3D Discriminator network.
        
        Args:
            input_size: Dimension of the input voxel grid
            feat_dims: Feature dimensions for each convolution layer
            voxel_dim: Backward compatibility for voxel_dim parameter (uses input_size if provided)
        """
        super(Discriminator3D, self).__init__()
        
        # Use voxel_dim for backward compatibility, but prefer input_size if provided
        self.voxel_dim = input_size if voxel_dim is None else voxel_dim
        
        # Calculate how many downsampling layers we need based on input size
        # Each layer reduces resolution by half, and we want to end with 2x2x2 feature maps
        start_size = self.voxel_dim
        target_size = 2
        num_downsample_layers = int(np.log2(start_size / target_size))
        
        # Ensure we have enough feature dimensions for the number of layers
        if len(feat_dims) < num_downsample_layers:
            raise ValueError(f"Not enough feature dimensions for input size {self.voxel_dim}. Need at least {num_downsample_layers} layers.")
        
        # Use only the number of feature dimensions we need
        feat_dims = feat_dims[:num_downsample_layers]
        
        # Create convolution layers
        layers = []
        
        # Input shape: [batch, 1, voxel_dim, voxel_dim, voxel_dim]
        in_channels = 1
        
        for out_channels in feat_dims:
            layers.append(nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ))
            in_channels = out_channels
        
        self.layers = nn.Sequential(*layers)
        
        # Calculate the feature map size after convolutions
        final_spatial_size = self.voxel_dim // (2 ** num_downsample_layers)
        final_features = feat_dims[-1] * (final_spatial_size ** 3)
        
        # Add the final linear layer
        self.fc = nn.Linear(final_features, 1)
        
        logger.info(f"Initialized 3D Discriminator with input dimension {self.voxel_dim} and final feature size {final_features}")
    
    def forward(self, x):
        """
        Forward pass through the discriminator.
        
        Args:
            x: Voxel grid of shape [batch_size, 1, voxel_dim, voxel_dim, voxel_dim]
            
        Returns:
            Scalar value indicating real (1) or fake (0)
        """
        batch_size = x.size(0)
        
        # Apply convolution layers
        x = self.layers(x)
        
        # Flatten and apply final linear layer
        x = x.view(batch_size, -1)
        x = self.fc(x)
        
        return x


class PointCloudGenerator(nn.Module):
    """
    Generator network for point cloud generation.
    Transforms a random latent vector into a point cloud.
    """
    
    def __init__(self, latent_dim=100, point_dim=3, num_points=2048, hidden_dims=[512, 512, 512, 1024]):
        """
        Initialize the Point Cloud Generator network.
        
        Args:
            latent_dim: Dimensionality of the latent space
            point_dim: Dimensionality of points (typically 3 for xyz)
            num_points: Number of points to generate
            hidden_dims: Hidden layer dimensions
        """
        super(PointCloudGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        self.point_dim = point_dim
        self.num_points = num_points
        
        # Create MLP layers
        layers = []
        
        in_dim = latent_dim
        for out_dim in hidden_dims:
            layers.append(nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True)
            ))
            in_dim = out_dim
        
        # Final layer to produce point coordinates
        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_dim, num_points * point_dim)
        
        logger.info(f"Initialized Point Cloud Generator with latent dimension {latent_dim} and {num_points} points")
    
    def forward(self, z):
        """
        Forward pass through the generator.
        
        Args:
            z: Latent vector of shape [batch_size, latent_dim]
            
        Returns:
            Point cloud of shape [batch_size, num_points, point_dim]
        """
        batch_size = z.size(0)
        
        # Apply MLP layers
        x = self.layers(z)
        
        # Generate point coordinates
        x = self.output_layer(x)
        x = x.view(batch_size, self.num_points, self.point_dim)
        
        # Apply tanh to constrain coordinates between -1 and 1
        x = torch.tanh(x)
        
        return x
    
    def generate_sample(self, batch_size=1, device=None):
        """
        Generate random point cloud samples.
        
        Args:
            batch_size: Number of samples to generate
            device: PyTorch device
            
        Returns:
            Generated point clouds
        """
        if device is None:
            device = next(self.parameters()).device
            
        z = torch.randn(batch_size, self.latent_dim, device=device)
        with torch.no_grad():
            point_clouds = self.forward(z)
        
        return point_clouds


class PointCloudDiscriminator(nn.Module):
    """
    Discriminator network for point cloud GAN.
    Determines if a point cloud is real or generated.
    """
    
    def __init__(self, point_dim=3, num_points=2048, hidden_dims=[64, 128, 256, 512, 1024]):
        """
        Initialize the Point Cloud Discriminator network.
        
        Args:
            point_dim: Dimensionality of points
            num_points: Number of points in the point cloud
            hidden_dims: Hidden layer dimensions
        """
        super(PointCloudDiscriminator, self).__init__()
        
        self.point_dim = point_dim
        self.num_points = num_points
        
        # Feature extraction layers (PointNet-like)
        self.feat_extraction = nn.ModuleList()
        
        in_dim = point_dim
        for out_dim in hidden_dims:
            self.feat_extraction.append(nn.Sequential(
                nn.Conv1d(in_dim, out_dim, 1),
                nn.BatchNorm1d(out_dim),
                nn.LeakyReLU(0.2, inplace=True)
            ))
            in_dim = out_dim
        
        # MLP for classification
        self.global_feat_dim = hidden_dims[-1]
        self.classifier = nn.Sequential(
            nn.Linear(self.global_feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )
        
        logger.info(f"Initialized Point Cloud Discriminator for {num_points} points")
    
    def forward(self, x):
        """
        Forward pass through the discriminator.
        
        Args:
            x: Point cloud of shape [batch_size, num_points, point_dim]
            
        Returns:
            Scalar value indicating real (1) or fake (0)
        """
        batch_size = x.size(0)
        
        # Transpose for 1D convolution: [batch_size, point_dim, num_points]
        x = x.transpose(1, 2)
        
        # Apply feature extraction
        for layer in self.feat_extraction:
            x = layer(x)
        
        # Global max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(batch_size, -1)
        
        # Apply classifier
        x = self.classifier(x)
        
        return x
