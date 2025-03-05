"""
GAN trainer for 3D mesh generation.
This module implements the training loop for the 3D GAN models.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
import time
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
import json

from .gan_models import Generator3D, Discriminator3D, PointCloudGenerator, PointCloudDiscriminator
from .mesh_dataset import MeshDataset, PointCloudDataset, create_dataloader
from .mesh_processor import MeshProcessor

logger = logging.getLogger(__name__)

class GANTrainer:
    """Trainer for 3D GANs."""
    
    def __init__(self, dataset=None, model_type='voxel', batch_size=32, output_dir='training_output', 
                 latent_dim=100, use_wandb=False, device=None, config=None):
        """
        Initialize the GAN trainer.
        
        Args:
            dataset: Dataset for training
            model_type: Type of model to train ('voxel' or 'point_cloud')
            batch_size: Batch size for training
            output_dir: Directory to save outputs
            latent_dim: Dimensionality of the latent space
            use_wandb: Whether to use Weights & Biases for logging
            device: PyTorch device
            config: Additional configuration parameters (optional)
        """
        self.model_type = model_type
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.latent_dim = latent_dim
        self.use_wandb = use_wandb
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create config dictionary
        self.config = config or {}
        self.config.update({
            'model_type': model_type,
            'batch_size': batch_size,
            'latent_dim': latent_dim
        })
        
        # Create output directories
        self.models_dir = self.output_dir / 'models'
        self.samples_dir = self.output_dir / 'samples'
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.samples_dir, exist_ok=True)
        
        # Set up dataset and dataloader if provided
        self.dataset = dataset
        if dataset:
            self.dataloader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=2,
                drop_last=True
            )
        
        # Initialize models based on the model type
        if self.model_type == 'voxel':
            # Get resolution from dataset or default to 64
            resolution = 64
            if hasattr(dataset, 'resolution'):
                resolution = dataset.resolution
            
            self.generator = Generator3D(
                latent_dim=latent_dim,
                output_size=resolution
            ).to(self.device)
            
            self.discriminator = Discriminator3D(
                input_size=resolution
            ).to(self.device)
        else:  # point cloud
            num_points = 2048
            if hasattr(dataset, 'num_points'):
                num_points = dataset.num_points
                
            self.generator = PointCloudGenerator(
                latent_dim=latent_dim,
                num_points=num_points
            ).to(self.device)
            
            self.discriminator = PointCloudDiscriminator(
                num_points=num_points
            ).to(self.device)
        
        # Initialize optimizers
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=self.config.get('lr_g', 0.0002),
            betas=(self.config.get('beta1', 0.5), self.config.get('beta2', 0.999))
        )
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.get('lr_d', 0.0002),
            betas=(self.config.get('beta1', 0.5), self.config.get('beta2', 0.999))
        )
        
        # Initialize loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # For mesh conversion
        self.mesh_processor = MeshProcessor()
        
        # Initialize training history
        self.history = {
            'g_loss': [],
            'd_loss': [],
            'd_real_accuracy': [],
            'd_fake_accuracy': []
        }
        
        # Initialize wandb if requested
        if self.use_wandb:
            wandb.init(
                project=self.config.get('wandb_project', '3d-gan'),
                config=self.config
            )
            wandb.watch(self.generator)
            wandb.watch(self.discriminator)
        
        logger.info(f"Initialized GAN trainer with model type: {self.model_type}")
        logger.info(f"Using device: {self.device}")
    
    def train(self, num_epochs=10, dataloader=None, max_steps=None, generate_samples=True):
        """
        Train the GAN.
        
        Args:
            num_epochs: Number of epochs to train
            dataloader: DataLoader for training data (optional, uses self.dataloader if None)
            max_steps: Maximum number of steps per epoch (for testing)
            generate_samples: Whether to generate samples during training
            
        Returns:
            Training history
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        start_time = time.time()
        
        # Use provided dataloader or class dataloader
        if dataloader is None:
            if not hasattr(self, 'dataloader') or self.dataloader is None:
                if not hasattr(self, 'dataset') or self.dataset is None:
                    raise ValueError("No dataset or dataloader provided for training")
                self.dataloader = DataLoader(
                    self.dataset, 
                    batch_size=self.batch_size, 
                    shuffle=True, 
                    num_workers=2
                )
            dataloader = self.dataloader
        
        # Labels for real and fake data
        real_label = 1
        fake_label = 0
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Initialize metrics for this epoch
            g_losses = []
            d_losses = []
            d_real_acc = []
            d_fake_acc = []
            
            # Training loop
            step_count = 0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in pbar:
                # Check if we've reached max_steps
                if max_steps is not None and step_count >= max_steps:
                    break
                
                # Get real data based on model type
                if self.model_type == 'voxel':
                    if isinstance(batch, dict):
                        real_data = batch.get('voxel_grid')
                    else:
                        real_data = batch
                else:  # point cloud
                    if isinstance(batch, dict):
                        real_data = batch.get('points')
                    else:
                        real_data = batch
                
                if real_data is None:
                    logger.warning("Received batch with no data, skipping")
                    continue
                
                batch_size = real_data.size(0)
                real_data = real_data.to(self.device)
                
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                
                # Train with real data
                self.discriminator.zero_grad()
                real_output = self.discriminator(real_data)
                label = torch.full((batch_size,), real_label, dtype=torch.float, device=self.device)
                d_loss_real = self.criterion(real_output.view(-1), label)
                d_loss_real.backward()
                
                # Calculate accuracy for real data
                pred_real = (real_output.view(-1) > 0).float()
                d_real_accuracy = (pred_real == label).float().mean().item()
                d_real_acc.append(d_real_accuracy)
                
                # Train with fake data
                noise = torch.randn(batch_size, self.latent_dim, device=self.device)
                fake_data = self.generator(noise)
                label.fill_(fake_label)
                fake_output = self.discriminator(fake_data.detach())
                d_loss_fake = self.criterion(fake_output.view(-1), label)
                d_loss_fake.backward()
                
                # Calculate accuracy for fake data
                pred_fake = (fake_output.view(-1) > 0).float()
                d_fake_accuracy = (pred_fake == label).float().mean().item()
                d_fake_acc.append(d_fake_accuracy)
                
                # Update discriminator
                d_loss = d_loss_real + d_loss_fake
                self.optimizer_d.step()
                
                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                
                self.generator.zero_grad()
                # Since we updated D, perform another forward pass
                label.fill_(real_label)  # We want to fool the discriminator
                fake_output = self.discriminator(fake_data)
                g_loss = self.criterion(fake_output.view(-1), label)
                g_loss.backward()
                self.optimizer_g.step()
                
                # Record losses
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())
                
                # Update progress bar
                pbar.set_postfix({
                    'g_loss': g_loss.item(),
                    'd_loss': d_loss.item(),
                    'd_real_acc': d_real_accuracy,
                    'd_fake_acc': d_fake_accuracy
                })
                
                step_count += 1
            
            # Calculate average losses and accuracies for the epoch
            epoch_g_loss = np.mean(g_losses) if g_losses else 0
            epoch_d_loss = np.mean(d_losses) if d_losses else 0
            epoch_d_real_acc = np.mean(d_real_acc) if d_real_acc else 0
            epoch_d_fake_acc = np.mean(d_fake_acc) if d_fake_acc else 0
            
            # Record epoch metrics
            self.history['g_loss'].append(epoch_g_loss)
            self.history['d_loss'].append(epoch_d_loss)
            self.history['d_real_accuracy'].append(epoch_d_real_acc)
            self.history['d_fake_accuracy'].append(epoch_d_fake_acc)
            
            # Log to wandb if enabled
            if self.use_wandb:
                wandb.log({
                    'g_loss': epoch_g_loss,
                    'd_loss': epoch_d_loss,
                    'd_real_accuracy': epoch_d_real_acc,
                    'd_fake_accuracy': epoch_d_fake_acc,
                    'epoch': epoch
                })
            
            # Save a checkpoint
            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                self.save_checkpoint(epoch + 1)
            
            # Generate and save some samples
            if generate_samples and ((epoch + 1) % 5 == 0 or epoch == num_epochs - 1):
                self.generate_samples(epoch + 1, num_samples=1 if max_steps is not None else 4)
            
            # Report epoch stats
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s - "
                        f"G loss: {epoch_g_loss:.4f}, D loss: {epoch_d_loss:.4f}, "
                        f"D real acc: {epoch_d_real_acc:.4f}, D fake acc: {epoch_d_fake_acc:.4f}")
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/60:.2f} minutes")
        
        # Save final models
        self.save_checkpoint('final')
        
        # Plot training curves
        if generate_samples:
            self.plot_training_curves()
        
        return self.history
    
    def save_checkpoint(self, epoch):
        """Save model checkpoint."""
        # Create a clean checkpoint structure that's compatible with weights_only=True
        checkpoint = {
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'metadata': {
                'epoch': epoch,
                'optimizer_g_state_dict': self.optimizer_g.state_dict(),
                'optimizer_d_state_dict': self.optimizer_d.state_dict(),
                'config': self.config,
                'history': self.history
            }
        }
        
        checkpoint_path = self.models_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=True)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save to wandb
        if self.use_wandb:
            wandb.save(str(checkpoint_path))
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['metadata']['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['metadata']['optimizer_d_state_dict'])
        self.history = checkpoint['metadata']['history']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint['metadata']['epoch']})")
        
        return checkpoint['metadata']['epoch']
    
    def generate_samples(self, epoch, num_samples=4):
        """Generate and save sample outputs."""
        self.generator.eval()
        
        with torch.no_grad():
            # Generate samples
            noise = torch.randn(num_samples, self.latent_dim, device=self.device)
            if self.model_type == 'voxel':
                samples = self.generator(noise)
                
                # Save samples as meshes
                for i in range(num_samples):
                    voxel_grid = samples[i, 0].cpu().numpy()
                    
                    # Convert voxel grid to mesh
                    mesh = self.mesh_processor.voxel_to_mesh(voxel_grid)
                    
                    # Save mesh
                    mesh_path = self.samples_dir / f'sample_epoch_{epoch}_model_{i+1}.obj'
                    self.mesh_processor.save_mesh(mesh, mesh_path)
                    
                    # Save voxel grid visualization
                    plt.figure(figsize=(8, 8))
                    plt.subplot(111, projection='3d')
                    ax = plt.gca()
                    ax.voxels(voxel_grid > 0.5, edgecolor='k')
                    plt.title(f'Sample {i+1} at Epoch {epoch}')
                    plt.savefig(self.samples_dir / f'voxel_epoch_{epoch}_model_{i+1}.png')
                    plt.close()
            
            else:  # point cloud
                point_clouds = self.generator(noise)
                
                # Save point clouds as PLY files
                for i in range(num_samples):
                    points = point_clouds[i].cpu().numpy()
                    
                    # Save as PLY
                    cloud_path = self.samples_dir / f'pointcloud_epoch_{epoch}_model_{i+1}.ply'
                    self.mesh_processor.save_point_cloud(points, cloud_path)
                    
                    # Save visualization
                    plt.figure(figsize=(8, 8))
                    ax = plt.subplot(111, projection='3d')
                    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    plt.title(f'Point Cloud {i+1} at Epoch {epoch}')
                    plt.savefig(self.samples_dir / f'pc_vis_epoch_{epoch}_model_{i+1}.png')
                    plt.close()
        
        # Log to wandb
        if self.use_wandb:
            sample_images = [
                wandb.Image(str(self.samples_dir / f'voxel_epoch_{epoch}_model_{i+1}.png'))
                for i in range(num_samples)
            ] if self.model_type == 'voxel' else [
                wandb.Image(str(self.samples_dir / f'pc_vis_epoch_{epoch}_model_{i+1}.png'))
                for i in range(num_samples)
            ]
            
            wandb.log({
                'epoch': epoch,
                'samples': sample_images
            })
        
        self.generator.train()
        logger.info(f"Generated {num_samples} samples at epoch {epoch}")
    
    def plot_training_curves(self):
        """Plot training curves."""
        # Plot losses
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history['g_loss'], label='Generator')
        plt.plot(self.history['d_loss'], label='Discriminator')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('GAN Losses')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['d_real_accuracy'], label='Real Accuracy')
        plt.plot(self.history['d_fake_accuracy'], label='Fake Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Discriminator Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png')
        plt.close()
        
        # Save history to JSON
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        logger.info(f"Saved training curves plot to {self.output_dir / 'training_curves.png'}")
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({
                'training_curves': wandb.Image(str(self.output_dir / 'training_curves.png'))
            })


def train_gan(config_path=None, config=None):
    """
    Train a 3D GAN model.
    
    Args:
        config_path: Path to configuration file
        config: Configuration dictionary (used if config_path is None)
        
    Returns:
        Trained GAN trainer instance
    """
    # Load configuration
    if config_path is not None:
        with open(config_path, 'r') as f:
            config = json.load(f)
    elif config is None:
        raise ValueError("Either config_path or config must be provided")
    
    # Set up logging
    log_level = getattr(logging, config.get('log_level', 'INFO'))
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(config.get('log_file', 'gan_training.log'))
        ]
    )
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() and config.get('use_gpu', True) else 'cpu')
    
    # Initialize trainer
    trainer = GANTrainer(
        config=config,
        output_dir=config.get('output_dir', 'gan_training_output'),
        device=device,
        use_wandb=config.get('use_wandb', False)
    )
    
    # Create dataset and dataloader
    if config.get('model_type', 'voxel') == 'voxel':
        dataset = MeshDataset(
            data_dir=config.get('data_dir', 'sample_data'),
            voxel_resolution=config.get('voxel_dim', 64)
        )
    else:  # point cloud
        dataset = PointCloudDataset(
            data_dir=config.get('data_dir', 'sample_data'),
            num_points=config.get('num_points', 2048)
        )
    
    dataloader = create_dataloader(
        dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=True,
        num_workers=config.get('num_workers', 4)
    )
    
    # Train the model
    trainer.train(
        dataloader=dataloader,
        num_epochs=config.get('num_epochs', 100)
    )
    
    return trainer
