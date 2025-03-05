import os
import sys
import json
import torch
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gan_training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("GAN Training")

def generate_sample_data(output_dir, count=100):
    """
    Generate sample 3D shapes for training the GAN.
    
    Args:
        output_dir (str): Directory to save generated samples
        count (int): Number of samples to generate
    """
    from generator.mesh_dataset import SampleDataGenerator
    
    logger.info(f"Generating {count} sample 3D shapes in {output_dir}")
    generator = SampleDataGenerator(output_dir)
    generator.generate_samples(count)
    logger.info(f"Sample generation complete: {count} samples created.")


def train_gan(config):
    """
    Train the GAN model using the configuration provided.
    
    Args:
        config (dict): Training configuration parameters
    """
    from generator.gan_trainer import GANTrainer
    from generator.mesh_dataset import MeshDataset, PointCloudDataset
    
    # Ensure output directory exists
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Log training start
    logger.info(f"Starting GAN training with config: {json.dumps(config, indent=2)}")
    logger.info(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # Create dataset based on model type
    if config['model_type'] == 'pointcloud':
        dataset = PointCloudDataset(config['data_dir'])
        logger.info(f"Created PointCloud dataset with {len(dataset)} samples")
    else:  # default to voxel
        dataset = MeshDataset(config['data_dir'])
        logger.info(f"Created Mesh dataset with {len(dataset)} samples")
    
    # Create and run trainer
    trainer = GANTrainer(
        dataset=dataset,
        model_type=config['model_type'],
        batch_size=config['batch_size'],
        use_wandb=config['use_wandb'],
        output_dir=config['output_dir']
    )
    
    # Start training
    logger.info(f"Starting training for {config['num_epochs']} epochs...")
    trainer.train(config['num_epochs'])
    logger.info("Training complete!")
    
    # Generate a few samples from the trained model
    logger.info("Generating samples from trained model...")
    trainer.generate_samples(10)
    
    return trainer


def run(*args):
    """
    Main entry point for the script.
    
    Args:
        *args: Command-line arguments
    """
    logger.info(f"Starting GAN training script with args: {args}")
    
    # Parse arguments
    config_file = None
    generate_samples = False
    
    for arg in args:
        if arg.startswith('config_file='):
            config_file = arg.split('=')[1]
        elif arg == 'generate_samples=True':
            generate_samples = True
    
    # Load configuration
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            'model_type': 'voxel',
            'num_epochs': 100,
            'batch_size': 16,
            'use_wandb': False,
            'output_dir': 'media/gan_output',
            'data_dir': 'media/sample_data'
        }
    
    # Ensure data directory exists
    os.makedirs(config['data_dir'], exist_ok=True)
    
    # Check if data directory is empty or if we need to generate samples
    if generate_samples or not os.listdir(config['data_dir']):
        generate_sample_data(config['data_dir'])
    
    # Train the GAN model
    trainer = train_gan(config)
    
    # Log completion
    logger.info("GAN training script completed successfully.")
    return trainer

if __name__ == "__main__":
    # This part enables running the script directly for testing
    run(*sys.argv[1:])
