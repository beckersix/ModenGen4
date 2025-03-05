"""
Command-line script to train the 3D GAN model.
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('gan_training.log')
    ]
)
logger = logging.getLogger(__name__)

def setup_args():
    """Set up command line arguments."""
    parser = argparse.ArgumentParser(description='Train 3D GAN for mesh generation')
    parser.add_argument('--config', type=str, default='gan_config.json',
                        help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='gan_training_output',
                        help='Directory to save output files')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory containing training data (overrides config)')
    parser.add_argument('--model-type', type=str, choices=['voxel', 'pointcloud'], default=None,
                        help='Type of model to train (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs to train (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size for training (overrides config)')
    parser.add_argument('--use-wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--generate-samples', action='store_true',
                        help='Generate sample data before training')
    return parser.parse_args()


def main():
    """Main function."""
    args = setup_args()
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.info(f"Configuration file {args.config} not found, using default configuration")
        config = {}
    
    # Override config with command line arguments
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.data_dir:
        config['data_dir'] = args.data_dir
    if args.model_type:
        config['model_type'] = args.model_type
    if args.epochs:
        config['num_epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.use_wandb:
        config['use_wandb'] = True
    
    # Set default values if not specified
    if 'output_dir' not in config:
        config['output_dir'] = 'gan_training_output'
    if 'data_dir' not in config:
        config['data_dir'] = 'sample_data'
    if 'model_type' not in config:
        config['model_type'] = 'voxel'
    if 'num_epochs' not in config:
        config['num_epochs'] = 100
    if 'batch_size' not in config:
        config['batch_size'] = 32
    if 'use_wandb' not in config:
        config['use_wandb'] = False
    if 'log_level' not in config:
        config['log_level'] = 'INFO'
    
    # Create output directories
    output_dir = Path(config['output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the configuration
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Generate sample data if requested
    if args.generate_samples:
        logger.info("Generating sample data...")
        from .mesh_dataset import SampleDataGenerator
        sample_generator = SampleDataGenerator(output_dir=config['data_dir'])
        sample_generator.generate_basic_shapes()
    
    # Train the model
    logger.info(f"Starting training with configuration: {config}")
    from .gan_trainer import train_gan
    trainer = train_gan(config=config)
    
    logger.info("Training completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
