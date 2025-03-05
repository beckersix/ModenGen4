"""
Training script for the text-to-3D generation system.
This script trains the pipeline on sample data and evaluates results.
"""

import os
import logging
import argparse
import torch
import json
import time
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

def setup_args():
    """Set up command line arguments."""
    parser = argparse.ArgumentParser(description='Train text-to-3D generation pipeline')
    parser.add_argument('--output-dir', type=str, default='media/training_output',
                        help='Directory to save output files')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Path to dataset file (if None, a sample dataset will be generated)')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of samples to generate if no dataset is provided')
    parser.add_argument('--use-gpu', action='store_true',
                        help='Use GPU for training if available')
    parser.add_argument('--save-intermediate', action='store_true',
                        help='Save intermediate results during training')
    return parser.parse_args()

def main():
    """Main training function."""
    args = setup_args()
    
    # Set up device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU for training")
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Set the 'media' directory relative to the Django project
    base_dir = Path(__file__).resolve().parent.parent
    media_dir = os.path.join(base_dir, 'media')
    os.makedirs(media_dir, exist_ok=True)
    
    # Import here to avoid circular imports
    from .sample_data import SampleDataGenerator
    from .training_pipeline import TrainingPipeline
    
    # Generate or load dataset
    data_generator = SampleDataGenerator()
    if args.dataset:
        try:
            dataset = data_generator.load_dataset(args.dataset)
            logger.info(f"Loaded dataset from {args.dataset} with {len(dataset)} samples")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            sys.exit(1)
    else:
        logger.info(f"Generating sample dataset with {args.num_samples} samples")
        dataset = data_generator.generate_test_dataset(
            simple_count=max(1, args.num_samples - 2),
            complex_count=min(2, args.num_samples)
        )
        
        # Save the generated dataset
        dataset_path = os.path.join(output_dir, 'sample_dataset.json')
        data_generator.save_dataset(dataset, dataset_path)
    
    # Initialize training pipeline
    pipeline = TrainingPipeline(output_dir=output_dir, device=device)
    
    # Train on dataset
    logger.info(f"Starting training on {len(dataset)} samples")
    start_time = time.time()
    
    # Limit the dataset size for demonstration purposes
    if len(dataset) > args.num_samples:
        logger.info(f"Limiting dataset to {args.num_samples} samples for demonstration")
        dataset = dataset[:args.num_samples]
    
    # Run the training
    try:
        history = pipeline.train_on_dataset(
            dataset,
            num_epochs=1,
            batch_size=1,
            save_intermediate=args.save_intermediate
        )
        
        # Analyze results
        stats = pipeline.analyze_training_results()
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Results saved to {output_dir}")
        
        # Print summary statistics
        logger.info("Training Summary:")
        logger.info(f"Total samples: {stats['total_samples']}")
        logger.info(f"Average generation time: {stats['average_times']['total']:.2f} seconds")
        logger.info(f"Average mesh complexity: {stats['average_complexity']['vertices']:.0f} vertices, "
                   f"{stats['average_complexity']['faces']:.0f} faces")
        
        return history
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
