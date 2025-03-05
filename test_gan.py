import os
import sys
import logging
import torch
import trimesh
import numpy as np
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GAN Test")

# Make sure we're using the project's directory
# Use the current directory as the base path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Import project modules
from generator.gan_models import Generator3D, Discriminator3D, PointCloudGenerator, PointCloudDiscriminator
from generator.mesh_dataset import MeshDataset, PointCloudDataset, SampleDataGenerator
from generator.gan_trainer import GANTrainer

def test_sample_data_generation():
    """Test the sample data generator."""
    logger.info("Testing sample data generation...")
    
    # Create a temporary output directory
    output_dir = os.path.join(BASE_DIR, "media", "test_samples")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a sample data generator
    generator = SampleDataGenerator(output_dir)
    
    # Generate a few samples
    logger.info("Generating 5 sample meshes...")
    generator.generate_samples(count=5)
    
    # Check if the files were created
    files = list(Path(output_dir).glob("*.obj"))
    logger.info(f"Generated {len(files)} sample files")
    
    if len(files) > 0:
        logger.info("Sample data generation test passed.")
        return True
    else:
        logger.error("Sample data generation failed - no files created.")
        return False

def test_mesh_dataset():
    """Test the mesh dataset."""
    logger.info("Testing mesh dataset...")
    
    # Create a temporary output directory for samples
    output_dir = os.path.join(BASE_DIR, "media", "test_samples")
    os.makedirs(output_dir, exist_ok=True)
    
    # Make sure we have some sample data
    if len(list(Path(output_dir).glob("*.obj"))) < 3:
        logger.info("Generating sample data for testing dataset...")
        generator = SampleDataGenerator(output_dir)
        generator.generate_samples(count=3)
    
    # Create a dataset
    dataset = MeshDataset(output_dir, resolution=32)
    
    # Try to get an item
    if len(dataset) > 0:
        sample = dataset[0]
        logger.info(f"Retrieved sample from dataset with shape: {sample.shape}")
        
        if isinstance(sample, torch.Tensor) and len(sample.shape) == 4:
            logger.info("Mesh dataset test passed.")
            return True
        else:
            logger.error(f"Invalid sample format: {type(sample)}, shape: {sample.shape if hasattr(sample, 'shape') else 'unknown'}")
            return False
    else:
        logger.error("Dataset is empty, test failed.")
        return False

def test_generator_model():
    """Test the 3D generator model."""
    logger.info("Testing generator model...")
    
    # Create a generator
    latent_dim = 128
    generator = Generator3D(latent_dim=latent_dim, output_size=32)
    
    # Generate a sample
    batch_size = 2
    z = torch.randn(batch_size, latent_dim)
    
    try:
        with torch.no_grad():
            output = generator(z)
        
        logger.info(f"Generated output with shape: {output.shape}")
        
        if output.shape == (batch_size, 1, 32, 32, 32):
            logger.info("Generator model test passed.")
            return True
        else:
            logger.error(f"Unexpected output shape: {output.shape}")
            return False
    except Exception as e:
        logger.error(f"Error running generator model: {str(e)}")
        return False

def test_discriminator_model():
    """Test the 3D discriminator model."""
    logger.info("Testing discriminator model...")
    
    # Create a discriminator
    discriminator = Discriminator3D(input_size=32)
    
    # Generate a sample input
    batch_size = 2
    voxel_grid = torch.rand(batch_size, 1, 32, 32, 32)
    
    try:
        with torch.no_grad():
            output = discriminator(voxel_grid)
        
        logger.info(f"Discriminator output: {output}")
        
        if output.shape == (batch_size, 1):
            logger.info("Discriminator model test passed.")
            return True
        else:
            logger.error(f"Unexpected output shape: {output.shape}")
            return False
    except Exception as e:
        logger.error(f"Error running discriminator model: {str(e)}")
        return False

def test_gan_trainer():
    """Test the GAN trainer class."""
    logger.info("Testing GAN trainer...")
    
    # Create a temporary output directory for samples
    output_dir = os.path.join(BASE_DIR, "media", "test_samples")
    data_dir = os.path.join(BASE_DIR, "media", "test_samples")
    os.makedirs(output_dir, exist_ok=True)
    
    # Make sure we have some sample data
    if len(list(Path(data_dir).glob("*.obj"))) < 3:
        logger.info("Generating sample data for testing trainer...")
        generator = SampleDataGenerator(data_dir)
        generator.generate_samples(count=3)
    
    # Create a dataset
    dataset = MeshDataset(data_dir, resolution=32)
    
    try:
        # Create a trainer
        trainer = GANTrainer(
            dataset=dataset,
            model_type='voxel',
            batch_size=1,
            use_wandb=False,
            output_dir=output_dir
        )
        
        # Test a single training step
        logger.info("Running a single training step...")
        trainer.train(num_epochs=1, max_steps=1, generate_samples=False)
        
        logger.info("GAN trainer test passed.")
        return True
    except Exception as e:
        logger.error(f"Error in GAN trainer: {str(e)}")
        return False

if __name__ == "__main__":
    # Print system info
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Run tests
    tests = [
        ("Sample data generation", test_sample_data_generation),
        ("Mesh dataset", test_mesh_dataset),
        ("Generator model", test_generator_model),
        ("Discriminator model", test_discriminator_model),
        ("GAN trainer", test_gan_trainer)
    ]
    
    results = []
    for name, test_func in tests:
        logger.info(f"\n{'='*40}\nRunning test: {name}\n{'='*40}")
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"Exception in {name} test: {str(e)}")
            results.append((name, False))
    
    # Print summary
    logger.info("\n\n" + "="*60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("="*60)
    
    passed = 0
    for name, result in results:
        status = "PASSED" if result else "FAILED"
        logger.info(f"{name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nPassed {passed}/{len(tests)} tests")
    
    if passed == len(tests):
        logger.info("\nAll tests passed! The GAN implementation is ready to use.")
    else:
        logger.info("\nSome tests failed. Please check the logs for details.")
