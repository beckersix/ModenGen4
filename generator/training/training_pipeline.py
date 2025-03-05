"""
Training pipeline for the text-to-3D generation system.
"""

import torch
import numpy as np
import os
import logging
import time
import json
from tqdm import tqdm
from .point_cloud_generator import PointCloudGenerator
from .mesh_processor import MeshProcessor
from .texture_generator import TextureGenerator
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class TrainingPipeline:
    """Pipeline for training and refining the text-to-3D generation models."""
    
    def __init__(self, output_dir='training_output', device=None):
        """
        Initialize the training pipeline.
        
        Args:
            output_dir: Directory to save training outputs
            device: PyTorch device to use
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Initializing training pipeline with device: {self.device}")
        
        # Initialize components
        self.point_cloud_generator = PointCloudGenerator(device=self.device)
        self.mesh_processor = MeshProcessor()
        self.texture_generator = TextureGenerator(device=self.device)
        
        # Training metadata
        self.training_history = {
            'models': [],
            'metrics': {
                'point_cloud_generation_times': [],
                'mesh_processing_times': [],
                'texture_generation_times': []
            }
        }
        
        logger.info("Training pipeline initialized")
    
    def train_on_dataset(self, dataset, num_epochs=1, batch_size=1, save_intermediate=True):
        """
        Train the model on a dataset of text prompts.
        
        Args:
            dataset: List of text prompts or dataset object
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            save_intermediate: Whether to save intermediate results
            
        Returns:
            Training history
        """
        logger.info(f"Starting training on dataset with {len(dataset)} samples for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            epoch_start_time = time.time()
            
            # Process each sample in the dataset
            for i, prompt in enumerate(tqdm(dataset)):
                sample_id = f"epoch{epoch+1}_sample{i+1}"
                
                # Generate outputs for this sample
                try:
                    self._process_sample(prompt, sample_id, save_intermediate)
                except Exception as e:
                    logger.error(f"Error processing sample {sample_id}: {e}")
                    continue
            
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")
            
            # Save training history
            self._save_training_history()
        
        logger.info("Training completed")
        return self.training_history
    
    def _process_sample(self, prompt, sample_id, save_intermediate):
        """Process a single sample."""
        logger.info(f"Processing sample {sample_id}: '{prompt}'")
        
        # 1. Generate point cloud
        pc_start_time = time.time()
        point_cloud, pc_gen_time = self.point_cloud_generator.generate_from_text(prompt)
        self.training_history['metrics']['point_cloud_generation_times'].append(pc_gen_time)
        
        # Save point cloud if requested
        if save_intermediate:
            pc_path = os.path.join(self.output_dir, f"{sample_id}_point_cloud.npy")
            self.point_cloud_generator.save_point_cloud(point_cloud, pc_path)
        
        # 2. Convert to mesh
        mesh_start_time = time.time()
        mesh = self.mesh_processor.point_cloud_to_mesh(point_cloud)
        mesh_time = time.time() - mesh_start_time
        self.training_history['metrics']['mesh_processing_times'].append(mesh_time)
        
        # Save mesh if requested
        if save_intermediate:
            mesh_path = os.path.join(self.output_dir, f"{sample_id}_mesh.obj")
            self.mesh_processor.save_mesh(mesh, mesh_path)
        
        # 3. Generate texture
        texture_start_time = time.time()
        texture, texture_time = self.texture_generator.generate_texture(prompt)
        self.training_history['metrics']['texture_generation_times'].append(texture_time)
        
        # Save texture if requested
        if save_intermediate:
            texture_path = os.path.join(self.output_dir, f"{sample_id}_texture.png")
            self.texture_generator.save_texture(texture, texture_path)
        
        # Record model
        self.training_history['models'].append({
            'id': sample_id,
            'prompt': prompt,
            'point_cloud_size': len(point_cloud.pos),
            'mesh_vertices': len(mesh.vertices),
            'mesh_faces': len(mesh.faces),
            'generation_times': {
                'point_cloud': pc_gen_time,
                'mesh': mesh_time,
                'texture': texture_time,
                'total': pc_gen_time + mesh_time + texture_time
            }
        })
        
        logger.info(f"Sample {sample_id} processed successfully")
    
    def _save_training_history(self):
        """Save training history to a file."""
        history_path = os.path.join(self.output_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        logger.info(f"Training history saved to {history_path}")
    
    def analyze_training_results(self):
        """Analyze training results and generate plots."""
        if not self.training_history['models']:
            logger.warning("No training data to analyze")
            return
        
        logger.info("Analyzing training results")
        
        # Prepare data for plots
        prompts = [model['prompt'] for model in self.training_history['models']]
        total_times = [model['generation_times']['total'] for model in self.training_history['models']]
        pc_times = [model['generation_times']['point_cloud'] for model in self.training_history['models']]
        mesh_times = [model['generation_times']['mesh'] for model in self.training_history['models']]
        texture_times = [model['generation_times']['texture'] for model in self.training_history['models']]
        
        # Create plots directory
        plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot generation times
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(prompts)), pc_times, label='Point Cloud')
        plt.bar(range(len(prompts)), mesh_times, bottom=pc_times, label='Mesh')
        plt.bar(range(len(prompts)), texture_times, bottom=[p+m for p, m in zip(pc_times, mesh_times)], label='Texture')
        plt.xlabel('Sample')
        plt.ylabel('Generation Time (s)')
        plt.title('Generation Times by Component')
        plt.legend()
        plt.xticks(range(len(prompts)), [f"Sample {i+1}" for i in range(len(prompts))], rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "generation_times.png"))
        
        # Plot mesh complexity
        plt.figure(figsize=(10, 6))
        vertices = [model['mesh_vertices'] for model in self.training_history['models']]
        faces = [model['mesh_faces'] for model in self.training_history['models']]
        plt.bar(range(len(prompts)), vertices, label='Vertices')
        plt.bar(range(len(prompts)), faces, alpha=0.7, label='Faces')
        plt.xlabel('Sample')
        plt.ylabel('Count')
        plt.title('Mesh Complexity')
        plt.legend()
        plt.xticks(range(len(prompts)), [f"Sample {i+1}" for i in range(len(prompts))], rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "mesh_complexity.png"))
        
        logger.info(f"Analysis plots saved to {plots_dir}")
        
        # Generate summary statistics
        stats = {
            'total_samples': len(self.training_history['models']),
            'average_times': {
                'point_cloud': np.mean(pc_times),
                'mesh': np.mean(mesh_times),
                'texture': np.mean(texture_times),
                'total': np.mean(total_times)
            },
            'average_complexity': {
                'vertices': np.mean(vertices),
                'faces': np.mean(faces)
            }
        }
        
        # Save statistics
        stats_path = os.path.join(self.output_dir, "training_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Training statistics saved to {stats_path}")
        return stats
