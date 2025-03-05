"""
Dataset utilities for 3D mesh and point cloud data.
This module handles loading, processing, and augmenting 3D data for training.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import trimesh
import random
import logging
import math
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

class MeshDataset(Dataset):
    """Dataset for 3D meshes."""
    
    def __init__(self, data_dir, transform=None, voxel_resolution=64):
        """
        Initialize the mesh dataset.
        
        Args:
            data_dir: Directory containing mesh files (.obj, .ply, etc.)
            transform: Optional transform to apply to the data
            voxel_resolution: Resolution of voxel grid for conversion
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.voxel_resolution = voxel_resolution
        
        self.file_paths = []
        self.mesh_metadata = []
        
        # Find all mesh files
        extensions = ['.obj', '.ply', '.stl', '.off', '.glb', '.gltf']
        for ext in extensions:
            self.file_paths.extend(list(self.data_dir.glob(f'**/*{ext}')))
        
        # Load metadata if available
        metadata_path = self.data_dir / 'metadata.json'
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    self.mesh_metadata = json.load(f)
                logger.info(f"Loaded metadata for {len(self.mesh_metadata)} meshes")
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        
        logger.info(f"Found {len(self.file_paths)} mesh files in {data_dir}")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        try:
            # Load mesh
            mesh = trimesh.load(file_path)
            
            # Convert to voxel grid
            voxel_grid = self._mesh_to_voxel(mesh)
            
            # Get metadata if available
            metadata = {}
            if idx < len(self.mesh_metadata):
                metadata = self.mesh_metadata[idx]
            
            # Apply transform if provided
            if self.transform:
                voxel_grid = self.transform(voxel_grid)
            
            return {
                'voxel_grid': voxel_grid,
                'file_path': str(file_path),
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error loading mesh {file_path}: {e}")
            # Return a placeholder voxel grid
            return {
                'voxel_grid': torch.zeros((1, self.voxel_resolution, self.voxel_resolution, self.voxel_resolution)),
                'file_path': str(file_path),
                'metadata': {}
            }
    
    def _mesh_to_voxel(self, mesh):
        """
        Convert a mesh to a voxel grid.
        
        Args:
            mesh: Trimesh mesh object
            
        Returns:
            Voxel grid as torch tensor
        """
        # Ensure mesh is watertight
        if not mesh.is_watertight:
            mesh = mesh.fill_holes()
        
        # Normalize mesh to unit cube
        mesh.vertices -= mesh.bounding_box.centroid
        max_dim = np.max(mesh.bounding_box.extents)
        mesh.vertices /= max_dim
        
        # Voxelize the mesh
        voxels = mesh.voxelized(self.voxel_resolution)
        voxel_grid = voxels.matrix.astype(np.float32)
        
        # Convert to torch tensor
        voxel_tensor = torch.from_numpy(voxel_grid).float()
        
        # Add channel dimension
        voxel_tensor = voxel_tensor.unsqueeze(0)
        
        return voxel_tensor


class PointCloudDataset(Dataset):
    """Dataset for 3D point clouds."""
    
    def __init__(self, data_dir, transform=None, num_points=2048):
        """
        Initialize the point cloud dataset.
        
        Args:
            data_dir: Directory containing point cloud files (.npy, .ply, etc.)
            transform: Optional transform to apply to the data
            num_points: Number of points to sample from each point cloud
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.num_points = num_points
        
        self.file_paths = []
        self.point_cloud_metadata = []
        
        # Find all point cloud files
        self.file_paths.extend(list(self.data_dir.glob('**/*.npy')))
        self.file_paths.extend(list(self.data_dir.glob('**/*.ply')))
        
        # Load metadata if available
        metadata_path = self.data_dir / 'metadata.json'
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    self.point_cloud_metadata = json.load(f)
                logger.info(f"Loaded metadata for {len(self.point_cloud_metadata)} point clouds")
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        
        logger.info(f"Found {len(self.file_paths)} point cloud files in {data_dir}")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        try:
            # Load point cloud
            if file_path.suffix == '.npy':
                point_cloud = np.load(file_path)
            else:  # .ply file
                mesh = trimesh.load(file_path)
                point_cloud = mesh.vertices
            
            # Sample points if needed
            if len(point_cloud) > self.num_points:
                indices = np.random.choice(len(point_cloud), self.num_points, replace=False)
                point_cloud = point_cloud[indices]
            elif len(point_cloud) < self.num_points:
                # Duplicate points if not enough
                indices = np.random.choice(len(point_cloud), self.num_points - len(point_cloud), replace=True)
                extra_points = point_cloud[indices]
                point_cloud = np.concatenate([point_cloud, extra_points], axis=0)
            
            # Normalize to unit cube
            centroid = np.mean(point_cloud, axis=0)
            point_cloud = point_cloud - centroid
            max_dist = np.max(np.sqrt(np.sum(point_cloud**2, axis=1)))
            point_cloud = point_cloud / max_dist
            
            # Convert to torch tensor
            point_tensor = torch.from_numpy(point_cloud).float()
            
            # Get metadata if available
            metadata = {}
            if idx < len(self.point_cloud_metadata):
                metadata = self.point_cloud_metadata[idx]
            
            # Apply transform if provided
            if self.transform:
                point_tensor = self.transform(point_tensor)
            
            return {
                'points': point_tensor,
                'file_path': str(file_path),
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error loading point cloud {file_path}: {e}")
            # Return a placeholder point cloud
            return {
                'points': torch.zeros((self.num_points, 3)),
                'file_path': str(file_path),
                'metadata': {}
            }


class MeshDataset(Dataset):
    """Dataset for loading 3D meshes for GAN training."""
    
    def __init__(self, data_dir, resolution=64, transform=None):
        """
        Initialize the mesh dataset.
        
        Args:
            data_dir (str): Directory containing mesh files (.obj, .ply, etc.)
            resolution (int): Voxel grid resolution
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data_dir = data_dir
        self.resolution = resolution
        self.transform = transform
        self.logger = logging.getLogger("MeshDataset")
        
        # Get all mesh files
        self.mesh_files = []
        for extension in ['.obj', '.ply', '.stl']:
            self.mesh_files.extend(list(Path(data_dir).glob(f'**/*{extension}')))
        
        self.mesh_files = sorted([str(f) for f in self.mesh_files])
        self.logger.info(f"Found {len(self.mesh_files)} mesh files in {data_dir}")
    
    def __len__(self):
        return len(self.mesh_files)
    
    def __getitem__(self, idx):
        mesh_path = self.mesh_files[idx]
        
        try:
            # Load the mesh
            mesh = trimesh.load(mesh_path)
            
            # Convert to voxel grid
            voxel_grid = self._mesh_to_voxels(mesh)
            
            # Convert to tensor
            voxel_tensor = torch.FloatTensor(voxel_grid)
            
            if self.transform:
                voxel_tensor = self.transform(voxel_tensor)
                
            return voxel_tensor.unsqueeze(0)  # Add channel dimension
            
        except Exception as e:
            self.logger.error(f"Error loading mesh {mesh_path}: {str(e)}")
            # Return a random voxel grid as fallback
            return torch.rand(1, self.resolution, self.resolution, self.resolution)
    
    def _mesh_to_voxels(self, mesh, pad=True):
        """
        Convert a mesh to a voxel grid using trimesh's voxelization.
        
        Args:
            mesh (trimesh.Trimesh): Input mesh
            pad (bool): Whether to pad the voxel grid
            
        Returns:
            numpy.ndarray: Voxel grid representation
        """
        # Center and normalize the mesh
        mesh = mesh.copy()
        mesh.vertices -= mesh.bounding_box.centroid
        scale = max(mesh.bounding_box.extents)
        if scale > 0:
            mesh.vertices /= scale
            mesh.vertices *= 0.9  # Slight scaling to ensure it fits in the voxel grid
        
        # Voxelize the mesh
        try:
            voxelized = mesh.voxelized(pitch=2.0/self.resolution)
            # Convert to dense matrix representation
            voxel_grid = voxelized.matrix
            
            # Check if voxelization produced a valid result
            if voxel_grid is None or np.sum(voxel_grid) == 0:
                raise ValueError("Empty voxel grid")
                
        except Exception as e:
            self.logger.warning(f"Voxelization failed: {str(e)}, using fallback")
            # Create a simple cube as fallback
            fallback_grid = np.zeros((self.resolution, self.resolution, self.resolution), dtype=bool)
            center = self.resolution // 2
            size = self.resolution // 4
            fallback_grid[center-size:center+size, center-size:center+size, center-size:center+size] = True
            return fallback_grid.astype(np.float32)
        
        # Resize to target resolution if needed
        if voxel_grid.shape[0] != self.resolution:
            from scipy.ndimage import zoom
            factors = [self.resolution / s for s in voxel_grid.shape]
            voxel_grid = zoom(voxel_grid.astype(float), factors, order=1)
            voxel_grid = (voxel_grid > 0.5).astype(np.float32)
        
        return voxel_grid.astype(np.float32)


class PointCloudDataset(Dataset):
    """Dataset for loading 3D point clouds for GAN training."""
    
    def __init__(self, data_dir, num_points=2048, transform=None):
        """
        Initialize the point cloud dataset.
        
        Args:
            data_dir (str): Directory containing mesh files (.obj, .ply, etc.)
            num_points (int): Number of points to sample from each mesh
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data_dir = data_dir
        self.num_points = num_points
        self.transform = transform
        self.logger = logging.getLogger("PointCloudDataset")
        
        # Get all mesh files (we'll convert to point clouds)
        self.mesh_files = []
        for extension in ['.obj', '.ply', '.stl']:
            self.mesh_files.extend(list(Path(data_dir).glob(f'**/*{extension}')))
        
        self.mesh_files = sorted([str(f) for f in self.mesh_files])
        self.logger.info(f"Found {len(self.mesh_files)} mesh files in {data_dir}")
    
    def __len__(self):
        return len(self.mesh_files)
    
    def __getitem__(self, idx):
        mesh_path = self.mesh_files[idx]
        
        try:
            # Load the mesh and sample points
            mesh = trimesh.load(mesh_path)
            points = self._sample_points_from_mesh(mesh, self.num_points)
            
            # Convert to tensor
            point_tensor = torch.FloatTensor(points)
            
            if self.transform:
                point_tensor = self.transform(point_tensor)
                
            return point_tensor
            
        except Exception as e:
            self.logger.error(f"Error loading mesh {mesh_path}: {str(e)}")
            # Return random points as fallback
            return torch.rand(self.num_points, 3) * 2 - 1  # Range [-1, 1]
    
    def _sample_points_from_mesh(self, mesh, num_points):
        """
        Sample points from a mesh surface.
        
        Args:
            mesh (trimesh.Trimesh): Input mesh
            num_points (int): Number of points to sample
            
        Returns:
            numpy.ndarray: Point cloud with shape (num_points, 3)
        """
        # Center and normalize the mesh
        mesh = mesh.copy()
        mesh.vertices -= mesh.bounding_box.centroid
        scale = max(mesh.bounding_box.extents)
        if scale > 0:
            mesh.vertices /= scale
        
        # Sample points from the mesh surface
        try:
            points, _ = trimesh.sample.sample_surface(mesh, num_points)
        except Exception as e:
            self.logger.warning(f"Could not sample from mesh: {str(e)}, using random points")
            # Create random points as fallback
            return np.random.rand(num_points, 3) * 2 - 1
        
        # If we couldn't sample enough points, duplicate some
        if len(points) < num_points:
            indices = np.random.choice(len(points), num_points - len(points))
            extra_points = points[indices]
            points = np.vstack([points, extra_points])
        
        return points


def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    """
    Create a dataloader from a dataset.
    
    Args:
        dataset: Dataset object
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader object
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


class SampleDataGenerator:
    """Generate sample 3D data for testing and development."""
    
    def __init__(self, output_dir='sample_data'):
        """
        Initialize the sample data generator.
        
        Args:
            output_dir: Directory to save generated data
        """
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Basic shapes for sample generation
        self.basic_shapes = {
            'cube': self._generate_cube,
            'sphere': self._generate_sphere,
            'cylinder': self._generate_cylinder,
            'cone': self._generate_cone,
            'torus': self._generate_torus,
            'pyramid': self._generate_pyramid
        }
    
    def generate_basic_shapes(self):
        """
        Generate basic 3D shapes and save as mesh files.
        
        Returns:
            List of generated file paths
        """
        mesh_dir = self.output_dir / 'meshes'
        os.makedirs(mesh_dir, exist_ok=True)
        
        file_paths = []
        metadata = []
        
        for shape_name, shape_func in tqdm(self.basic_shapes.items(), desc="Generating shapes"):
            for i in range(3):  # Generate variations of each shape
                scale = 0.5 + i * 0.5  # Different sizes
                
                # Generate mesh
                mesh = shape_func(scale)
                
                # Save mesh
                file_path = mesh_dir / f"{shape_name}_{i+1}.obj"
                mesh.export(file_path)
                file_paths.append(str(file_path))
                
                # Add metadata
                metadata.append({
                    'shape': shape_name,
                    'scale': scale,
                    'vertices': len(mesh.vertices),
                    'faces': len(mesh.faces)
                })
        
        # Save metadata
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Generated {len(file_paths)} sample shapes in {mesh_dir}")
        return file_paths
    
    def _generate_cube(self, scale=1.0):
        """Generate a cube mesh."""
        return trimesh.creation.box(extents=[scale, scale, scale])
    
    def _generate_sphere(self, scale=1.0):
        """Generate a sphere mesh."""
        return trimesh.creation.icosphere(radius=scale)
    
    def _generate_cylinder(self, scale=1.0):
        """Generate a cylinder mesh."""
        return trimesh.creation.cylinder(radius=scale/2, height=scale)
    
    def _generate_cone(self, scale=1.0):
        """Generate a cone mesh."""
        return trimesh.creation.cone(radius=scale/2, height=scale)
    
    def _generate_torus(self, scale=1.0):
        """Generate a torus mesh."""
        # Use parametric equation to create a torus
        r1, r2 = scale/2, scale/4  # Major and minor radii
        n = 50  # Resolution
        
        # Parametric coordinates
        u = np.linspace(0, 2*np.pi, n)
        v = np.linspace(0, 2*np.pi, n)
        u, v = np.meshgrid(u, v)
        u, v = u.flatten(), v.flatten()
        
        # Compute points
        x = (r1 + r2 * np.cos(v)) * np.cos(u)
        y = (r1 + r2 * np.cos(v)) * np.sin(u)
        z = r2 * np.sin(v)
        
        # Create mesh using points
        vertices = np.vstack((x, y, z)).T
        
        # Create faces by connecting adjacent vertices
        faces = []
        for i in range(n-1):
            for j in range(n-1):
                idx = i * n + j
                faces.append([idx, idx+1, idx+n])
                faces.append([idx+1, idx+n+1, idx+n])
        
        return trimesh.Trimesh(vertices=vertices, faces=faces)
    
    def _generate_pyramid(self, scale=1.0):
        """Generate a pyramid mesh."""
        # Create vertices
        vertices = np.array([
            [0, 0, scale],          # Apex
            [-scale/2, -scale/2, 0], # Base corner 1
            [scale/2, -scale/2, 0],  # Base corner 2
            [scale/2, scale/2, 0],   # Base corner 3
            [-scale/2, scale/2, 0]   # Base corner 4
        ])
        
        # Create faces
        faces = np.array([
            [0, 1, 2],  # Side 1
            [0, 2, 3],  # Side 2
            [0, 3, 4],  # Side 3
            [0, 4, 1],  # Side 4
            [1, 3, 2],  # Base 1
            [1, 4, 3]   # Base 2
        ])
        
        return trimesh.Trimesh(vertices=vertices, faces=faces)


class SampleDataGenerator:
    """Generates sample 3D shapes for GAN training."""
    
    def __init__(self, output_dir):
        """
        Initialize the sample data generator.
        
        Args:
            output_dir (str): Directory to save generated samples
        """
        self.output_dir = output_dir
        self.logger = logging.getLogger("SampleDataGenerator")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_samples(self, count=100):
        """
        Generate sample 3D shapes.
        
        Args:
            count (int): Number of samples to generate
        """
        self.logger.info(f"Generating {count} sample shapes in {self.output_dir}")
        
        for i in range(count):
            # Choose a random shape type
            shape_type = random.choice(['cube', 'sphere', 'cylinder', 'torus', 'pyramid', 'combined'])
            
            # Generate the shape
            if shape_type == 'cube':
                mesh = self._generate_cube()
            elif shape_type == 'sphere':
                mesh = self._generate_sphere()
            elif shape_type == 'cylinder':
                mesh = self._generate_cylinder()
            elif shape_type == 'torus':
                mesh = self._generate_torus()
            elif shape_type == 'pyramid':
                mesh = self._generate_pyramid()
            else:  # combined
                mesh = self._generate_combined_shape()
            
            # Apply random transformations
            mesh = self._apply_random_transforms(mesh)
            
            # Save the mesh
            filename = os.path.join(self.output_dir, f"sample_{i:04d}_{shape_type}.obj")
            mesh.export(filename)
            
            if (i + 1) % 10 == 0:
                self.logger.info(f"Generated {i+1}/{count} samples")
        
        self.logger.info(f"Sample generation complete: {count} samples created in {self.output_dir}")
    
    def _generate_cube(self):
        """Generate a cube with random dimensions."""
        size_x = random.uniform(0.5, 1.5)
        size_y = random.uniform(0.5, 1.5)
        size_z = random.uniform(0.5, 1.5)
        return trimesh.creation.box(extents=[size_x, size_y, size_z])
    
    def _generate_sphere(self):
        """Generate a sphere with random radius."""
        radius = random.uniform(0.5, 1.0)
        return trimesh.creation.icosphere(radius=radius, subdivisions=3)
    
    def _generate_cylinder(self):
        """Generate a cylinder with random dimensions."""
        radius = random.uniform(0.3, 0.8)
        height = random.uniform(0.5, 2.0)
        return trimesh.creation.cylinder(radius=radius, height=height)
    
    def _generate_torus(self):
        """Generate a torus with random dimensions."""
        tube_radius = random.uniform(0.1, 0.3)
        ring_radius = random.uniform(0.5, 1.0)
        return trimesh.creation.annulus(r_min=ring_radius - tube_radius, 
                                        r_max=ring_radius + tube_radius, 
                                        height=tube_radius * 2)
    
    def _generate_pyramid(self):
        """Generate a pyramid with random dimensions."""
        base_size = random.uniform(0.5, 1.5)
        height = random.uniform(0.5, 2.0)
        
        # Create a pyramid manually
        vertices = np.array([
            [-base_size/2, -base_size/2, -height/2],  # Base corner 1
            [base_size/2, -base_size/2, -height/2],   # Base corner 2
            [base_size/2, base_size/2, -height/2],    # Base corner 3
            [-base_size/2, base_size/2, -height/2],   # Base corner 4
            [0, 0, height/2]                          # Apex
        ])
        
        faces = np.array([
            [0, 1, 4],  # Side face 1
            [1, 2, 4],  # Side face 2
            [2, 3, 4],  # Side face 3
            [3, 0, 4],  # Side face 4
            [0, 3, 2],  # Base face half 1
            [0, 2, 1]   # Base face half 2
        ])
        
        return trimesh.Trimesh(vertices=vertices, faces=faces)
    
    def _generate_combined_shape(self):
        """Generate a combined shape from multiple primitives."""
        # Create a random number of primitives
        num_primitives = random.randint(2, 4)
        meshes = []
        
        for _ in range(num_primitives):
            # Choose a random shape type
            shape_type = random.choice(['cube', 'sphere', 'cylinder', 'pyramid'])
            
            # Generate the shape
            if shape_type == 'cube':
                mesh = self._generate_cube()
            elif shape_type == 'sphere':
                mesh = self._generate_sphere()
            elif shape_type == 'cylinder':
                mesh = self._generate_cylinder()
            else:  # pyramid
                mesh = self._generate_pyramid()
            
            # Apply random position within a bounding box
            pos_x = random.uniform(-0.5, 0.5)
            pos_y = random.uniform(-0.5, 0.5)
            pos_z = random.uniform(-0.5, 0.5)
            
            transform = np.eye(4)
            transform[:3, 3] = [pos_x, pos_y, pos_z]
            mesh.apply_transform(transform)
            
            meshes.append(mesh)
        
        # Combine all meshes
        if len(meshes) > 1:
            try:
                combined = trimesh.util.concatenate(meshes)
                return combined
            except:
                # Fallback if concatenation fails
                return meshes[0]
        else:
            return meshes[0]
    
    def _apply_random_transforms(self, mesh):
        """Apply random transformations to a mesh."""
        # Random rotation
        angle_x = random.uniform(0, 2 * math.pi)
        angle_y = random.uniform(0, 2 * math.pi)
        angle_z = random.uniform(0, 2 * math.pi)
        
        rotation_matrix = trimesh.transformations.euler_matrix(angle_x, angle_y, angle_z, 'sxyz')
        mesh.apply_transform(rotation_matrix)
        
        # Random scaling (subtle)
        scale_factor = random.uniform(0.8, 1.2)
        scale_matrix = np.eye(4)
        scale_matrix[:3, :3] *= scale_factor
        mesh.apply_transform(scale_matrix)
        
        return mesh
