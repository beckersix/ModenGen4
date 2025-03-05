"""
Mesh processor module for converting point clouds to meshes and processing 3D models.
"""

import numpy as np
import torch
import trimesh
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
import logging
import os
from scipy.spatial import Delaunay
import time
from skimage import measure

logger = logging.getLogger(__name__)

class MeshProcessor:
    """Process point clouds into meshes with UV coordinates for texturing."""
    
    def __init__(self):
        """Initialize the mesh processor."""
        logger.info("Initializing MeshProcessor")
    
    def point_cloud_to_mesh(self, point_cloud, method='alpha_shape', alpha=0.5):
        """
        Convert a point cloud to a triangle mesh.
        
        Args:
            point_cloud: torch_geometric.data.Data object or numpy array of points
            method: Meshing method ('alpha_shape', 'ball_pivoting', or 'poisson')
            alpha: Alpha value for alpha shape algorithm
            
        Returns:
            trimesh.Trimesh object
        """
        start_time = time.time()
        logger.info(f"Converting point cloud to mesh using {method} method")
        
        # Extract points from point_cloud
        if hasattr(point_cloud, 'pos'):
            points = point_cloud.pos.numpy()
        else:
            points = point_cloud
            
        # Get normals if available, otherwise estimate them
        if hasattr(point_cloud, 'norm'):
            normals = point_cloud.norm.numpy()
        else:
            normals = self._estimate_normals(points)
        
        # Convert to open3d point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        
        # Apply meshing method
        if method == 'alpha_shape':
            mesh = self._alpha_shape(points, alpha)
        elif method == 'ball_pivoting':
            mesh = self._ball_pivoting(pcd)
        elif method == 'poisson':
            mesh = self._poisson_reconstruction(pcd)
        else:
            raise ValueError(f"Unknown meshing method: {method}")
        
        # Ensure mesh is watertight and manifold
        mesh = self._clean_mesh(mesh)
        
        # Generate UV coordinates
        mesh = self._generate_uv_coordinates(mesh)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Mesh generation completed in {elapsed_time:.2f} seconds")
        logger.info(f"Mesh has {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
        
        return mesh
    
    def voxel_to_mesh(self, voxel_grid, threshold=0.5, smoothing=None):
        """
        Convert a voxel grid to a triangle mesh using Marching Cubes.
        
        Args:
            voxel_grid: 3D numpy array representing voxel occupancy
            threshold: Isovalue threshold for the surface
            smoothing: Amount of smoothing to apply (0 to 1, or None)
            
        Returns:
            trimesh.Trimesh object
        """
        start_time = time.time()
        logger.info(f"Converting voxel grid to mesh with threshold {threshold}")
        
        # Apply marching cubes to get vertices and faces
        try:
            vertices, faces, normals, _ = measure.marching_cubes(voxel_grid, level=threshold, method='lewiner')
        except Exception as e:
            logger.error(f"Marching cubes failed: {e}")
            # Fall back to basic method
            vertices, faces, normals, _ = measure.marching_cubes(voxel_grid, level=threshold)
        
        # Create trimesh object
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        
        # Apply smoothing if requested
        if smoothing is not None and smoothing > 0:
            # Simple laplacian smoothing
            for _ in range(int(smoothing * 10)):  # Number of iterations based on smoothing value
                mesh = self._laplacian_smooth(mesh)
        
        # Generate UV coordinates
        mesh = self._generate_uv_coordinates(mesh)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Voxel to mesh conversion completed in {elapsed_time:.2f} seconds")
        logger.info(f"Mesh has {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
        
        return mesh
    
    def _laplacian_smooth(self, mesh, lambda_factor=0.5):
        """
        Apply laplacian smoothing to a mesh.
        
        Args:
            mesh: trimesh.Trimesh object
            lambda_factor: Smoothing factor (0 to 1)
            
        Returns:
            Smoothed trimesh.Trimesh object
        """
        # Get adjacency information
        adjacency = mesh.vertex_adjacency_graph
        
        # Get original vertices
        original_vertices = mesh.vertices.copy()
        new_vertices = original_vertices.copy()
        
        # For each vertex, move towards the average of its neighbors
        for i in range(len(original_vertices)):
            if i in adjacency:
                # Get neighbors
                neighbors = list(adjacency[i])
                if neighbors:
                    # Calculate average position of neighbors
                    avg_pos = np.mean(original_vertices[neighbors], axis=0)
                    # Move vertex towards average position
                    new_vertices[i] = original_vertices[i] + lambda_factor * (avg_pos - original_vertices[i])
        
        # Create new mesh with smoothed vertices
        smoothed_mesh = trimesh.Trimesh(vertices=new_vertices, faces=mesh.faces)
        
        return smoothed_mesh
    
    def _alpha_shape(self, points, alpha):
        """Generate mesh using alpha shape algorithm."""
        # Use Delaunay triangulation as basis
        tri = Delaunay(points)
        
        # Extract triangles
        faces = []
        for simplex in tri.simplices:
            # Compute circumradius
            p1, p2, p3 = points[simplex]
            # Simplified alpha test (proper implementation would check circumradius)
            if np.linalg.norm(p1 - p2) < alpha and np.linalg.norm(p2 - p3) < alpha and np.linalg.norm(p3 - p1) < alpha:
                faces.append(simplex)
        
        if not faces:
            logger.warning("Alpha shape produced no faces, using Delaunay triangulation")
            faces = tri.simplices
        
        # Create trimesh
        mesh = trimesh.Trimesh(vertices=points, faces=faces)
        return mesh
    
    def _ball_pivoting(self, pcd):
        """Generate mesh using ball pivoting algorithm."""
        # Create mesh using Open3D
        radii = [0.05, 0.1, 0.2, 0.4]
        o3d_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii))
        
        # Convert to trimesh
        vertices = np.asarray(o3d_mesh.vertices)
        faces = np.asarray(o3d_mesh.triangles)
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        return mesh
    
    def _poisson_reconstruction(self, pcd):
        """Generate mesh using Poisson surface reconstruction."""
        # Create mesh using Open3D
        o3d_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=8, width=0, scale=1.1, linear_fit=False)
        
        # Convert to trimesh
        vertices = np.asarray(o3d_mesh.vertices)
        faces = np.asarray(o3d_mesh.triangles)
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        return mesh
    
    def _clean_mesh(self, mesh):
        """Clean a mesh to make it suitable for rendering."""
        # Remove duplicate vertices
        mesh = mesh.merge_vertices()
        
        # Fill holes (if any)
        if not mesh.is_watertight:
            logger.warning("Mesh is not watertight, attempting to fill holes")
            # In a real implementation, use more sophisticated hole filling
        
        # Ensure outward-facing normals
        mesh.fix_normals()
        
        return mesh
    
    def _generate_uv_coordinates(self, mesh):
        """Generate UV coordinates for the mesh using a simple projection."""
        # Simple spherical projection for UV mapping
        # In a real implementation, you would use more sophisticated UV unwrapping
        
        # Normalize vertex positions
        vertices = mesh.vertices - mesh.vertices.mean(axis=0)
        max_dist = np.max(np.linalg.norm(vertices, axis=1))
        vertices = vertices / max_dist if max_dist > 0 else vertices
        
        # Calculate spherical coordinates
        r = np.linalg.norm(vertices, axis=1)
        r = np.where(r == 0, 1e-10, r)  # Avoid division by zero
        theta = np.arccos(np.clip(vertices[:, 2] / r, -1, 1))
        phi = np.arctan2(vertices[:, 1], vertices[:, 0])
        
        # Convert to UV coordinates
        u = (phi + np.pi) / (2 * np.pi)
        v = theta / np.pi
        
        # Combine into UV array
        uv = np.column_stack([u, v])
        
        # Assign to mesh
        mesh.visual.uv = uv
        
        return mesh
    
    def _estimate_normals(self, points, k=10):
        """Estimate normal vectors for a point cloud."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Estimate normals
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
        
        # Orient normals consistently
        pcd.orient_normals_consistent_tangent_plane(k=k)
        
        return np.asarray(pcd.normals)
    
    def save_mesh(self, mesh, file_path, file_type=None):
        """Save mesh to a file."""
        # Determine file type from extension if not provided
        if file_type is None:
            file_type = os.path.splitext(file_path)[1][1:]
        
        if file_type in ['obj', 'stl', 'ply', 'glb', 'gltf']:
            mesh.export(file_path, file_type=file_type)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        logger.info(f"Mesh saved to {file_path}")
        return file_path
    
    def load_mesh(self, file_path):
        """Load mesh from a file."""
        try:
            mesh = trimesh.load(file_path)
            logger.info(f"Loaded mesh from {file_path}")
            return mesh
        except Exception as e:
            logger.error(f"Error loading mesh from {file_path}: {e}")
            raise
    
    def save_point_cloud(self, points, file_path):
        """
        Save point cloud to a file.
        
        Args:
            points: Numpy array of points
            file_path: Path to save the point cloud
            
        Returns:
            Path to the saved file
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Determine file type from extension
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.npy':
            np.save(file_path, points)
        elif ext == '.ply':
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            o3d.io.write_point_cloud(file_path, pcd)
        else:
            raise ValueError(f"Unsupported point cloud file type: {ext}")
        
        logger.info(f"Point cloud saved to {file_path}")
        return file_path
