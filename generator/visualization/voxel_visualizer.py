"""
Voxel visualization utilities for 3D model generation

This module provides helper functions to visualize voxel grids and debug the
generation process in the Text-to-3D system.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
from skimage import measure
import io
import base64
from PIL import Image

def visualize_voxel_grid(voxel_grid, threshold=0.5, figsize=(10, 10), show=True):
    """
    Visualize a voxel grid as a 3D scatter plot
    
    Args:
        voxel_grid: 3D numpy array representing voxel occupancy
        threshold: Value threshold for voxel visibility
        figsize: Figure size tuple (width, height)
        show: Whether to display the plot (set False to return fig only)
        
    Returns:
        fig: The matplotlib figure (if show=False)
    """
    # Create a figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Get coordinates of filled voxels
    x, y, z = np.where(voxel_grid > threshold)
    
    # Check if there are any voxels above threshold
    if len(x) == 0:
        # No voxels above threshold, show a dummy visualization
        # Create grid of all positions to display grid shape
        grid_shape = voxel_grid.shape
        center_x, center_y, center_z = grid_shape[0] // 2, grid_shape[1] // 2, grid_shape[2] // 2
        
        # Plot grid box edges
        ax.scatter([0, 0, 0, 0, grid_shape[0], grid_shape[0], grid_shape[0], grid_shape[0]], 
                   [0, 0, grid_shape[1], grid_shape[1], 0, 0, grid_shape[1], grid_shape[1]], 
                   [0, grid_shape[2], 0, grid_shape[2], 0, grid_shape[2], 0, grid_shape[2]], 
                   c='gray', alpha=0.3, s=10)
        
        # Plot a warning text in the center
        ax.text(center_x, center_y, center_z, "Empty Grid", color='red', fontsize=16, 
                ha='center', va='center')
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Empty Voxel Grid (threshold={threshold})')
        
        # Set limits
        ax.set_xlim(0, grid_shape[0])
        ax.set_ylim(0, grid_shape[1])
        ax.set_zlim(0, grid_shape[2])
    else:
        # Get values at those coordinates for coloring
        values = voxel_grid[x, y, z]
        
        # Normalize values for coloring - handle case when all values are the same
        if values.min() == values.max():
            colors = plt.cm.viridis(np.full_like(values, 0.5))
        else:
            colors = plt.cm.viridis((values - values.min()) / (values.max() - values.min()))
        
        # Plot voxels
        ax.scatter(x, y, z, c=colors, marker='o', alpha=0.7)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Voxel Grid (threshold={threshold})')
        
        # Equal aspect ratio
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    # Show the plot or return the figure
    if show:
        plt.tight_layout()
        plt.show()
        return None
    else:
        return fig

def voxel_grid_to_image(voxel_grid, threshold=0.5, figsize=(10, 10), dpi=100):
    """
    Convert a voxel grid visualization to an image
    
    Args:
        voxel_grid: 3D numpy array representing voxel occupancy
        threshold: Value threshold for voxel visibility
        figsize: Figure size tuple (width, height)
        dpi: DPI for the rendered image
        
    Returns:
        image_base64: Base64 encoded PNG image of the visualization
    """
    try:
        fig = visualize_voxel_grid(voxel_grid, threshold, figsize, show=False)
        
        # Save figure to in-memory buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        # Convert to base64
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"Error converting voxel grid to image: {str(e)}")
        # Create a simple error image
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Error visualizing voxel grid:\n{str(e)}", 
                ha='center', va='center', fontsize=14, color='red')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Save figure to in-memory buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        # Convert to base64
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_str}"

def voxel_grid_to_mesh(voxel_grid, threshold=0.5, smooth=False):
    """
    Convert a voxel grid to a mesh using marching cubes
    
    Args:
        voxel_grid: 3D numpy array representing voxel occupancy
        threshold: Isosurface threshold
        smooth: Whether to smooth the resulting mesh
        
    Returns:
        mesh: Trimesh object representing the voxel grid or None if no voxels above threshold
    """
    # Check if there are any voxels above threshold
    if np.max(voxel_grid) <= threshold:
        print(f"Warning: No voxels above threshold {threshold}. Cannot create mesh.")
        # Create a tiny default cube as a placeholder
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ])
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 5, 6], [4, 6, 7],  # top
            [0, 1, 5], [0, 5, 4],  # front
            [2, 3, 7], [2, 7, 6],  # back
            [0, 3, 7], [0, 7, 4],  # left
            [1, 2, 6], [1, 6, 5]   # right
        ])
        
        # Scale the cube to be very small (almost invisible)
        vertices = vertices * 0.001
        
        # Place it at the center of the voxel grid
        center = np.array(voxel_grid.shape) / 2
        vertices = vertices + center
        
        # Create empty mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        return mesh
    
    try:
        # Create a padded copy of the voxel grid to ensure boundary constraints
        # Add a single cell of padding with values below threshold to ensure boundaries are respected
        padded_grid = np.pad(voxel_grid, 1, mode='constant', constant_values=0.0)
        
        # Reverse the voxel grid values - this is a workaround to force correct normal orientation
        # When values are inverted (1.0 - value), marching cubes generates faces with opposite orientation
        padded_grid = 1.0 - padded_grid
        
        # Apply marching cubes algorithm to create a surface mesh
        # Make sure voxel data has values within a valid range for marching cubes
        if np.max(padded_grid) <= (1.0 - threshold) or np.min(padded_grid) > (1.0 - threshold):
            print(f"Warning: Inverted voxel grid values not appropriate for threshold {1.0 - threshold}. Adjusting values.")
            # Adjust voxel grid to ensure we have values on both sides of the threshold
            if np.max(padded_grid) <= (1.0 - threshold):
                # Scale up the grid to ensure max values exceed threshold
                scale_factor = ((1.0 - threshold) + 0.1) / np.max(padded_grid) if np.max(padded_grid) > 0 else 1.0
                padded_grid = padded_grid * scale_factor
            if np.min(padded_grid) > (1.0 - threshold):
                # Add small offset to ensure some values below threshold
                padded_grid = padded_grid - (np.min(padded_grid) - (1.0 - threshold) + 0.1)
        
        # Now run marching cubes with adjusted values
        vertices, faces, normals, _ = measure.marching_cubes(padded_grid, level=(1.0 - threshold))
        
        # Adjust vertices to account for padding (subtract 1 from all coordinates)
        vertices = vertices - 1.0
        
        # Create mesh with correct normal orientation due to inverted grid values
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        print("Using inverted values technique for correct normal orientation")
        
        # Smooth the mesh if requested
        if smooth:
            mesh = mesh.smoothed()
        
        return mesh
    except Exception as e:
        print(f"Error in marching cubes algorithm: {str(e)}")
        # Return a default simple mesh (cube) as a fallback
        return trimesh.primitives.Box(extents=[1, 1, 1])

def save_voxel_grid_visualization(voxel_grid, filepath, threshold=0.5, figsize=(10, 10), dpi=100):
    """
    Save a visualization of a voxel grid to a file
    
    Args:
        voxel_grid: 3D numpy array representing voxel occupancy
        filepath: Path to save the image
        threshold: Value threshold for voxel visibility
        figsize: Figure size tuple (width, height)
        dpi: DPI for the rendered image
    """
    try:
        fig = visualize_voxel_grid(voxel_grid, threshold, figsize, show=False)
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Visualization saved to {filepath}")
    except Exception as e:
        print(f"Error saving voxel grid visualization: {str(e)}")
        
        # Create an error message image
        fig, ax = plt.subplots(figsize=figsize)
        if np.max(voxel_grid) <= threshold:
            message = f"Empty voxel grid (no values above threshold {threshold})"
        else:
            message = f"Error visualizing voxel grid: {str(e)}"
            
        ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=14, color='red')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Save the error message image
        try:
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            print(f"Error message image saved to {filepath}")
        except Exception as save_error:
            print(f"Failed to save error message image: {str(save_error)}")

def create_voxel_grid_slices(voxel_grid, threshold=0.5, figsize=(15, 10), show=True):
    """
    Create a 2D visualization of slices through the voxel grid
    
    Args:
        voxel_grid: 3D numpy array representing voxel occupancy
        threshold: Value threshold for voxel visibility
        figsize: Figure size tuple (width, height)
        show: Whether to display the plot (set False to return fig only)
        
    Returns:
        fig: The matplotlib figure (if show=False)
    """
    # Create a figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Get grid dimensions
    grid_shape = voxel_grid.shape
    
    # Get middle slices
    x_mid = grid_shape[0] // 2
    y_mid = grid_shape[1] // 2
    z_mid = grid_shape[2] // 2
    
    # Check if the grid is empty (all values below threshold)
    if np.max(voxel_grid) <= threshold:
        for i, (title, plane) in enumerate([
            (f'X-axis slice (x={x_mid})', 'Empty'),
            (f'Y-axis slice (y={y_mid})', 'Empty'),
            (f'Z-axis slice (z={z_mid})', 'Empty')
        ]):
            axes[i].text(0.5, 0.5, "Empty Grid", ha='center', va='center', fontsize=14, color='red')
            axes[i].set_title(title)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
    else:
        # Plot the three orthogonal slices
        axes[0].imshow(voxel_grid[x_mid, :, :].T, cmap='viridis', vmin=0, vmax=1)
        axes[0].set_title(f'X-axis slice (x={x_mid})')
        axes[0].set_xlabel('Y')
        axes[0].set_ylabel('Z')
        
        axes[1].imshow(voxel_grid[:, y_mid, :].T, cmap='viridis', vmin=0, vmax=1)
        axes[1].set_title(f'Y-axis slice (y={y_mid})')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Z')
        
        axes[2].imshow(voxel_grid[:, :, z_mid], cmap='viridis', vmin=0, vmax=1)
        axes[2].set_title(f'Z-axis slice (z={z_mid})')
        axes[2].set_xlabel('X')
        axes[2].set_ylabel('Y')
    
    plt.tight_layout()
    
    # Show the plot or return the figure
    if show:
        plt.show()
        return None
    else:
        return fig

def analyze_voxel_grid(voxel_grid, threshold=0.5):
    """
    Analyze a voxel grid and return statistics
    
    Args:
        voxel_grid: 3D numpy array representing voxel occupancy
        threshold: Value threshold for voxel count
        
    Returns:
        stats: Dictionary of statistics about the voxel grid
    """
    # Get basic stats
    occupied = voxel_grid > threshold
    occupied_count = np.sum(occupied)
    total_voxels = voxel_grid.size
    occupancy_ratio = occupied_count / total_voxels
    
    # Find the bounding box of the occupied voxels
    if occupied_count > 0:
        x, y, z = np.where(occupied)
        bbox = {
            'min_x': np.min(x),
            'max_x': np.max(x),
            'min_y': np.min(y),
            'max_y': np.max(y),
            'min_z': np.min(z),
            'max_z': np.max(z),
        }
        bbox_volume = (bbox['max_x'] - bbox['min_x'] + 1) * \
                     (bbox['max_y'] - bbox['min_y'] + 1) * \
                     (bbox['max_z'] - bbox['min_z'] + 1)
        bbox_density = occupied_count / bbox_volume
    else:
        bbox = {
            'min_x': 0,
            'max_x': 0,
            'min_y': 0,
            'max_y': 0,
            'min_z': 0,
            'max_z': 0,
        }
        bbox_volume = 0
        bbox_density = 0
    
    # Return statistics
    return {
        'shape': voxel_grid.shape,
        'min_value': float(np.min(voxel_grid)),
        'max_value': float(np.max(voxel_grid)),
        'mean_value': float(np.mean(voxel_grid)),
        'std_value': float(np.std(voxel_grid)),
        'threshold': threshold,
        'occupied_voxels': int(occupied_count),
        'total_voxels': int(total_voxels),
        'occupancy_ratio': float(occupancy_ratio),
        'bounding_box': bbox,
        'bbox_volume': int(bbox_volume),
        'bbox_density': float(bbox_density),
        'is_empty': occupied_count == 0,
    }
