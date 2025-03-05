"""
Text-to-3D Manager Module

This module provides a complete implementation for generating 3D models from text descriptions
using a small language model (SLM) to interpret user input and a GAN to generate 3D shapes.
"""

import torch
import numpy as np
import logging
import json
import os
import re
import time
import traceback
import collections  # Added for OrderedDict import in model loading
import types  # Added for CodeType in pickle loading
import inspect  # Added for signature inspection
import io  # Added for BytesIO operations
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..models.gan_models import Generator3D
import torch_geometric
from torch_geometric.data import Data
from scipy.ndimage import binary_erosion, binary_dilation, zoom
from skimage import measure  # Added for marching cubes algorithm
from tqdm import tqdm
import trimesh

# Import voxel visualization utilities
from ..visualization.voxel_visualizer import (
    visualize_voxel_grid, 
    voxel_grid_to_image, 
    voxel_grid_to_mesh,
    save_voxel_grid_visualization,
    create_voxel_grid_slices,
    analyze_voxel_grid
)

logger = logging.getLogger(__name__)

class TextTo3DManager:
    """
    Manages the complete pipeline from text descriptions to 3D models
    using a small language model and GAN-based generator.
    """
    
    def __init__(self, 
                 llm_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                 gan_model_path=None,
                 latent_dim=128,
                 voxel_size=64,
                 device=None,
                 debug_mode=False,
                 trained_model_id=None):
        """
        Initialize the Text-to-3D manager.
        
        Args:
            llm_model_name: Name of the language model to use
            gan_model_path: Path to the pretrained GAN generator model
            latent_dim: Dimension of the latent vector for the GAN
            voxel_size: Size of the voxel grid (resolution)
            device: PyTorch device to use (cuda/cpu)
            debug_mode: Enable additional debugging features
            trained_model_id: ID of a trained model from database (if applicable)
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"TextTo3DManager using device: {self.device}")
        
        # Save parameters
        self.latent_dim = latent_dim
        self.voxel_size = voxel_size
        self.debug_mode = debug_mode
        
        # Create debug output directory if needed
        if self.debug_mode:
            self.debug_dir = Path("debug_output")
            self.debug_dir.mkdir(exist_ok=True)
            logger.info(f"Debug mode enabled, output will be saved to {self.debug_dir}")
        
        # Load language model
        logger.info(f"Loading language model: {llm_model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                llm_model_name, 
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
            ).to(self.device)
        except Exception as e:
            logger.error(f"Failed to load language model: {e}")
            logger.error(traceback.format_exc())
            raise
        
        # Initialize GAN generator
        logger.info("Initializing GAN generator")
        self.generator = Generator3D(latent_dim=latent_dim, output_size=voxel_size).to(self.device)
        self.using_mock_generator = False
        
        # Load pretrained GAN model if provided
        if gan_model_path:
            logger.info(f"Loading pretrained GAN model from: {gan_model_path}")
            
            # Check if a weights-only compatible version exists
            weights_only_path = Path(str(gan_model_path).replace(".pth", "_weights.pth"))
            if weights_only_path.exists():
                logger.info(f"Found weights-only compatible version at: {weights_only_path}")
                gan_model_path = weights_only_path
            
            try:
                # Try both ways to load the model
                try:
                    # First try with pickle module
                    logger.info("Attempting to load model with pickle module")
                    import pickle
                    
                    # Define a persistent_load function for pickle
                    def persistent_load(pid):
                        logger.info(f"Persistent load called with: {pid}")
                        # Handle protocol 0 (ASCII strings) vs other protocols
                        if isinstance(pid, str):
                            # Protocol 0 - must be ASCII string
                            logger.info("Protocol 0 persistent ID (ASCII string)")
                            return None
                        else:
                            # Other protocols
                            logger.info(f"Protocol {'>0'} persistent ID")
                            return None
                    
                    # Create an unpickler with persistent_load
                    with open(gan_model_path, 'rb') as f:
                        unpickler = pickle.Unpickler(f)
                        unpickler.persistent_load = persistent_load
                        checkpoint = unpickler.load()
                    logger.info("Successfully loaded model with pickle module")
                except Exception as e:
                    logger.error(f"Error loading model with pickle: {e}")
                    
                    try:
                        # Create a custom pickle module with persistent_load
                        class CustomPickle:
                            @staticmethod
                            def load(file_obj):
                                unpickler = pickle.Unpickler(file_obj)
                                # Handle both string and non-string persistent IDs
                                def safe_persistent_load(pid):
                                    logger.info(f"Custom persistent load called with: {pid}")
                                    return None
                                unpickler.persistent_load = safe_persistent_load
                                return unpickler.load()
                            
                            @staticmethod
                            def Unpickler(file_obj):
                                unpickler = pickle.Unpickler(file_obj)
                                # Handle both string and non-string persistent IDs
                                def safe_persistent_load(pid):
                                    logger.info(f"Custom persistent load called with: {pid}")
                                    return None
                                unpickler.persistent_load = safe_persistent_load
                                return unpickler
                        
                        # Then try standard PyTorch loading with custom pickle module
                        logger.info("Attempting to load model with torch.load and custom pickle module")
                        logger.warning("Using weights_only=False which allows arbitrary code execution. Only use with trusted model files!")
                        checkpoint = torch.load(gan_model_path, map_location=self.device, weights_only=False, 
                                              pickle_module=CustomPickle)
                        logger.info("Successfully loaded model with custom pickle module")
                    except Exception as e:
                        logger.error(f"Error loading model with custom pickle module: {e}")
                        
                        try:
                            # Check for specific "code must be code, not str" error
                            if "argument 'code' must be code, not str" in str(e):
                                try:
                                    logger.info("Detected code object error, attempting special loading approach")
                                    
                                    # Add a custom dispatch table to handle CodeType serialization
                                    class CodeTypeDispatcher:
                                        def __init__(self):
                                            self.dispatch = {}
                                            
                                        def register(self, code, func):
                                            self.dispatch[code] = func
                                            
                                        def dispatch_table(self):
                                            return self.dispatch
                                    
                                    code_dispatcher = CodeTypeDispatcher()
                                    
                                    def dummy_code(*args, **kwargs):
                                        # Create an empty code object as a placeholder
                                        # Handle different Python versions
                                        try:
                                            # Try Python 3.8+ signature first
                                            return types.CodeType(
                                                0, 0, 0, 0, 0, 0, b'', (), (), (), '', '', 0, b'', (), ()
                                            )
                                        except TypeError:
                                            try:
                                                # Python 3.7 and earlier
                                                return types.CodeType(
                                                    0, 0, 0, 0, 0, b'', (), (), (), '', '', 0, b'', (), ()
                                                )
                                            except TypeError:
                                                # Final fallback with fewer arguments
                                                return types.CodeType(
                                                    0, 0, 0, 0, 0, b'', (), (), (), '', '', 0, b''
                                                )
                                    
                                    # Handle the specific pickle opcodes related to code objects
                                    code_dispatcher.register(pickle.GLOBAL, lambda *args: None)
                                    
                                    # Create a special loader with custom dispatch
                                    with open(gan_model_path, 'rb') as f:
                                        buffer = f.read()
                                        
                                    # Modify buffer to handle problematic code sections if needed
                                    buffer_io = io.BytesIO(buffer)
                                    
                                    # Create a custom unpickler that avoids code object issues
                                    class CodeFixUnpickler(pickle.Unpickler):
                                        def find_class(self, module, name):
                                            if module == 'types' and name == 'CodeType':
                                                # Return a dummy factory that creates empty code objects
                                                return dummy_code
                                            try:
                                                return super().find_class(module, name)
                                            except:
                                                logger.warning(f"Could not load {module}.{name}, returning None")
                                                return None
                                        
                                        def load_code(self):
                                            # Skip code loading, return a dummy code object
                                            return dummy_code()
                                    
                                    unpickler = CodeFixUnpickler(buffer_io)
                                    checkpoint = unpickler.load()
                                    logger.info("Successfully loaded model with code object fix approach")
                                except Exception as code_err:
                                    logger.error(f"Code fix approach failed: {code_err}")
                            
                            # Then try standard PyTorch loading without weights_only
                            logger.info("Attempting to load model with torch.load and weights_only=False")
                            logger.warning("Using weights_only=False which allows arbitrary code execution. Only use with trusted model files!")
                            checkpoint = torch.load(gan_model_path, map_location=self.device, weights_only=False)
                            logger.info("Successfully loaded model with weights_only=False")
                        except Exception as e:
                            logger.error(f"Error loading model with weights_only=False: {e}")
                            
                            try:
                                # Then try standard PyTorch loading with weights_only
                                logger.info("Attempting to load model with weights_only=True")
                                checkpoint = torch.load(gan_model_path, map_location=self.device, weights_only=True)
                                logger.info("Successfully loaded model weights only")
                            except Exception as e:
                                logger.error(f"Error loading model with weights_only=True: {e}")
                                
                                try:
                                    # For older PyTorch versions that don't have weights_only parameter
                                    logger.info("Attempting to load model with legacy PyTorch (without weights_only)")
                                    checkpoint = torch.load(gan_model_path, map_location=self.device)
                                    logger.info("Successfully loaded model with legacy PyTorch")
                                except Exception as e:
                                    logger.error(f"Standard loading methods failed: {e}")
                                    
                                    # Try a more advanced approach with BytesIO to skip problematic sections
                                    try:
                                        import io
                                        import struct
                                        
                                        logger.info("Attempting bytewise loading to skip problematic sections")
                                        # Read the file as binary
                                        with open(gan_model_path, 'rb') as f:
                                            model_data = f.read()
                                        
                                        # Check for zip file signature
                                        if model_data[:4] == b'PK\x03\x04':
                                            logger.info("Model appears to be in zip format")
                                            import zipfile
                                            with zipfile.ZipFile(io.BytesIO(model_data), 'r') as z:
                                                # Extract the main model data
                                                if 'data.pkl' in z.namelist():
                                                    logger.info("Extracting data.pkl from zip")
                                                    model_data = z.read('data.pkl')
                                        
                                        # Process the model data to handle persistent IDs
                                        buffer = io.BytesIO(model_data)
                                        
                                        # Create a custom unpickler
                                        class SafeUnpickler(pickle.Unpickler):
                                            def persistent_load(self, pid):
                                                logger.info(f"Skipping persistent ID: {pid}")
                                                return None
                                            
                                            def find_class(self, module, name):
                                                # For safety, restrict the classes that can be loaded
                                                if module == 'torch.storage' and name == '_load_from_bytes':
                                                    return torch.storage._load_from_bytes
                                                if module == 'collections' and name == 'OrderedDict':
                                                    return collections.OrderedDict
                                                if module == 'torch._utils' and name == '_rebuild_tensor_v2':
                                                    return torch._utils._rebuild_tensor_v2
                                                # Accept all torch and numpy classes
                                                if module.startswith('torch') or module.startswith('numpy'):
                                                    try:
                                                        return super().find_class(module, name)
                                                    except:
                                                        logger.warning(f"Could not load {module}.{name}")
                                                        return None
                                                # Handle types module for code objects
                                                if module == 'types' and name == 'CodeType':
                                                    return types.CodeType
                                                return None  # Return None for unknown classes
                                        
                                        # Try to load with our custom unpickler
                                        unpickler = SafeUnpickler(buffer)
                                        checkpoint = unpickler.load()
                                        logger.info("Successfully loaded model with custom bytewise loader")
                                    except Exception as bytewise_err:
                                        logger.error(f"All loading attempts failed. Final error: {bytewise_err}")
                                        
                                        # Last resort: binary hex editor approach
                                        try:
                                            logger.info("Attempting binary hex editor approach")
                                            with open(gan_model_path, 'rb') as f:
                                                data = f.read()
                                            
                                            # Check for specific problematic markers and fix them
                                            
                                            # 1. Look for 'MARK' sequences in the pickle data
                                            # The c0 (MARK) opcode is problematic in some cases
                                            mark_positions = []
                                            for i in range(len(data) - 1):
                                                if data[i] == pickle.MARK[0]:  # MARK opcode
                                                    mark_positions.append(i)
                                            
                                            logger.info(f"Found {len(mark_positions)} MARK opcodes in pickle data")
                                            
                                            # 2. Look for 'c' (GLOBAL opcode) for types.CodeType
                                            codetype_positions = []
                                            code_type_marker = b'c' + b'types\n' + b'CodeType\n'
                                            
                                            pos = 0
                                            while True:
                                                pos = data.find(code_type_marker, pos)
                                                if pos == -1:
                                                    break
                                                codetype_positions.append(pos)
                                                pos += len(code_type_marker)
                                            
                                            logger.info(f"Found {len(codetype_positions)} CodeType references")
                                            
                                            # If we find problematic patterns, create a modified version
                                            if codetype_positions:
                                                logger.info("Creating modified pickle data without problematic code objects")
                                                # Create a modified pickle stream skipping code objects
                                                new_data = bytearray()
                                                skip_ranges = []
                                                
                                                # Mark ranges to skip (around code object definitions)
                                                for pos in codetype_positions:
                                                    # Find pickle MARK before the code object (conservative approach)
                                                    mark_before = -1
                                                    for m in mark_positions:
                                                        if m < pos:
                                                            mark_before = m
                                                    
                                                    # Find pickle STOP after code object (conservative approach)
                                                    stop_after = data.find(bytes([pickle.STOP]), pos)
                                                    if stop_after == -1:
                                                        stop_after = len(data)
                                                    
                                                    if mark_before != -1:
                                                        skip_ranges.append((mark_before, stop_after))
                                                
                                                # Create modified data by skipping problematic ranges
                                                last_pos = 0
                                                for start, end in sorted(skip_ranges):
                                                    # Add data before problematic range
                                                    new_data.extend(data[last_pos:start])
                                                    # Add placeholder for problematic range (simple None value)
                                                    new_data.extend(pickle.NONE)  # Use pickle.NONE opcode
                                                    last_pos = end
                                                
                                                # Add remaining data after last problematic range
                                                new_data.extend(data[last_pos:])
                                                
                                                # Try to load the modified pickle data
                                                buffer = io.BytesIO(new_data)
                                                unpickler = SafeUnpickler(buffer)
                                                checkpoint = unpickler.load()
                                                logger.info("Successfully loaded model with binary hex fixes")
                                            else:
                                                logger.warning("No CodeType references found in pickle data")
                                                raise ValueError("Binary approach ineffective for this model format")
                                        except Exception as hex_err:
                                            logger.error(f"All advanced loading approaches failed: {hex_err}")
                                            raise ValueError(f"Failed to load GAN model after all approaches: {e}")
                
                # Extract generator state dict from checkpoint
                state_dict = None
                
                # Try different key possibilities to extract state_dict
                logger.info(f"Checkpoint keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Not a dictionary'}")
                
                # Try to extract state_dict with multiple possible key structures
                if isinstance(checkpoint, dict):
                    # Common key patterns for state dicts
                    possible_keys = [
                        'generator', 
                        'netG', 
                        'model', 
                        'generator_state_dict',
                        'model_state_dict',
                        'state_dict',
                        'G_state_dict',
                        'network',
                        'net',
                        'G',
                    ]
                    
                    # First check if any of our expected keys exist
                    for key in possible_keys:
                        if key in checkpoint:
                            if isinstance(checkpoint[key], dict):
                                state_dict = checkpoint[key]
                                logger.info(f"Found state_dict under key: {key}")
                                break
                            elif hasattr(checkpoint[key], 'state_dict') and callable(getattr(checkpoint[key], 'state_dict')):
                                # Handle case where the checkpoint contains the model object
                                state_dict = checkpoint[key].state_dict()
                                logger.info(f"Extracted state_dict from model object under key: {key}")
                                break
                    
                    # If no expected keys found, check if the checkpoint itself is a state_dict
                    if state_dict is None and any(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                        state_dict = checkpoint
                        logger.info("Checkpoint itself appears to be a state_dict")
                
                # If all else fails and checkpoint is already a state dict, use it directly
                if state_dict is None and isinstance(checkpoint, dict):
                    # Check if checkpoint looks like a state dict (has tensor values)
                    if any(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                        state_dict = checkpoint
                        logger.info("Using checkpoint directly as state_dict")
                
                if state_dict is None:
                    logger.error(f"Could not extract generator state dict from checkpoint")
                    raise ValueError("Failed to extract generator state dict from checkpoint")

                logger.info(f"State dict keys: {state_dict.keys()}")
                
                # Log metadata if available
                if isinstance(checkpoint, dict) and "metadata" in checkpoint:
                    logger.info("Found metadata in checkpoint")
                    try:
                        if "model_config" in checkpoint["metadata"]:
                            config = checkpoint["metadata"]["model_config"]
                            logger.info(f"Model config: latent_dim={config.get('latent_dim')}, "
                                        f"voxel_size={config.get('voxel_size')}, "
                                        f"model_type={config.get('model_type')}")
                        if "training_info" in checkpoint["metadata"]:
                            info = checkpoint["metadata"]["training_info"]
                            logger.info(f"Training info: epochs={info.get('epochs')}, "
                                        f"batch_size={info.get('batch_size')}")
                    except Exception as e:
                        logger.warning(f"Error processing metadata: {e}")
                
                # Try to load the model with adaptation for mismatched dimensions
                try:
                    if state_dict is not None:
                        self._load_adapted_state_dict(state_dict)
                        logger.info("Adapted GAN model loaded successfully")
                    else:
                        raise ValueError("Could not extract a valid state dictionary from the checkpoint")
                except Exception as e:
                    logger.error(f"Failed to load adapted model: {e}")
                    logger.error(traceback.format_exc())
                    logger.warning("Using mock generator due to loading failure")
                    self.using_mock_generator = True
                    
            except Exception as e:
                logger.error(f"Failed to load GAN model: {e}")
                logger.error(traceback.format_exc())
                logger.warning("Falling back to mock generator for development")
                self.using_mock_generator = True
                # We won't raise the exception here, just continue with an untrained model
        else:
            logger.warning("No GAN model path provided, using untrained generator")
            self.using_mock_generator = True
        
        # Set generator to evaluation mode
        self.generator.eval()
        
        # Shape attributes for feature vector generation
        self.shape_attributes_file = Path(__file__).parent / "shape_attributes.json"

        if self.shape_attributes_file.exists():
            with open(self.shape_attributes_file, 'r') as f:
                self.shape_attributes = json.load(f)
            logger.info(f"Loaded shape attributes from {self.shape_attributes_file}")
        else:
            self.shape_attributes = {
                "basic_shapes": ["sphere", "cube", "cylinder", "cone", "torus", "plane"],
                "sizes": ["small", "medium", "large"],
                "textures": ["smooth", "rough", "bumpy"],
                "colors": ["red", "green", "blue", "yellow", "white", "black"],
                "features": ["hollow", "solid", "sharp", "rounded", "flat", "curved"]
            }
            logger.warning("Shape attributes file not found, using defaults")
        
        logger.info("TextTo3DManager initialized successfully")
    
    def _load_adapted_state_dict(self, state_dict):
        """
        Load a state dictionary with dimension adaptation for mismatched tensors.
        
        Args:
            state_dict: The state dictionary to load
            
        Returns:
            bool: True if loading was successful, False otherwise
        """
        # Create a new state dict for the adapted weights
        adapted_state_dict = {}
        
        # Get the current model state dict for reference
        model_state_dict = self.generator.state_dict()
        
        # Iterate through all keys in the model state dict
        for key in model_state_dict:
            if key in state_dict:
                # Check if shapes match
                current_shape = model_state_dict[key].shape
                loaded_shape = state_dict[key].shape
                
                if current_shape == loaded_shape:
                    # Shapes match, use the loaded weights
                    adapted_state_dict[key] = state_dict[key]
                else:
                    # Shapes don't match, try to adapt
                    logger.info(f"Adapting weights for {key}: {loaded_shape} -> {current_shape}")
                    
                    # Handle projection layer (typically handles latent_dim mismatch)
                    if key == "projection.weight":
                        # Loaded: [32768, 100], Current: [32768, 128]
                        # Expand or trim the second dimension
                        if loaded_shape[1] < current_shape[1]:
                            # Expand with zeros
                            expanded = torch.zeros(current_shape, device=self.device)
                            expanded[:, :loaded_shape[1]] = state_dict[key]
                            adapted_state_dict[key] = expanded
                        else:
                            # Trim
                            adapted_state_dict[key] = state_dict[key][:, :current_shape[1]]
                    
                    # Handle convolutional layer weights
                    elif "layers" in key and ".weight" in key and len(current_shape) == 5:
                        # For 3D convolutional layers
                        if "layers.2.0.weight" in key:
                            # Special case for problematic layer we saw in the error
                            # Loaded: [128, 1, 4, 4, 4], Current: [128, 64, 4, 4, 4]
                            if loaded_shape[1] < current_shape[1]:
                                # Expand the input channels
                                expanded = torch.zeros(current_shape, device=self.device)
                                expanded[:, 0:1] = state_dict[key]  # Copy the single channel to first channel
                                # Replicate to other channels with small random variation for better features
                                for i in range(1, current_shape[1]):
                                    expanded[:, i:i+1] = state_dict[key] * (0.9 + 0.2 * torch.rand(1, device=self.device))
                                adapted_state_dict[key] = expanded
                            else:
                                # Use first N channels
                                adapted_state_dict[key] = state_dict[key][:, :current_shape[1]]
                        else:
                            # Try a general adaptation for other convolution layers
                            try:
                                # Handle different number of output channels (first dim)
                                if loaded_shape[0] != current_shape[0]:
                                    if loaded_shape[0] < current_shape[0]:
                                        # Expand output channels
                                        expanded = torch.zeros(current_shape, device=self.device)
                                        expanded[:loaded_shape[0]] = state_dict[key]
                                        adapted_state_dict[key] = expanded
                                    else:
                                        # Use first N channels
                                        adapted_state_dict[key] = state_dict[key][:current_shape[0]]
                                
                                # Handle different number of input channels (second dim)
                                elif loaded_shape[1] != current_shape[1]:
                                    if loaded_shape[1] < current_shape[1]:
                                        # Expand input channels
                                        expanded = torch.zeros(current_shape, device=self.device)
                                        expanded[:, :loaded_shape[1]] = state_dict[key]
                                        adapted_state_dict[key] = expanded
                                    else:
                                        # Use first N channels
                                        adapted_state_dict[key] = state_dict[key][:, :current_shape[1]]
                                else:
                                    # Same number of channels but different kernel size
                                    # Use a complex padding/trimming strategy (not implemented here)
                                    # For now, just initialize with random
                                    logger.warning(f"Complex shape mismatch for {key}, using random initialization")
                                    adapted_state_dict[key] = model_state_dict[key]
                            except Exception as e:
                                logger.warning(f"Failed to adapt {key}: {e}")
                                adapted_state_dict[key] = model_state_dict[key]
                    
                    # Handle bias terms
                    elif ".bias" in key:
                        if loaded_shape[0] < current_shape[0]:
                            # Expand
                            expanded = torch.zeros(current_shape, device=self.device)
                            expanded[:loaded_shape[0]] = state_dict[key]
                            adapted_state_dict[key] = expanded
                        else:
                            # Trim
                            adapted_state_dict[key] = state_dict[key][:current_shape[0]]
                    
                    # For other layers (batch norm, etc.)
                    else:
                        logger.warning(f"Cannot adapt {key} with shape mismatch: {loaded_shape} vs {current_shape}")
                        # Use the model's random initialization
                        adapted_state_dict[key] = model_state_dict[key]
            else:
                # Key doesn't exist in loaded state dict, keep model's random initialization
                logger.info(f"Missing key in state dict: {key}")
                adapted_state_dict[key] = model_state_dict[key]
        
        # Load the adapted weights
        self.generator.load_state_dict(adapted_state_dict, strict=False)
        return True
    
    def generate_from_text(self, prompt, output_formats=None, detail_level=3, debug=False):
        """
        Generate a 3D model from a text prompt.
        
        Args:
            prompt: Text description of the desired 3D model
            output_formats: List of output formats ("voxel", "mesh", "point_cloud", "debug_image")
            detail_level: Level of detail (1-5)
            debug: Enable debugging for this specific generation
            
        Returns:
            Dictionary containing the generated model in requested formats
            and metadata about the generation process
        """
        if output_formats is None:
            output_formats = ["voxel", "mesh", "point_cloud"]
        
        # Enable debugging for this generation if requested
        debug_this_generation = debug or self.debug_mode
        
        # Record start time
        start_time = time.time()
        logger.info(f"Generating 3D model from prompt: '{prompt}'")
        
        try:
            # Step 1: Process the text with the language model
            self.shape_info = self._extract_shape_from_text(prompt)
            logger.info(f"Extracted shape information: {self.shape_info}")
            
            # Step 2: Convert shape information to latent vector
            latent_vector = self._convert_info_to_latent(self.shape_info)
            
            # Check if we should use procedural shape generation for certain shapes
            basic_shape = self.shape_info.get("basic_shape", "").lower()
            if basic_shape in ["sphere", "cylinder", "cone", "torus"]:
                logger.info(f"Using procedural generation for {basic_shape} shape")
                voxel_grid = self._generate_procedural_shape(basic_shape, latent_vector)
            else:
                # Step 3: Generate voxel grid with the GAN
                voxel_grid = self._generate_voxel_grid(latent_vector)
            
            # Debug: Analyze and visualize the voxel grid
            if debug_this_generation:
                # Analyze voxel grid
                grid_stats = analyze_voxel_grid(voxel_grid)
                logger.info(f"Voxel grid analysis: {grid_stats}")
                
                # Save voxel grid visualizations if in debug mode
                timestamp = int(time.time())
                safe_prompt = re.sub(r'[^a-zA-Z0-9]', '_', prompt)[:30]
                
                # Create debug output directory if it doesn't exist
                debug_dir = self.debug_dir if hasattr(self, 'debug_dir') else Path("debug_output")
                debug_dir.mkdir(exist_ok=True)
                
                # Save 3D visualization
                visualization_path = debug_dir / f"{safe_prompt}_{timestamp}_voxel_3d.png"
                save_voxel_grid_visualization(voxel_grid, visualization_path)
                logger.info(f"Saved 3D voxel visualization to {visualization_path}")
                
                # Save 2D slices
                slices_fig = create_voxel_grid_slices(voxel_grid, show=False)
                slices_path = debug_dir / f"{safe_prompt}_{timestamp}_voxel_slices.png"
                slices_fig.savefig(slices_path, dpi=100, bbox_inches='tight')
                logger.info(f"Saved voxel grid slices to {slices_path}")
        
            # Step 4: Process outputs in requested formats
            results = {"metadata": {}}
            
            # Include voxel grid if requested
            if "voxel" in output_formats:
                results["voxel_grid"] = voxel_grid
            
            # Generate debug image if requested
            if "debug_image" in output_formats:
                results["debug_image"] = voxel_grid_to_image(voxel_grid)
            
            # Generate point cloud if requested
            if "point_cloud" in output_formats:
                num_points = 2048 * detail_level
                point_cloud = self._convert_to_point_cloud(voxel_grid, num_points)
                results["point_cloud"] = point_cloud
            
            # Generate mesh if requested
            if "mesh" in output_formats:
                mesh = self._convert_to_mesh(voxel_grid, detail_level)
                results["mesh"] = mesh
            
            # Add metadata
            generation_time = time.time() - start_time
            results["metadata"] = {
                "prompt": prompt,
                "shape_info": self.shape_info,
                "detail_level": detail_level,
                "voxel_resolution": self.voxel_size,
                "generation_time": generation_time,
                "used_mock_generator": self.using_mock_generator,
                "voxel_stats": analyze_voxel_grid(voxel_grid) if debug_this_generation else None
            }
            
            if "point_cloud" in results:
                results["metadata"]["point_count"] = len(results["point_cloud"].pos)
            
            if "mesh" in results:
                # Handle mesh as dictionary with vertices and faces keys
                results["metadata"]["vertex_count"] = len(results["mesh"]["vertices"])
                results["metadata"]["face_count"] = len(results["mesh"]["faces"])
            
            logger.info(f"3D model generated in {generation_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            logger.error(f"Error generating 3D model: {e}")
            logger.error(traceback.format_exc())
            
            # Return error information in the results
            return {
                "error": str(e),
                "metadata": {
                    "prompt": prompt,
                    "detail_level": detail_level,
                    "generation_time": time.time() - start_time,
                    "error": str(e),
                    "error_traceback": traceback.format_exc()
                }
            }
    
    def _generate_voxel_grid(self, latent_vector):
        """
        Generate a voxel grid from a latent vector using the generator model.
        
        Args:
            latent_vector: Latent vector input for the generator (can be tensor or numpy array)
            
        Returns:
            Voxel grid as a numpy array of shape [voxel_size, voxel_size, voxel_size]
        """
        try:
            # Convert to PyTorch tensor if it's not already
            if not isinstance(latent_vector, torch.Tensor):
                latent_tensor = torch.from_numpy(latent_vector).float().to(self.device)
            else:
                latent_tensor = latent_vector.to(self.device)
            
            # Ensure proper batch dimension
            if len(latent_tensor.shape) == 1:
                latent_tensor = latent_tensor.unsqueeze(0)  # [1, latent_dim]
            
            # Generate voxel grid using the generator
            with torch.no_grad():
                voxel_grid = self.generator(latent_tensor)
            
            # Convert to numpy and remove extra dimensions
            voxel_grid = voxel_grid.squeeze().cpu().numpy()
            
            if self.debug_mode:
                # Save visualization of voxel grid
                visualization_path = self.debug_dir / "voxel_grid_generated.png"
                self._visualize_voxel_grid(voxel_grid, visualization_path)
                logger.info(f"Voxel grid visualization saved to {visualization_path}")
            
            if voxel_grid.size == 0 or np.all(voxel_grid < 0.1):
                logger.warning("Generator produced empty or near-empty voxel grid, using procedural shape instead")
                return self._generate_procedural_shape(latent_vector)
            
            return voxel_grid
                
        except Exception as e:
            logger.error(f"Error generating voxel grid: {e}")
            logger.error(traceback.format_exc())
            logger.warning("Falling back to procedural shape generation")
            return self._generate_procedural_shape(latent_vector)
    
    def _generate_procedural_shape(self, shape_type, latent_vector=None):
        """
        Generate a procedural 3D shape voxel grid.
        
        Args:
            shape_type: Type of shape to generate
            latent_vector: Optional latent vector to use for size and variation
            
        Returns:
            numpy.ndarray: Voxel grid of the shape
        """
        # Initialize an empty voxel grid
        voxel_grid = np.zeros((self.voxel_size, self.voxel_size, self.voxel_size))
        
        # Calculate the center of the voxel grid
        center = self.voxel_size // 2
        
        # Determine a base radius - make it proportional to the voxel size
        # so shapes don't get too small or too large
        base_radius = self.voxel_size * 0.35  # 35% of the voxel grid size
        
        # Add randomness to the size using the latent vector's first value if available
        size_factor = 1.0
        if latent_vector is not None:
            try:
                # Extract a single scalar value from the latent vector's first element
                if isinstance(latent_vector, torch.Tensor):
                    size_value = latent_vector[0, 0].item() if latent_vector.dim() > 1 else latent_vector[0].item()
                else:
                    size_value = float(latent_vector[0])
                
                # Convert the size value to a factor between 0.7 and 1.3
                size_factor = 1.0 + (size_value * 0.3)  # Range: 0.7 to 1.3
            except (IndexError, ValueError, TypeError) as e:
                logger.warning(f"Could not use latent vector for size: {e}")
                size_factor = 1.0
        
        # Apply the size factor to the radius
        radius = base_radius * size_factor
        logger.info(f"Generating procedural {shape_type} with radius {radius} (size factor: {size_factor})")
        
        # Generate different shapes
        if shape_type == "sphere":
            # Generate a sphere with smoother edges
            for x in range(self.voxel_size):
                for y in range(self.voxel_size):
                    for z in range(self.voxel_size):
                        # Calculate distance from center
                        distance = np.sqrt((x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2)
                        # Apply sigmoid-like falloff at the sphere's surface
                        # This creates a smoother transition at the edge
                        value = 1.0 - 1.0 / (1.0 + np.exp(-(distance - radius) * 2))
                        voxel_grid[x, y, z] = value
        
        elif shape_type == "cube":
            # Generate a cube with slightly rounded edges
            cube_size = radius
            for x in range(self.voxel_size):
                for y in range(self.voxel_size):
                    for z in range(self.voxel_size):
                        # Check if within cube bounds with some smoothing at edges
                        dx = abs(x - center)
                        dy = abs(y - center)
                        dz = abs(z - center)
                        
                        # Solid inside
                        if dx <= cube_size and dy <= cube_size and dz <= cube_size:
                            # Calculate distance from edge for smoothing
                            edge_dx = cube_size - dx
                            edge_dy = cube_size - dy
                            edge_dz = cube_size - dz
                            min_edge = min(edge_dx, edge_dy, edge_dz)
                            
                            # Smooth falloff near edges (smoother for smaller edge distances)
                            if min_edge < 3:
                                falloff = min_edge / 3.0
                                voxel_grid[x, y, z] = max(0.2, falloff)
                            else:
                                voxel_grid[x, y, z] = 1.0
        
        elif shape_type == "cylinder":
            # Generate a cylinder with rounded edges
            height = radius * 2  # Height of cylinder
            for x in range(self.voxel_size):
                for y in range(self.voxel_size):
                    for z in range(self.voxel_size):
                        # Calculate distance from central axis (y-axis)
                        axial_distance = np.sqrt((x - center) ** 2 + (z - center) ** 2)
                        height_distance = abs(y - center)
                        
                        # Apply sigmoid falloff at the edges
                        if height_distance <= height / 2:
                            # Inside height bounds, check radial distance
                            radial_value = 1.0 - 1.0 / (1.0 + np.exp(-(axial_distance - radius) * 2))
                            
                            # Apply a smoother transition at the top/bottom
                            height_edge = (height / 2) - height_distance
                            if height_edge < 3:
                                height_falloff = height_edge / 3.0
                                edge_value = max(0.2, height_falloff)
                                # Blend the radial and height values
                                value = min(radial_value, edge_value)
                            else:
                                value = radial_value
                                
                            voxel_grid[x, y, z] = value
        
        elif shape_type == "cone":
            # Generate a cone with smooth edges
            height = radius * 2  # Height of cone
            for x in range(self.voxel_size):
                for y in range(self.voxel_size):
                    for z in range(self.voxel_size):
                        # Calculate normalized height from bottom to top
                        y_normalized = (y - (center - height/2)) / height
                        
                        # Ensure y is within height bounds
                        if 0 <= y_normalized <= 1:
                            # Calculate current radius at this height (linear decrease)
                            current_radius = radius * (1 - y_normalized)
                            
                            # Calculate distance from central axis at this height
                            axial_distance = np.sqrt((x - center) ** 2 + (z - center) ** 2)
                            
                            # Check if within the cone
                            if axial_distance <= current_radius:
                                # Apply smooth falloff near edges
                                edge_distance = current_radius - axial_distance
                                if edge_distance < 3:
                                    value = max(0.2, edge_distance / 3.0)
                                else:
                                    value = 1.0
                                
                                # Also smooth at the very tip and base
                                if y_normalized < 0.1:  # Base
                                    base_factor = y_normalized / 0.1
                                    value = min(value, base_factor)
                                elif y_normalized > 0.9:  # Tip
                                    tip_factor = (1 - y_normalized) / 0.1
                                    value = min(value, tip_factor)
                                    
                                voxel_grid[x, y, z] = value
        
        elif shape_type == "torus":
            # Generate a torus with smooth edges
            major_radius = radius * 0.75  # Distance from center to center of tube
            minor_radius = radius * 0.25  # Radius of the tube
            
            for x in range(self.voxel_size):
                for y in range(self.voxel_size):
                    for z in range(self.voxel_size):
                        # Calculate distance from the torus central axis (ignoring y)
                        central_distance = np.sqrt((x - center) ** 2 + (z - center) ** 2)
                        
                        # Calculate distance from the torus ring
                        ring_distance = abs(central_distance - major_radius)
                        
                        # Calculate 3D distance from the torus surface
                        point_distance = np.sqrt(ring_distance ** 2 + (y - center) ** 2)
                        
                        # Apply sigmoid-like falloff for smoother surface
                        value = 1.0 - 1.0 / (1.0 + np.exp(-(point_distance - minor_radius) * 2))
                        voxel_grid[x, y, z] = value
        
        else:
            # Default to a simple cube
            logger.warning(f"Unknown shape type: {shape_type}, defaulting to cube")
            cube_size = radius
            for x in range(self.voxel_size):
                for y in range(self.voxel_size):
                    for z in range(self.voxel_size):
                        # Check if within cube bounds
                        if (abs(x - center) <= cube_size and
                            abs(y - center) <= cube_size and
                            abs(z - center) <= cube_size):
                            voxel_grid[x, y, z] = 1.0
        
        logger.info(f"Procedural {shape_type} generated: value range {np.min(voxel_grid):.3f} to {np.max(voxel_grid):.3f}")
        return voxel_grid
    
    def _visualize_voxel_grid(self, voxel_grid, output_path):
        """
        Create a visualization of the voxel grid and save it to an image file.
        
        Args:
            voxel_grid: 3D numpy array containing the voxel grid
            output_path: Path where the visualization will be saved
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            # Create a figure with multiple views of the voxel grid
            fig = plt.figure(figsize=(12, 8))
            
            # Create three different projections (front, side, top)
            # Front view (YZ projection)
            ax1 = fig.add_subplot(131)
            front_projection = np.max(voxel_grid, axis=0)
            ax1.imshow(front_projection.T, cmap='viridis', origin='lower')
            ax1.set_title("Front View (YZ)")
            
            # Side view (XZ projection)
            ax2 = fig.add_subplot(132)
            side_projection = np.max(voxel_grid, axis=1)
            ax2.imshow(side_projection.T, cmap='viridis', origin='lower')
            ax2.set_title("Side View (XZ)")
            
            # Top view (XY projection)
            ax3 = fig.add_subplot(133)
            top_projection = np.max(voxel_grid, axis=2)
            ax3.imshow(top_projection, cmap='viridis', origin='lower')
            ax3.set_title("Top View (XY)")
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close(fig)
            
            # Create a 3D visualization of non-zero voxels
            if np.count_nonzero(voxel_grid) > 0:
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                # Get coordinates of all non-zero voxels
                threshold = 0.5  # Only show voxels with values >= 0.5
                x, y, z = np.where(voxel_grid >= threshold)
                
                # Get colors based on voxel values
                colors = plt.cm.viridis(voxel_grid[x, y, z])
                
                # Plot each voxel as a small cube
                size = 0.8  # Size of each voxel cube
                ax.scatter(x, y, z, c=colors, marker='o', alpha=0.7, s=100*size)
                
                # Set limits and labels
                vsize = voxel_grid.shape[0]
                ax.set_xlim(0, vsize)
                ax.set_ylim(0, vsize)
                ax.set_zlim(0, vsize)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title("3D Voxel Visualization")
                
                # Save the 3D figure
                output_path_3d = str(output_path).replace('.png', '_3d.png')
                plt.tight_layout()
                plt.savefig(output_path_3d)
                plt.close(fig)
        
        except Exception as e:
            logger.error(f"Error creating voxel visualization: {e}")
            logger.error(traceback.format_exc())
    
    def _extract_shape_from_text(self, prompt):
        """
        Extract shape information from the text prompt using the language model.
        
        Args:
            prompt: Text description of the 3D model
            
        Returns:
            Dictionary with shape information
        """
        # Create a structured prompt for the language model
        system_prompt = "You are a 3D model interpreter. Extract key features from the text to help generate a 3D model."
        instruction = f"""Extract the following information from this text description of a 3D object:

1. Basic shape (e.g., sphere, cube, cylinder, cone, torus, plane)
2. Size (small, medium, large)
3. Main colors
4. Texture (smooth, rough, bumpy)
5. Any distinctive features or parts

Text description: {prompt}

Output as JSON with these fields only: "basic_shape", "size", "colors", "texture", "features"."""

        # Format prompt based on model type
        if hasattr(self, 'tokenizer') and hasattr(self.tokenizer, 'name_or_path') and "TinyLlama" in self.tokenizer.name_or_path:
            formatted_prompt = f"<|system|>\n{system_prompt}\n<|reserved_special_token_1|>\n{instruction}"
        else:
            formatted_prompt = f"{system_prompt}\n\n{instruction}"
        
        try:
            # Get response from language model
            input_ids = self.tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(self.device)
            response = self.llm_model.generate(input_ids, max_length=256)
            response_text = self.tokenizer.decode(response[0], skip_special_tokens=True)
            
            logger.debug(f"Raw LLM response: {response_text}")
            
            # Parse JSON response
            try:
                # Try to find and extract JSON from the response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    response_json = json_match.group(0)
                    # Attempt to clean up the JSON if needed
                    response_json = response_json.replace("'", '"')  # Replace single quotes with double quotes
                    shape_info = json.loads(response_json)
                else:
                    # Fallback: try to extract shape information from non-JSON response
                    logger.warning("No JSON found in response, using pattern matching fallback")
                    shape_info = self._extract_shape_info_from_text(response_text, prompt)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.debug(f"Problematic JSON: {response_json if 'response_json' in locals() else 'None'}")
                # Fallback to pattern matching
                shape_info = self._extract_shape_info_from_text(response_text, prompt)
            
            # Ensure the shape_info has the expected structure
            shape_info = self._validate_shape_info(shape_info)
            
            return shape_info
            
        except Exception as e:
            logger.error(f"Error extracting shape from text: {e}")
            logger.error(traceback.format_exc())
            # Return default shape info
            return self._validate_shape_info({})
    
    def _extract_shape_info_from_text(self, text, original_prompt):
        """
        Extract shape information from raw text using pattern matching.
        Used as a fallback when JSON parsing fails.
        
        Args:
            text: Raw text from language model
            original_prompt: Original user prompt
            
        Returns:
            Dictionary with shape information
        """
        # Initialize default shape info
        shape_info = {
            "basic_shape": "cube",
            "size": "medium",
            "colors": ["blue"],
            "texture": "smooth",
            "features": []
        }
        
        # Basic shape detection
        basic_shapes = ["sphere", "cube", "cylinder", "cone", "torus", "plane", "pyramid", "box"]
        for shape in basic_shapes:
            if re.search(r'\b' + shape + r'\b', text.lower()) or re.search(r'\b' + shape + r'\b', original_prompt.lower()):
                shape_info["basic_shape"] = shape
                break
                
        # Size detection
        sizes = ["small", "medium", "large", "tiny", "huge", "big"]
        for size in sizes:
            if re.search(r'\b' + size + r'\b', text.lower()) or re.search(r'\b' + size + r'\b', original_prompt.lower()):
                shape_info["size"] = size
                break
                
        # Color detection
        colors = ["red", "green", "blue", "yellow", "white", "black", "purple", "orange", 
                "gray", "grey", "brown", "pink", "gold", "silver"]
        detected_colors = []
        for color in colors:
            if re.search(r'\b' + color + r'\b', text.lower()) or re.search(r'\b' + color + r'\b', original_prompt.lower()):
                detected_colors.append(color)
        if detected_colors:
            shape_info["colors"] = detected_colors
                
        # Texture detection
        textures = ["smooth", "rough", "bumpy", "shiny", "matte", "glossy", "textured", "soft", "hard"]
        for texture in textures:
            if re.search(r'\b' + texture + r'\b', text.lower()) or re.search(r'\b' + texture + r'\b', original_prompt.lower()):
                shape_info["texture"] = texture
                break
                
        # Feature detection
        features = ["hollow", "solid", "sharp", "rounded", "flat", "curved", "transparent", 
                  "opaque", "perforated", "symmetrical", "asymmetrical"]
        detected_features = []
        for feature in features:
            if re.search(r'\b' + feature + r'\b', text.lower()) or re.search(r'\b' + feature + r'\b', original_prompt.lower()):
                detected_features.append(feature)
        if detected_features:
            shape_info["features"] = detected_features
            
        logger.info(f"Extracted shape info via pattern matching: {shape_info}")
        return shape_info
        
    def _validate_shape_info(self, shape_info):
        """
        Validate and normalize shape information structure.
        
        Args:
            shape_info: Dictionary with shape information
            
        Returns:
            Validated and normalized shape information
        """
        # Create a new dict with default values
        validated = {
            "basic_shape": "cube",
            "size": "medium",
            "colors": ["blue"],
            "texture": "smooth",
            "features": []
        }
        
        # Update with provided values if they exist and are valid
        if "basic_shape" in shape_info and isinstance(shape_info["basic_shape"], str):
            validated["basic_shape"] = shape_info["basic_shape"].lower()
            
        if "size" in shape_info and isinstance(shape_info["size"], str):
            validated["size"] = shape_info["size"].lower()
            
        if "colors" in shape_info:
            if isinstance(shape_info["colors"], list):
                validated["colors"] = [c.lower() for c in shape_info["colors"] if isinstance(c, str)]
            elif isinstance(shape_info["colors"], str):
                # Handle case where colors is a string instead of a list
                validated["colors"] = [shape_info["colors"].lower()]
                
        if "texture" in shape_info and isinstance(shape_info["texture"], str):
            validated["texture"] = shape_info["texture"].lower()
            
        if "features" in shape_info:
            if isinstance(shape_info["features"], list):
                validated["features"] = [f.lower() for f in shape_info["features"] if isinstance(f, str)]
            elif isinstance(shape_info["features"], str):
                # Handle case where features is a string instead of a list
                validated["features"] = [shape_info["features"].lower()]
        
        return validated
        
    def _convert_info_to_latent(self, shape_info):
        """
        Convert shape information to a latent vector that can be used as input to the generator.
        
        Args:
            shape_info: Dictionary with shape information extracted from text
            
        Returns:
            torch.Tensor: Latent vector
        """
        # For simplicity in this initial implementation, we'll generate a random latent vector
        # biased towards the basic shape and features detected
        
        # Initialize a random latent vector
        z = torch.randn(1, self.latent_dim, device=self.device)
        
        # Apply bias based on basic shape
        basic_shape = shape_info.get("basic_shape", "cube").lower()
        shape_weights = None
        
        # Check if we have pre-defined shape weights
        if hasattr(self, 'shape_attributes') and 'shape_weights' in self.shape_attributes:
            if basic_shape in self.shape_attributes["shape_weights"]:
                shape_weights = torch.tensor(self.shape_attributes["shape_weights"][basic_shape], 
                                         device=self.device)
        
        # If no pre-defined weights, create basic weights based on shape type
        if shape_weights is None:
            # Simple mapping from basic shapes to indices
            shape_to_idx = {"sphere": 0, "cube": 1, "cylinder": 2, "cone": 3, "torus": 4, "plane": 5}
            shape_idx = shape_to_idx.get(basic_shape, 1)  # Default to cube (1) if shape not found
            
            # Create a one-hot-like encoding for the first 6 dimensions
            shape_weights = torch.zeros(min(6, self.latent_dim), device=self.device)
            if shape_idx < len(shape_weights):
                shape_weights[shape_idx] = 1.0
        
        # Apply shape bias to first few dimensions
        bias_dims = min(6, self.latent_dim)
        actual_weights_dims = min(bias_dims, len(shape_weights) if shape_weights is not None else 0)
        
        if actual_weights_dims > 0:
            z[0, :actual_weights_dims] = z[0, :actual_weights_dims] * 0.3 + shape_weights[:actual_weights_dims] * 0.7
        
        # Apply bias based on texture
        texture = shape_info.get("texture", "smooth").lower()
        texture_weights = None
        
        # Check if we have pre-defined texture weights
        if hasattr(self, 'shape_attributes') and 'feature_vectors' in self.shape_attributes:
            if texture in self.shape_attributes["feature_vectors"]:
                texture_weights = torch.tensor(self.shape_attributes["feature_vectors"][texture], 
                                           device=self.device)
        
        # Apply texture bias to next dimensions if we have weights
        if texture_weights is not None:
            start_dim = min(6, self.latent_dim)
            end_dim = min(12, self.latent_dim)
            if end_dim > start_dim:
                bias_dims = end_dim - start_dim
                feature_dims = min(bias_dims, len(texture_weights))
                z[0, start_dim:start_dim+feature_dims] = z[0, start_dim:start_dim+feature_dims] * 0.3 + texture_weights[:feature_dims] * 0.7
        
        # Important: Add a scalar value in the first position for procedural shape generation
        # This will be used for size calculation
        if z.shape[1] > 0:
            z[0, 0] = torch.randn(1).item()  # Random value between -1 and 1
            
        return z

    def _convert_to_point_cloud(self, voxel_grid, num_points=2048):
        """
        Convert a voxel grid to a point cloud representation.
        
        Args:
            voxel_grid: 3D numpy array representing voxel occupancy
            num_points: Number of points to sample for the point cloud
            
        Returns:
            torch_geometric.data.Data: Point cloud object
        """
        try:
            from torch_geometric.data import Data
            import torch
            import numpy as np
            
            # Make a copy of the voxel grid to avoid modifying the original
            voxel_data = voxel_grid.copy()
            
            # Check if the voxel grid is empty
            if np.max(voxel_data) <= 0.1:
                logger.warning("Empty voxel grid detected in point cloud conversion. Using procedural shape instead.")
                # Generate a procedural shape as fallback
                voxel_data = self._generate_procedural_shape(None)
                
                # If it's still empty, create a simple cube point cloud
                if np.max(voxel_data) <= 0.1:
                    logger.warning("Procedural shape generation failed. Using simple cube as fallback.")
                    # Create a simple cube point cloud
                    x = np.linspace(-0.5, 0.5, 10)
                    y = np.linspace(-0.5, 0.5, 10)
                    z = np.linspace(-0.5, 0.5, 10)
                    
                    # Create points on the surface of the cube
                    surface_points = []
                    
                    # Generate points on the 6 faces of the cube
                    for i in range(10):
                        for j in range(10):
                            # Front face (z = -0.5)
                            surface_points.append([x[i], y[j], -0.5])
                            # Back face (z = 0.5)
                            surface_points.append([x[i], y[j], 0.5])
                            # Left face (x = -0.5)
                            surface_points.append([-0.5, y[i], z[j]])
                            # Right face (x = 0.5)
                            surface_points.append([0.5, y[i], z[j]])
                            # Bottom face (y = -0.5)
                            surface_points.append([x[i], -0.5, z[j]])
                            # Top face (y = 0.5)
                            surface_points.append([x[i], 0.5, z[j]])
                    
                    # Convert to numpy array
                    surface_points = np.array(surface_points)
                    
                    # Randomly select num_points from the surface points
                    if len(surface_points) > num_points:
                        indices = np.random.choice(len(surface_points), num_points, replace=False)
                        points = surface_points[indices]
                    else:
                        # Use all points and duplicate if necessary
                        indices = np.random.choice(len(surface_points), num_points, replace=True)
                        points = surface_points[indices]
                    
                    # Create a mock normals array (pointing outward from the center)
                    normals = points / np.linalg.norm(points, axis=1, keepdims=True)
                    
                    # Convert to PyTorch tensors
                    pos = torch.tensor(points, dtype=torch.float)
                    norm = torch.tensor(normals, dtype=torch.float)
                    
                    # Create the Data object
                    point_cloud = Data(pos=pos, norm=norm)
                    
                    return point_cloud
            
            # Get coordinates of all voxels above threshold
            threshold = 0.5
            x, y, z = np.where(voxel_data > threshold)
            
            # Initialize weights with a default value
            weights = None
            
            if len(x) == 0:
                # No voxels above threshold, lower the threshold
                threshold = voxel_data.mean()
                x, y, z = np.where(voxel_data > threshold)
                
                if len(x) == 0:
                    # Still no voxels, use all voxels
                    x, y, z = np.meshgrid(
                        np.arange(voxel_data.shape[0]),
                        np.arange(voxel_data.shape[1]),
                        np.arange(voxel_data.shape[2]),
                        indexing='ij'
                    )
                    x = x.flatten()
                    y = y.flatten()
                    z = z.flatten()
                    
                    # Apply weights based on voxel values
                    weights = voxel_data.flatten()
                    if weights.sum() > 0:  # Avoid division by zero
                        weights = weights / weights.sum()
                    else:
                        weights = np.ones_like(weights) / len(weights)
            else:
                # Use weights based on voxel values for sampling
                weights = voxel_data[x, y, z]
                if weights.sum() > 0:  # Avoid division by zero
                    weights = weights / weights.sum()
                else:
                    weights = np.ones_like(weights) / len(weights)
            
            # Safety check to ensure weights is initialized
            if weights is None or len(weights) == 0:
                # Fallback to uniform weights if something went wrong
                weights = np.ones(len(x)) / len(x)
            
            # Sample points based on weights
            indices = np.random.choice(len(x), min(num_points, len(x)), replace=len(x) < num_points, p=weights)
            sampled_x = x[indices]
            sampled_y = y[indices]
            sampled_z = z[indices]
            
            # Stack coordinates to form points
            points = np.column_stack((sampled_x, sampled_y, sampled_z))
            
            # Normalize to [-0.5, 0.5] range
            points = points / max(voxel_data.shape) - 0.5
            
            # Compute normal vectors using central differences
            normals = np.zeros_like(points)
            
            # Calculate normal for each point
            for i, (px, py, pz) in enumerate(zip(sampled_x, sampled_y, sampled_z)):
                # Sample gradients using central difference
                dx = self._safe_gradient(voxel_data, px, py, pz, 0)
                dy = self._safe_gradient(voxel_data, px, py, pz, 1)
                dz = self._safe_gradient(voxel_data, px, py, pz, 2)
                
                # Create normal vector (gradient of the voxel field)
                normal = np.array([dx, dy, dz])
                
                # Normalize the vector
                norm = np.linalg.norm(normal)
                if norm > 0:
                    normal = normal / norm
                    
                normals[i] = normal
            
            # Convert to PyTorch tensors
            pos = torch.tensor(points, dtype=torch.float)
            norm = torch.tensor(normals, dtype=torch.float)
            
            # Create the Data object
            point_cloud = Data(pos=pos, norm=norm)
            
            return point_cloud
            
        except Exception as e:
            logger.error(f"Error in point cloud generation: {e}")
            logger.error(traceback.format_exc())
            
            # Create a simple fallback point cloud (cube)
            logger.warning("Using fallback cube point cloud")
            
            # Create a simple cube
            x = np.linspace(-0.5, 0.5, 10)
            y = np.linspace(-0.5, 0.5, 10)
            z = np.linspace(-0.5, 0.5, 10)
            
            # Create points on the surface of the cube
            surface_points = []
            for i in range(10):
                for j in range(10):
                    # Six faces of the cube
                    surface_points.append([x[i], y[j], -0.5])  # Front face
                    surface_points.append([x[i], y[j], 0.5])   # Back face
                    surface_points.append([-0.5, y[i], z[j]])  # Left face
                    surface_points.append([0.5, y[i], z[j]])   # Right face
                    surface_points.append([x[i], -0.5, z[j]])  # Bottom face
                    surface_points.append([x[i], 0.5, z[j]])   # Top face
            
            # Convert to numpy array
            surface_points = np.array(surface_points)
            
            # Randomly select num_points
            if len(surface_points) > num_points:
                indices = np.random.choice(len(surface_points), num_points, replace=False)
                points = surface_points[indices]
            else:
                indices = np.random.choice(len(surface_points), num_points, replace=True)
                points = surface_points[indices]
            
            # Compute normals (pointing outward from the center)
            normals = points / np.linalg.norm(points, axis=1, keepdims=True)
            
            # Convert to PyTorch tensors
            pos = torch.tensor(points, dtype=torch.float)
            norm = torch.tensor(normals, dtype=torch.float)
            
            # Create the Data object
            point_cloud = Data(pos=pos, norm=norm)
            
            return point_cloud
    
    def _safe_gradient(self, voxel_data, x, y, z, axis):
        """
        Compute gradient safely using central differences.
        
        Args:
            voxel_data: 3D numpy array of voxel values
            x, y, z: Voxel coordinates
            axis: Axis along which to compute gradient (0=x, 1=y, 2=z)
            
        Returns:
            float: Gradient value
        """
        shape = voxel_data.shape
        
        if axis == 0:
            if x > 0 and x < shape[0]-1:
                return (voxel_data[x+1, y, z] - voxel_data[x-1, y, z]) / 2.0
            elif x > 0:
                return voxel_data[x, y, z] - voxel_data[x-1, y, z]
            elif x < shape[0]-1:
                return voxel_data[x+1, y, z] - voxel_data[x, y, z]
            else:
                return 0.0
                
        elif axis == 1:
            if y > 0 and y < shape[1]-1:
                return (voxel_data[x, y+1, z] - voxel_data[x, y-1, z]) / 2.0
            elif y > 0:
                return voxel_data[x, y, z] - voxel_data[x, y-1, z]
            elif y < shape[1]-1:
                return voxel_data[x, y+1, z] - voxel_data[x, y, z]
            else:
                return 0.0
                
        else:  # axis == 2
            if z > 0 and z < shape[2]-1:
                return (voxel_data[x, y, z+1] - voxel_data[x, y, z-1]) / 2.0
            elif z > 0:
                return voxel_data[x, y, z] - voxel_data[x, y, z-1]
            elif z < shape[2]-1:
                return voxel_data[x, y, z+1] - voxel_data[x, y, z]
            else:
                return 0.0
    
    def _convert_to_mesh(self, voxel_grid, detail_level=3):
        """
        Convert a voxel grid to a mesh using marching cubes algorithm.
        
        Args:
            voxel_grid: Voxel grid as numpy array (either binary or continuous values)
            detail_level: Level of detail (1-5)
                
        Returns:
            dict: Dictionary containing mesh data (vertices, faces)
        """
        try:
            from skimage import measure
            
            # Downsample for lower detail levels
            if detail_level < 5:
                factor = 6 - detail_level  # 5->1, 4->2, 3->3, 2->4, 1->5
                target_size = min(voxel_grid.shape) // factor
                target_size = max(target_size, 32)  # Ensure minimum size
                
                # Calculate new dimensions
                h, w, d = voxel_grid.shape
                h_new = max(int(h * target_size / min(voxel_grid.shape)), 16)
                w_new = max(int(w * target_size / min(voxel_grid.shape)), 16)
                d_new = max(int(d * target_size / min(voxel_grid.shape)), 16)
                
                from scipy.ndimage import zoom
                zoom_factors = (h_new/h, w_new/w, d_new/d)
                logger.info(f"Downsampling voxel grid from {voxel_grid.shape} to ({h_new}, {w_new}, {d_new})")
                voxel_grid = zoom(voxel_grid, zoom_factors, order=1)
            
            # Convert to float for normalization
            voxel_grid = voxel_grid.astype(np.float32)
            
            # Get the vertices and faces using our new helper method
            vertices, faces = self._create_mesh_from_voxels(voxel_grid, threshold=0.5)
            
            # Center the mesh (important for rendering)
            if vertices is not None and len(vertices) > 0:
                vertices = vertices - np.mean(vertices, axis=0)
                
                # Scale to unit cube
                max_extent = np.max(np.abs(vertices))
                if max_extent > 0:
                    vertices = vertices / max_extent
            else:
                # If vertices is empty, create a fallback shape
                logger.warning("Empty vertices array returned from marching cubes, using fallback shape")
                vertices, faces = self._create_fallback_shape()
            
            return {
                'vertices': vertices.tolist(),
                'faces': faces.tolist()
            }
        except Exception as e:
            logger.error(f"Error converting voxels to mesh: {e}")
            logger.error(traceback.format_exc())
            
            # Return a fallback shape
            vertices, faces = self._create_fallback_shape()
            
            return {
                'vertices': vertices.tolist(),
                'faces': faces.tolist(),
                'error': str(e)
            }
    
    def _create_mesh_from_voxels(self, voxel_grid, threshold=0.5):
        """
        Create a mesh from voxel grid using marching cubes algorithm.
        
        Args:
            voxel_grid: Voxel grid as a numpy array
            threshold: Isosurface threshold value
            
        Returns:
            tuple: Vertices and faces of the mesh
        """
        try:
            # Ensure voxel grid has values that can work with the threshold
            vmin, vmax = np.min(voxel_grid), np.max(voxel_grid)
            logger.info(f"Voxel grid value range: {vmin} to {vmax}")
            
            # Apply a small amount of smoothing to get cleaner surfaces
            from scipy.ndimage import gaussian_filter
            voxel_grid = gaussian_filter(voxel_grid, sigma=0.7)
            
            # Check if the threshold is outside the value range
            if threshold <= vmin or threshold >= vmax:
                old_threshold = threshold
                
                # Set threshold to something in the range
                # Use 40% of the range from min
                threshold = vmin + 0.4 * (vmax - vmin)
                logger.warning(f"Adjusted threshold from {old_threshold} to {threshold} to be within voxel range")
                
            # Skip if the range is too small (nearly uniform grid)
            if vmax - vmin < 0.01:
                logger.warning("Voxel grid has almost uniform values, cannot extract mesh")
                return self._create_fallback_shape()
                
            # If the voxel grid is too sparse (mostly zeros or ones), it's better to
            # adjust the threshold to get a meaningful surface
            if np.mean(voxel_grid > threshold) < 0.01:
                # Less than 1% of voxels above threshold - lower it
                old_threshold = threshold
                threshold = np.percentile(voxel_grid, 90)  # Use 90th percentile as threshold
                logger.warning(f"Grid too sparse at threshold {old_threshold}, lowering to {threshold}")
            elif np.mean(voxel_grid > threshold) > 0.99:
                # More than 99% of voxels above threshold - raise it
                old_threshold = threshold
                threshold = np.percentile(voxel_grid, 10)  # Use 10th percentile as threshold
                logger.warning(f"Grid too dense at threshold {old_threshold}, raising to {threshold}")
            
            # Double-check the threshold is still within range
            threshold = max(vmin + 1e-5, min(vmax - 1e-5, threshold))
            
            # Run marching cubes algorithm to extract the mesh
            try:
                verts, faces, normals, values = measure.marching_cubes(voxel_grid, level=threshold)
            except Exception as e:
                logger.error(f"Error in marching cubes algorithm: {e}")
                logger.error(traceback.format_exc())
                return self._create_fallback_shape()
            
            # Check if we have valid vertices and faces
            if len(verts) == 0 or len(faces) == 0:
                logger.warning("Marching cubes returned empty mesh, using fallback shape")
                return self._create_fallback_shape()
                
            # Apply Laplacian smoothing to vertices to reduce stair-stepping artifacts
            if len(verts) > 100:
                try:
                    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                    trimesh.smoothing.filter_laplacian(mesh, iterations=2)
                    verts = np.array(mesh.vertices)
                    faces = np.array(mesh.faces)
                except Exception as e:
                    logger.warning(f"Error applying Laplacian smoothing: {e}")
                    # Continue with original verts/faces
            
            logger.info(f"Mesh created with {len(verts)} vertices and {len(faces)} faces")
            return verts, faces
            
        except Exception as e:
            logger.error(f"Error creating mesh from voxels: {e}")
            logger.error(traceback.format_exc())
            return self._create_fallback_shape()
            
    def _create_fallback_shape(self):
        """
        Create a fallback shape when mesh creation fails
        
        Returns:
            tuple: Vertices and faces of a simple shape
        """
        # Create a simple cube as fallback
        logger.info("Creating fallback shape (cube)")
        
        # Cube vertices
        vertices = np.array([
            [0, 0, 0],  # 0
            [1, 0, 0],  # 1
            [1, 1, 0],  # 2
            [0, 1, 0],  # 3
            [0, 0, 1],  # 4
            [1, 0, 1],  # 5
            [1, 1, 1],  # 6
            [0, 1, 1]   # 7
        ])
        
        # Cube faces (triangles)
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # Bottom
            [4, 5, 6], [4, 6, 7],  # Top
            [0, 1, 5], [0, 5, 4],  # Front
            [2, 3, 7], [2, 7, 6],  # Back
            [0, 3, 7], [0, 7, 4],  # Left
            [1, 2, 6], [1, 6, 5]   # Right
        ])
        
        return vertices, faces

    def get_debug_visualizations(self, voxel_grid):
        """
        Generate debug visualizations for a voxel grid.
        
        Args:
            voxel_grid: Binary voxel grid (numpy.ndarray)
            
        Returns:
            dict: Dictionary containing visualization data
        """
        visualizations = {}
        
        try:
            # Get 3D visualization as image
            visualizations["voxel_3d"] = voxel_grid_to_image(voxel_grid)
            
            # Get 2D slices
            slices_fig = create_voxel_grid_slices(voxel_grid, show=False)
            buf = io.BytesIO()
            slices_fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            import base64
            slices_b64 = base64.b64encode(buf.read()).decode('utf-8')
            visualizations["voxel_slices"] = f"data:image/png;base64,{slices_b64}"
            
            # Get voxel stats
            visualizations["voxel_stats"] = analyze_voxel_grid(voxel_grid)
            
        except Exception as e:
            logger.error(f"Error generating debug visualizations: {e}")
            logger.error(traceback.format_exc())
            visualizations["error"] = str(e)
            
        return visualizations

def analyze_voxel_grid(voxel_grid):
    """
    Analyze voxel grid and return statistics.
    
    Args:
        voxel_grid: Binary voxel grid (numpy.ndarray)
        
    Returns:
        dict: Dictionary with voxel grid statistics
    """
    stats = {
        "shape": list(voxel_grid.shape),
        "min": float(np.min(voxel_grid)),
        "max": float(np.max(voxel_grid)),
        "mean": float(np.mean(voxel_grid)),
        "std": float(np.std(voxel_grid)),
        "nonzero": int(np.count_nonzero(voxel_grid)),
        "total": int(voxel_grid.size),
        "nonzero_percent": float(np.count_nonzero(voxel_grid) / voxel_grid.size * 100),
    }
    return stats

def voxel_grid_to_image(voxel_grid, size=(400, 400)):
    """
    Convert voxel grid to image for visualization.
    
    Args:
        voxel_grid: Binary voxel grid (numpy.ndarray)
        size: Target image size
        
    Returns:
        str: Base64 encoded PNG image
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        import io
        import base64
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
        
        # Create 3D figure
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get voxel positions
        x, y, z = np.indices(voxel_grid.shape)
        
        # Set the colors with transparency
        colors = np.zeros(voxel_grid.shape + (4,))
        colors[..., 0] = 0.5  # Red
        colors[..., 1] = 0.5  # Green
        colors[..., 2] = 1.0  # Blue
        colors[..., 3] = voxel_grid * 0.8  # Alpha depends on voxel value
        
        # Only plot voxels above a threshold
        threshold = 0.2
        mask = voxel_grid > threshold
        
        # If no voxels above threshold, create some dummy voxels
        if not np.any(mask):
            center = np.array(voxel_grid.shape) // 2
            radius = min(voxel_grid.shape) // 4
            x, y, z = np.indices(voxel_grid.shape)
            sphere_mask = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 < radius**2
            colors[sphere_mask, 0] = 1.0  # Red
            colors[sphere_mask, 1] = 0.5  # Green
            colors[sphere_mask, 2] = 0.0  # Blue
            colors[sphere_mask, 3] = 0.8  # Alpha
            mask = sphere_mask
        
        # Plot the voxels
        ax.voxels(mask, facecolors=colors, edgecolor='k', linewidth=0.1)
        
        # Set view angle
        ax.view_init(30, 30)
        
        # Remove axes for cleaner visualization
        ax.set_axis_off()
        
        # Save to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', transparent=True)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        logger.error(f"Error generating voxel image: {e}")
        return ""

def create_voxel_grid_slices(voxel_grid, show=False):
    """
    Create visualization of voxel grid slices.
    
    Args:
        voxel_grid: Binary voxel grid (numpy.ndarray)
        show: Whether to show the plot or return the figure
        
    Returns:
        matplotlib.figure.Figure: Figure with voxel grid slices
    """
    import matplotlib.pyplot as plt
    
    # Create figure with 3x3 grid of subplots
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    
    # Get shape
    h, w, d = voxel_grid.shape
    
    # Slice positions
    h_slices = [h//4, h//2, 3*h//4]
    w_slices = [w//4, w//2, 3*w//4]
    d_slices = [d//4, d//2, 3*d//4]
    
    # Show slices along height
    for i, h_slice in enumerate(h_slices):
        img = voxel_grid[h_slice, :, :]
        axes[0, i].imshow(img, cmap='viridis')
        axes[0, i].set_title(f'Height: {h_slice}/{h}')
        axes[0, i].axis('off')
    
    # Show slices along width
    for i, w_slice in enumerate(w_slices):
        img = voxel_grid[:, w_slice, :]
        axes[1, i].imshow(img, cmap='viridis')
        axes[1, i].set_title(f'Width: {w_slice}/{w}')
        axes[1, i].axis('off')
    
    # Show slices along depth
    for i, d_slice in enumerate(d_slices):
        img = voxel_grid[:, :, d_slice]
        axes[2, i].imshow(img, cmap='viridis')
        axes[2, i].set_title(f'Depth: {d_slice}/{d}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig
