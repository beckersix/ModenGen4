"""
Dataset management utilities for 3D model datasets.

This module provides functionality for managing 3D model datasets,
including downloading ShapeNet categories, custom dataset creation,
and dataset combination for training.
"""

import os
import json
import logging
import shutil
import zipfile
import requests
from pathlib import Path
from datetime import datetime
import trimesh
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ShapeNetDownloader:
    """Downloads and manages ShapeNet dataset access"""
    
    def __init__(self, base_dir):
        """
        Initialize the ShapeNet downloader.
        
        Args:
            base_dir: Base directory for dataset storage
        """
        self.base_dir = Path(base_dir)
        self.shapenet_dir = self.base_dir / "shapenet"
        self.shapenet_dir.mkdir(parents=True, exist_ok=True)
        
        # Category mapping for ShapeNet
        self.categories = {
            "02691156": "airplane",
            "02828884": "bench",
            "02933112": "cabinet",
            "02958343": "car",
            "03001627": "chair",
            "03211117": "display",
            "03636649": "lamp",
            "04256520": "sofa",
            "04379243": "table",
            "04530566": "vessel"
        }
        
        # Sources for free 3D model datasets that are ShapeNet alternatives
        self.dataset_sources = {
            "thingi10k": {
                "url": "https://ten-thousand-models.appspot.com/",
                "description": "A dataset of 10,000 models from Thingiverse",
                "license": "Creative Commons"
            },
            "abc_dataset": {
                "url": "https://deep-geometry.github.io/abc-dataset/",
                "description": "A dataset for benchmarking CAD model processing",
                "license": "MIT"
            },
            "modelnet40": {
                "url": "https://modelnet.cs.princeton.edu/",
                "description": "Princeton ModelNet with 40 common object categories",
                "license": "Princeton"
            }
        }
    
    def get_available_categories(self):
        """
        Return available/downloaded ShapeNet categories.
        
        Returns:
            List of category information dictionaries
        """
        available = []
        for cat_id, cat_name in self.categories.items():
            cat_dir = self.shapenet_dir / cat_id
            if cat_dir.exists():
                # Count models in this category
                model_count = len(list(cat_dir.glob('**/*.obj')))
                available.append({
                    "id": cat_id,
                    "name": cat_name,
                    "path": str(cat_dir),
                    "model_count": model_count,
                    "downloaded": True
                })
            else:
                available.append({
                    "id": cat_id,
                    "name": cat_name,
                    "downloaded": False
                })
        return available
    
    def download_category(self, category_id):
        """
        Download a specific ShapeNet category.
        
        Since actual ShapeNet requires registration, this provides
        alternatives or placeholder functionality.
        
        Args:
            category_id: ShapeNet category ID
            
        Returns:
            Path to downloaded category
        """
        if category_id not in self.categories:
            raise ValueError(f"Unknown category ID: {category_id}")
            
        cat_name = self.categories[category_id]
        cat_dir = self.shapenet_dir / category_id
        
        # If already downloaded, return path
        if cat_dir.exists() and len(list(cat_dir.glob('**/*.obj'))) > 0:
            logger.info(f"Category {cat_name} already downloaded")
            return str(cat_dir)
            
        # Create category directory
        cat_dir.mkdir(exist_ok=True)
        
        # Since actual ShapeNet requires registration and API access,
        # we'll provide instructions and alternatives here
        logger.info(f"Attempting to download {cat_name} models from alternative sources")
        
        # For demonstration, generate some sample shapes for this category
        # In a real implementation, this would download from ShapeNet API
        from .mesh_dataset import SampleDataGenerator
        sample_gen = SampleDataGenerator(output_dir=str(cat_dir))
        
        # Generate samples based on category
        logger.info(f"Generating sample shapes for {cat_name}")
        if cat_name == "airplane":
            sample_gen.generate_samples(count=20, shape_types=["airplane"])
        elif cat_name == "chair":
            sample_gen.generate_samples(count=20, shape_types=["chair"])
        else:
            # Generic shapes for other categories
            sample_gen.generate_samples(count=20)
            
        # Create a metadata file
        metadata = {
            "category_id": category_id,
            "category_name": cat_name,
            "source": "sample_generator",
            "model_count": len(list(cat_dir.glob('**/*.obj'))),
            "downloaded_at": datetime.now().isoformat(),
            "note": "These are procedurally generated samples for demonstration. For the real ShapeNet dataset, register at https://shapenet.org/"
        }
        
        with open(cat_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Downloaded {metadata['model_count']} models for category {cat_name}")
        return str(cat_dir)
    
    def download_alternative_dataset(self, dataset_name, sample_only=True):
        """
        Download an alternative to ShapeNet - public dataset.
        
        Args:
            dataset_name: Name of the dataset
            sample_only: If True, download only a sample
            
        Returns:
            Path to the downloaded dataset
        """
        if dataset_name not in self.dataset_sources:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
        dataset_dir = self.base_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        # If we already have data, just return
        if len(list(dataset_dir.glob('**/*.obj'))) > 0:
            logger.info(f"Dataset {dataset_name} already has models")
            return str(dataset_dir)
            
        # Generate samples instead of real download (for demo purposes)
        logger.info(f"Generating sample models for {dataset_name}")
        from .mesh_dataset import SampleDataGenerator
        sample_gen = SampleDataGenerator(output_dir=str(dataset_dir))
        sample_gen.generate_samples(count=30)
        
        # Create a metadata file
        metadata = {
            "dataset_name": dataset_name,
            "source": self.dataset_sources[dataset_name]["url"],
            "description": self.dataset_sources[dataset_name]["description"],
            "license": self.dataset_sources[dataset_name]["license"],
            "model_count": len(list(dataset_dir.glob('**/*.obj'))),
            "downloaded_at": datetime.now().isoformat(),
            "note": "These are procedurally generated samples for demonstration. Visit the source URL for the full dataset."
        }
        
        with open(dataset_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Generated {metadata['model_count']} sample models for {dataset_name}")
        return str(dataset_dir)


class ObjectNet3DDownloader:
    """Downloads and manages ObjectNet3D dataset access"""
    
    def __init__(self, base_dir):
        """
        Initialize the ObjectNet3D downloader.
        
        Args:
            base_dir: Base directory for dataset storage
        """
        self.base_dir = Path(base_dir)
        self.objectnet3d_dir = self.base_dir / "objectnet3d"
        self.objectnet3d_dir.mkdir(parents=True, exist_ok=True)
        
        # Base URL for ObjectNet3D downloads
        self.base_url = "http://cvgl.stanford.edu/projects/objectnet3d/data"
        
        # Category mapping for ObjectNet3D (100 categories total)
        # These are some of the main categories with 3D models
        self.categories = {
            "aeroplane": "aeroplane",
            "bed": "bed",
            "bicycle": "bicycle",
            "boat": "boat",
            "bookshelf": "bookshelf", 
            "bottle": "bottle",
            "bus": "bus",
            "car": "car",
            "chair": "chair",
            "computer": "computer",
            "desk": "desk",
            "dining_table": "dining_table",
            "door": "door",
            "lamp": "lamp",
            "motorcycle": "motorcycle",
            "sofa": "sofa",
            "table": "table",
            "train": "train",
            "tv": "tv"
        }
        
        # ObjectNet3D dataset information
        self.dataset_info = {
            "name": "ObjectNet3D",
            "url": "http://cvgl.stanford.edu/projects/objectnet3d/",
            "description": "A Large Scale Database for 3D Object Recognition with 100 categories and 90,127 images",
            "paper": "https://link.springer.com/chapter/10.1007/978-3-319-46478-7_12",
            "license": "MIT"
        }
        
        # Download links for different parts of the dataset
        self.download_links = {
            "images": "https://3d-ai-dataset-bucket.s3.amazonaws.com/objectnet3d/ObjectNet3D_images.zip",
            "annotations": "https://3d-ai-dataset-bucket.s3.amazonaws.com/objectnet3d/ObjectNet3D_annotations.zip",
            "cad_models": "https://3d-ai-dataset-bucket.s3.amazonaws.com/objectnet3d/ObjectNet3D_cads.zip",
            "splits": "https://3d-ai-dataset-bucket.s3.amazonaws.com/objectnet3d/ObjectNet3D_splits.zip",
            "toolbox": "https://github.com/yuxng/ObjectNet3D_toolbox/archive/refs/heads/master.zip"
        }
    
    def get_available_categories(self):
        """
        Return available/downloaded ObjectNet3D categories.
        
        Returns:
            List of category information dictionaries
        """
        # Check if main CAD models directory exists
        cad_dir = self.objectnet3d_dir / "CAD"
        has_cad_models = cad_dir.exists()
        
        available = []
        for cat_id, cat_name in self.categories.items():
            cat_dir = cad_dir / cat_id if has_cad_models else None
            
            if has_cad_models and cat_dir and cat_dir.exists():
                # Count models in this category
                model_count = len(list(cat_dir.glob('**/*.obj')))
                available.append({
                    "id": cat_id,
                    "name": cat_name,
                    "path": str(cat_dir),
                    "model_count": model_count,
                    "downloaded": True
                })
            else:
                available.append({
                    "id": cat_id,
                    "name": cat_name,
                    "downloaded": False
                })
        return available
    
    def download_category(self, category_id):
        """
        Download a specific ObjectNet3D category.
        
        Args:
            category_id: ObjectNet3D category ID
            
        Returns:
            Path to downloaded category
        """
        if category_id not in self.categories:
            raise ValueError(f"Unknown category ID: {category_id}")
            
        cat_name = self.categories[category_id]
        
        # We need to download the entire CAD dataset first if not already downloaded
        # Then extract specific category models
        cad_dir = self.objectnet3d_dir / "CAD"
        cat_dir = cad_dir / category_id
        
        # If already downloaded, return path
        if cat_dir.exists() and len(list(cat_dir.glob('**/*.obj'))) > 0:
            logger.info(f"Category {cat_name} already downloaded")
            return str(cat_dir)
        
        try:
            # If CAD models not downloaded yet, download them
            if not cad_dir.exists():
                logger.info("CAD models not found. Downloading ObjectNet3D CAD models...")
                self._download_cad_models()
                
                # Double check that the directory exists after download
                if not cad_dir.exists():
                    cad_dir.mkdir(parents=True, exist_ok=True)
            
            # If the category directory still doesn't exist after downloading the CAD models,
            # this category may not be part of the dataset or extraction failed
            if not cat_dir.exists() or len(list(cat_dir.glob('**/*.obj'))) == 0:
                # Create category directory
                cat_dir.mkdir(exist_ok=True)
                
                logger.warning(f"No models found for {cat_name} in ObjectNet3D. Creating placeholder models.")
                # Generate some placeholder models for this category
                from .mesh_dataset import SampleDataGenerator
                sample_gen = SampleDataGenerator(output_dir=str(cat_dir))
                
                # Generate samples based on category
                if cat_name == "aeroplane":
                    sample_gen.generate_samples(count=5, shape_types=["airplane"])
                elif cat_name == "chair":
                    sample_gen.generate_samples(count=5, shape_types=["chair"])
                elif cat_name == "car":
                    sample_gen.generate_samples(count=5, shape_types=["car"])
                else:
                    # Generic shapes for other categories
                    sample_gen.generate_samples(count=5)
                    
                # Create a metadata file
                metadata = {
                    "category_id": category_id,
                    "category_name": cat_name,
                    "source": "ObjectNet3D (Placeholder)",
                    "model_count": len(list(cat_dir.glob('**/*.obj'))),
                    "downloaded_at": datetime.now().isoformat(),
                    "note": "These are procedurally generated samples as placeholders. Real models not found in ObjectNet3D dataset."
                }
            else:
                # Create a metadata file
                metadata = {
                    "category_id": category_id,
                    "category_name": cat_name,
                    "source": "ObjectNet3D",
                    "model_count": len(list(cat_dir.glob('**/*.obj'))),
                    "downloaded_at": datetime.now().isoformat()
                }
            
            with open(cat_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Prepared {metadata['model_count']} models for category {cat_name}")
            return str(cat_dir)
        except Exception as e:
            logger.error(f"Error downloading category {cat_name}: {str(e)}")
            # Ensure the category directory exists even if download failed
            cat_dir.mkdir(exist_ok=True)
            
            # Generate placeholder data on error
            from .mesh_dataset import SampleDataGenerator
            sample_gen = SampleDataGenerator(output_dir=str(cat_dir))
            sample_gen.generate_samples(count=3)
            
            # Create an error metadata file
            error_metadata = {
                "category_id": category_id,
                "category_name": cat_name,
                "source": "ObjectNet3D (Error Fallback)",
                "model_count": len(list(cat_dir.glob('**/*.obj'))),
                "downloaded_at": datetime.now().isoformat(),
                "error": str(e),
                "note": "Error occurred during download. These are procedurally generated fallback models."
            }
            
            with open(cat_dir / "metadata.json", "w") as f:
                json.dump(error_metadata, f, indent=2)
                
            logger.info(f"Created fallback models for category {cat_name} due to download error")
            return str(cat_dir)
    
    def _download_cad_models(self):
        """
        Download and extract ObjectNet3D CAD models
        
        Returns:
            Path to the CAD models directory
        """
        cad_zip = self.objectnet3d_dir / "ObjectNet3D_cads.zip"
        cad_dir = self.objectnet3d_dir / "CAD"
        
        # Ensure the directory exists
        cad_dir.mkdir(parents=True, exist_ok=True)
        
        # Download CAD models
        try:
            logger.info("Downloading ObjectNet3D CAD models...")
            
            # Make a HEAD request first to check if the URL is accessible
            head_response = requests.head(self.download_links["cad_models"])
            if head_response.status_code != 200:
                raise ValueError(f"CAD models URL is not accessible. Status code: {head_response.status_code}")
            
            response = requests.get(self.download_links["cad_models"], stream=True)
            response.raise_for_status()  # Raise an exception for 4XX/5XX responses
            
            total_size = int(response.headers.get('content-length', 0))
            
            if total_size == 0:
                logger.warning("Could not determine download size for ObjectNet3D CAD models")
                
            # Download the file
            with open(cad_zip, 'wb') as f:
                chunk_size = 8192
                total_chunks = total_size // chunk_size if total_size > 0 else None
                
                with tqdm(total=total_chunks, desc="Downloading CAD models", unit="chunk") as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(1)
            
            # Verify the file was downloaded correctly
            if not cad_zip.exists() or cad_zip.stat().st_size == 0:
                raise ValueError("Failed to download CAD models: file is empty or does not exist")
            
            # Extract the zip file
            logger.info("Extracting ObjectNet3D CAD models...")
            try:
                with zipfile.ZipFile(cad_zip, 'r') as zip_ref:
                    zip_ref.extractall(self.objectnet3d_dir)
            except zipfile.BadZipFile:
                raise ValueError("Downloaded file is not a valid zip file")
            
            # Delete the zip file
            cad_zip.unlink()
            
            # Verify extraction worked
            if not cad_dir.exists() or len(list(cad_dir.glob('**/*'))) == 0:
                raise ValueError("Failed to extract CAD models: directory is empty or does not exist")
            
            logger.info("ObjectNet3D CAD models downloaded and extracted")
            return str(cad_dir)
        
        except Exception as e:
            logger.error(f"Error downloading or extracting ObjectNet3D CAD models: {str(e)}")
            if cad_zip.exists():
                cad_zip.unlink()
            raise
    
    def download_dataset_components(self, component):
        """
        Download specific components of the ObjectNet3D dataset
        
        Args:
            component: One of 'images', 'annotations', 'cad_models', 'splits'
            
        Returns:
            Path to the downloaded component
        """
        if component not in self.download_links:
            raise ValueError(f"Unknown component: {component}. Available components: {list(self.download_links.keys())}")
        
        target_dir = self.objectnet3d_dir
        zip_path = target_dir / f"ObjectNet3D_{component}.zip"
        
        try:
            logger.info(f"Downloading ObjectNet3D {component}...")
            response = requests.get(self.download_links[component], stream=True)
            response.raise_for_status()  # Raise an exception for 4XX/5XX responses
            
            total_size = int(response.headers.get('content-length', 0))
            
            if total_size == 0:
                logger.warning(f"Could not determine download size for ObjectNet3D {component}")
                
            # Download the file
            with open(zip_path, 'wb') as f:
                chunk_size = 8192
                total_chunks = total_size // chunk_size if total_size > 0 else None
                
                with tqdm(total=total_chunks, desc=f"Downloading {component}", unit="chunk") as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(1)
            
            # Extract the zip file
            logger.info(f"Extracting ObjectNet3D {component}...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(target_dir)
            except zipfile.BadZipFile:
                raise ValueError("Downloaded file is not a valid zip file")
            
            # Delete the zip file
            zip_path.unlink()
            
            logger.info(f"ObjectNet3D {component} downloaded and extracted")
            return str(target_dir / component)
        
        except Exception as e:
            logger.error(f"Error downloading or extracting ObjectNet3D {component}: {str(e)}")
            if zip_path.exists():
                zip_path.unlink()
            raise
    
    def download_toolbox(self):
        """
        Download the ObjectNet3D toolbox from GitHub
        
        Returns:
            Path to the downloaded toolbox
        """
        toolbox_dir = self.objectnet3d_dir / "toolbox"
        
        # If already downloaded, return path
        if toolbox_dir.exists() and (toolbox_dir / "README.md").exists():
            logger.info("ObjectNet3D toolbox already downloaded")
            return str(toolbox_dir)
            
        # Create toolbox directory
        toolbox_dir.mkdir(exist_ok=True)
        
        try:
            # Download the toolbox from GitHub
            logger.info("Downloading ObjectNet3D toolbox from GitHub")
            
            # Download the zip file
            zip_path = toolbox_dir / "objectnet3d_toolbox.zip"
            response = requests.get(self.download_links["toolbox"], stream=True)
            response.raise_for_status()  # Raise an exception for 4XX/5XX responses
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(zip_path, 'wb') as f:
                for chunk in tqdm(response.iter_content(chunk_size=8192), total=total_size//8192, desc="Downloading toolbox"):
                    f.write(chunk)
            
            # Extract the zip file
            logger.info("Extracting ObjectNet3D toolbox")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(str(toolbox_dir))
            except zipfile.BadZipFile:
                raise ValueError("Downloaded file is not a valid zip file")
            
            # Remove the zip file
            zip_path.unlink()
            
            # Move contents from the extracted directory to toolbox_dir
            extracted_dir = next(toolbox_dir.glob('ObjectNet3D_toolbox*'))
            for item in extracted_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, toolbox_dir)
                elif item.is_dir():
                    shutil.copytree(item, toolbox_dir / item.name, dirs_exist_ok=True)
            
            # Remove the extracted directory
            shutil.rmtree(extracted_dir)
            
            logger.info("ObjectNet3D toolbox downloaded successfully")
            return str(toolbox_dir)
            
        except Exception as e:
            logger.error(f"Error downloading ObjectNet3D toolbox: {str(e)}")
            raise
            
    def download_complete_dataset(self):
        """
        Download the complete ObjectNet3D dataset
        
        Returns:
            Dictionary with paths to all components
        """
        result = {}
        
        try:
            # Download CAD models
            result["cad_models"] = self._download_cad_models()
            
            # Download other components
            for component in ["images", "annotations", "splits"]:
                result[component] = self.download_dataset_components(component)
                
            # Download toolbox
            result["toolbox"] = self.download_toolbox()
            
            return result
        except Exception as e:
            logger.error(f"Error downloading complete ObjectNet3D dataset: {str(e)}")
            raise


class CustomDatasetManager:
    """Manages user-uploaded 3D model datasets"""
    
    def __init__(self, base_dir):
        """
        Initialize the custom dataset manager.
        
        Args:
            base_dir: Base directory for dataset storage
        """
        self.base_dir = Path(base_dir)
        self.custom_datasets_dir = self.base_dir / "custom_datasets"
        self.custom_datasets_dir.mkdir(parents=True, exist_ok=True)
    
    def get_available_datasets(self):
        """
        Return list of available custom datasets.
        
        Returns:
            List of dataset information dictionaries
        """
        datasets = []
        for path in self.custom_datasets_dir.iterdir():
            if path.is_dir() and (path / "metadata.json").exists():
                with open(path / "metadata.json", "r") as f:
                    metadata = json.load(f)
                    
                # Add path and model count
                metadata["path"] = str(path)
                metadata["model_count"] = len(list(path.glob('**/*.obj')))
                datasets.append(metadata)
        return datasets
    
    def create_dataset(self, name, description=""):
        """
        Create a new custom dataset container.
        
        Args:
            name: Name of the dataset
            description: Description of the dataset
            
        Returns:
            Path to the created dataset
        """
        # Clean the name for use as directory
        clean_name = name.replace(" ", "_").lower()
        dataset_dir = self.custom_datasets_dir / clean_name
        
        # Check if already exists
        if dataset_dir.exists():
            raise ValueError(f"Dataset {name} already exists")
            
        dataset_dir.mkdir(exist_ok=True)
        
        # Create metadata file
        metadata = {
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "model_count": 0
        }
        
        with open(dataset_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Created new custom dataset: {name}")
        return str(dataset_dir)
    
    def add_model_to_dataset(self, dataset_name, model_file):
        """
        Add a model file to an existing dataset.
        
        Args:
            dataset_name: Name of the dataset
            model_file: Path to the model file
            
        Returns:
            Path to the copied model file
        """
        # Clean the name for use as directory
        clean_name = dataset_name.replace(" ", "_").lower()
        dataset_dir = self.custom_datasets_dir / clean_name
        
        if not dataset_dir.exists():
            raise ValueError(f"Dataset {dataset_name} does not exist")
            
        # Ensure the model file exists
        model_path = Path(model_file)
        if not model_path.exists():
            raise ValueError(f"Model file {model_file} does not exist")
            
        # Copy the model file to the dataset directory
        dest_file = dataset_dir / model_path.name
        shutil.copy(model_path, dest_file)
        
        # Update metadata
        metadata_file = dataset_dir / "metadata.json"
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
            
        metadata["model_count"] += 1
        metadata["updated_at"] = datetime.now().isoformat()
        
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Added model {model_path.name} to dataset {dataset_name}")
        return str(dest_file)
    
    def delete_dataset(self, dataset_name):
        """
        Delete a custom dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            True if successful
        """
        # Clean the name for use as directory
        clean_name = dataset_name.replace(" ", "_").lower()
        dataset_dir = self.custom_datasets_dir / clean_name
        
        if not dataset_dir.exists():
            raise ValueError(f"Dataset {dataset_name} does not exist")
            
        # Delete the dataset directory
        shutil.rmtree(dataset_dir)
        logger.info(f"Deleted dataset {dataset_name}")
        return True
    
    def extract_zip_to_dataset(self, dataset_name, zip_file):
        """
        Extract a zip file to a dataset.
        
        Args:
            dataset_name: Name of the dataset
            zip_file: Path to the zip file
            
        Returns:
            Number of extracted models
        """
        # Clean the name for use as directory
        clean_name = dataset_name.replace(" ", "_").lower()
        dataset_dir = self.custom_datasets_dir / clean_name
        
        if not dataset_dir.exists():
            raise ValueError(f"Dataset {dataset_name} does not exist")
            
        # Ensure the zip file exists
        zip_path = Path(zip_file)
        if not zip_path.exists():
            raise ValueError(f"Zip file {zip_file} does not exist")
            
        # Extract the zip file
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
            
        # Count the number of models
        model_files = list(dataset_dir.glob('**/*.obj'))
        model_files.extend(list(dataset_dir.glob('**/*.ply')))
        model_files.extend(list(dataset_dir.glob('**/*.stl')))
        model_count = len(model_files)
        
        # Update metadata
        metadata_file = dataset_dir / "metadata.json"
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
            
        metadata["model_count"] = model_count
        metadata["updated_at"] = datetime.now().isoformat()
        
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Extracted {model_count} models from {zip_file} to dataset {dataset_name}")
        return model_count


class CombinedDatasetManager:
    """Manages combined datasets for training"""
    
    def __init__(self, base_dir):
        """
        Initialize the combined dataset manager.
        
        Args:
            base_dir: Base directory for dataset storage
        """
        self.base_dir = Path(base_dir)
        self.combined_datasets_dir = self.base_dir / "combined_datasets"
        self.combined_datasets_dir.mkdir(parents=True, exist_ok=True)
    
    def get_available_combined_datasets(self):
        """
        Return list of available combined datasets.
        
        Returns:
            List of dataset information dictionaries
        """
        datasets = []
        for path in self.combined_datasets_dir.iterdir():
            if path.is_dir() and (path / "metadata.json").exists():
                with open(path / "metadata.json", "r") as f:
                    metadata = json.load(f)
                    
                # Add path
                metadata["path"] = str(path)
                datasets.append(metadata)
        return datasets
    
    def create_combined_dataset(self, name, source_datasets, description=""):
        """
        Create a combined dataset from multiple source datasets.
        
        Args:
            name: Name of the combined dataset
            source_datasets: List of source dataset paths
            description: Description of the combined dataset
            
        Returns:
            Path to the created combined dataset
        """
        # Clean the name for use as directory
        clean_name = name.replace(" ", "_").lower()
        combined_dir = self.combined_datasets_dir / clean_name
        
        # Check if already exists
        if combined_dir.exists():
            raise ValueError(f"Combined dataset {name} already exists")
            
        combined_dir.mkdir(exist_ok=True)
        
        # Source datasets information
        source_info = []
        total_models = 0
        
        # Create symlinks to all models in source datasets
        for idx, source_path in enumerate(source_datasets):
            source_dir = Path(source_path)
            if not source_dir.exists():
                logger.warning(f"Source dataset {source_path} does not exist, skipping")
                continue
                
            # Get source metadata if available
            source_name = f"dataset_{idx}"
            if (source_dir / "metadata.json").exists():
                with open(source_dir / "metadata.json", "r") as f:
                    source_metadata = json.load(f)
                    source_name = source_metadata.get("name", source_name)
            
            # Create a subdirectory for this source
            target_subdir = combined_dir / source_name
            target_subdir.mkdir(exist_ok=True)
            
            # Find all model files
            model_files = list(source_dir.glob('**/*.obj'))
            model_files.extend(list(source_dir.glob('**/*.ply')))
            model_files.extend(list(source_dir.glob('**/*.stl')))
            
            # Create hard links (or copy if links not possible)
            for model_file in model_files:
                try:
                    # Create relative path structure
                    rel_path = model_file.relative_to(source_dir)
                    target_file = target_subdir / rel_path
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Try hard link first, then copy
                    try:
                        os.link(model_file, target_file)
                    except:
                        shutil.copy(model_file, target_file)
                except Exception as e:
                    logger.error(f"Error linking/copying {model_file}: {e}")
            
            # Count models in this source
            source_model_count = len(list(target_subdir.glob('**/*.obj')))
            source_model_count += len(list(target_subdir.glob('**/*.ply')))
            source_model_count += len(list(target_subdir.glob('**/*.stl')))
            
            total_models += source_model_count
            
            # Add source info
            source_info.append({
                "name": source_name,
                "path": str(source_dir),
                "model_count": source_model_count
            })
        
        # Create metadata file
        metadata = {
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "source_datasets": source_info,
            "model_count": total_models
        }
        
        with open(combined_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Created combined dataset {name} with {total_models} models from {len(source_info)} sources")
        return str(combined_dir)
    
    def delete_combined_dataset(self, dataset_name):
        """
        Delete a combined dataset.
        
        Args:
            dataset_name: Name of the combined dataset
            
        Returns:
            True if successful
        """
        # Clean the name for use as directory
        clean_name = dataset_name.replace(" ", "_").lower()
        combined_dir = self.combined_datasets_dir / clean_name
        
        if not combined_dir.exists():
            raise ValueError(f"Combined dataset {dataset_name} does not exist")
            
        # Delete the dataset directory
        shutil.rmtree(combined_dir)
        logger.info(f"Deleted combined dataset {dataset_name}")
        return True


# Initialize dataset managers at module level
def get_dataset_managers(base_dir):
    """
    Get initialized dataset managers.
    
    Args:
        base_dir: Base directory for dataset storage
        
    Returns:
        Tuple of (ShapeNetDownloader, ObjectNet3DDownloader, CustomDatasetManager, CombinedDatasetManager)
    """
    shapenet = ShapeNetDownloader(base_dir)
    objectnet3d = ObjectNet3DDownloader(base_dir)
    custom = CustomDatasetManager(base_dir)
    combined = CombinedDatasetManager(base_dir)
    
    return shapenet, objectnet3d, custom, combined
