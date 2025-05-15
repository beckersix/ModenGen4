# Text-to-3D Generator

This project uses natural language processing and generative AI to create 3D models from text descriptions. It combines:

- A language model to interpret text prompts
- PyTorch Geometric for point cloud generation and manipulation
- Stable Diffusion for texture generation and model detailing

## Features

- Generate 3D point clouds from simple text descriptions
- Detail and refine 3D models using Stable Diffusion
- Convert point clouds to renderable meshes with UV coordinates
- Generate textures based on the same text prompts

## Setup

1. Install dependencies:
   You may have to install venv for windows / mac

   ```
   python -m venv venv
 ```
 ```


   .\venv\Scripts\activate - WINDOWS
   source venv/bin/activate - MAC
 ```
 ```

   pip install -r requirements.txt
 ```
 ```

   python manage.py migrate
   ```

2. Run the application:
   ```
   python manage.py runserver

   Go to chrome and navigate to 127.0.0.1:8000

   ```

## Architecture

The application follows a modular pipeline:
1. Text interpretation using a small language model
2. Basic shape generation as point clouds
3. Point cloud refinement with geometric operations
4. Mesh generation from point clouds
5. Texture generation using Stable Diffusion
6. Export to standard 3D formats
