"""
Test script for the Text-to-3D API

This script tests the text-to-3D endpoints to generate a 3D model from text.
"""

import requests
import json
import time
import logging
import sys
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("text_to_3d_tester")

# API endpoint URLs
BASE_URL = "http://localhost:8000"
GENERATE_URL = f"{BASE_URL}/api/text-to-3d/generate/"
STATUS_URL = f"{BASE_URL}/api/text-to-3d/status/"
DOWNLOAD_URL = f"{BASE_URL}/api/text-to-3d/download/"

def test_text_to_3d_generation(prompt="A simple cube"):
    """
    Test the text-to-3D model generation API
    
    Args:
        prompt (str): Text description of the desired 3D model
    """
    logger.info(f"Testing text-to-3D generation with prompt: '{prompt}'")
    
    # Request payload
    payload = {
        "prompt": prompt,
        "detail_level": 3,
        "output_formats": ["obj", "glb"]
    }
    
    try:
        # Send generation request
        logger.info("Sending generation request...")
        response = requests.post(GENERATE_URL, json=payload)
        
        if not response.ok:
            logger.error(f"Generation request failed with status {response.status_code}")
            logger.error(f"Response: {response.text}")
            return
        
        # Parse response
        generation_data = response.json()
        model_id = generation_data.get("model_id")
        
        if not model_id:
            logger.error("No model_id in response")
            return
        
        logger.info(f"Generation started with model_id: {model_id}")
        
        # Poll status until complete
        status = "pending"
        start_time = time.time()
        timeout = 120  # 2 minutes timeout
        
        while status in ["pending", "processing"]:
            # Check timeout
            if time.time() - start_time > timeout:
                logger.error("Generation timed out")
                return
            
            # Poll status
            logger.info("Checking generation status...")
            status_response = requests.get(f"{STATUS_URL}{model_id}/")
            
            if not status_response.ok:
                logger.error(f"Status check failed with status {status_response.status_code}")
                logger.error(f"Response: {status_response.text}")
                return
            
            status_data = status_response.json()
            status = status_data.get("status", "unknown")
            progress = status_data.get("progress", 0)
            
            logger.info(f"Status: {status}, Progress: {progress}%")
            
            if status == "completed":
                logger.info("Generation completed successfully!")
                logger.info(f"Result: {json.dumps(status_data, indent=2)}")
                
                # Download the generated model files
                download_models(model_id, status_data.get("available_formats", []))
                break
            elif status == "failed":
                logger.error(f"Generation failed: {status_data.get('error_message', 'Unknown error')}")
                return
            
            # Wait before next poll
            time.sleep(2)
    
    except Exception as e:
        logger.error(f"Error during test: {e}")

def download_models(model_id, formats):
    """
    Download the generated model files
    
    Args:
        model_id (str): Model ID
        formats (list): List of available formats
    """
    for fmt in formats:
        try:
            logger.info(f"Downloading {fmt} file...")
            download_url = f"{DOWNLOAD_URL}{model_id}/{fmt}/"
            
            response = requests.get(download_url)
            if not response.ok:
                logger.error(f"Download failed with status {response.status_code}")
                continue
            
            # Save the file
            download_dir = "downloaded_models"
            os.makedirs(download_dir, exist_ok=True)
            
            filename = f"{download_dir}/{model_id}.{fmt}"
            with open(filename, "wb") as f:
                f.write(response.content)
            
            logger.info(f"Model saved to {filename}")
        
        except Exception as e:
            logger.error(f"Error downloading {fmt} file: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the Text-to-3D API")
    parser.add_argument("--prompt", type=str, default="A blue cube", 
                      help="Text description of the desired 3D model")
    
    args = parser.parse_args()
    
    test_text_to_3d_generation(args.prompt)
