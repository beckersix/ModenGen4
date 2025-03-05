"""
Test script for the text-to-3D functionality.
This script sends a request to the text-to-3D API endpoint to generate a model from a text prompt.
"""

import requests
import json
import time
import sys
import os

# Ensure we can import from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Base URL for the API
BASE_URL = "http://127.0.0.1:8000/api/text-to-3d"

def test_text_to_3d_generation():
    """Test the text-to-3D generation endpoint with basic shapes."""
    
    # List of basic shapes to test
    test_prompts = [
        "A blue cube",
        "A red sphere",
        "A green cylinder",
        "A yellow cone",
        "A purple torus"
    ]
    
    for prompt in test_prompts:
        print(f"\nTesting prompt: '{prompt}'")
        
        # Prepare the request data
        data = {
            "prompt": prompt,
            "detail_level": 3,
            "output_formats": ["mesh", "point_cloud"]
        }
        
        # Send the request to generate the model
        print("Sending generation request...")
        response = requests.post(f"{BASE_URL}/generate", json=data)
        
        if response.status_code == 200:
            result = response.json()
            model_id = result.get("model_id")
            print(f"Generation started. Model ID: {model_id}")
            
            # Poll for completion
            print("Polling for completion...")
            status = "pending"
            while status in ["pending", "processing"]:
                time.sleep(2)  # Wait 2 seconds between polls
                status_response = requests.get(f"{BASE_URL}/status/{model_id}")
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    status = status_data.get("status")
                    print(f"Current status: {status}")
                else:
                    print(f"Error checking status: {status_response.status_code}")
                    break
            
            if status == "completed":
                print(f"Model generation complete!")
                print(f"Files available at: {status_data.get('files', {})}")
            else:
                print(f"Generation failed or timed out. Final status: {status}")
        else:
            print(f"Error generating model: {response.status_code}")
            print(response.text)

if __name__ == "__main__":
    test_text_to_3d_generation()
