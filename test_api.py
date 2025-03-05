import requests
import json

def test_available_models():
    response = requests.get("http://localhost:8000/api/text-to-3d/available-models/")
    print("Status Code:", response.status_code)
    
    try:
        data = response.json()
        print("Response Structure:", json.dumps(data[0] if isinstance(data, list) else data, indent=2)[:500])
        print(f"Number of models: {len(data) if isinstance(data, list) else 'unknown'}")
    except Exception as e:
        print("Error parsing response:", e)
        print("Raw response:", response.text[:1000])

def test_generate_model():
    data = {
        "prompt": "Test model generation",
        "trained_model_id": "",  # Empty string for default
        "detail_level": 3,
        "output_formats": {
            "mesh": True,
            "point_cloud": True
        }
    }
    
    response = requests.post(
        "http://localhost:8000/api/text-to-3d/generate/",
        json=data,
        headers={"X-CSRFToken": "dummy-token"}  # Django might need a CSRF token
    )
    
    print("Generate Status Code:", response.status_code)
    try:
        data = response.json()
        print("Generate Response:", json.dumps(data, indent=2))
    except Exception as e:
        print("Error parsing generate response:", e)
        print("Raw generate response:", response.text[:1000])

if __name__ == "__main__":
    print("Testing Available Models API...")
    test_available_models()
    
    print("\nTesting Generate Model API...")
    test_generate_model()
