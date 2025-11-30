"""
Example script to test the API endpoints.
"""
import requests
import json

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health check endpoint."""
    print("Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")


def test_models():
    """Test models list endpoint."""
    print("Testing /models endpoint...")
    response = requests.get(f"{BASE_URL}/models")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_generate():
    """Test code generation endpoint."""
    print("Testing /generate endpoint...")
    payload = {
        "prompt": "Write a Python function to check if a string is a palindrome",
        "model": "qwen2.5-coder:14b-instruct",
        "temperature": 0.3
    }
    
    response = requests.post(f"{BASE_URL}/generate", json=payload)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Model: {result['model']}")
    print(f"Response:\n{result['response']}\n")


def test_complete():
    """Test code completion endpoint."""
    print("Testing /complete endpoint...")
    payload = {
        "prompt": """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        """,
        "model": "qwen2.5-coder:14b-instruct"
    }
    
    response = requests.post(f"{BASE_URL}/complete", json=payload)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Completion:\n{result['response']}\n")


def test_chat():
    """Test chat endpoint."""
    print("Testing /chat endpoint...")
    payload = {
        "prompt": "What is the time complexity of quicksort?",
        "model": "qwen2.5-coder:14b-instruct"
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Answer:\n{result['response']}\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Local Code LLM API")
    print("=" * 60 + "\n")
    
    try:
        test_health()
        test_models()
        test_generate()
        test_complete()
        test_chat()
        
        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API. Make sure the server is running:")
        print("  uv run python api.py")
    except Exception as e:
        print(f"Error: {e}")
