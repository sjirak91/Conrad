import pytest
import requests
import time
import os
import subprocess
import signal
from typing import Dict, List

# Define the API URL
API_URL = os.environ.get("API_URL", "http://localhost:8000")

class TestNERAPIIntegration:
    @classmethod
    def setup_class(cls):
        """
        Set up for the entire test class - start the API server if needed.
        This is only used when running tests directly, not when API is already running.
        """
        # Check if the API is already running
        try:
            response = requests.get(f"{API_URL}/health", timeout=2)
            if response.status_code == 200:
                print("API is already running, skipping server start")
                cls.server_process = None
                return
        except requests.exceptions.ConnectionError:
            print("API not running, starting server for tests...")
        
        # Start the API server as a subprocess
        cls.server_process = subprocess.Popen(
            ["python", "app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for the server to start
        max_retries = 10
        for i in range(max_retries):
            try:
                response = requests.get(f"{API_URL}/health", timeout=2)
                if response.status_code == 200:
                    print(f"API server started and responding after {i+1} attempts")
                    break
            except requests.exceptions.ConnectionError:
                print(f"Waiting for API server to start (attempt {i+1}/{max_retries})...")
                time.sleep(2)
        else:
            cls.teardown_class()
            raise RuntimeError("Failed to start API server for tests")
    
    @classmethod
    def teardown_class(cls):
        """Clean up after all tests - stop the API server if we started it."""
        if cls.server_process:
            print("Stopping API server...")
            cls.server_process.terminate()
            cls.server_process.wait(timeout=5)
            print("API server stopped")
    
    def test_health_endpoint(self):
        """Test that the health endpoint returns correct status."""
        response = requests.get(f"{API_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "model_loaded" in data
    
    def test_root_endpoint(self):
        """Test that the root endpoint returns API information."""
        response = requests.get(f"{API_URL}/")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "api_version" in data
        assert "endpoints" in data
        assert "model_loaded" in data
    
    def test_entities_endpoint(self):
        """Test that the entities endpoint returns supported entity types."""
        response = requests.get(f"{API_URL}/entities")
        assert response.status_code == 200
        data = response.json()
        assert "entity_types" in data
        assert isinstance(data["entity_types"], list)
        assert len(data["entity_types"]) > 0
    
    def test_predict_simple_text(self):
        """Test prediction with a simple text that should contain entities."""
        sample_text = "John Smith works at Microsoft in Seattle and has a meeting on January 15th."
        
        response = requests.post(
            f"{API_URL}/predict",
            json={"text": sample_text}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check the response structure
        assert "text" in data
        assert "entities" in data
        assert "tagged_text" in data
        
        # Original text should be returned
        assert data["text"] == sample_text
        
        # There should be entities detected
        assert isinstance(data["entities"], list)
        
        # Check entity structure if any were found
        if data["entities"]:
            entity = data["entities"][0]
            assert "text" in entity
            assert "label" in entity
            assert "start" in entity
            assert "end" in entity
            assert "confidence" in entity
            assert 0 <= entity["confidence"] <= 1
    
    def test_predict_with_confidence_threshold(self):
        """Test prediction with different confidence thresholds."""
        sample_text = "John Smith works at Microsoft in Seattle and has a meeting on January 15th."
        
        # First request with default threshold
        default_response = requests.post(
            f"{API_URL}/predict",
            json={"text": sample_text}
        )
        
        # Second request with high threshold
        high_threshold_response = requests.post(
            f"{API_URL}/predict",
            json={"text": sample_text, "confidence_threshold": 0.9}
        )
        
        # Third request with low threshold
        low_threshold_response = requests.post(
            f"{API_URL}/predict",
            json={"text": sample_text, "confidence_threshold": 0.1}
        )
        
        assert default_response.status_code == 200
        assert high_threshold_response.status_code == 200
        assert low_threshold_response.status_code == 200
        
        default_entities = default_response.json()["entities"]
        high_threshold_entities = high_threshold_response.json()["entities"]
        low_threshold_entities = low_threshold_response.json()["entities"]
        
        # High threshold should return fewer or equal entities than default
        assert len(high_threshold_entities) <= len(default_entities)
        
        # Low threshold should return more or equal entities than default
        assert len(low_threshold_entities) >= len(default_entities)
    
    def test_batch_predict(self):
        """Test batch prediction endpoint."""
        batch_data = [
            {"text": "John Smith works at Microsoft.", "confidence_threshold": 0.5},
            {"text": "Apple released a new iPhone in September.", "confidence_threshold": 0.3}
        ]
        
        response = requests.post(f"{API_URL}/batch_predict", json=batch_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "results" in data
        assert isinstance(data["results"], list)
        assert len(data["results"]) == 2
        
        for result in data["results"]:
            assert "text" in result
            if "error" not in result:
                assert "entities" in result
                assert "tagged_text" in result
    
    def test_predict_no_entities(self):
        """Test prediction with text that doesn't contain entities."""
        sample_text = "This is a simple text without any named entities."
        
        response = requests.post(
            f"{API_URL}/predict",
            json={"text": sample_text}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Original text should be returned
        assert data["text"] == sample_text
        
        # Empty list of entities is expected
        assert isinstance(data["entities"], list)
        
        # Tagged text should be provided
        assert "tagged_text" in data
    
    def test_predict_error_handling(self):
        """Test API error handling with invalid input."""
        # Test with empty request
        response = requests.post(f"{API_URL}/predict", json={})
        assert response.status_code != 200, "Empty request should fail"
        
        # Test with null text
        response = requests.post(f"{API_URL}/predict", json={"text": None})
        assert response.status_code != 200, "Null text should fail"
        
        # Test with very long text to check if API handles it properly
        very_long_text = "A" * 100000  # 100K characters
        response = requests.post(f"{API_URL}/predict", json={"text": very_long_text})
        assert response.status_code == 200, "API should handle very long text"
        
        # Test with invalid confidence threshold
        response = requests.post(
            f"{API_URL}/predict", 
            json={"text": "Sample text", "confidence_threshold": 2.0}
        )
        assert response.status_code != 200, "Invalid confidence threshold should fail"
        
        response = requests.post(
            f"{API_URL}/predict", 
            json={"text": "Sample text", "confidence_threshold": -0.5}
        )
        assert response.status_code != 200, "Negative confidence threshold should fail"

if __name__ == "__main__":
    pytest.main(["-xvs"]) 