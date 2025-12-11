"""Test API endpoint when running locally."""

import warnings
import pytest
import requests


def is_api_running():
    """Check if API is running on localhost:8000."""
    try:
        response = requests.get("http://127.0.0.1:8000", timeout=2)
        return response.status_code < 500
    except (requests.ConnectionError, requests.Timeout):
        return False


@pytest.mark.skipif(
    not is_api_running(),
    reason="API is not running on 127.0.0.1:8000"
)
def test_api():
    """Test /scan/permalint endpoint returns expected keys."""
    try:
        response = requests.get(
            "http://127.0.0.1:8000/scan/permalint",
            timeout=10
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        result = response.json()
        
        # Assert required keys are present
        required_keys = {"package", "raw_data", "features", "prediction"}
        assert required_keys.issubset(result.keys()), (
            f"Missing keys. Expected {required_keys}, "
            f"got {set(result.keys())}"
        )
    except requests.ConnectionError:
        warnings.warn("did not test the API")
        pytest.skip("API connection failed")
    except requests.Timeout:
        warnings.warn("did not test the API")
        pytest.skip("API timeout")
