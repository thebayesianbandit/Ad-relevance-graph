import pytest
import polars as pl
import random

@pytest.fixture
def mock_raw_data():
    """Generates a small, deterministic synthetic Polars DataFrame for testing."""
    random.seed(42)
    
    # Simulating a small subset of your real-world data schema
    data = {
        "user_id": [101, 102, 101, 103, 102],
        "target_id": [501, 502, 901, 501, 902],
        "is_ad": [True, True, False, True, False],
        "click": [1, 0, 0, 1, 0],
        "age_group": ["18-25", "26-34", "18-25", "44+", "26-34"],
        "main_interest": ["Fashion", "Food", "Fashion", "Home", "Food"],
        "category": ["Fashion", "Exercise", "Home", "Fashion", "Animals"]
    }
    return pl.DataFrame(data)