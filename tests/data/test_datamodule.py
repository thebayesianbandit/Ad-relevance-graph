import pytest
import torch
from src.data.datamodule import AdRelevanceDataModule

def test_ad_relevance_datamodule_shape_integrity(mock_raw_data, tmp_path):
    """
    Verifies that the processing factory transforms raw tables 
    into aligned, mathematically valid tensor shapes.
    """
    # 1. Create a temporary CSV path using pytest's built-in tmp_path fixture
    # This prevents creating leftover garbage files on your machine or CI runner
    data_path = tmp_path / "test_snapshot.csv"
    mock_raw_data.write_csv(data_path)

    # 2. Initialize and run the production datamodule pipeline
    dm = AdRelevanceDataModule(data_path=str(data_path))
    dm.prepare_data()
    dm.setup()

    # 3. Defensive Assertions (The structural contract)
    assert dm.graph_data is not None, "Data module failed to generate the graph structure."
    
    # Verify coordinates match targets perfectly (Number of Ad edges == Number of Labels)
    num_ad_edges = dm.graph_data.edge_index_ad.size(1)
    num_labels = dm.graph_data.y_ads.size(0)
    assert num_ad_edges == num_labels, (
        f"Dimension Mismatch: Had {num_ad_edges} ad edges but {num_labels} click labels."
    )

    # Verify tensor feature constraints (Must be Float32 numbers, not strings)
    assert dm.graph_data.x_user.dtype == torch.float32
    assert dm.graph_data.x_ad.dtype == torch.float32
    
    # Assert expected user shape matches unique users in mock data (101, 102, 103 = 3 nodes)
    assert dm.graph_data.x_user.size(0) == 3