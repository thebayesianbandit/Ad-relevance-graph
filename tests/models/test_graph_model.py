import pytest
import torch
from src.models.graph_model import AdRelevanceLightningModule 

def test_model_forward_pass_shape_integrity():
    """
    Ensures that passing arbitrary tensor profiles through the network
    correctly yields a formatted array of click probabilities.
    """
    # 1. Initialize random dummy tensors representing independent node dimensions
    num_users, user_dim = 10, 8
    num_ads, ad_dim = 5, 4
    num_edges = 12
    hidden_dim = 16
    
    x_user = torch.randn(num_users, user_dim)
    x_ad = torch.randn(num_ads, ad_dim)
    x_non_ad = torch.randn(num_ads, ad_dim)
    
    # Generate arbitrary matching coordinate indices
    edge_index_ad = torch.randint(0, min(num_users, num_ads), (2, num_edges), dtype=torch.long)
    edge_index_non_ad = torch.randint(0, min(num_users, num_ads), (2, num_edges), dtype=torch.long)
    
    # 2. Instantiate our Production GAT network module
    model = AdRelevanceLightningModule(
        user_dim=user_dim, 
        ad_dim=ad_dim, 
        hidden_dim=hidden_dim
    )
    
    # 3. Execute the forward pass
    predictions = model(x_user, x_ad, x_non_ad, edge_index_ad, edge_index_non_ad)
    
    # 4. Assert structural output validity
    # The network must spit out exactly one probability score per target edge interaction
    assert predictions.shape == (num_edges, 1), \
        f"Output dimension mismatch. Expected {(num_edges, 1)}, got {predictions.shape}."
        
    # All scores must represent safe probabilities bounded between 0.0 and 1.0
    assert torch.all(predictions >= 0.0) and torch.all(predictions <= 1.0), \
        "Model generated invalid scoring bounds outside [0, 1]."