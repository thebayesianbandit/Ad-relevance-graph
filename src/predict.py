# src/predict.py
import yaml
import torch
import mlflow.pytorch
from torch_geometric.data import Data

def load_production_model(config_path: str = "config/config.yaml"):
    """
    Connects to the centralized tracking database and pulls down the latest 
    approved version of your Graph Attention Network from the Model Registry.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Bind to our local tracking database
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    
    # Query the central registry by the name we assigned in Phase 2
    # The 'latest' tag automatically routes to the newest approved version
    model_name = "ad_relevance_gat"
    model_uri = f"models:/{model_name}/latest"
    
    print(f"📦 Fetching active model from central registry: {model_uri}...")
    
    try:
        # MLflow automatically rebuilds the object back as a fully functional PyTorch Lightning module
        model = mlflow.pytorch.load_model(model_uri)
        model.eval()  # Turn off gradients and dropout for deterministic inference
        print("✅ Model successfully loaded and initialized for real-time serving.")
        return model
    except Exception as e:
        print(f"❌ Failed to pull model from registry: {e}")
        print("\n💡 Tip: Ensure you clicked 'Register Model' in the MLflow web UI dashboard")
        print("   and named the registered model exactly: 'Ad_Relevance_GAT_Production'")
        return None

def serve_real_time_prediction(model, user_dim=8, ad_dim=4):
    """
    Simulates a live production request: a user requests a page, 
    and we must score a candidate ad instantly.
    """
    print("\n⚡ Incoming ad slot request received from web gateway...")
    
    # Simulate fetching raw structural features for 1 user and 1 candidate ad.
    # In a live company like Uber or Pinterest, these embeddings are pulled 
    # in single-digit milliseconds from an in-memory feature store like Redis.
    live_user_features = torch.randn(1, user_dim)
    live_ad_features = torch.randn(1, ad_dim)
    live_non_ad_features = torch.randn(1, ad_dim)  # Baseline node to satisfy tensor shapes
    
    # Construct the edge index layout for this single user-ad pair
    # User index 0 is connected to Ad index 0 
    edge_index_ad = torch.tensor([[0], [0]], dtype=torch.long)
    edge_index_non_ad = torch.tensor([[0], [0]], dtype=torch.long)
    
    # Package into a real-time geometric payload
    live_payload = Data(
        x_user=live_user_features,
        x_ad=live_ad_features,
        x_non_ad=live_non_ad_features,
        edge_index_ad=edge_index_ad,
        edge_index_non_ad=edge_index_non_ad,
        num_nodes=2
    )
    
    # Completely disable the autograd engine to optimize memory and minimize request latency
    with torch.no_grad():
        # Unpack the tensors directly to match the model's forward() signature
        click_probability = model(
            live_payload.x_user,
            live_payload.x_ad,
            live_payload.x_non_ad,
            live_payload.edge_index_ad,
            live_payload.edge_index_non_ad
        )
        
    score = click_probability.item()
    print(f"🎯 Inference Complete | Calculated Click Probability: {score:.4f}")
    
    # Business Logic Decision Gate
    if score >= 0.5:
        print("🚀 DECISION: HIGH RELEVANCE. Serve this ad immediately to maximize monetization!")
    else:
        print("🛑 DECISION: LOW RELEVANCE. Skip this ad slot; look for alternative inventory.")

if __name__ == "__main__":
    gat_model = load_production_model()
    if gat_model:
        serve_real_time_prediction(gat_model)