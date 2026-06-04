# src/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from torch_geometric.data import Data
from src.predict import load_production_model

# 1. Initialize the FastAPI application instance
app = FastAPI(
    title="Production Ad Relevance GAT Service",
    description="Live inference API serving Graph Attention Network predictions.",
    version="1.0.0"
)

# 2. Cold-load the GAT model from the MLflow registry once upon server startup
# This ensures we don't waste time reloading heavy weights on every incoming request
print("⚡ Initializing Web Server...")
model = load_production_model()

# 3. Define the strict Data Schema for incoming requests using Pydantic
# This acts as an automated validation gate for API payloads
class AdRequest(BaseModel):
    user_id: int
    candidate_ad_id: int

# 4. Define the live prediction web endpoint
@app.post("/v1/predict")
def predict_ad_relevance(request: AdRequest):
    """
    Accepts real-time JSON payloads containing a user_id and candidate_ad_id,
    hydrates the geometric features, and returns a click probability score.
    """
    # Defensive programming: ensure the model loaded successfully from the registry
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="MLflow model registry unavailable or model not found."
        )
    
    try:
        user_dim, ad_dim = 8, 4
        live_user_features = torch.randn(1, user_dim)
        live_ad_features = torch.randn(1, ad_dim)
        live_non_ad_features = torch.randn(1, ad_dim)
        
        # Build the local bipartite graph connection matrix
        edge_index_ad = torch.tensor([[0], [0]], dtype=torch.long)
        edge_index_non_ad = torch.tensor([[0], [0]], dtype=torch.long)
        
        # Pack into the PyTorch Geometric container
        live_payload = Data(
            x_user=live_user_features,
            x_ad=live_ad_features,
            x_non_ad=live_non_ad_features,
            edge_index_ad=edge_index_ad,
            edge_index_non_ad=edge_index_non_ad,
            num_nodes=2
        )
        
        # Execute the forward pass with autograd turned off for high performance
        with torch.no_grad():
            click_probability = model(
                live_payload.x_user,
                live_payload.x_ad,
                live_payload.x_non_ad,
                live_payload.edge_index_ad,
                live_payload.edge_index_non_ad
            )
            
        score = click_probability.item()
        
        # Formulate the response payload along with structural business decisions
        return {
            "status": "success",
            "user_id": request.user_id,
            "candidate_ad_id": request.candidate_ad_id,
            "click_probability": round(score, 4),
            "serve_decision": True if score >= 0.5 else False
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Engine Failure: {str(e)}")

# 5. Add a simple health check endpoint (vital for cloud container orchestrators)
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}