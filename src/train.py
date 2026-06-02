# src/train.py
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
import mlflow
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from models.graph_model import AdRelevanceLightningModule

def generate_in_memory_batch(user_dim=8, ad_dim=4, num_edges=100):
    """
    Bypasses the DataModule entirely by generating the final PyTorch Geometric 
    tensor shapes directly in memory, matching your notebook's end results.
    """
    num_users = 50
    num_ads = 30
    
    # 1. Create clean, pre-shaped node feature matrices
    x_user = torch.randn(num_users, user_dim)
    x_ad = torch.randn(num_ads, ad_dim)
    x_non_ad = torch.randn(num_ads, ad_dim)
    
    # 2. Generate random graph edge connections (coordination index arrays)
    edge_index_ad = torch.randint(0, min(num_users, num_ads), (2, num_edges), dtype=torch.long)
    edge_index_non_ad = torch.randint(0, min(num_users, num_ads), (2, num_edges), dtype=torch.long)
    
    # 3. Create binary target labels for whether an ad was clicked
    y_ads = torch.randint(0, 2, (num_edges, 1), dtype=torch.float32)
    
    # 4. Pack everything into a standardized PyTorch Geometric Data container
    graph_data = Data(
        x_user=x_user,
        x_ad=x_ad,
        x_non_ad=x_non_ad,
        edge_index_ad=edge_index_ad,
        edge_index_non_ad=edge_index_non_ad,
        y_ads=y_ads,
        num_nodes=num_users + num_ads
    )
    
    # Return as a simple list inside a PyTorch Geometric DataLoader
    return DataLoader([graph_data], batch_size=1)

def run_training(config_path: str = "config/config.yaml"):
    # 1. Load the runtime configurations
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Configure MLflow Tracking Environment
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow_logger = MLFlowLogger(
        experiment_name=config["mlflow"]["experiment_name"],
        tracking_uri=config["mlflow"]["tracking_uri"]
    )

    # 3. Bypass DataModule: Generate tensor dimensions manually
    user_feature_dim = 8
    ad_feature_dim = 4
    train_dataloader = generate_in_memory_batch(user_dim=user_feature_dim, ad_dim=ad_feature_dim)

    # 4. Instantiate the model layer
    model = AdRelevanceLightningModule(
        user_dim=user_feature_dim,
        ad_dim=ad_feature_dim,
        hidden_dim=config["model"]["hidden_dim"],
        lr=config["model"]["lr"]
    )

    # 5. Instantiate PyTorch Lightning's Trainer
    trainer = pl.Trainer(
        max_epochs=config["trainer"]["max_epochs"],
        accelerator=config["trainer"]["accelerator"],
        logger=mlflow_logger,
        enable_checkpointing=True
    )

    # 6. Run the tracked training run
    print("🚀 Commencing tracked training pipeline bypassing the DataModule...")
    with mlflow.start_run(run_id=mlflow_logger.run_id):
        mlflow.log_params(config["model"])
        mlflow.log_params(config["trainer"])
        
        # Pass our in-memory loader directly to fit instead of a data module!
        trainer.fit(model, train_dataloaders=train_dataloader)
        
        print("💾 Saving model artifact to immutable registry...")
        mlflow.pytorch.log_model(model, artifact_path="model")
        
    print("🏁 Tracked training complete! Open your MLflow dashboard.")

if __name__ == "__main__":
    run_training()