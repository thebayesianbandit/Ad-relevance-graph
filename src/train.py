# src/train.py
import yaml
import os
import pytorch_lightning as pl
from data.datamodule import AdRelevanceDataModule
from models.graph_model import AdRelevanceLightningModule

def run_training(config_path: str = "config/config.yaml"):
    # 1. Load the runtime configurations
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Instantiate our standardized data factory step
    datamodule = AdRelevanceDataModule(
        data_path=config["data"]["raw_snapshot_path"],
        batch_size=config["data"]["batch_size"]
    )
    
    # Run data preparation explicitly to extract network dimensions dynamically
    datamodule.prepare_data()
    datamodule.setup()

    # 3. Instantiate the model layer using parameters injected from our configuration file
    model = AdRelevanceLightningModule(
        user_dim=datamodule.user_feature_dim,
        ad_dim=datamodule.ad_feature_dim,
        hidden_dim=config["model"]["hidden_dim"],
        lr=config["model"]["lr"]
    )

    # 4. Instantiate PyTorch Lightning's production execution engine
    # Notice how we completely eliminate manual 'for epoch in range()' script loops!
    trainer = pl.Trainer(
        max_epochs=config["trainer"]["max_epochs"],
        accelerator=config["trainer"]["accelerator"],
        enable_checkpointing=True  # Automatically saves the best model states for you
    )

    # 5. Kick off the automated execution pipeline
    print("🚀 Commencing production training pipeline...")
    trainer.fit(model, datamodule=datamodule)
    print("🏁 Training complete. Model artifacts secured.")

if __name__ == "__main__":
    run_training()