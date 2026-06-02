import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class AdRelevanceLightningModule(pl.LightningModule):
    def __init__(self, user_dim: int, ad_dim: int, hidden_dim: int = 16, lr: float = 0.01):
        """
        Initializes the model architecture using modular feature dimensions
        instead of relying on hardcoded global notebook states.
        """
        super().__init__()
        # Automatically saves hyperparams for production tracking and audits
        self.save_hyperparameters()
        
        self.lr = lr
        
        # 1. Feature Projection Layers (Matching your notebook strategy)
        self.user_proj = nn.Linear(user_dim, hidden_dim)
        self.ad_proj = nn.Linear(ad_dim, hidden_dim)
        self.non_ad_proj = nn.Linear(ad_dim, hidden_dim)
        
        # 2. Graph Attention Networks
        self.gat_user = GATConv(hidden_dim, hidden_dim, heads=1, concat=False)
        self.gat_ad = GATConv(hidden_dim, hidden_dim, heads=1, concat=False)
        
        # 3. Final Prediction Multi-Layer Perceptron (MLP)
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 4. Standardized Loss Objective
        self.loss_fn = nn.BCELoss()

    def forward(self, x_user, x_ad, x_non_ad, edge_index_ad, edge_index_non_ad) -> torch.Tensor:
        """
        The pure forward computation pass. Completely agnostic of hardware
        targets (.to('cuda')) or outer optimization steps.
        """
        # Step A: Project distinct raw properties into a uniform latent space
        u_proj = self.user_proj(x_user)
        ad_proj = self.ad_proj(x_ad)
        non_ad_proj = self.non_ad_proj(x_non_ad)
        
        # Step B: Execute Message-Passing via Graph Attention convolutions
        u_gnn = self.gat_user(u_proj, edge_index_ad)
        ad_gnn = self.gat_ad(ad_proj, edge_index_ad)
        
        # Step C: Extract user/ad coordinate coordinates matching active interactions
        row, col = edge_index_ad
        edge_user_feats = u_gnn[row]
        edge_ad_feats = ad_gnn[col]
        
        # Step D: Concat embeddings and score the probability of a click interaction
        edge_feats = torch.cat([edge_user_feats, edge_ad_feats], dim=1)
        return self.prediction_head(edge_feats)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Automated execution step for a training batch. PyTorch Lightning handles 
        loss.backward(), optimizer.step(), and device transfer automatically.
        """
        # PyTorch Geometric automatically packs the attributes inside the batch object
        out = self(
            batch.x_user, 
            batch.x_ad, 
            batch.x_non_ad, 
            batch.edge_index_ad, 
            batch.edge_index_non_ad
        )
        
        loss = self.loss_fn(out, batch.y_ads)
        
        # Automated logging engine hook
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Declares the optimization algorithms to apply across parameters."""
        return torch.optim.Adam(self.parameters(), lr=self.lr)