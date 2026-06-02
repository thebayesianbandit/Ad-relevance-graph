import pytorch_lightning as pl
import polars as pl_lib
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import Optional

class AdRelevanceDataModule(pl.LightningDataModule):
    def __init__(self, data_path: pl_lib.DataFrame, batch_size: int = 1):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size

        # Placeholders for tracking shapes and states programmatically
        self.raw_df: Optional[pl_lib.DataFrame] = None
        self.graph_data: Optional[Data] = None
        self.num_user_nodes: int = 0
        self.num_ad_nodes: int = 0
        self.user_feature_dim: int = 0
        self.ad_feature_dim: int = 0

    def prepare_data(self):
        self.raw_df = pl_lib.read_csv(self.data_path)

    def setup(self, stage: Optional[str] = None):
        """
        The core transformation layer: builds vocabularies, maps IDs to internal coordinate
        spaces, processes features with Polars, and packages tensors.
        """
        df = self.raw_df

        # 1. Isolate Node Mappings exactly like your notebook blueprint
        user_mapping = df.select("user_id").unique().with_row_index("user_gnn_id")
        ad_mapping = df.filter(pl_lib.col("is_ad") == True).select("target_id").unique().with_row_index("ad_ids")
        non_ad_mapping = df.filter(pl_lib.col("is_ad") == False).select("target_id").unique().with_row_index("non_ad_ids")

        self.num_user_nodes = user_mapping.height
        self.num_ad_nodes = ad_mapping.height

        # Join mappings back to original dataframe to align coordinate space
        df_with_u = df.join(user_mapping, on="user_id", how="left")
        df_ads = df_with_u.filter(pl_lib.col("is_ad") == True).join(ad_mapping, on="target_id", how="left")
        df_non_ads = df_with_u.filter(pl_lib.col("is_ad") == False).join(non_ad_mapping, on="target_id", how="left")

        # 2. Build Structural Edge Indices
        u_ad_edges = torch.tensor(
            df_ads.select(["user_gnn_id", "ad_ids"]).to_numpy(), dtype=torch.long
        ).t().contiguous()

        u_non_ad_edges = torch.tensor(
            df_non_ads.select(["user_gnn_id", "non_ad_ids"]).to_numpy(), dtype=torch.long
        ).t().contiguous()

        # 3. Vectorize Features using your Polars strategy
        # User feature vector generation
        user_features_df = df_with_u.select(["user_gnn_id", "age_group", "main_interest"]).unique().sort("user_gnn_id")
        user_features_encoded = user_features_df.to_dummies(["age_group", "main_interest"]).drop("user_gnn_id")
        x_user = torch.tensor(user_features_encoded.to_numpy(), dtype=torch.float)
        self.user_feature_dim = x_user.size(1)

        # Ad feature vector generation
        ad_features_df = df_ads.select(["ad_ids", "category"]).unique().sort("ad_ids")
        ad_features_encoded = ad_features_df.to_dummies("category").drop("ad_ids")
        x_ad = torch.tensor(ad_features_encoded.to_numpy(), dtype=torch.float)
        self.ad_feature_dim = x_ad.size(1)

        # Non-Ad feature vector generation
        non_ad_features_df = df_non_ads.select(["non_ad_ids", "category"]).unique().sort("non_ad_ids")
        non_ad_features_encoded = non_ad_features_df.to_dummies("category").drop("non_ad_ids")
        x_non_ad = torch.tensor(non_ad_features_encoded.to_numpy(), dtype=torch.float)

        # 4. Target Generation (Labels for click probability)
        y_ads = torch.tensor(df_ads["click"].to_numpy(), dtype=torch.float).view(-1, 1)

        # 5. Pack everything cleanly into a PyTorch Geometric Custom Data Container
        self.graph_data = Data(
            x_user=x_user,
            x_ad=x_ad,
            x_non_ad=x_non_ad,
            edge_index_ad=u_ad_edges,
            edge_index_non_ad=u_non_ad_edges,
            y_ads=y_ads
        )

    def train_dataloader(self) -> DataLoader:
        """Exposes the training dataset iterator to the training execution engine."""
        # For full graph training (like your GAT), we treat the graph as a single massive batch item
        return DataLoader([self.graph_data], batch_size=self.batch_size, shuffle=False)