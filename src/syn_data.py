# src/seed_data.py
import os
import polars as pl
import numpy as np

def generate_and_save_synthetic_data(output_path: str = "data/raw_snapshot.csv"):
    """
    Generates mock user-ad interaction data exactly matching your notebook's 
    synthetic structure and saves it to disk as a production data seed.
    """
    print("🧪 Generating synthetic ad-relevance dataset...")
    np.random.seed(42)
    
    num_records = 1000
    
    data = {
        "user_id": np.random.randint(1, 101, size=num_records),
        "target_id": np.random.randint(500, 600, size=num_records),
        "is_ad": np.random.choice([0, 1], size=num_records, p=[0.6, 0.4]),
        "clicked": np.random.choice([0, 1], size=num_records, p=[0.8, 0.2]),
        "age_group": np.random.randint(0, 4, size=num_records)
    }
    
    df = pl.DataFrame(data)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.write_csv(output_path)
    print(f"✅ Data seed secured successfully at: {output_path} ({num_records} rows)")

if __name__ == "__main__":
    generate_and_save_synthetic_data()