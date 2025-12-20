import pandas as pd

# csv_path = "data/clean/data.csv"
csv_path = "data/clean/data.csv"
parquet_path = "data/clean/data.parquet"

df = pd.read_csv(csv_path)

df.to_parquet(
    parquet_path,
    engine="pyarrow",   
    compression="snappy"  
)
