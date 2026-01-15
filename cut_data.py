import pandas as pd

df = pd.read_parquet("data_full.parquet")

city_grouped = (
    df.groupby("CityName", as_index=False)
      .agg(total_rides=("total_rides", "sum"))
      .sort_values("total_rides", ascending=False)
      .reset_index(drop=True)
)

# Select ranks 5 to 15
selected_cities = city_grouped.iloc[4:9]["CityName"]

df_small = df[df["CityName"].isin(selected_cities)].copy()

df_small.to_parquet("data_small.parquet", index=False)
