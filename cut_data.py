import pandas as pd

df = pd.read_parquet('full_data.parquet')

city_grouped = (
    df.groupby("CityName", as_index=False)
      .agg(total_rides=("total_rides", "sum"))
)

top_10_cities = city_grouped.nlargest(10, "total_rides")["CityName"]
df_small = df[df["CityName"].isin(top_10_cities)].copy()
df_small.to_parquet("data.parquet", index=False)