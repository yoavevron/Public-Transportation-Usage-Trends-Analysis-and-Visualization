import pandas as pd
import numpy as np
import os
import time
import calendar



p = 'datasets/תיקופי תחבצ'
raw_data_path = 'datasets'

clean_data_path = os.path.join(raw_data_path, 'clean', 'data.csv')

data_type = 'category'

def day_to_weekday(year, month, day):
    return calendar.weekday(year, month, day)

# # Load public-transportation stations data
stations = pd.read_csv('stations.csv', sep="|")
stations['StationId'] = stations['StationId'].astype(data_type)
stations_idx = stations.set_index('StationId')

# Load public-transportation usage data (tickets/travels)
tickets_datasets = []
tickets_data_path = os.path.join(raw_data_path, 'Raw', 'Tickets')

for file in os.listdir(tickets_data_path):
    if not file.endswith(".csv"):
        continue
    d = pd.read_csv(os.path.join(tickets_data_path, file), sep="|")
    d['StationId'] = d['StationId'].astype(data_type)
    # join with stations dataframe each travel dataframe
    tickets_datasets.append(d.join(stations_idx, on='StationId'))

print(f'{len(tickets_datasets)} ".csv" files successfully loaded!')

# concat the different travels by year into one unified dataframe
df = pd.concat(tickets_datasets, ignore_index=True)
print(f"There are total of : {len(df)} rows in the dataset")


# calculate the total travels in each city to keep only the 30 cities with the highest travels
day_cols = [c for c in df.columns if c.startswith("day_")]
df["total_rides"] = df[day_cols].sum(axis=1, skipna=True)

city_rides = (
    df.groupby("CityName", as_index=False)
      .agg(total_rides=("total_rides", "sum"))
      .sort_values("total_rides", ascending=False)
)

top_30_cities = (
    city_rides
    .head(30)["CityName"]
    .tolist()
)

df_top30 = df[df["CityName"].isin(top_30_cities)]
print(f"There are {len(df_top30)} stations in the 30 top cities")

# remove midnight travels
keep_periods = [
    "06:00 - 08:59 - שיא בוקר",
    "09:00 - 11:59 - שפל יום 1",
    "12:00 - 14:59 - שפל יום 2",
    "15:00 - 18:59 - שיא ערב",
    "19:00 - 23:59 - שפל ערב",
]

df_top30 = df_top30[df_top30["LowOrPeakDescFull"].isin(keep_periods)]

df_top30["df_top30"] = (
    df_top30["LowOrPeakDescFull"]
    .str.extract(r"^(\d{2}:\d{2}\s*-\s*\d{2}:\d{2})")
)

df = df_top30.copy()

# melt the days
day_cols = [c for c in df.columns if c.startswith("day_")]

df_long = df.melt(
    id_vars=[
        "StationId",
        "StationName",
        "CityName",
        "LowOrPeakDescFull",
        "year_key",
        "month_key",
        "Lat",
        "Long",
    ],
    value_vars=day_cols,
    var_name="day_in_month",
    value_name="rides"
)
df_long = df_long.dropna(subset=["rides"])

df_long["day_in_month"] = df_long["day_in_month"].str[4:].astype(int)
dates = pd.to_datetime(
    dict(
        year=df_long["year_key"],
        month=df_long["month_key"],
        day=df_long["day_in_month"]
    ),
    errors="coerce"
)
# sunday=1 ... saturday=7
df_long["day_in_week"] = ((dates.dt.weekday + 1) % 7) + 1

# aggregate sum of the days into groups by the new key (stationid, year, month, day, hour)
df_final = (
    df_long
    .groupby(
        [
            "StationId",
            "StationName",
            "CityName",
            "LowOrPeakDescFull",
            "year_key",
            "month_key",
            "day_in_week",
            "Lat",
            "Long",
        ],
        as_index=False
    )
    .agg(total_rides=("rides", "sum"))
)

print(len(df_final))
print(df_final.head())

# save compressed version of the dataframe (around 3GB of csv file comapred ro 130MB in parquet binary)
df_final.to_parquet(
    "data.parquet",
    engine="pyarrow",   
    compression="snappy"  
)
