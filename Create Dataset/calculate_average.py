import pandas as pd

DATA_PATH = "data/clean/data.parquet"

df = pd.read_parquet(DATA_PATH)

hours_dict = {
    "06:00 - 08:59 - שיא בוקר": 3,
    "09:00 - 11:59 - שפל יום 1": 3,
    "12:00 - 14:59 - שפל יום 2": 3,
    "15:00 - 18:59 - שיא ערב": 4,
    "19:00 - 23:59 - שפל ערב": 5,
}

df["hours_in_group"] = df["LowOrPeakDescFull"].map(hours_dict)
df["avg_rides_per_hour"] = df["total_rides"] / df["hours_in_group"]
df = df.drop(["hours_in_group", "Unnamed: 0"], axis=1)
df.to_parquet('data/clean/final_data.parquet')