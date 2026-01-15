import pyarrow.parquet as pq

table = pq.read_table("city_grouped_data_old.parquet")
pq.write_table(table, "city_grouped_data.parquet")