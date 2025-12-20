import streamlit as st
import plotly.express as px
import pandas as pd

@st.cache_data
def load_data():
    return pd.read_csv("../Create Dataset/data/clean/data.csv")

df = load_data()
print(df.columns)

year_range = st.slider(
    "בחר טווח שנים",
    int(df.year_key.min()),
    int(df.year_key.max()),
    (int(df.year_key.min()), int(df.year_key.max()))
)

filtered = df[df.year_key.between(*year_range)]

fig = px.bar(
    filtered
    .groupby(["CityName", "LowOrPeakDescFull"])["total_rides"]
    .sum()
    .reset_index(),
    x="CityName",
    y="total_rides",
    color="LowOrPeakDescFull",
    title="נסיעות לפי עיר"
)


st.plotly_chart(fig, use_container_width=True)
