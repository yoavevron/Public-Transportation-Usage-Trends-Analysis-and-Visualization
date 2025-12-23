import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np

st.set_page_config(layout="wide")

DATA_PATH = "../Create Dataset/data/clean/data.parquet"

DAY_MAP = {
    1: "ראשון",
    2: "שני",
    3: "שלישי",
    4: "רביעי",
    5: "חמישי",
    6: "שישי",
    7: "שבת",
}

# =========================================================
# Load + prepare (cached)
# =========================================================
@st.cache_data(show_spinner=True)
def load_and_prepare(path: str):
    df = pd.read_parquet(path)

    df = df[
        [
            "StationId",
            "StationName",
            "CityName",
            "Lat",
            "Long",
            "LowOrPeakDescFull",
            "day_in_week",
            "year_key",
            "total_rides",
        ]
    ].copy()

    stations = (
        df[["StationId", "StationName", "CityName", "Lat", "Long"]]
        .drop_duplicates(subset=["StationId"])
        .reset_index(drop=True)
    )

    fact = (
        df.groupby(
            ["StationId", "year_key", "LowOrPeakDescFull", "day_in_week"],
            as_index=False,
        )
        .agg(total_rides=("total_rides", "sum"))
    )

    return (
        fact,
        stations,
        int(fact.year_key.min()),
        int(fact.year_key.max()),
        sorted(fact.LowOrPeakDescFull.unique()),
        sorted(fact.day_in_week.unique()),
        sorted(stations.CityName.dropna().unique()),
    )


fact, stations, year_min, year_max, time_values, day_values, city_values = load_and_prepare(
    DATA_PATH
)

# =========================================================
# Page selector
# =========================================================
page = st.sidebar.radio("Page", ["🗺️ Map", "📊 Other (placeholder)"])

# =========================================================
# ======================= PAGE 1 ==========================
# =========================================================
if page == "🗺️ Map":

    # -----------------------------
    # Sidebar – Filters
    # -----------------------------
    st.sidebar.header("Filters")

    years = st.sidebar.slider("Years", year_min, year_max, (year_min, year_max))

    st.sidebar.divider()

    # ---- Time of day
    st.sidebar.subheader("Time of day")
    selected_hours = [
        v for v in time_values if st.sidebar.checkbox(v, value=True, key=f"tod_{v}")
    ]

    st.sidebar.divider()

    # ---- Days of week (with select/clear all)
    st.sidebar.subheader("Day of week")

    if "days" not in st.session_state:
        st.session_state["days"] = day_values[:]

    d1, d2 = st.sidebar.columns(2)
    if d1.button("Select all days", use_container_width=True):
        st.session_state["days"] = day_values[:]
    if d2.button("Clear all days", use_container_width=True):
        st.session_state["days"] = []

    day_labels = [DAY_MAP[d] for d in day_values]
    selected_day_labels = st.sidebar.multiselect(
        " ",
        options=day_labels,
        default=[DAY_MAP[d] for d in st.session_state["days"]],
        key="day_labels_internal",
    )

    selected_days = [d for d, lbl in DAY_MAP.items() if lbl in selected_day_labels]
    st.session_state["days"] = selected_days

    st.sidebar.divider()

    # ---- Cities (with select/clear all)
    st.sidebar.subheader("Cities")

    if "cities" not in st.session_state:
        st.session_state["cities"] = city_values[:]

    c1, c2 = st.sidebar.columns(2)
    if c1.button("Select all cities", use_container_width=True):
        st.session_state["cities"] = city_values[:]
    if c2.button("Clear all cities", use_container_width=True):
        st.session_state["cities"] = []

    selected_cities = st.sidebar.multiselect(
        " ",
        options=city_values,
        key="cities",
    )

    st.sidebar.divider()

    # Visual controls
    radius = st.sidebar.slider("Column radius (meters)", 30, 150, 60, 10)
    elev_scale = st.sidebar.slider("Elevation scale", 0.0001, 0.01, 0.001, step=0.0001)

    # -----------------------------
    # Filter + aggregate
    # -----------------------------
    filtered_fact = fact[
        (fact.year_key.between(*years))
        & (fact.LowOrPeakDescFull.isin(selected_hours))
        & (fact.day_in_week.isin(selected_days))
    ]

    agg_station = (
        filtered_fact.groupby("StationId", as_index=False)
        .agg(total_rides=("total_rides", "sum"))
    )

    map_df = agg_station.merge(stations, on="StationId")

    if selected_cities:
        map_df = map_df[map_df.CityName.isin(selected_cities)]
    else:
        map_df = map_df.iloc[0:0]

    map_df = map_df.dropna(subset=["Lat", "Long"])

    if map_df.empty:
        st.warning("אין נתונים להצגה עבור הפילטרים שנבחרו.")
        st.stop()

    # -----------------------------
    # Top-N stations (slider + textbox)
    # -----------------------------
    max_stations = len(map_df)

    if "top_n" not in st.session_state:
        st.session_state["top_n"] = min(1000, max_stations)

    top_n_slider = st.sidebar.slider(
        "Top stations (slider)",
        min_value=1,
        max_value=max_stations,
        value=st.session_state["top_n"],
        step=50 if max_stations > 50 else 1,
    )

    top_n_input = st.sidebar.number_input(
        "Top stations (exact number)",
        min_value=1,
        max_value=max_stations,
        value=top_n_slider,
        step=1,
    )

    top_n = min(top_n_input, max_stations)
    st.session_state["top_n"] = top_n

    map_df = (
        map_df.sort_values("total_rides", ascending=False)
        .head(top_n)
        .copy()
    )

    # -----------------------------
    # Color scale (blue → green)
    # -----------------------------
    rides = map_df["total_rides"].values
    norm = (rides - rides.min()) / (rides.max() - rides.min() + 1e-9)

    map_df["color"] = [
        [int(40 + 40*n), int(120 + 120*n), int(180 - 60*n), 170]
        for n in norm
    ]

    map_df["rides_fmt"] = map_df["total_rides"].apply(lambda x: f"{int(x):,}")

    # -----------------------------
    # Metrics row
    # -----------------------------
    st.title("שימוש בתחבורה ציבורית לפי תחנה")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stations shown", f"{len(map_df):,}")
    m2.metric("Cities selected", f"{len(selected_cities):,}")
    m3.metric("Years", f"{years[0]}–{years[1]}")
    m4.metric("Days selected", f"{len(selected_days):,}")

    # -----------------------------
    # ColumnLayer map
    # -----------------------------
    layer = pdk.Layer(
        "ColumnLayer",
        data=map_df,
        get_position=["Long", "Lat"],
        get_elevation="total_rides",
        elevation_scale=elev_scale,
        radius=radius,
        get_fill_color="color",
        pickable=True,
        auto_highlight=True,
    )

    view_state = pdk.ViewState(
        latitude=float(map_df.Lat.mean()),
        longitude=float(map_df.Long.mean()),
        zoom=10,
        pitch=60,
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={
            "html": (
                "<b>{StationName}</b><br/>"
                "Station ID: {StationId}<br/>"
                "City: {CityName}<br/>"
                "Total rides: {rides_fmt}"
            )
        },
    )

    st.pydeck_chart(deck, use_container_width=True, height=780)

# =========================================================
# ======================= PAGE 2 ==========================
# =========================================================
else:
    st.title("📊 Coming soon")
    st.info("כאן ייכנס גרף נוסף (טרנדים, התפלגות, השוואות וכו׳).")
