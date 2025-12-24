import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import time

st.set_page_config(layout="wide")

# Align everything on the hteml final page to right because we think its more elegant in hebrew as the data is relevant to Israel only
st.markdown(
    """
    <style>
    html, body, [class*="st-"] {
        direction: rtl;
        text-align: right;
    }

    h1, h2, h3, h4, h5, h6 {
        direction: rtl;
        text-align: right;
    }

    .stMarkdown {
        direction: rtl;
        text-align: right;
    }

    .deck-tooltip {
        direction: rtl;
        text-align: right;
    }
    
    .stMarkdown ul {
        padding-right: 1.2em;
        padding-left: 0;
        list-style-position: inside;
    }

    .stMarkdown li {
        text-align: right;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)


data_path = "../Create Dataset/data/clean/data.parquet"

day_names_map = {
    1: "ראשון",
    2: "שני",
    3: "שלישי",
    4: "רביעי",
    5: "חמישי",
    6: "שישי",
    7: "שבת",
}


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
            "month_key",     
            "total_rides",
        ]
    ].copy()

    stations = (
        df[["StationId", "StationName", "CityName", "Lat", "Long"]]
        .drop_duplicates(subset=["StationId"])
        .reset_index(drop=True)
    )

    travels = (
        df.groupby(
            ["StationId", "year_key", "month_key", "LowOrPeakDescFull", "day_in_week"],
            as_index=False,
        )
        .agg(total_rides=("total_rides", "sum"))
    )

    travels["LowOrPeakDescFull"] = travels["LowOrPeakDescFull"].astype("category")
    travels["day_in_week"] = travels["day_in_week"].astype("int8")
    travels["StationId"] = travels["StationId"].astype("int32")
    travels["year_key"] = travels["year_key"].astype("int16")
    travels["month_key"] = travels["month_key"].astype("int8")

    return (
        travels,
        stations,
        int(travels.year_key.min()),
        int(travels.year_key.max()),
        int(travels.month_key.min()),
        int(travels.month_key.max()),
        sorted(travels.LowOrPeakDescFull.unique()),
        sorted(travels.day_in_week.unique()),
        sorted(stations.CityName.dropna().unique()),
    )

#region Initalize
(travels, stations,
 year_min, year_max,
 month_min, month_max,
 time_values, day_values,
 city_values,
 ) = load_and_prepare(data_path)

travels = travels.merge(
    stations[["StationId", "StationName", "CityName", "Lat", "Long"]],
    on="StationId",
    how="left"
)
travels = travels.dropna(subset=["Lat", "Long"])
#endregion

# Page selector
page = st.sidebar.radio("תפריט", [
    "🏠 מסך הבית",
    "מפה גיאוגרפית",
    "שימוש לפי שעות וחודשים",
    "מגמות לאורך זמן",
    "ערים מובילות"
])
st.sidebar.divider()

# Home page
if page == '🏠 מסך הבית':
    st.title("שימוש בתחבורה ציבורית לפי תחנה")

    st.info("הסברים")

# Map Page
elif page == "מפה גיאוגרפית":

    #region Map Filters GUI
    st.sidebar.header("סינון")

    # Years and months sliders
    years = st.sidebar.slider("שנים", year_min, year_max, (year_min, year_max))
    months = st.sidebar.slider("חודשים", month_min, month_max, (month_min, month_max))

    st.sidebar.divider()

    # Time in day checkboxes
    st.sidebar.subheader("זמן ביום")
    selected_hours = [
        v for v in time_values if st.sidebar.checkbox(v, value=True, key=f"tod_{v}")
    ]

    st.sidebar.divider()

    # Day of week multiselect
    st.sidebar.subheader("יום בשבוע")

    day_labels = [day_names_map[d] for d in day_values]
    inverse_day_names_map = {day_names_map[d]: d for d in day_values}

    if "day_labels_internal" not in st.session_state:
        st.session_state["day_labels_internal"] = day_labels[:]

    d1, d2 = st.sidebar.columns(2)
    if d1.button("בחר כל הימים", use_container_width=True):
        st.session_state["day_labels_internal"] = day_labels[:]
    if d2.button("הסר כל הימים", use_container_width=True):
        st.session_state["day_labels_internal"] = []

    selected_day_labels = st.sidebar.multiselect(
        " ",
        options=day_labels,
        key="day_labels_internal",
    )

    selected_days = [inverse_day_names_map[lbl] for lbl in selected_day_labels]
    st.sidebar.divider()

    # ities multiselect
    st.sidebar.subheader("ערים")

    if "cities" not in st.session_state:
        st.session_state["cities"] = city_values[:]

    c1, c2 = st.sidebar.columns(2)
    if c1.button("בחר כל הערים", use_container_width=True):
        st.session_state["cities"] = city_values[:]
    if c2.button("הסר כל הערים", use_container_width=True):
        st.session_state["cities"] = []

    selected_cities = st.sidebar.multiselect(
        " ",
        options=city_values,
        key="cities",
    )

    st.sidebar.divider()

    # Visual controls
    radius = st.sidebar.slider("רדיוס תחנה (מ')", 30, 150, 100)
    elev_scale = st.sidebar.slider("פרופורצית מתיחה (גובה)", 0.0001, 0.01, 0.001, step=0.0001)
    #endregion

    #region Handle change in GUI elements
    # s = time.time()
    filtered_travels = travels[
        (travels.year_key.between(*years))
        & (travels.month_key.between(*months))
        & (travels.LowOrPeakDescFull.isin(selected_hours))
        & (travels.day_in_week.isin(selected_days))
    ]
    # print(f"Filter + aggregate: {round(time.time()-s,1)} [s]")

    # s = time.time()
    map_df = (
        filtered_travels
        .groupby(
            ["StationId", "StationName", "CityName", "Lat", "Long"],
            as_index=False
        )
        .agg(total_rides=("total_rides", "sum"))
    )
    # print(f"groupby: {round(time.time()-s,1)} [s]")

    if selected_cities:
        map_df = map_df[map_df.CityName.isin(selected_cities)]
    else:
        map_df = map_df.iloc[0:0]


    if map_df.empty:
        st.warning("אין נתונים להצגה עבור הפילטרים שנבחרו.")
        st.stop()
    #endregion

    #region Top-N stations
    amount_stations = len(map_df)

    if "top_n" not in st.session_state:
        st.session_state["top_n"] = amount_stations

    st.session_state["top_n"] = int(np.clip(st.session_state["top_n"], 1, amount_stations))

    # Slider
    top_n_slider = st.sidebar.slider(
        "מספר תחנות להצגה",
        min_value=1,
        max_value=amount_stations,
        value=st.session_state["top_n"],
        step=50 if amount_stations > 50 else 1,
    )

    # Textbox
    top_n_input = st.sidebar.number_input(
        "הקלד מספר תחנות",
        min_value=1,
        max_value=amount_stations,
        value=int(top_n_slider),
        step=1,
    )

    top_n = min(int(top_n_input), amount_stations)
    st.session_state["top_n"] = int(top_n_slider)

    map_df = (
        map_df.sort_values("total_rides", ascending=False)
        .head(top_n)
        .copy()
    )
    #endregion

    #region Color scale
    rides = map_df["total_rides"].values
    log_rides = np.log1p(rides)
    norm = (log_rides - log_rides.min()) / (log_rides.max() - log_rides.min() + 1e-9)

    map_df["color"] = [
        [int(255 * n), int(255 * (1 - abs(n - 0.5) * 2)), int(255 * (1 - n)), 180]
        for n in norm
    ]
    map_df["rides_fmt"] = map_df["total_rides"].apply(lambda x: f"{int(x):,}")
    #endregion




    #region Statistics above the map
    st.markdown(
        """
        # איפה נמצאות התחנות העמוסות ביותר?

        המפה מציגה תחנות תחבורה ציבורית בישראל, כאשר כל תחנה מיוצגת על־ידי עמוד תלת־ממדי.
        **גובה העמוד** - מייצג את סך הנסיעות בתחנה בפרק זמן.
        **צבע העמוד** - מייצג את סך הנסיעות באופן יחסי.

        ### מדריך שימוש
        
        - השתמש בלחצן השמאלי של העכבר לתנועה בתוך המפה (גרירה). בנוסף ניתן ללחוץ על העכבר בזמן שלוחצים Ctrl לשינוי זווית המבט, בשביל לשנות זום ניתן להשתמש בגלגלת.
        - אפשר להעביר את העכבר מעל תחנה כדי לצפות בפרטים שלה כולל סך הנסיעות בפרק הזמן הנבחר.
        - השתמש בסרגל הצד כדי לסנן את התחנות לפי קריטריונים שונים (שנים, חודשים, ימים, שעות וערים).
        - בתחתית הסרגל ניתן להגביל את כמות התחנות המוצגות (בהתאם לסינון שנבחר) ע"י שימוש בסליידר או בתיבת הטקסט.
        - ניתן לשנות את גובה העמודים והרדיוס שלהם מהסרגל.
    """
    )
    stations_stat, cities_stat, years_stat, months_stat, days_stat = st.columns(5)

    stations_stat.metric("תחנות מוצגות", f"{len(map_df):,}")
    cities_stat.metric("ערים נבחרו", f"{len(selected_cities):,}")
    if years[0] != years[1]:
        years_stat.metric("שנים", f"{years[0]}–{years[1]}")
    else:
        years_stat.metric("שנים", f"{years[0]}")
    if months[0] != months[1]:
        months_stat.metric("חודשים", f"{months[0]}–{months[1]}")
    else:
        months_stat.metric("חודשים", f"{months[0]}")
    days_stat.metric("ימים נבחרו", f"{len(selected_days):,}")
    #endregion

    #region ColumnLayer map
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
        zoom=9,
        pitch=90,
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={
            "html": (
                "<b>{StationName}</b><br/>"
                "מספר תחנה: {StationId}<br/>"
                "עיר: {CityName}<br/>"
                "סך הנסיעות: {rides_fmt}"
            )
        },
    )

    st.pydeck_chart(deck, use_container_width=True, height=780)
    #endregion

# Page 2
elif page == 'שימוש לפי שעות וחודשים':
    st.title("עמוד 2")
    st.info("כאן ייכנס גרף נוסף (טרנדים, התפלגות, השוואות וכו׳).")

# Page 3
elif page == 'מגמות לאורך זמן':
    st.title("עמוד 4")
    st.info("כאן ייכנס גרף נוסף (טרנדים, התפלגות, השוואות וכו׳).")

# Page 4
elif page == 'ערים מובילות':
    st.title("עמוד 5")
    st.info("כאן ייכנס גרף נוסף (טרנדים, התפלגות, השוואות וכו׳).")

