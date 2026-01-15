import os
os.environ["STREAMLIT_RUNTIME_LOG_LEVEL"] = "debug"

import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import time
from markdowns import *
from configuration import *
import sys


def debug(*args):
    print(*args, file=sys.stderr, flush=True)
debug("start")

data_path = "data_small.parquet"

# region Configuration
st.set_page_config(layout="wide")

# Align everything on the hteml final page to right because we think its more elegant in hebrew as the data is relevant to Israel only
st.markdown(STYLE_MARKDOWN, unsafe_allow_html=True)

#region Initalize
@st.cache_data
def aggregate_map(df):
    return (
        df.groupby(
            ["StationId", "StationName", "CityName", "Lat", "Long"],
            as_index=False
        )
        .agg(total_rides=("total_rides", "sum"))
    )

@st.cache_data(show_spinner=True)
def load_city_grouped_data(path: str):
    # return pd.read_parquet(path)
    return pd.read_parquet(data_path)


@st.cache_data(show_spinner=True)
def load_prepare_enriched(path: str):
    df = pd.read_parquet(data_path)
    debug(df.head())
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
    debug("a")
    stations = (
        df[["StationId", "StationName", "CityName", "Lat", "Long"]]
        .drop_duplicates(subset=["StationId"])
        .reset_index(drop=True)
    )
    debug("b")

    travels = (
        df.groupby(
            ["StationId", "year_key", "month_key", "LowOrPeakDescFull", "day_in_week"],
            as_index=False,
        )
        .agg(total_rides=("total_rides", "sum"))
    )
    debug("c")

    # data types
    travels["LowOrPeakDescFull"] = travels["LowOrPeakDescFull"].astype("category")
    travels["day_in_week"] = travels["day_in_week"].astype("int8")
    travels["StationId"] = travels["StationId"].astype("int32")
    travels["year_key"] = travels["year_key"].astype("int16")
    travels["month_key"] = travels["month_key"].astype("int8")
    debug("d")

    # merge + drop na
    travels = travels.merge(
        stations[["StationId", "StationName", "CityName", "Lat", "Long"]],
        on="StationId",
        how="left",
    ).dropna(subset=["Lat", "Long"])
    debug("e")

    return (
        travels,
        int(travels.year_key.min()),
        int(travels.year_key.max()),
        int(travels.month_key.min()),
        int(travels.month_key.max()),
        sorted(travels.LowOrPeakDescFull.unique()),
        sorted(travels.day_in_week.unique()),
        sorted(travels.CityName.dropna().unique()),
    )

@st.cache_data
def filter_travels(travels, years, months, selected_hours, selected_days, selected_cities):
    print("filter_travels")
    df = travels[
        travels.year_key.between(*years)
        & travels.month_key.between(*months)
        & travels.LowOrPeakDescFull.isin(selected_hours)
        & travels.day_in_week.isin(selected_days)
    ]

    if selected_cities:
        df = df[df.CityName.isin(selected_cities)]

    return aggregate_map(df)
(travels, year_min, year_max, month_min, month_max, time_values, day_values, city_values) = load_prepare_enriched(data_path)
debug("f")

#load the city gouped data
city_grouped = load_city_grouped_data("city_grouped_data.parquet")
#endregion
#endregion
debug("g")

# Page selector
page = st.sidebar.radio("תפריט", [
    "🏠 מסך הבית",
    "🗺️ מפה",
    "📆 תקופות ושעות עמוסות",
    "📈 מגמות",
    "📍 דירוג ערים"
])
st.sidebar.divider()
debug("h")

# Home page
if page == '🏠 מסך הבית':
    st.title("שימוש בתחבורה ציבורית לפי תחנה")
    st.markdown(HOME_PAGE_MARKDOWN)

# Map Page
elif page == "🗺️ מפה":
    #Infomation and guidance paragraph
    print("1")
    st.markdown(MAP_MARKDOWN)
    print("0")

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
    if d1.button("בחר כל הימים", width='stretch'):
        st.session_state["day_labels_internal"] = day_labels[:]
    if d2.button("הסר כל הימים", width='stretch'):
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
    if c1.button("בחר כל הערים", width='stretch'):
        st.session_state["cities"] = city_values[:]
    if c2.button("הסר כל הערים", width='stretch'):
        st.session_state["cities"] = []

    selected_cities = st.sidebar.multiselect(
        " ",
        options=city_values,
        key="cities",
    )

    st.sidebar.divider()

    # Visual controls
    radius_scale = st.sidebar.slider(
        "רדיוס תחנה",
        min_value=0.2,
        max_value=8.0,
        value=1.0,
        step=0.1
    )    
    #endregion

    #region Handle change in GUI elements
    map_df = filter_travels(
        travels,
        years,
        months,
        selected_hours,
        selected_days,
        selected_cities
    )

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

    display_df = map_df.nlargest(top_n, "total_rides").sort_values("total_rides", ascending=True).copy()
    display_df["radius"] = 80 * radius_scale

    #endregion

    #region Color and scale
    rides = display_df["total_rides"].values
    log_rides = np.log1p(rides)

    # this normalization calculation is meant to make the top stations very red in comapre to the other
    norm = (log_rides - log_rides.min()) / (log_rides.max() - log_rides.min() + 1e-9)
    gamma = 3   
    saturation = norm ** gamma


    display_df["color"] = [
        [
            int(255 * (1 - s)),
            int(255 * (1 - s)), 
            255,
            180                             
        ]
        for s in saturation
    ]

    display_df["rides_fmt"] = display_df["total_rides"].apply(lambda x: f"{int(x):,}")

    #endregion

    #region Statistics above the map
    stations_stat, cities_stat, years_stat, months_stat, days_stat = st.columns(5)

    stations_stat.metric("תחנות מוצגות", f"{len(display_df):,}")
    cities_stat.metric("ערים נבחרו", f"{len(selected_cities):,}")
    if years[0] != years[1]:
        years_stat.metric("שנים", f"{years[1]}–{years[0]}")
    else:
        years_stat.metric("שנים", f"{years[0]}")
    if months[0] != months[1]:
        months_stat.metric("חודשים", f"{months[1]}–{months[0]}")
    else:
        months_stat.metric("חודשים", f"{months[0]}")
    days_stat.metric("ימים נבחרו", f"{len(selected_days):,}")
    #endregion

    #region ColumnLayer map
    # scatter map
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=display_df,
        get_position=["Long", "Lat"],
        get_radius="radius",
        get_fill_color="color",
        pickable=True,
        auto_highlight=True,
    )

    # heatmap
    # layer = pdk.Layer(
    #     "HeatmapLayer",
    #     data=map_df,
    #     get_position=["Long", "Lat"],
    #     get_weight="total_rides",   
    #     radiusPixels=38,            
    #     intensity=1,
    #     threshold=0.02,
    # )

    view_state = pdk.ViewState(
        latitude=float(map_df.Lat.mean()),
        longitude=float(map_df.Long.mean()),
        zoom=9,
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

    st.pydeck_chart(deck, width='stretch', height=780)
    #endregion

# Congestion Page
elif page == '📆 תקופות ושעות עמוסות':

    st.title("ניתוח עומסים: ממוצעי תיקופים ארצי בתחבורה ציבורית")

    #  Region: Data Preparation

    # Monthly Data
    all_months = pd.DataFrame({"month_key": range(1, 13)})
    real_counts = (
        travels.groupby("month_key", as_index=False)
        .agg(sum_rides=("total_rides", "sum"))
    )
    # Simple average per year
    real_counts["avg_val"] = real_counts["sum_rides"] / YEARS_IN_DATA
    month_counts = pd.merge(all_months, real_counts, on="month_key", how="left").fillna(0)

    month_counts["month_name"] = month_counts["month_key"].map(month_map)
    # Theta/Width not strictly needed for Bar, but keeping for data consistency if needed
    month_counts["theta_val"] = month_counts["month_key"] * 30
    month_counts["width_val"] = 25

    # Time Data
    time_counts = (
        travels.groupby("LowOrPeakDescFull", as_index=False, observed=False)
        .agg(sum_rides=("total_rides", "sum"))
    )

    time_data = time_counts["LowOrPeakDescFull"].astype(str).apply(parse_time_range)
    time_counts["start_time"] = [x[0] for x in time_data]
    time_counts["duration"] = [x[1] for x in time_data]

    # Calc normalized hourly avg
    time_counts["avg_val"] = (time_counts["sum_rides"] / ESTIMATED_NON_SAT_DAYS) / time_counts["duration"]

    time_counts["range_only_name"] = time_counts["LowOrPeakDescFull"].apply(get_time_range_only)
    time_counts["theta_val"] = (time_counts["start_time"] + time_counts["duration"] / 2) * 15
    time_counts["width_val"] = time_counts["duration"] * 15

    month_counts["text_display"] = month_counts["avg_val"].apply(format_millions)
    month_counts["tooltip_val"] = month_counts["avg_val"].apply(format_comma)

    time_counts["text_display"] = time_counts["avg_val"].apply(format_millions)
    time_counts["tooltip_val"] = time_counts["avg_val"].apply(format_comma)


    def create_improved_bar(df, x_col, y_col, title, x_label, y_label, hover_col=None):
        fig = px.bar(
            df,
            x=x_col,
            y=y_col,
            text=df["text_display"],
            color=y_col,
            color_continuous_scale=CUSTOM_BLUE_SCALE,
            custom_data=[df[hover_col] if hover_col else df[x_col], df["tooltip_val"]]
        )

        fig.update_layout(
            title=dict(text=title, x=1),
            xaxis_title=x_label,
            yaxis=dict(
                title=y_label,
                title_standoff=30,
                title_font=dict(size=14)
            ),
            coloraxis_showscale=False,
            font=dict(family="Rubik, sans-serif"),
            margin=dict(l=80, r=20, t=50, b=50)
        )
        fig.update_xaxes(tickangle=0)
        fig.update_traces(
            textposition='outside',
            hovertemplate="<b>%{customdata[0]}</b><br>כמות: %{customdata[1]}<extra></extra>"
        )
        return fig

    def create_real_time_clock(df, r_col, title):
        fig = go.Figure()
        max_val = df[r_col].max() if not df.empty else 1

        fig.add_trace(go.Barpolar(
            r=df[r_col],
            theta=df["theta_val"],
            width=df["width_val"],
            text=df["text_display"],
            customdata=np.stack((df['LowOrPeakDescFull'], df['tooltip_val']), axis=-1),
            hovertemplate="<b>%{customdata[0]}</b><br>ממוצע לשעה: %{customdata[1]}<extra></extra>",
            marker=dict(
                color=df[r_col],
                colorscale=[[0, '#BDD7EE'], [1, '#08519C']],
                cmin=df[r_col].min() * 0.3,
                cmax=max_val,
                line=dict(color='white', width=1)
            ),
        ))

        tick_vals = [h * 15 for h in range(0, 24, 3)]
        tick_text = [f"{h:02d}:00" for h in range(0, 24, 3)]

        fig.update_layout(
            title=dict(text=title, x=1),
            polar=dict(
                radialaxis=dict(visible=False),
                angularaxis=dict(
                    direction="clockwise", rotation=90,
                    tickmode="array", tickvals=tick_vals, ticktext=tick_text,
                    tickfont=dict(size=12), showline=True,
                    linewidth=1, linecolor='rgba(0,0,0,0.1)', gridcolor='rgba(0,0,0,0.1)'
                ),
                hole=0.35
            ),
            annotations=[dict(
                text="שעון<br>24 שעות", x=0.5, y=0.5,
                font=dict(size=14, color='#555'), showarrow=False, xref="paper", yref="paper"
            )],
            font=dict(family="Rubik, sans-serif"),
            margin=dict(t=60, b=40, l=40, r=40)
        )
        return fig

    #  Render GUI
    tab_times, tab_months = st.tabs(["⏰ לפי שעות", "📅 לפי חודשים"])

    with tab_months:
        st.subheader("ממוצע תיקופים חודשי ב-5 השנים האחרונות")
        fig_m_bar = create_improved_bar(
            month_counts[month_counts.avg_val > 0],
            "month_name",
            "avg_val",
            "ממוצע תיקופים לחודש (השוואה כמותית)",
            "חודש",
            "כמות ממוצעת"
        )
        st.plotly_chart(fig_m_bar, width='stretch')

    with tab_times:
        st.subheader("ממוצע תיקופים לשעה בפלחי זמן שונים ביום")

        st.markdown(CLOCK_MARKDOWN, unsafe_allow_html=True)

        fig_t_clock = create_real_time_clock(time_counts, "avg_val", "שעון עומס שעתי (ממוצע)")
        st.plotly_chart(fig_t_clock, width='stretch')

# Trends Page
elif page == '📈 מגמות':
    st.title("מגמות שימוש בתחבורה הציבורית לאורך זמן")
    #region GUI
    st.markdown(TRENDS_MARKDOWN)

    st.sidebar.header("סינון למגמות")

    years_trend = st.sidebar.slider(
        "טווח שנים", year_min, year_max, (year_min, year_max), key="y_trend"
    )

    st.sidebar.subheader("בחר ערים לניתוח המגמה")

    # init state
    if "trend_cities" not in st.session_state:
        st.session_state["trend_cities"] = city_values[:]  # default: all

    c1, c2 = st.sidebar.columns(2)
    if c1.button("בחר את כל הערים", width='stretch', key="trend_all"):
        st.session_state["trend_cities"] = city_values[:]
    if c2.button("נקה הכל", width='stretch', key="trend_none"):
        st.session_state["trend_cities"] = []

    st.sidebar.multiselect(" ", options=city_values, key="trend_cities")
    selected_cities_trend = st.session_state.get("trend_cities", [])

    if not selected_cities_trend:
        st.warning("אנא בחר לפחות עיר אחת כדי לצפות במגמות.")
        st.stop()

    #endregion

    # columns: CityName, year_key, month_key, day_in_week, LowOrPeakDescFull, total_rides
    df_filtered = city_grouped[
        (city_grouped.year_key.between(*years_trend)) &
        (city_grouped.CityName.isin(selected_cities_trend))
    ].copy()

    if df_filtered.empty:
        st.error("לא נמצאו נתונים התואמים את הסינון שנבחר.")
        st.stop()

    # Aggregate to MONTHLY total
    # city_grouped contains duplicates per (month, city) because of day_in_week + time-range
    # so we must sum over them to get monthly totals.
    df_trend = (
        df_filtered
        .groupby(["year_key", "month_key"], as_index=False)["total_rides"]
        .sum()
    )

    # Create date axis
    df_trend["Full_Date"] = pd.to_datetime(
        df_trend["year_key"].astype(str) + "-" +
        df_trend["month_key"].astype(str) + "-01"
    )

    # Sort for clean line
    df_trend = df_trend.sort_values("Full_Date")

    if df_trend.empty:
        st.error("לא נמצאו נתונים התואמים את הסינון שנבחר.")
        st.stop()

    max_val = df_trend["total_rides"].max()
    max_date = df_trend.loc[df_trend["total_rides"].idxmax(), "Full_Date"]

    st.metric('סה"כ נסיעות בתקופה הנבחרת', f'{df_trend["total_rides"].sum():,.0f}')

    fig = px.line(
        df_trend,
        x="Full_Date",
        y="total_rides",
        title='סה"כ נסיעות חודשיות לאורך זמן',
        labels={"Full_Date": "תאריך", "total_rides": "כמות נסיעות"},
        markers=True
    )

    fig.update_traces(
        line_width=3,
        marker=dict(size=6, opacity=0.7),
        hovertemplate="<b>תאריך:</b> %{x|%B %Y}<br><b>נסיעות:</b> %{y:,.0f}<extra></extra>"
    )

    fig.add_annotation(
        x=max_date,
        y=max_val,
        text=f"נקודת שיא: {max_val:,.0f}",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-40,
        font=dict(color="red", size=12, family="Arial")
    )

    fig.update_layout(
        xaxis=dict(
            showgrid=False,
            title="ציר זמן",
            rangeslider_visible=False
        ),
        yaxis=dict(
            title='סה"כ נסיעות',
            title_standoff=70,
            automargin=True,
            gridcolor='lightgray',
            tickformat=",.0f",
            rangemode="tozero"
        ),
        title_x=0.5,
        hovermode="x unified"
    )

    st.plotly_chart(fig, width='stretch')

# Top Cities Page
elif page == '📍 דירוג ערים':
    st.title("דירוג ערים לפי סה״כ נסיעות")
    st.sidebar.header("סינון")

    years_rank = st.sidebar.slider(
        "שנים ", year_min, year_max, (year_min, year_max), key="years_rank"
    )
    months_rank = st.sidebar.slider(
        "חודשים", month_min, month_max, (month_min, month_max), key="months_rank"
    )

    st.sidebar.divider()

    # Days multiselect
    st.sidebar.subheader("יום בשבוע")
    day_labels_rank = [day_names_map[d] for d in day_values]
    inv_day_rank = {day_names_map[d]: d for d in day_values}

    if "day_labels_rank_internal" not in st.session_state:
        st.session_state["day_labels_rank_internal"] = day_labels_rank[:]

    d1, d2 = st.sidebar.columns(2)
    if d1.button("בחר כל הימים", width='stretch', key="rank_days_all"):
        st.session_state["day_labels_rank_internal"] = day_labels_rank[:]
    if d2.button("נקה ימים", width='stretch', key="rank_days_none"):
        st.session_state["day_labels_rank_internal"] = []

    selected_day_labels_rank = st.sidebar.multiselect(
        " ", options=day_labels_rank, key="day_labels_rank_internal"
    )
    selected_days_rank = [inv_day_rank[lbl] for lbl in selected_day_labels_rank]

    st.sidebar.divider()

    # Cities multiselect
    st.sidebar.subheader("ערים")

    if "rank_cities" not in st.session_state:
        st.session_state["rank_cities"] = city_values[:]

    c1, c2 = st.sidebar.columns(2)
    if c1.button("בחר כל הערים", width='stretch', key="rank_cities_all"):
        st.session_state["rank_cities"] = city_values[:]
    if c2.button("נקה ערים", width='stretch', key="rank_cities_none"):
        st.session_state["rank_cities"] = []

    selected_cities_rank = st.sidebar.multiselect(
        " ", options=city_values, key="rank_cities"
    )

    st.sidebar.divider()

    # Top N control
    max_cities = min(30, len(selected_cities_rank)) if selected_cities_rank else 30
    top_n = st.sidebar.slider(
        "כמה ערים להציג",
        min_value=5,
        max_value=max(5, max_cities),
        value=min(15, max_cities),
        step=1,
        key="rank_top_n",
    )

    # Validate selections
    if not selected_days_rank:
        st.warning("בחר/י לפחות יום אחד בשבוע כדי להציג דירוג.")
        st.stop()

    if not selected_cities_rank:
        st.warning("בחר/י לפחות עיר אחת כדי להציג דירוג.")
        st.stop()

    # Filter data
    df = city_grouped[
        (city_grouped.year_key.between(*years_rank))
        & (city_grouped.month_key.between(*months_rank))
        & (city_grouped.day_in_week.isin(selected_days_rank))
        & (city_grouped.CityName.isin(selected_cities_rank))
        ].copy()

    if df.empty:
        st.warning("אין נתונים להצגה עבור הפילטרים שנבחרו.")
        st.stop()

    city_time_summary = (
        df.groupby(["CityName", "LowOrPeakDescFull"], as_index=False)["total_rides"]
        .sum()
    )

    city_totals = (
        city_time_summary.groupby("CityName", as_index=False)["total_rides"]
        .sum()
        .sort_values("total_rides", ascending=False)
    )

    top_cities = city_totals.head(top_n)["CityName"].tolist()

    top_cities_data = city_time_summary[city_time_summary["CityName"].isin(top_cities)].copy()
    other_cities_data = city_time_summary[~city_time_summary["CityName"].isin(top_cities)].copy()

    if not other_cities_data.empty:
        others_summary = (
            other_cities_data.groupby("LowOrPeakDescFull", as_index=False)["total_rides"]
            .sum()
        )
        others_summary["CityName"] = "כל השאר"

        city_time_summary_final = pd.concat([top_cities_data, others_summary], ignore_index=True)
    else:
        city_time_summary_final = top_cities_data

    city_order = top_cities + ["כל השאר"]

    max_rides = city_time_summary_final["total_rides"].max()

    # bar plot
    fig = px.bar(
        city_time_summary_final,
        x="CityName",
        y="total_rides",
        animation_frame="LowOrPeakDescFull",
        title=f"אלו {top_n} הערים המובילות לפי סה״כ נסיעות (לפי שעות ביום)",
        labels={"CityName": "עיר", "total_rides": "סה״כ נסיעות", "LowOrPeakDescFull": "טווח שעות"},
        color="total_rides",
        color_continuous_scale="Blues",
        category_orders={
            "LowOrPeakDescFull": time_order,
            "CityName": city_order
        },
        range_y=[0, max_rides * 1.1]
    )

    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>נסיעות: %{y:,.0f}<extra></extra>"
    )

    fig.update_layout(
        height=600,
        title_x=0.5,
        template="plotly_white",
        showlegend=False,
        xaxis_tickangle=-45,
        coloraxis_showscale=False,
        transition={"duration": 600, "easing": "cubic-in-out"},
        xaxis=dict(
            title=dict(
                text="",
                font=dict(size=16, family="Arial", color="black"),
                standoff=10
            ),
            tickfont=dict(size=12, family="Arial")
        ),
        yaxis=dict(
            title=dict(
                text="",
                font=dict(size=12, family="Arial", color="black"),
                standoff=10
            ),
            tickfont=dict(size=12, family="Arial"),
            range=[0, max_rides * 1.1]
        )
    )

    for frame in fig.frames:
        frame.layout.margin = dict(l=90, r=30, t=80, b=140)
        nice_label = time_labels.get(frame.name, frame.name)
        frame.layout.title = {"text": f"אלו {top_n} הערים המובילות בשעות - {nice_label}", "x": 0.5}
        frame.layout.yaxis = {
            "range": [0, max_rides * 1.1],
            "title": {
                "text": "",
                "font": {"size": 16, "family": "Arial", "color": "black"},
                "standoff": 15
            }
        }
        frame.layout.xaxis = {
            "title": {
                "text": "",
                "font": {"size": 16, "family": "Arial", "color": "black"},
                "standoff": 25
            },
            "tickangle": -45
        }

    if fig.layout.sliders and len(fig.layout.sliders) > 0:
        slider = fig.layout.sliders[0]
        slider.currentvalue.prefix = "טווח שעות: "
        for step in slider.steps:
            full = step.label
            step.label = time_labels.get(full, full)
    fig.update_layout(
        xaxis_title="",
        yaxis_title=""
    )

    st.plotly_chart(fig, width='stretch')

