import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import time


data_path = "data.parquet"

# region Configuration
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
    
    /* sliders*/
    input[type="range"] {
        direction: ltr;
    }
    div[data-baseweb="slider"] {
        direction: ltr;
    }
    div[data-baseweb="slider"] * {
        direction: ltr;
        text-align: left;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# A dictionary that maps each day number to its hebrew name for example "1:ראשון"
day_names_map = {
    1: "ראשון",
    2: "שני",
    3: "שלישי",
    4: "רביעי",
    5: "חמישי",
    6: "שישי",
    7: "שבת",
}


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
def load_prepare_enriched(path: str):
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

    # data types
    travels["LowOrPeakDescFull"] = travels["LowOrPeakDescFull"].astype("category")
    travels["day_in_week"] = travels["day_in_week"].astype("int8")
    travels["StationId"] = travels["StationId"].astype("int32")
    travels["year_key"] = travels["year_key"].astype("int16")
    travels["month_key"] = travels["month_key"].astype("int8")

    # merge + drop na
    travels = travels.merge(
        stations[["StationId", "StationName", "CityName", "Lat", "Long"]],
        on="StationId",
        how="left",
    ).dropna(subset=["Lat", "Long"])

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

(travels,
 year_min, year_max,
 month_min, month_max,
 time_values, day_values,
 city_values) = load_prepare_enriched(data_path)

#endregion
#endregion

# Page selector
page = st.sidebar.radio("תפריט", [
    "🏠 מסך הבית",
    "🗺️ מפה",
    "📆 תקופות ושעות עמוסות",
    "📈 מגמות",
    "📍 דירוג ערים"
])
st.sidebar.divider()

# Home page
if page == '🏠 מסך הבית':
    st.title("שימוש בתחבורה ציבורית לפי תחנה")

    st.info("הסברים")

# Map Page
elif page == "🗺️ מפה":

    #Infomation and guidance paragraph
    st.markdown(
        """
        # איפה נמצאות התחנות העמוסות ביותר?

        המפה מציגה תחנות תחבורה ציבורית בישראל, כאשר כל תחנה מיוצגת על־ידי עיגול.
        **צבע העיגול** - מייצג את סך הנסיעות באופן יחסי.

        ### מדריך שימוש
        
        - השתמש בלחצן השמאלי של העכבר לתנועה בתוך המפה ("גרור" את המפה). בשביל לשנות זום ניתן להשתמש בגלגלת.
        - אפשר להעביר את העכבר מעל תחנה כדי לצפות בפרטים שלה כגון סך הנסיעות (בפרק הזמן הנבחר לפי הפילטרים).
        - השתמש בסרגל הצד כדי לסנן את התחנות לפי קריטריונים שונים (שנים, חודשים, ימים, שעות וערים).
        - בתחתית הסרגל ניתן להגביל את כמות התחנות המוצגות (בהתאם לסינון שנבחר) ע"י שימוש בסליידר או בתיבת הטקסט.
        - ניתן לשנות את רדיוס העיגולים מהסרגל.

        דוגמא להבנת הנתונים:
         
        - אם נסנן את השנים 2024-2025, יום ראשון בלבד, בשיא הבוקר ונראה שבתחנה מסוימת היו 100,000 נסיעות, סימן שסך הנסיעות שבוצעו בתחנה זו בשנים 2024-2025 בכל חודשי השנה, בכל ימי ראשון - רק בשיא הבוקר זה 100,000
        - אם נסנן ערים ונשאיר רק ירושלים ותל אביב, ונבחר להציג רק את 50 התחנות העמוסות ביותר, זה יציג לנו מתוך כל התחנות שהיו בירושלים ותל אביב את 50 התחנות העמוסות ביותר
    """
    )

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
        max_value=2.0,
        value=1.0,
        step=0.1
    )    
    #endregion

    #region Handle change in GUI elements
    filtered_travels = travels[
        (travels.year_key.between(*years))
        & (travels.month_key.between(*months))
        & (travels.LowOrPeakDescFull.isin(selected_hours))
        & (travels.day_in_week.isin(selected_days))
    ]

    map_df = aggregate_map(filtered_travels)


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
        map_df
        .nlargest(top_n, "total_rides")
        .sort_values("total_rides", ascending=True)
        .copy()
    )

    #endregion

    #region Color and scale
    rides = map_df["total_rides"].values
    log_rides = np.log1p(rides)

    # this normalization calculation is meant to make the top stations very red in comapre to the other
    norm = (log_rides - log_rides.min()) / (log_rides.max() - log_rides.min() + 1e-9)
    gamma = 3   
    saturation = norm ** gamma


    map_df["color"] = [
        [
            255,
            int(255 * (1 - s)), 
            int(255 * (1 - s)),             
            180                             
        ]
        for s in saturation
    ]

    map_df["rides_fmt"] = map_df["total_rides"].apply(lambda x: f"{int(x):,}")

    # min_r = 30
    # max_r = 180
    # base_radius = min_r + norm * (max_r - min_r)

    map_df["radius"] = 80 * radius_scale

    #endregion

    #region Statistics above the map
    
    stations_stat, cities_stat, years_stat, months_stat, days_stat = st.columns(5)

    stations_stat.metric("תחנות מוצגות", f"{len(map_df):,}")
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
        data=map_df,
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

elif page == '📆 תקופות ושעות עמוסות':

    st.title("ניתוח עומסים: ממוצעי תיקופים ארצי בתחבורה ציבורית")

    # Constants
    YEARS_IN_DATA = 5
    ESTIMATED_NON_SAT_DAYS = 1566

    # --- Region: Data Preparation ---

    # 1. Monthly Data
    all_months = pd.DataFrame({"month_key": range(1, 13)})
    real_counts = (
        travels.groupby("month_key", as_index=False)
        .agg(sum_rides=("total_rides", "sum"))
    )
    # Simple average per year
    real_counts["avg_val"] = real_counts["sum_rides"] / YEARS_IN_DATA
    month_counts = pd.merge(all_months, real_counts, on="month_key", how="left").fillna(0)

    month_map = {
        1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MAY', 6: 'JUN',
        7: 'JUL', 8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'
    }
    month_counts["month_name"] = month_counts["month_key"].map(month_map)
    # Theta/Width not strictly needed for Bar, but keeping for data consistency if needed
    month_counts["theta_val"] = month_counts["month_key"] * 30
    month_counts["width_val"] = 25

    # 2. Time Data
    time_counts = (
        travels.groupby("LowOrPeakDescFull", as_index=False, observed=False)
        .agg(sum_rides=("total_rides", "sum"))
    )

    # Time Parsing
    def parse_time_range(desc):
        matches = re.findall(r'(\d{2}):(\d{2})', str(desc))
        if len(matches) >= 2:
            start_h, start_m = int(matches[0][0]), int(matches[0][1])
            end_h, end_m = int(matches[1][0]), int(matches[1][1])
            start_decimal = start_h + (start_m / 60.0)
            end_decimal = end_h + (end_m / 60.0)
            if end_decimal < start_decimal: end_decimal += 24
            duration = end_decimal - start_decimal
            if duration <= 0: duration = 1
            return start_decimal, duration
        return 0, 1

    def get_time_range_only(desc):
        match = re.search(r'(\d{2}:\d{2}\s*-\s*\d{2}:\d{2})', str(desc))
        if match: return match.group(1).strip()
        return str(desc)

    time_data = time_counts["LowOrPeakDescFull"].astype(str).apply(parse_time_range)
    time_counts["start_time"] = [x[0] for x in time_data]
    time_counts["duration"] = [x[1] for x in time_data]

    # Calc normalized hourly avg
    time_counts["avg_val"] = (time_counts["sum_rides"] / ESTIMATED_NON_SAT_DAYS) / time_counts["duration"]

    time_counts["range_only_name"] = time_counts["LowOrPeakDescFull"].apply(get_time_range_only)
    time_counts["theta_val"] = (time_counts["start_time"] + time_counts["duration"] / 2) * 15
    time_counts["width_val"] = time_counts["duration"] * 15

    # Formatting
    def format_millions(x):
        if x >= 1_000_000:
            return f'{x / 1_000_000:.1f}M'
        elif x >= 1_000:
            return f'{x / 1_000:.0f}K'
        return "" if x == 0 else str(int(x))

    def format_comma(x):
        return f"{int(x):,}"

    month_counts["text_display"] = month_counts["avg_val"].apply(format_millions)
    month_counts["tooltip_val"] = month_counts["avg_val"].apply(format_comma)

    time_counts["text_display"] = time_counts["avg_val"].apply(format_millions)
    time_counts["tooltip_val"] = time_counts["avg_val"].apply(format_comma)

    # --- Plotting Functions ---


    CUSTOM_BLUE_SCALE = ['#BDD7EE', '#6BAED6', '#3182BD', '#08519C']

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

    # --- Render GUI (Selected Alternatives Only) ---

    tab_months, tab_times = st.tabs(["📅 לפי חודשים", "⏰ לפי שעות"])

    with tab_months:
        st.subheader("ממוצע תיקופים חודשי ב-5 השנים האחרונות")
        # הצגת החלופה הנבחרת: גרף עמודות
        fig_m_bar = create_improved_bar(
            month_counts[month_counts.avg_val > 0],
            "month_name",
            "avg_val",
            "ממוצע תיקופים לחודש (השוואה כמותית)",
            "חודש",
            "כמות ממוצעת"
        )
        st.plotly_chart(fig_m_bar, use_container_width=True)

    with tab_times:
        st.subheader("ממוצע תיקופים לשעה בפלחי זמן שונים ביום")

        st.markdown("""
        <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>
        <strong>מדריך לשימוש בגרף:</strong><br>
        הגרף מציג את המחזוריות היומית של התחבורה הציבורית במודל "שעון".<br>
        הגוון הכחול ואורך הגזרה מתחזקים ככל שהעומס הממוצע לשעה עולה.<br>
        הניחו את העכבר על פלח זמן כדי לראות את המספר המדויק.
        </div>
        """, unsafe_allow_html=True)

        # הצגת החלופה הנבחרת: גרף שעון
        fig_t_clock = create_real_time_clock(time_counts, "avg_val", "שעון עומס שעתי (ממוצע)")
        st.plotly_chart(fig_t_clock, use_container_width=True)
# Page 3
elif page == '📈 מגמות':
    st.title("עמוד 4")
    st.markdown("""
        גרף זה מציג את השינויים בכמות הנסיעות לאורך ציר הזמן.
        באמצעות הוויזואליזציה ניתן לזהות דפוסים תקופתיים, השפעות של אירועים חיצוניים (כמו חגים או מצבים ביטחוניים)
        ואת קצב הגידול בשימוש בתחבורה הציבורית בישראל.
    """)

    st.sidebar.header("סינון למגמות")

    years_trend = st.sidebar.slider("טווח שנים", year_min, year_max, (year_min, year_max), key="y_trend")

    st.sidebar.subheader("בחר ערים לניתוח המגמה")

    # init state
    if "trend_cities" not in st.session_state:
        st.session_state["trend_cities"] = city_values[:]  # ברירת מחדל: כל הערים

    c1, c2 = st.sidebar.columns(2)
    if c1.button("בחר את כל הערים", use_container_width=True, key="trend_all"):
        st.session_state["trend_cities"] = city_values[:]

    if c2.button("נקה הכל", use_container_width=True, key="trend_none"):
        st.session_state["trend_cities"] = []

    selected_cities_trend = st.sidebar.multiselect(
        " ",
        options=city_values,
        key="trend_cities"
    )

    if not selected_cities_trend:
        st.warning("אנא בחר לפחות עיר אחת כדי לצפות במגמות.")
    else:

        df_filtered = travels[
            (travels.year_key.between(*years_trend)) &
            (travels.CityName.isin(selected_cities_trend))
        ]


        df_filtered['Full_Date'] = pd.to_datetime(
            df_filtered['year_key'].astype(str) + '-' +
            df_filtered['month_key'].astype(str) + '-01'
        )

        #
        df_trend = df_filtered.groupby('Full_Date', as_index=False)['total_rides'].sum()

        if df_trend.empty:
            st.error("לא נמצאו נתונים התואמים את הסינון שנבחר.")
        else:

            max_val = df_trend['total_rides'].max()
            max_date = df_trend.loc[df_trend['total_rides'].idxmax(), 'Full_Date']


            fig = px.line(
                df_trend,
                x='Full_Date',
                y='total_rides',
                title='סה"כ נסיעות חודשיות לאורך זמן',
                labels={'Full_Date': 'תאריך', 'total_rides': 'כמות נסיעות'},
                markers=True
            )




            fig.update_traces(
                line_color='#1f77b4',
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
                arrowcolor="red",
                ax=0,
                ay=-40,
                font=dict(color="red", size=12, family="Arial")
            )

            fig.update_layout(
                plot_bgcolor='white',
                xaxis=dict(
                    showgrid=False,
                    title="ציר זמן",
                    rangeslider_visible=False  #
                ),
                yaxis=dict(
                    gridcolor='lightgray',
                    title="סה\"כ נסיעות",
                    tickformat=",.0f",
                    rangemode="tozero"
                ),
                title_x=0.5,
                hovermode="x unified"
            )
            fig.update_layout(
                yaxis=dict(
                    title='סה"כ נסיעות',
                    title_standoff=70,
                    automargin=True,
                    gridcolor='lightgray',
                    tickformat=",.0f",
                    rangemode="tozero"
                )
            )

            st.plotly_chart(fig, use_container_width=True)

            # הצגת נתון מספרי בולט מתחת לגרף
            st.metric("סה\"כ נסיעות בתקופה הנבחרת", f"{df_trend['total_rides'].sum():,.0f}")
# Page 4
elif page == '📍 דירוג ערים':
    st.title("דירוג ערים ושימוש לפי זמן")
    st.sidebar.header("סינון (דירוג ערים)")

    years_rank = st.sidebar.slider(
        "שנים (דירוג)", year_min, year_max, (year_min, year_max), key="years_rank"
    )
    months_rank = st.sidebar.slider(
        "חודשים (דירוג)", month_min, month_max, (month_min, month_max), key="months_rank"
    )

    st.sidebar.divider()

    # ---------- Days multiselect ----------
    st.sidebar.subheader("יום בשבוע (דירוג)")
    day_labels_rank = [day_names_map[d] for d in day_values]
    inv_day_rank = {day_names_map[d]: d for d in day_values}

    if "day_labels_rank_internal" not in st.session_state:
        st.session_state["day_labels_rank_internal"] = day_labels_rank[:]

    d1, d2 = st.sidebar.columns(2)
    if d1.button("בחר כל הימים", use_container_width=True, key="rank_days_all"):
        st.session_state["day_labels_rank_internal"] = day_labels_rank[:]
    if d2.button("נקה ימים", use_container_width=True, key="rank_days_none"):
        st.session_state["day_labels_rank_internal"] = []

    selected_day_labels_rank = st.sidebar.multiselect(
        " ", options=day_labels_rank, key="day_labels_rank_internal"
    )
    selected_days_rank = [inv_day_rank[lbl] for lbl in selected_day_labels_rank]

    st.sidebar.divider()

    # ---------- Cities multiselect ----------
    st.sidebar.subheader("ערים (דירוג)")

    if "rank_cities" not in st.session_state:
        st.session_state["rank_cities"] = city_values[:]  # default: all

    c1, c2 = st.sidebar.columns(2)
    if c1.button("בחר כל הערים", use_container_width=True, key="rank_cities_all"):
        st.session_state["rank_cities"] = city_values[:]
    if c2.button("נקה ערים", use_container_width=True, key="rank_cities_none"):
        st.session_state["rank_cities"] = []

    selected_cities_rank = st.sidebar.multiselect(
        " ", options=city_values, key="rank_cities"
    )

    # ---------- Validate selections ----------
    if not selected_days_rank:
        st.warning("בחר/י לפחות יום אחד בשבוע כדי להציג דירוג.")
        st.stop()

    if not selected_cities_rank:
        st.warning("בחר/י לפחות עיר אחת כדי להציג דירוג.")
        st.stop()

    # ---------- Filter data ----------
    df = travels[
        (travels.year_key.between(*years_rank))
        & (travels.month_key.between(*months_rank))
        & (travels.day_in_week.isin(selected_days_rank))
        & (travels.CityName.isin(selected_cities_rank))
    ].copy()

    if df.empty:
        st.warning("אין נתונים להצגה עבור הפילטרים שנבחרו.")
        st.stop()

    # ---------- Group for chart ----------
    df_grouped = (
        df.groupby(["CityName", "LowOrPeakDescFull"], as_index=False)["total_rides"]
          .sum()
          .rename(columns={
              "LowOrPeakDescFull": "TimeRange",
              "CityName": "City"
          })
    )

    # ---------- Define TimeRange order + display labels (NO data change) ----------
    time_order = [
        "06:00 - 08:59 - שיא בוקר",
        "09:00 - 11:59 - שפל יום 1",
        "12:00 - 14:59 - שפל יום 2",
        "15:00 - 18:59 - שיא ערב",
        "19:00 - 23:59 - שפל ערב",
    ]

    time_labels = {
        "06:00 - 08:59 - שיא בוקר": "06:00 - 08:59",
        "09:00 - 11:59 - שפל יום 1": "09:00 - 11:59",
        "12:00 - 14:59 - שפל יום 2": "12:00 - 14:59",
        "15:00 - 18:59 - שיא ערב": "15:00 - 18:59",
        "19:00 - 23:59 - שפל ערב": "19:00 - 23:59",
    }

    # Fallback: keep any extra categories (if exist) after the known ones
    extra_times = [t for t in df_grouped["TimeRange"].unique().tolist() if t not in time_order]
    category_order = time_order + sorted(extra_times)

    # ---------- Plotly animated bar ----------
    fig = px.bar(
        df_grouped,
        x="total_rides",
        y="City",
        animation_frame="TimeRange",
        orientation="h",
        title="שימוש בתחבורה ציבורית לפי עיר וזמן ביום",
        labels={
            "total_rides": "סה״כ נסיעות",
            "City": "",
            "TimeRange": "טווח שעות",
        },
        category_orders={"TimeRange": category_order},
        text="total_rides",
        range_x=[0, df_grouped["total_rides"].max() * 1.2],
    )

    fig.update_traces(texttemplate="%{text:.2s}", textposition="outside")
    fig.update_layout(
        title_x=0.5,
        template="plotly_white",
        showlegend=False,
        transition={"duration": 800},
        margin=dict(l=200),
    )
    fig.update_yaxes(title_text="", automargin=True)

    # ---------- Replace the animation frame name in the title (display-only) ----------
    # px uses the frame name (f.name) in the per-frame title; we swap it to numeric-only.
    def _swap_frame_title(frame):
        # frame.name is the original TimeRange string
        nice = time_labels.get(frame.name, frame.name)

        # ensure layout.title exists
        if frame.layout.title and frame.layout.title.text:
            frame.layout.title.text = frame.layout.title.text.replace(frame.name, nice)
        else:
            frame.layout.title = {"text": nice}


    for frame in fig.frames:
        nice = time_labels.get(frame.name, frame.name)

        if frame.layout.title and frame.layout.title.text:
            frame.layout.title.text = frame.layout.title.text.replace(frame.name, nice)
        else:
            frame.layout.title = {"text": nice}

    # ---  slider  ---
    if fig.layout.sliders and len(fig.layout.sliders) > 0:
        slider = fig.layout.sliders[0]

        slider.currentvalue.prefix = "טווח שעות: "

        for step in slider.steps:
            full = step.label
            step.label = time_labels.get(full, full)

    st.plotly_chart(fig, use_container_width=True)