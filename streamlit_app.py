```python
import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

# =====================================================
# CONFIGURATION
# =====================================================

API_BASE = "https://api.spacexdata.com/v4"


# =====================================================
# LOAD & PREPARE DATA
# =====================================================

@st.cache_data(ttl=300)
def load_data():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "SpaceX-Launch-Dashboard/1.0"
    })

    def fetch_json(url, method="GET", body=None):
        response = session.request(
            method,
            url,
            json=body,
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    def get_records(payload):
        if isinstance(payload, list):
            return payload

        if isinstance(payload, dict):
            for key in ("docs", "launches", "results", "data"):
                value = payload.get(key)

                if isinstance(value, list):
                    return value

                if isinstance(value, dict):
                    for nested_key in ("docs", "launches", "results"):
                        nested = value.get(nested_key)
                        if isinstance(nested, list):
                            return nested

        raise ValueError(
            "Unexpected API response format. "
            "The API did not return a list of records."
        )

    try:
        # Request both real outcome classes from the official API.
        successful_payload = fetch_json(
            f"{API_BASE}/launches/query",
            method="POST",
            body={
                "query": {"success": True},
                "options": {
                    "limit": 1000,
                    "pagination": False,
                    "sort": {"date_utc": "asc"}
                }
            }
        )

        failed_payload = fetch_json(
            f"{API_BASE}/launches/query",
            method="POST",
            body={
                "query": {"success": False},
                "options": {
                    "limit": 1000,
                    "pagination": False,
                    "sort": {"date_utc": "asc"}
                }
            }
        )

        successful_launches = get_records(successful_payload)
        failed_launches = get_records(failed_payload)

        launches = successful_launches + failed_launches

        # Load rocket information.
        rockets_payload = fetch_json(f"{API_BASE}/rockets")
        rockets = get_records(rockets_payload)

        rocket_map = {}

        for rocket in rockets:
            if not isinstance(rocket, dict):
                continue

            rocket_name = rocket.get("name", "Unknown")
            rocket_id = rocket.get("id", rocket.get("_id", rocket_name))

            rocket_map[str(rocket_id)] = rocket_name
            rocket_map[str(rocket_name)] = rocket_name

        # Load launchpad names and coordinates.
        launchpads_payload = fetch_json(f"{API_BASE}/launchpads")
        launchpads = get_records(launchpads_payload)

        launchpad_map = {}

        for pad in launchpads:
            if not isinstance(pad, dict):
                continue

            pad_id = pad.get("id", pad.get("_id"))
            pad_name = pad.get("name", "Unknown")
            latitude = pad.get("latitude")
            longitude = pad.get("longitude")

            launchpad_map[str(pad_id)] = {
                "name": pad_name,
                "latitude": latitude,
                "longitude": longitude
            }

            launchpad_map[str(pad_name)] = {
                "name": pad_name,
                "latitude": latitude,
                "longitude": longitude
            }

        rows = []

        for launch in launches:
            if not isinstance(launch, dict):
                continue

            success = launch.get("success")

            # Only accept confirmed Boolean outcomes.
            if not isinstance(success, bool):
                continue

            date_value = launch.get("date_utc")

            if not date_value:
                continue

            launch_date = pd.to_datetime(
                date_value,
                utc=True,
                errors="coerce"
            )

            if pd.isna(launch_date):
                continue

            # Resolve rocket name.
            rocket_value = launch.get("rocket", "Unknown")

            if isinstance(rocket_value, dict):
                rocket_value = rocket_value.get(
                    "id",
                    rocket_value.get("_id", rocket_value.get("name", "Unknown"))
                )

            rocket_name = rocket_map.get(
                str(rocket_value),
                str(rocket_value)
            )

            # Resolve launchpad and coordinates.
            pad_value = launch.get("launchpad", "Unknown")

            if isinstance(pad_value, dict):
                pad_value = pad_value.get(
                    "id",
                    pad_value.get("_id", pad_value.get("name", "Unknown"))
                )

            pad_info = launchpad_map.get(str(pad_value), {})

            pad_name = pad_info.get("name", str(pad_value))
            latitude = pad_info.get("latitude")
            longitude = pad_info.get("longitude")

            try:
                latitude = (
                    float(latitude) if latitude is not None else None
                )
                longitude = (
                    float(longitude) if longitude is not None else None
                )
            except (TypeError, ValueError):
                latitude = None
                longitude = None

            rows.append({
                "flight_number": launch.get("flight_number", 0),
                "name": launch.get("name", "Unknown"),
                "date_utc": str(date_value),
                "launch_year": int(launch_date.year),
                "rocket_id": rocket_name,
                "rocket_name": rocket_name,
                "launchpad_name": pad_name,
                "latitude": latitude,
                "longitude": longitude,
                "success": int(success),
                "details": launch.get("details") or ""
            })

        df = pd.DataFrame(rows)

        if df.empty:
            raise ValueError(
                "The API returned no launches with confirmed outcomes."
            )

        df = df.drop_duplicates(
            subset=["flight_number", "date_utc"]
        ).reset_index(drop=True)

        # Display actual data counts in the sidebar.
        successful_count = int((df["success"] == 1).sum())
        failed_count = int((df["success"] == 0).sum())

        st.sidebar.caption(
            f"Loaded launches: {len(df)}"
        )
        st.sidebar.caption(
            f"Successful: {successful_count}"
        )
        st.sidebar.caption(
            f"Unsuccessful: {failed_count}"
        )

        return df

    except requests.exceptions.RequestException as error:
        st.error(f"Unable to connect to the SpaceX API: {error}")
        st.info(
            "The API may be unavailable or the connection may be blocked. "
            "Please try again later."
        )
        st.stop()

    except (ValueError, TypeError, KeyError) as error:
        st.error(f"Unable to process SpaceX data: {error}")
        st.stop()


df = load_data()


# =====================================================
# BUILD PREDICTION MODEL
# =====================================================

if df["rocket_id"].nunique() < 1 or df["success"].nunique() < 2:
    st.error(
        "The prediction model needs launches from at least one rocket "
        "and both successful and unsuccessful outcomes. "
        f"Currently loaded: {len(df)} launches, "
        f"{int((df['success'] == 1).sum())} successful and "
        f"{int((df['success'] == 0).sum())} unsuccessful."
    )
    st.stop()

le = LabelEncoder()

df["encoded_rocket"] = le.fit_transform(
    df["rocket_id"].astype(str)
)

X = df[["launch_year", "encoded_rocket"]]
y = df["success"]

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)


# =====================================================
# SIDEBAR & THEME
# =====================================================

st.sidebar.title("🚀 SpaceX Launch Dashboard")
st.sidebar.write("Filters & Prediction")

mode = st.sidebar.radio(
    "Theme Mode",
    ["Light", "Dark"]
)

if mode == "Dark":
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #121212 !important;
            color: #ffffff !important;
        }
        [data-testid="stSidebar"] {
            background-color: #1c1c1c !important;
        }
        .stButton > button {
            background-color: #4caf50;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        .stApp {
            background-color: white !important;
            color: black !important;
        }
        .stButton > button {
            background-color: #4caf50;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# =====================================================
# FILTERS
# =====================================================

years = sorted(df["launch_year"].unique().tolist())

selected_year = st.sidebar.selectbox(
    "Filter launches by Year",
    years
)

rocket_options = (
    df[["rocket_id", "rocket_name"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

rocket_names = rocket_options["rocket_name"].tolist()

rocket_name = st.sidebar.selectbox(
    "Select Rocket for Prediction",
    rocket_names
)

input_year = st.sidebar.number_input(
    "Enter Launch Year (Prediction)",
    min_value=2006,
    max_value=2030,
    value=2026
)

input_month = st.sidebar.slider(
    "Launch Month", 1, 12, 6
)

input_day = st.sidebar.slider(
    "Launch Day", 1, 31, 15
)

predict_btn = st.sidebar.button("Predict Success")


# =====================================================
# MAIN DASHBOARD
# =====================================================

st.title("🚀 SpaceX Launch Dashboard")

st.write(
    "Explore SpaceX launch history, visualize launch sites, "
    "and predict future launch success probabilities."
)

filtered = df[df["launch_year"] == selected_year]

# Launch outcomes
st.subheader(f"Launch Outcomes in {selected_year}")

success_counts = (
    filtered["success"]
    .value_counts()
    .reindex([1, 0], fill_value=0)
)

success_counts.index = ["Success", "Failure"]

st.bar_chart(success_counts)


# Launch trends
st.subheader("📈 Launch Trends Over Years")

launches_per_year = df.groupby("launch_year").size()

success_rate_per_year = (
    df.groupby("launch_year")["success"].mean()
)

fig, ax1 = plt.subplots(figsize=(10, 4))

ax1.set_xlabel("Year")
ax1.set_ylabel("Number of Launches")

ax1.plot(
    launches_per_year.index,
    launches_per_year.values,
    marker="o"
)

ax2 = ax1.twinx()

ax2.set_ylabel("Success Rate")

ax2.plot(
    success_rate_per_year.index,
    success_rate_per_year.values,
    marker="x"
)

ax2.set_ylim(0, 1)

fig.tight_layout()

st.pyplot(fig)

plt.close(fig)


# Launch sites map
st.subheader("🌍 Launch Sites Map")

map_data = filtered.dropna(
    subset=["latitude", "longitude"]
)

if not map_data.empty:
    m = folium.Map(
        location=[
            map_data["latitude"].mean(),
            map_data["longitude"].mean()
        ],
        zoom_start=3
    )

    for _, row in map_data.iterrows():
        popup_text = (
            f"{row['name']} ({row['launchpad_name']})"
            f"<br>Success: "
            f"{'Yes' if row['success'] == 1 else 'No'}"
        )

        color = "green" if row["success"] == 1 else "red"

        folium.CircleMarker(
            location=[
                row["latitude"],
                row["longitude"]
            ],
            radius=6,
            popup=popup_text,
            color=color,
            fill=True,
            fill_color=color
        ).add_to(m)

    st_folium(m, width=700, height=450)

else:
    st.info(
        "Launchpad coordinates are unavailable for the selected year."
    )


# Feature importance
st.subheader("🧠 Model Feature Importance")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

st.bar_chart(
    importance_df.set_index("Feature")
)


# =====================================================
# PREDICTION
# =====================================================

if predict_btn:
    encoded_id = le.transform([rocket_name])[0]

    input_data = pd.DataFrame(
        [[input_year, encoded_id]],
        columns=["launch_year", "encoded_rocket"]
    )

    probabilities = model.predict_proba(input_data)[0]

    success_class_index = list(model.classes_).index(1)

    probability = probabilities[success_class_index]

    st.success(
        f"Predicted Success Probability: {probability:.2%}"
    )

    st.caption(
        "This is a model estimate based on historical launch outcomes, "
        "not a guarantee. The selected month and day are not model inputs."
    )


# =====================================================
# DASHBOARD SUMMARY
# =====================================================

total_launches = len(df)

success_rate = df["success"].mean() * 100

st.markdown(
    f"""
    <div id="fixed-box" style="
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: rgba(30,30,30,0.90);
        border-radius: 12px;
        padding: 15px 25px;
        box-shadow: 0 0 10px 2px #4caf50;
        z-index: 9999;
        font-size: 14px;
        color: #d1d1d1;
        max-width: 250px;">
        <b>Dashboard Summary</b><br>
        Total Launches: {total_launches}<br>
        Overall Success Rate: {success_rate:.1f}%<br>
        Selected Year: {selected_year}<br>
        Rockets: {len(rocket_names)}
    </div>
    """,
    unsafe_allow_html=True
)


# =====================================================
# FOOTER
# =====================================================

st.markdown(
    "<br><hr><p style='text-align:center'>"
    "Made with ❤️ by Aaraiz Ali | Data from SpaceX API"
    "</p>",
    unsafe_allow_html=True
)
```
