
import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

# =====================================================
# LOAD & PREPARE DATA
# =====================================================

API_BASE = "https://gateway.pipeworx.io/spacex/v4"


@st.cache_data(ttl=300)
def load_data():
    def fetch_json(url):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()

    def get_records(payload, keys=()):
        if isinstance(payload, list):
            return payload

        if isinstance(payload, dict):
            for key in keys:
                if isinstance(payload.get(key), list):
                    return payload[key]

        raise ValueError(
            "Unexpected API response format. Please check the API."
        )

    try:
        launches_payload = fetch_json(
            f"{API_BASE}/launches/past?limit=1000"
        )
        rockets_payload = fetch_json(f"{API_BASE}/rockets")

        launches = get_records(
            launches_payload, ("launches", "docs", "results")
        )
        rockets = get_records(
            rockets_payload, ("rockets", "docs", "results")
        )

        rocket_map = {}
        for rocket in rockets:
            if isinstance(rocket, dict):
                rocket_name = rocket.get("name", "Unknown")
                rocket_id = rocket.get(
                    "id", rocket.get("_id", rocket_name)
                )
                rocket_map[str(rocket_id)] = rocket_name
                rocket_map[rocket_name] = rocket_name

        # Approximate coordinates for recognized SpaceX launch sites.
        # If the API does not identify a site, its coordinates stay empty.
        pad_locations = {
            "lc-39a": (28.6082, -80.6040),
            "39a": (28.6082, -80.6040),
            "slc-40": (28.5619, -80.5774),
            "40": (28.5619, -80.5774),
            "slc-4e": (34.6321, -120.6107),
            "4e": (34.6321, -120.6107),
            "omelek": (9.0477, 167.7431),
            "boca chica": (25.9973, -97.1566),
            "starbase": (25.9973, -97.1566),
        }

        def find_pad_coordinates(pad_name):
            normalized = str(pad_name).lower()

            for key, coordinates in pad_locations.items():
                if key in normalized:
                    return coordinates

            return None, None

        data = []

        for launch in launches:
            if not isinstance(launch, dict):
                continue

            success = launch.get("success")

            # Only train on confirmed outcomes.
            if success is None:
                continue

            date_utc = launch.get(
                "date_utc", launch.get("date")
            )

            if not date_utc:
                continue

            try:
                launch_date = pd.to_datetime(
                    date_utc, utc=True, errors="raise"
                )
                launch_year = int(launch_date.year)
            except (ValueError, TypeError):
                continue

            rocket_value = launch.get("rocket", "Unknown")

            if isinstance(rocket_value, dict):
                rocket_value = rocket_value.get(
                    "name",
                    rocket_value.get(
                        "id", rocket_value.get("_id", "Unknown")
                    ),
                )

            rocket_value = str(rocket_value or "Unknown")
            rocket_name = rocket_map.get(
                rocket_value, rocket_value
            )

            pad = launch.get(
                "launchpad", launch.get("pad", "Unknown")
            )

            if isinstance(pad, dict):
                pad_name = pad.get(
                    "name", pad.get("location", "Unknown")
                )
                latitude = pad.get("latitude")
                longitude = pad.get("longitude")

                if latitude is None or longitude is None:
                    latitude, longitude = find_pad_coordinates(
                        pad_name
                    )
            else:
                pad_name = str(pad or "Unknown")
                latitude, longitude = find_pad_coordinates(
                    pad_name
                )

            # Handle numeric coordinates if supplied by the API.
            try:
                latitude = (
                    float(latitude) if latitude is not None else None
                )
                longitude = (
                    float(longitude) if longitude is not None else None
                )
            except (TypeError, ValueError):
                latitude, longitude = None, None

            data.append({
                "flight_number": launch.get(
                    "flight_number", launch.get("id", 0)
                ),
                "name": launch.get("name", "Unknown"),
                "date_utc": str(date_utc),
                "launch_year": launch_year,
                "rocket_id": rocket_name,
                "rocket_name": rocket_name,
                "launchpad_name": pad_name,
                "latitude": latitude,
                "longitude": longitude,
                "success": int(bool(success)),
                "details": launch.get("details") or "",
            })

        df = pd.DataFrame(data)

        if df.empty:
            raise ValueError(
                "No launches with confirmed outcomes were returned."
            )

        return df

    except requests.exceptions.RequestException as error:
        st.error(f"Unable to connect to the SpaceX data API: {error}")
        st.info("Please try again later if the data service is unavailable.")
        st.stop()

    except (ValueError, TypeError, KeyError) as error:
        st.error(f"Unable to process SpaceX data: {error}")
        st.info(
            "The API response may have changed. "
            "Check the response format before retrying."
        )
        st.stop()


df = load_data()

# =====================================================
# BUILD PREDICTION MODEL
# =====================================================

if df["rocket_id"].nunique() < 1 or df["success"].nunique() < 2:
    st.error(
        "The prediction model needs launches from at least one rocket "
        "and both successful and unsuccessful outcomes."
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

mode = st.sidebar.radio("Theme Mode", ["Light", "Dark"])

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
        unsafe_allow_html=True,
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
        unsafe_allow_html=True,
    )

# =====================================================
# FILTERS
# =====================================================

years = sorted(df["launch_year"].unique().tolist())

selected_year = st.sidebar.selectbox(
    "Filter launches by Year", years
)

rocket_options = (
    df[["rocket_id", "rocket_name"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

rocket_names = rocket_options["rocket_name"].tolist()

rocket_name = st.sidebar.selectbox(
    "Select Rocket for Prediction", rocket_names
)

input_year = st.sidebar.number_input(
    "Enter Launch Year (Prediction)",
    min_value=2006,
    max_value=2030,
    value=2026,
)

input_month = st.sidebar.slider("Launch Month", 1, 12, 6)
input_day = st.sidebar.slider("Launch Day", 1, 31, 15)

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

# --- Launch Outcomes ---
st.subheader(f"Launch Outcomes in {selected_year}")

success_counts = (
    filtered["success"]
    .value_counts()
    .reindex([1, 0], fill_value=0)
)
success_counts.index = ["Success", "Failure"]

st.bar_chart(success_counts)

# --- Launch Trends ---
st.subheader("📈 Launch Trends Over Years")

launches_per_year = df.groupby("launch_year").size()
success_rate_per_year = df.groupby("launch_year")["success"].mean()

fig, ax1 = plt.subplots(figsize=(10, 4))

ax1.set_xlabel("Year")
ax1.set_ylabel("Number of Launches")
ax1.plot(
    launches_per_year.index,
    launches_per_year.values,
    marker="o",
)

ax2 = ax1.twinx()
ax2.set_ylabel("Success Rate")
ax2.plot(
    success_rate_per_year.index,
    success_rate_per_year.values,
    marker="x",
)
ax2.set_ylim(0, 1)

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# --- Launch Sites Map ---
st.subheader("🌍 Launch Sites Map")

map_data = filtered.dropna(
    subset=["latitude", "longitude"]
)

if not map_data.empty:
    m = folium.Map(
        location=[
            map_data["latitude"].mean(),
            map_data["longitude"].mean(),
        ],
        zoom_start=3,
    )

    for _, row in map_data.iterrows():
        popup_text = (
            f"{row['name']} ({row['launchpad_name']})"
            f"<br>Success: "
            f"{'Yes' if row['success'] == 1 else 'No'}"
        )

        color = "green" if row["success"] == 1 else "red"

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=6,
            popup=popup_text,
            color=color,
            fill=True,
            fill_color=color,
        ).add_to(m)

    st_folium(m, width=700, height=450)
else:
    st.info(
        "Launchpad coordinates are unavailable for the selected year."
    )

# --- Feature Importance ---
st.subheader("🧠 Model Feature Importance")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_,
}).sort_values(by="Importance", ascending=False)

st.bar_chart(importance_df.set_index("Feature"))

# =====================================================
# PREDICTION
# =====================================================

if predict_btn:
    encoded_id = le.transform([rocket_name])[0]

    input_data = pd.DataFrame(
        [[input_year, encoded_id]],
        columns=["launch_year", "encoded_rocket"],
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
    unsafe_allow_html=True,
)

# =====================================================
# FOOTER
# =====================================================

st.markdown(
    "<br><hr><p style='text-align:center'>"
    "Made with ❤️ by Aaraiz Ali | Data from SpaceX API"
    "</p>",
    unsafe_allow_html=True,
)
