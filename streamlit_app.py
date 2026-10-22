import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

API_BASE = "https://gateway.pipeworx.io/spacex/v4"


@st.cache_data(ttl=300)
def load_data():
    session = requests.Session()
    session.headers.update({"User-Agent": "SpaceX-Launch-Dashboard"})

    def fetch_json(url, method="GET", body=None):
        response = session.request(
            method, url, json=body, timeout=30
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
            "The API returned an unexpected launch data format."
        )

    def normalize_success(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            value = value.strip().lower()
            if value in ("true", "1", "yes", "success", "successful"):
                return True
            if value in ("false", "0", "no", "failure", "failed", "unsuccessful"):
                return False
        return None

    try:
        # SpaceX API query endpoint: request historical launches.
        try:
            payload = fetch_json(
                f"{API_BASE}/launches/query",
                method="POST",
                body={
                    "query": {},
                    "options": {
                        "limit": 1000,
                        "pagination": False,
                        "sort": {"date_utc": "asc"},
                    },
                },
            )
            launches = get_records(payload)
        except (requests.RequestException, ValueError):
            # Fallback for gateways that expose a normal launches endpoint.
            payload = fetch_json(f"{API_BASE}/launches")
            launches = get_records(payload)

        rockets_payload = fetch_json(f"{API_BASE}/rockets")
        rockets = get_records(rockets_payload)

        rocket_map = {}
        for rocket in rockets:
            if isinstance(rocket, dict):
                name = rocket.get("name", "Unknown")
                rocket_id = rocket.get("id", rocket.get("_id", name))
                rocket_map[str(rocket_id)] = name
                rocket_map[str(name)] = name

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

        def coordinates(pad_name):
            text = str(pad_name).lower()
            for key, coords in pad_locations.items():
                if key in text:
                    return coords
            return None, None

        rows = []
        missing_outcomes = 0

        for launch in launches:
            if not isinstance(launch, dict):
                continue

            success_value = launch.get("success")
            if success_value is None:
                success_value = launch.get("successful")

            success = normalize_success(success_value)
            if success is None:
                missing_outcomes += 1
                continue

            date_value = launch.get("date_utc", launch.get("date"))
            if not date_value:
                continue

            date = pd.to_datetime(
                date_value, utc=True, errors="coerce"
            )
            if pd.isna(date):
                continue

            rocket_value = launch.get(
                "rocket", launch.get("rocket_id", "Unknown")
            )
            if isinstance(rocket_value, dict):
                rocket_value = rocket_value.get(
                    "name", rocket_value.get(
                        "id", rocket_value.get("_id", "Unknown")
                    )
                )

            rocket_name = rocket_map.get(
                str(rocket_value), str(rocket_value)
            )

            pad = launch.get("launchpad", launch.get("pad", "Unknown"))
            latitude = longitude = None

            if isinstance(pad, dict):
                pad_name = pad.get(
                    "name", pad.get("full_name", pad.get("id", "Unknown"))
                )
                latitude = pad.get("latitude")
                longitude = pad.get("longitude")
            else:
                pad_name = str(pad)

            if latitude is None or longitude is None:
                lat, lon = coordinates(pad_name)
                latitude = latitude if latitude is not None else lat
                longitude = longitude if longitude is not None else lon

            try:
                latitude = float(latitude) if latitude is not None else None
                longitude = float(longitude) if longitude is not None else None
            except (TypeError, ValueError):
                latitude = longitude = None

            rows.append({
                "flight_number": launch.get(
                    "flight_number", launch.get("id", 0)
                ),
                "name": launch.get("name", "Unknown"),
                "date_utc": str(date_value),
                "launch_year": int(date.year),
                "rocket_id": rocket_name,
                "rocket_name": rocket_name,
                "launchpad_name": str(pad_name),
                "latitude": latitude,
                "longitude": longitude,
                "success": int(success),
                "details": launch.get("details") or "",
            })

        df = pd.DataFrame(rows)

        if df.empty:
            raise ValueError(
                f"No launches with confirmed outcomes. "
                f"API records: {len(launches)}; "
                f"missing outcomes: {missing_outcomes}."
            )

        st.sidebar.caption(
            f"Loaded {len(df)} launches | "
            f"Success: {(df['success'] == 1).sum()} | "
            f"Failure: {(df['success'] == 0).sum()}"
        )

        return df

    except (requests.RequestException, ValueError, TypeError, KeyError) as error:
        st.error(f"Unable to load SpaceX launch data: {error}")
        st.info(
            "Check whether the API endpoint is available and returns "
            "historical launches with real success/failure outcomes."
        )
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
        f"{int((df['success'] == 0).sum())} unsuccessful. "
        "The API must provide real records for both outcomes."
    )
    st.stop()

le = LabelEncoder()
df["encoded_rocket"] = le.fit_transform(df["rocket_id"].astype(str))

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
    st.markdown("""
    <style>
    .stApp {background-color:#121212 !important;color:white !important;}
    [data-testid="stSidebar"] {background-color:#1c1c1c !important;}
    .stButton>button {background-color:#4caf50;color:white;}
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .stApp {background-color:white !important;color:black !important;}
    .stButton>button {background-color:#4caf50;color:white;}
    </style>
    """, unsafe_allow_html=True)

# =====================================================
# FILTERS
# =====================================================

years = sorted(df["launch_year"].unique().tolist())
selected_year = st.sidebar.selectbox("Filter launches by Year", years)

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

st.subheader(f"Launch Outcomes in {selected_year}")
success_counts = (
    filtered["success"].value_counts().reindex([1, 0], fill_value=0)
)
success_counts.index = ["Success", "Failure"]
st.bar_chart(success_counts)

st.subheader("📈 Launch Trends Over Years")
launches_per_year = df.groupby("launch_year").size()
success_rate_per_year = df.groupby("launch_year")["success"].mean()

fig, ax1 = plt.subplots(figsize=(10, 4))
ax1.set_xlabel("Year")
ax1.set_ylabel("Number of Launches")
ax1.plot(launches_per_year.index, launches_per_year.values, marker="o")

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

st.subheader("🌍 Launch Sites Map")
map_data = filtered.dropna(subset=["latitude", "longitude"])

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
            f"<br>Success: {'Yes' if row['success'] == 1 else 'No'}"
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
        "This is an estimate based on historical launch outcomes, "
        "not a guarantee. Month and day are not model inputs."
    )

# =====================================================
# DASHBOARD SUMMARY
# =====================================================

total_launches = len(df)
success_rate = df["success"].mean() * 100

st.markdown(
    f"""
    <div id="fixed-box" style="
        position:fixed;bottom:20px;right:20px;
        background:rgba(30,30,30,0.90);
        border-radius:12px;padding:15px 25px;
        box-shadow:0 0 10px 2px #4caf50;
        z-index:9999;font-size:14px;color:#d1d1d1;
        max-width:250px;">
        <b>Dashboard Summary</b><br>
        Total Launches: {total_launches}<br>
        Overall Success Rate: {success_rate:.1f}%<br>
        Selected Year: {selected_year}<br>
        Rockets: {len(rocket_names)}
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<br><hr><p style='text-align:center'>"
    "Made with ❤️ by Aaraiz Ali | Data from SpaceX API"
    "</p>",
    unsafe_allow_html=True,
)
