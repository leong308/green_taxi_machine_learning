
# B9 - Predict a single NYC Green Taxi journey

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

NYC = "America/New_York"
artifact = Path("lesson_models/forest.joblib")

# Create the same 15 features used during model training
def make_features(pickup, origin, destination, zones):
    origin_borough = origin.map(zones)
    destination_borough = destination.map(zones)

    if origin_borough.isna().any() or destination_borough.isna().any():
        raise ValueError("Use two TLC zone IDs within NYC.")

    hour = pickup.dt.hour + pickup.dt.minute / 60
    day = pickup.dt.dayofweek
    pu = origin.astype(int).astype(str)
    do = destination.astype(int).astype(str)

    return pd.DataFrame({
        "pu_zone": pu,
        "do_zone": do,
        "zone_pair": pu + ">" + do,
        "pu_borough": origin_borough,
        "do_borough": destination_borough,
        "borough_pair": origin_borough + ">" + destination_borough,
        "time_band": pd.cut(
            hour, [0, 6, 10, 16, 20, 24],
            right=False,
            labels=["overnight", "morning", "midday",
                    "evening", "late"]
        ).astype(str),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "weekday_sin": np.sin(2 * np.pi * day / 7),
        "weekday_cos": np.cos(2 * np.pi * day / 7),
        "weekend": (day >= 5).astype("int8"),
        "same_zone": origin.eq(destination).astype("int8"),
        "cross_borough": origin_borough.ne(
            destination_borough
        ).astype("int8"),
        "airport_destination": destination.isin(
            [132, 138]
        ).astype("int8"),
    })


# Load the NYC taxi zone lookup
zone_table = pd.read_csv("data/lesson_raw/taxi_zone_lookup.csv")
zones = zone_table.set_index("LocationID")["Borough"]
zones = zones.loc[zones.isin([
    "Bronx", "Brooklyn", "Manhattan",
    "Queens", "Staten Island"
])]

# Example taxi journey
pickup_time = "2024-06-18 14:30"
pickup_zone = 74
destination_zone = 75

pickup = pd.Series([pd.Timestamp(pickup_time, tz=NYC)])

features = make_features(
    pickup,
    pd.Series([pickup_zone]),
    pd.Series([destination_zone]),
    zones
)

# Load the saved Random Forest model
saved_model = joblib.load(artifact)

# Predict the journey duration
duration = max(1.0, saved_model.predict(features)[0])

print(f"Estimated duration: {duration:.2f} minutes")
