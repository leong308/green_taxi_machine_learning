from pathlib import Path
from urllib.request import urlretrieve
import numpy as np
import pandas as pd
RAW = Path("data/lesson_raw")
RAW.mkdir(parents=True, exist_ok=True)
CDN = "https://d37ci6vzurychx.cloudfront.net/"
NYC = "America/New_York"
BOROUGHS = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
def fetch(remote_path):
    destination = RAW / remote_path.split("/")[-1]
    if not destination.exists():
        temporary = destination.with_suffix(".part")
        urlretrieve(CDN + remote_path, temporary)
        temporary.replace(destination)
    return destination
lookup = pd.read_csv(fetch("misc/taxi_zone_lookup.csv"))
zones = lookup.loc[lookup.Borough.isin(BOROUGHS)]
zones = zones.set_index("LocationID")["Borough"]
january_path = fetch("trip-data/green_tripdata_2024-01.parquet")
january = pd.read_parquet(january_path)
print("January rows and columns:", january.shape)
print("Mapped NYC zones:", len(zones))

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
        "time_band": pd.cut(hour, [0, 6, 10, 16, 20, 24], right=False,
                            labels=["overnight", "morning", "midday",
                                    "evening", "late"]).astype(str),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "weekday_sin": np.sin(2 * np.pi * day / 7),
        "weekday_cos": np.cos(2 * np.pi * day / 7),
        "weekend": (day >= 5).astype("int8"),
        "same_zone": origin.eq(destination).astype("int8"),
        "cross_borough": origin_borough.ne(destination_borough).astype("int8"),
        "airport_destination": destination.isin([132, 138]).astype("int8"),
    })

def clean_month(month, seed):
    path = fetch(f"trip-data/green_tripdata_{month}.parquet")
    raw = pd.read_parquet(path).drop_duplicates()
    start = pd.Timestamp(month, tz=NYC)
    end = start + pd.offsets.MonthBegin(1)
    pickup = raw.lpep_pickup_datetime.dt.tz_localize(
        NYC, ambiguous="NaT", nonexistent="NaT"
    )
    dropoff = raw.lpep_dropoff_datetime.dt.tz_localize(
        NYC, ambiguous="NaT", nonexistent="NaT"
    )
    minutes = (dropoff - pickup).dt.total_seconds() / 60
    in_month = pickup.ge(start) & pickup.lt(end)
    valid_zone = raw.PULocationID.isin(zones.index)
    valid_zone &= raw.DOLocationID.isin(zones.index)
    eligible = in_month & valid_zone & minutes.between(1, 120)
    selected = raw.loc[eligible].sample(n=15000, random_state=seed)
    rows = make_features(pickup.loc[selected.index], selected.PULocationID,
                         selected.DOLocationID, zones)
    rows["pickup"] = pickup.loc[selected.index]
    rows["duration_min"] = minutes.loc[selected.index]
    rows["record_id"] = month + "-" + rows.index.astype(str)
    print(month, int(eligible.sum()), "eligible before sampling")
    return rows

first_month = clean_month("2024-01", seed=6001)
print("January sample:", first_month.shape)

parts = [first_month]
for number in range(2, 7):
    month = f"2024-{number:02d}"
    parts.append(clean_month(month, seed=6000 + number))
trips = pd.concat(parts).sort_values(["pickup", "record_id"])
trips = trips.reset_index(drop=True)
assert len(trips) == 90000
assert trips.record_id.is_unique
assert trips.notna().all().all()
data_file = Path("data/lesson_trips.parquet")
trips.to_parquet(data_file, index=False)
print("Saved", data_file, "with", len(trips), "trips")