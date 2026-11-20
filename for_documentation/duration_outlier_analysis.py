from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "lesson_raw"
NYC = "America/New_York"
BOROUGHS = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
CANDIDATE_RANGES = [(1, 120), (0.5, 120), (2, 120), (1, 90), (1, 150)]

lookup = pd.read_csv(RAW / "taxi_zone_lookup.csv")
valid_zones = lookup.loc[lookup["Borough"].isin(BOROUGHS), "LocationID"]

# Use the raw development months. Apply C1's duplicate, time and zone checks.
# Keep all remaining durations, including negatives and zero, for the audit.
monthly_trips = []
for month in ("2024-01", "2024-02", "2024-03", "2024-04"):
    raw = pd.read_parquet(RAW / f"green_tripdata_{month}.parquet").drop_duplicates()
    pickup = raw["lpep_pickup_datetime"].dt.tz_localize(
        NYC, ambiguous="NaT", nonexistent="NaT"
    )
    dropoff = raw["lpep_dropoff_datetime"].dt.tz_localize(
        NYC, ambiguous="NaT", nonexistent="NaT"
    )
    start = pd.Timestamp(month, tz=NYC)
    end = start + pd.offsets.MonthBegin(1)
    time_ok = pickup.ge(start) & pickup.lt(end) & dropoff.notna()
    zone_ok = raw["PULocationID"].isin(valid_zones) & raw["DOLocationID"].isin(valid_zones)
    minutes = (dropoff - pickup).dt.total_seconds() / 60
    eligible = time_ok & zone_ok
    trip_distances = raw.loc[eligible, ["trip_distance"]]
    monthly_trips.append(trip_distances.assign(duration_min=minutes.loc[eligible]))

trips = pd.concat(monthly_trips, ignore_index=True)
durations = trips["duration_min"]
positive = durations.loc[durations.gt(0)]
if positive.empty:
    raise SystemExit("No positive durations were available to analyse.")

total = len(durations)
print("Raw January-April 2024 trips after duplicate, timestamp and zone checks")
print(f"Trips before duration filtering: {total:,}")

subminute = durations.gt(0) & durations.lt(1)
above_ceiling = durations.gt(120)
print("\nDuration band (minutes)             Trips     Share")
bands = [
    ("0 or less", durations.le(0)),
    ("Above 0 and below 1", subminute),
    ("1 to 120 inclusive", durations.between(1, 120)),
    ("Above 120", above_ceiling),
]
for label, mask in bands:
    count = int(mask.sum())
    print(f"{label:<31} {count:>9,} {count / total:>9.2%}")

zero_distance = int(trips.loc[subminute, "trip_distance"].eq(0).sum())
over_23_hours = int(durations.gt(23 * 60).sum())
print("\nChecks on the duration tails")
print(f"Positive subminute trips with zero recorded miles: {zero_distance:,} / {subminute.sum():,}")
print(f"Trips above 120 minutes that exceed 23 hours:      {over_23_hours:,} / {above_ceiling.sum():,}")

print(f"\nPercentiles of positive durations ({len(positive):,} trips)")
percentiles = positive.quantile([0.01, 0.50, 0.95, 0.99, 0.995, 0.999])
for percentile, minutes in percentiles.items():
    print(f"{percentile:>7.1%} {minutes:>8.2f} min")

q1, q3 = positive.quantile([0.25, 0.75])
iqr = q3 - q1
upper_fence = q3 + 1.5 * iqr
print(f"1.5 x IQR fences: {q1 - 1.5 * iqr:.2f} to {upper_fence:.2f} min")
above_fence_kept = int((durations.gt(upper_fence) & durations.between(1, 120)).sum())
print(f"Trips above the IQR upper fence retained by 1-120: {above_fence_kept:,}")

print("\nInclusive range (min)         Kept   Excluded   Kept share")
for lower, upper in CANDIDATE_RANGES:
    label = f"{lower:g} to {upper:g}"
    kept = int(durations.between(lower, upper).sum())
    print(f"{label:<22} {kept:>11,} {total - kept:>10,} {kept / total:>11.2%}")
