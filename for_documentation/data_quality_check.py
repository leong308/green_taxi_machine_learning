from pathlib import Path
import pandas as pd

# Find the project folder above for_documentation.
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "lesson_raw"
NYC = "America/New_York"

# Identify valid taxi zones within the five NYC boroughs.
lookup = pd.read_csv(RAW / "taxi_zone_lookup.csv")
boroughs = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
valid_zones = lookup.loc[lookup["Borough"].isin(boroughs), "LocationID"]

rows = []

for number in range(1, 7):
    month = f"2024-{number:02d}"
    original = pd.read_parquet(RAW / f"green_tripdata_{month}.parquet")

    # Check 1: remove exact duplicates within this monthly file.
    raw = original.drop_duplicates()
    duplicates = len(original) - len(raw)

    # Convert timestamps to New York time.
    # Ambiguous or nonexistent daylight-saving times become missing.
    pickup = raw["lpep_pickup_datetime"].dt.tz_localize(
        NYC, ambiguous="NaT", nonexistent="NaT"
    )
    dropoff = raw["lpep_dropoff_datetime"].dt.tz_localize(
        NYC, ambiguous="NaT", nonexistent="NaT"
    )

    # Check 2: require valid timestamps and pickup in the source month.
    start = pd.Timestamp(f"{month}-01", tz=NYC)
    end = start + pd.offsets.MonthBegin(1)
    time_ok = (
        pickup.notna()
        & dropoff.notna()
        & pickup.ge(start)
        & pickup.lt(end)
    )

    # Check 3: require both zones to be in the valid NYC zone list.
    zone_ok = (
        raw["PULocationID"].isin(valid_zones)
        & raw["DOLocationID"].isin(valid_zones)
    )

    # Check 4: require elapsed duration from 1 to 120 minutes, inclusive.
    duration = (dropoff - pickup).dt.total_seconds() / 60
    duration_ok = duration.between(1, 120)

    # Count each excluded record under its first failed check.
    rows.append({
        "Month": month,
        "Raw": len(original),
        "Duplicate": duplicates,
        "Time/month": int((~time_ok).sum()),
        "Zone": int((time_ok & ~zone_ok).sum()),
        "Duration": int((time_ok & zone_ok & ~duration_ok).sum()),
        "Pass": int((time_ok & zone_ok & duration_ok).sum()),
    })

# Add totals and print Table 4.
table = pd.DataFrame(rows).set_index("Month")
table.loc["Total"] = table.sum()

print(table.to_string(formatters={
    column: "{:,.0f}".format for column in table.columns
}))

excluded = int(table.loc["Total", "Raw"] - table.loc["Total", "Pass"])
print(f"\nRecords failing quality checks: {excluded:,}")