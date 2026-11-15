from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "lesson_raw"

# Use the seven original categorical attributes.
categorical = [
    "VendorID", "store_and_fwd_flag", "RatecodeID",
    "PULocationID", "DOLocationID", "payment_type", "trip_type",
]
zone_columns = ["PULocationID", "DOLocationID"]

# Reuse the same January-April development trip IDs.
prepared = pd.read_parquet(
    ROOT / "data" / "lesson_trips.parquet",
    columns=["pickup", "record_id"],
)
months = prepared["pickup"].dt.strftime("%Y-%m")
development_ids = prepared.loc[
    months.isin(["2024-01", "2024-02", "2024-03", "2024-04"]),
    "record_id",
]

# Retrieve original categorical values for these trips.
parts = []

for number in range(1, 5):
    month = f"2024-{number:02d}"
    raw = pd.read_parquet(RAW / f"green_tripdata_{month}.parquet")
    raw["record_id"] = month + "-" + raw.index.astype(str)
    raw = raw.set_index("record_id")
    parts.append(raw.loc[raw.index.isin(development_ids), categorical])

original = pd.concat(parts)
total = len(original)
rows = []

# Add one category's count and percentage to the printed table.
def add_row(attribute, category, count):
    rows.append({
        "Attribute": attribute,
        "Category": category,
        "Trips": int(count),
        "Share (%)": 100 * count / total,
    })

for column in categorical:
    counts = original[column].value_counts()

    # Show the five leading zone IDs; show all other categories.
    shown = counts.head(5) if column in zone_columns else counts

    for value, count in shown.items():
        label = value if isinstance(value, str) else str(int(value))
        add_row(column, label, count)

    # Group remaining zones for display only.
    if column in zone_columns and len(counts) > 5:
        other_count = counts.iloc[5:].sum()
        other_zones = len(counts) - 5
        add_row(column, f"Other zones ({other_zones})", other_count)

    # Keep missing values visible in the distribution.
    missing = int(original[column].isna().sum())
    if missing > 0:
        add_row(column, "Missing", missing)

table = pd.DataFrame(rows)

# Print Table 6 directly in the terminal.
print(f"\nJanuary-April 2024 development records: {total:,}")
print(f"Original categorical attributes: {len(categorical)}\n")
print(table.to_string(index=False, formatters={
    "Trips": "{:,.0f}".format,
    "Share (%)": "{:.3f}".format,
}))