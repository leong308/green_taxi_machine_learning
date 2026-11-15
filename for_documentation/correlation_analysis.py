from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "lesson_raw"

# List the 11 numerical attributes in the original dataset.
numerical = [
    "passenger_count", "trip_distance", "fare_amount",
    "extra", "mta_tax", "tip_amount", "tolls_amount",
    "ehail_fee", "improvement_surcharge", "total_amount",
    "congestion_surcharge",
]

# Reuse B1's trip IDs and target; do not load engineered predictors.
prepared = pd.read_parquet(
    ROOT / "data" / "lesson_trips.parquet",
    columns=["pickup", "record_id", "duration_min"],
)
months = prepared["pickup"].dt.strftime("%Y-%m")
development = prepared.loc[
    months.isin(["2024-01", "2024-02", "2024-03", "2024-04"])
].set_index("record_id")

# Retrieve the original attributes for the same development trips.
parts = []

for number in range(1, 5):
    month = f"2024-{number:02d}"
    raw = pd.read_parquet(RAW / f"green_tripdata_{month}.parquet")
    raw["record_id"] = month + "-" + raw.index.astype(str)
    parts.append(raw.set_index("record_id")[numerical])

original = pd.concat(parts).join(
    development[["duration_min"]], how="inner"
)

# Calculate correlations using available values for each attribute.
rows = []
skipped = []

for column in numerical:
    pairs = original[[column, "duration_min"]].dropna()

    # Correlation requires variation in both variables.
    if pairs[column].nunique() < 2 or pairs["duration_min"].nunique() < 2:
        skipped.append(column)
        continue

    coefficient = pairs[column].corr(
        pairs["duration_min"], method="spearman"
    )

    rows.append({
        "Original attribute": column,
        "Valid pairs": len(pairs),
        "Spearman correlation": coefficient,
    })

# Select the four strongest correlations, keeping their signs.
results = pd.DataFrame(rows)
top_four = results.loc[
    results["Spearman correlation"].abs().nlargest(4).index
]

# Print Table 5 without saving any files
print(f"January-April 2024 development records: {len(original):,}")
print(f"Original numerical attributes: {len(numerical)}")
print(f"Correlations available: {len(results)}\n")

print(top_four.to_string(index=False, formatters={
    "Valid pairs": "{:,.0f}".format,
    "Spearman correlation": "{:.3f}".format,
}))

print("Not usabel:", ", ".join(skipped) or "None")