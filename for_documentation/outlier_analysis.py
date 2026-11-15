from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "lesson_raw"

# Check the original numerical attributes, not engineered predictors.
numerical = [
    "passenger_count", "trip_distance", "fare_amount",
    "extra", "mta_tax", "tip_amount", "tolls_amount",
    "ehail_fee", "improvement_surcharge", "total_amount",
    "congestion_surcharge",
]

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

# Retrieve the original numerical values for these trips.
parts = []

for number in range(1, 5):
    month = f"2024-{number:02d}"
    raw = pd.read_parquet(RAW / f"green_tripdata_{month}.parquet")
    raw["record_id"] = month + "-" + raw.index.astype(str)
    raw = raw.set_index("record_id")
    parts.append(raw.loc[raw.index.isin(development_ids), numerical])

original = pd.concat(parts)
rows = []

for column in numerical:
    # Omit missing values only from this attribute's calculation.
    values = original[column].dropna()

    row = {
        "Attribute": column,
        "N": f"{len(values):,}",
        "Lower": "-",
        "Upper": "-",
        "Flagged": "-",
        "Flagged (%)": "-",
        "Status": "All missing",
    }

    if not values.empty:
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1

        if iqr == 0:
            # Avoid labelling every value outside a common value an outlier.
            row["Status"] = "Review: IQR=0"
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            flagged = int(((values < lower) | (values > upper)).sum())

            row.update({
                "Lower": f"{lower:.3f}",
                "Upper": f"{upper:.3f}",
                "Flagged": f"{flagged:,}",
                "Flagged (%)": f"{100 * flagged / len(values):.3f}",
                "Status": "IQR applied",
            })

    rows.append(row)

# Print Table 7 without saving or changing any data.
table = pd.DataFrame(rows)

print(f"\nJanuary-April 2024 development records: {len(original):,}\n")
print(table.to_string(index=False))
print(
    f"\nMaximum recorded trip_distance: "
    f"{original['trip_distance'].max():,.2f} miles"
)