from pathlib import Path
import pandas as pd

# Find the project folder above for_documentation.
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "lesson_raw"

# Count missing values across all six original monthly files.
raw_records = 0
missing_counts = None

for number in range(1, 7):
    month = f"2024-{number:02d}"
    raw = pd.read_parquet(RAW / f"green_tripdata_{month}.parquet")

    raw_records += len(raw)
    monthly_missing = raw.isna().sum()

    if missing_counts is None:
        missing_counts = monthly_missing
    else:
        missing_counts = missing_counts.add(monthly_missing, fill_value=0)

# Calculate the percentage using all original records.
missing_table = pd.DataFrame({
    "Missing records": missing_counts.astype(int),
    "Missing (%)": missing_counts / raw_records * 100,
})
missing_table.index.name = "Attribute"

# Print attributes containing missing values.
print(f"Original records checked: {raw_records:,}")
print(f"Original attributes checked: {len(missing_counts)}\n")

incomplete = missing_table.loc[missing_table["Missing records"] > 0]
print(incomplete.to_string(formatters={
    "Missing records": "{:,.0f}".format,
    "Missing (%)": "{:.2f}".format,
}))

complete_count = int(missing_counts.eq(0).sum())
print(f"\nRemaining {complete_count} attributes: 0 missing values (0.00%).")