from pathlib import Path
import pandas as pd

# Find the data folder beside this script.
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "lesson_raw"
PREPARED = ROOT / "data" / "lesson_trips.parquet"

# These codes represent categories, even when stored as numbers.
RAW_CATEGORICAL = {
    "VendorID", "store_and_fwd_flag", "RatecodeID",
    "PULocationID", "DOLocationID", "payment_type", "trip_type",
}
RAW_DATETIME = {"lpep_pickup_datetime", "lpep_dropoff_datetime"}

# Count records across the six original monthly files.
source_months = [f"2024-{month:02d}" for month in range(1, 7)]
raw_rows = 0
original_columns = None

for month in source_months:
    path = RAW / f"green_tripdata_{month}.parquet"
    raw = pd.read_parquet(path)

    if original_columns is None:
        original_columns = list(raw.columns)
    elif set(raw.columns) != set(original_columns):
        raise ValueError(f"Different columns found in {path.name}")

    raw_rows += len(raw)

# Count the original attributes by their analytical meaning.
columns = set(original_columns)
raw_categorical = len(columns & RAW_CATEGORICAL)
raw_datetime = len(columns & RAW_DATETIME)
raw_numerical = len(columns) - raw_categorical - raw_datetime

# Read the prepared data and identify the model inputs.
trips = pd.read_parquet(PREPARED)
excluded = {"pickup", "duration_min", "record_id"}
predictors = [column for column in trips.columns if column not in excluded]

numerical_predictors = len(
    trips[predictors].select_dtypes(include="number").columns
)
categorical_predictors = len(predictors) - numerical_predictors

# Count the records in each assessment period.
months = trips["pickup"].dt.strftime("%Y-%m")
development_rows = int(months.isin(source_months[:4]).sum())
may_rows = int(months.eq("2024-05").sum())
june_rows = int(months.eq("2024-06").sum())

# Display the summary for the report screenshot.
summary = {
    "Observation period": "January to June 2024",
    "Monthly source files": len(source_months),
    "Raw trip records": f"{raw_rows:,}",
    "Original attributes": len(original_columns),
    "Analytical data types": (
        f"{raw_numerical} numerical / "
        f"{raw_categorical} categorical / {raw_datetime} datetime"
    ),
    "Prepared trip records": f"{len(trips):,}",
    "Model predictors": (
        f"{len(predictors)} "
        f"({categorical_predictors} categorical / "
        f"{numerical_predictors} numerical)"
    ),
    "Target": "duration_min (elapsed trip minutes)",
    "Development records (Jan-Apr)": f"{development_rows:,}",
    "May records": f"{may_rows:,}",
    "June test records": f"{june_rows:,}",
}

print()
print("NYC GREEN TAXI DATASET - BRIEF CHARACTERISTICS")
print("=" * 68)

for label, value in summary.items():
    print(f"{label:<31} : {value}")

print("=" * 68)
print("Counts calculated from the existing project data files.")
print()