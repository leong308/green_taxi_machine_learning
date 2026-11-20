"""Benchmark the Ridge pipeline at three monthly sampling levels."""

from pathlib import Path
from statistics import median
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.base import clone
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "lesson_raw"
RIDGE_ARTIFACT = ROOT / "lesson_models" / "ridge.joblib"
NYC = "America/New_York"
BOROUGHS = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
TARGET = "duration_min"
ALPHA = 1
THREADS = 4
REPEATS = 7
FEBRUARY = pd.Timestamp("2024-02-01", tz=NYC)
CASE_SIZES = [("15,000", 15_000), ("30,000", 30_000), ("All eligible", None)]


def eligible_month(month, zones):
    """Match the shared-data builder's eligibility and feature rules in memory."""
    raw = pd.read_parquet(RAW / f"green_tripdata_{month}.parquet").drop_duplicates()
    start = pd.Timestamp(month, tz=NYC)
    end = start + pd.offsets.MonthBegin(1)
    pickup = raw.lpep_pickup_datetime.dt.tz_localize(NYC, ambiguous="NaT", nonexistent="NaT")
    dropoff = raw.lpep_dropoff_datetime.dt.tz_localize(NYC, ambiguous="NaT", nonexistent="NaT")
    duration = (dropoff - pickup).dt.total_seconds() / 60
    eligible = (
        pickup.ge(start) & pickup.lt(end) & duration.between(1, 120)
        & raw.PULocationID.isin(zones.index) & raw.DOLocationID.isin(zones.index)
    )
    pickup = pickup.loc[eligible]
    origin = raw.loc[eligible, "PULocationID"]
    destination = raw.loc[eligible, "DOLocationID"]
    origin_borough = origin.map(zones)
    destination_borough = destination.map(zones)
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
            hour, [0, 6, 10, 16, 20, 24], right=False,
            labels=["overnight", "morning", "midday", "evening", "late"],
        ).astype(str),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "weekday_sin": np.sin(2 * np.pi * day / 7),
        "weekday_cos": np.cos(2 * np.pi * day / 7),
        "weekend": (day >= 5).astype("int8"),
        "same_zone": origin.eq(destination).astype("int8"),
        "cross_borough": origin_borough.ne(destination_borough).astype("int8"),
        "airport_destination": destination.isin([132, 138]).astype("int8"),
        "pickup": pickup,
        TARGET: duration.loc[eligible],
        "record_id": month + "-" + pickup.index.astype(str),
    })


def benchmark(template, features, train, validation):
    x_train, y_train = train[features], train[TARGET]
    x_valid = validation[features]
    fit_times, predict_times = [], []
    for repeat in range(REPEATS + 1):
        model = clone(template)
        start = perf_counter()
        model.fit(x_train, y_train)
        fit_ms = 1000 * (perf_counter() - start)
        start = perf_counter()
        np.maximum(1.0, model.predict(x_valid))
        predict_ms = 1000 * (perf_counter() - start)
        if repeat > 0:
            fit_times.append(fit_ms)
            predict_times.append(predict_ms)

    # Inspect the fitted training matrix outside the timed work.
    encoded = model.named_steps["preprocess"].transform(x_train)
    if issparse(encoded):
        matrix_bytes = encoded.data.nbytes + encoded.indices.nbytes + encoded.indptr.nbytes
    else:
        matrix_bytes = encoded.nbytes
    return median(fit_times), median(predict_times), encoded.shape[1], matrix_bytes / 1024**2


def main():
    lookup = pd.read_csv(RAW / "taxi_zone_lookup.csv")
    zones = lookup.loc[lookup.Borough.isin(BOROUGHS)].set_index("LocationID")["Borough"]
    # One seeded order per month keeps the 15,000-trip sample inside the 30,000-trip sample.
    january = eligible_month("2024-01", zones).sample(frac=1, random_state=6001)
    february = eligible_month("2024-02", zones).sample(frac=1, random_state=6002)
    if min(len(january), len(february)) < 30_000:
        raise ValueError("Both months need at least 30,000 eligible trips for this benchmark.")

    # Reuse the project's pipeline definition and discard the saved fitted state.
    saved = joblib.load(RIDGE_ARTIFACT)
    features = list(saved.feature_names_in_)
    template = clone(saved).set_params(regression__alpha=ALPHA)
    cutoff = FEBRUARY - pd.Timedelta(hours=2)

    print(f"Ridge alpha={ALPHA} | January 2024 fit | February 2024 validation")
    print(f"Median of {REPEATS} runs per case after one warm-up | Thread limit: {THREADS}")
    print("Fit and predict times include pipeline preprocessing. Raw-data preparation is untimed.")
    print("Matrix MiB measures encoded training matrix storage.")
    print()
    print(f"{'Trips per month':<15} {'Fit rows':>9} {'Val. rows':>9} {'Fit ms':>10} "
          f"{'Predict ms':>11} {'Pred. ms/1k':>12} {'Encoded cols':>13} {'Matrix MiB':>11}")
    with threadpool_limits(limits=THREADS):
        for label, size in CASE_SIZES:
            train = january.iloc[:size].sort_values(["pickup", "record_id"])
            train = train.loc[train.pickup.lt(cutoff)]
            validation = february.iloc[:size].sort_values(["pickup", "record_id"])
            fit_ms, predict_ms, columns, matrix_mib = benchmark(template, features, train, validation)
            predict_per_1000 = predict_ms * 1000 / len(validation)
            print(f"{label:<15} {len(train):>9,} {len(validation):>9,} {fit_ms:>10.2f} "
                  f"{predict_ms:>11.2f} {predict_per_1000:>12.2f} {columns:>13,} {matrix_mib:>11.2f}")


if __name__ == "__main__":
    main()
