"""Benchmark the selected CatBoost configuration at three monthly sampling levels."""

import json
from statistics import median
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from benchmark_ridge_cost import (
    BOROUGHS, CASE_SIZES, FEBRUARY, RAW, ROOT, TARGET, eligible_month,
)
from compare_feature_engineering import (
    ENGINEERED_CATS, ENGINEERED_NUMS, THREADS, build_model,
)


REPEATS = 3
FEATURES = ENGINEERED_CATS + ENGINEERED_NUMS


def benchmark(params, train, validation):
    """Return median fit and batch-prediction times in milliseconds."""
    x_train, y_train = train[FEATURES], train[TARGET]
    x_valid = validation[FEATURES]
    fit_times, predict_times = [], []

    for repeat in range(REPEATS + 1):
        model = build_model("catboost", params, ENGINEERED_CATS, ENGINEERED_NUMS)
        start = perf_counter()
        model.fit(x_train, y_train)
        fit_ms = 1000 * (perf_counter() - start)

        start = perf_counter()
        np.maximum(1.0, model.predict(x_valid))
        predict_ms = 1000 * (perf_counter() - start)

        if repeat > 0:
            fit_times.append(fit_ms)
            predict_times.append(predict_ms)

    return median(fit_times), median(predict_times)


def main():
    selection = ROOT / "lesson_results" / "catboost_selection.json"
    params = json.loads(selection.read_text(encoding="utf-8"))["winner"]["params"]
    lookup = pd.read_csv(RAW / "taxi_zone_lookup.csv")
    zones = lookup.loc[lookup.Borough.isin(BOROUGHS)].set_index("LocationID")["Borough"]
    # Reuse the Ridge benchmark's eligibility, nested samples and split.
    january = eligible_month("2024-01", zones).sample(frac=1, random_state=6001)
    february = eligible_month("2024-02", zones).sample(frac=1, random_state=6002)
    if min(len(january), len(february)) < 30_000:
        raise ValueError("Both months need at least 30,000 eligible trips for this benchmark.")
    cutoff = FEBRUARY - pd.Timedelta(hours=2)

    print("CatBoost | January 2024 fit | February 2024 validation")
    print(f"Selected parameters: {params}")
    print(f"Median of {REPEATS} runs per case after one warm-up | Thread limit: {THREADS}")
    print("Times include CatBoost's internal processing. Raw-data preparation is untimed.")
    print()
    print(f"{'Trips per month':<15} {'Fit rows':>9} {'Val. rows':>9} {'Fit ms':>11} "
          f"{'Predict ms':>11} {'Pred. ms/1k':>12}", flush=True)
    with threadpool_limits(limits=THREADS):
        for label, size in CASE_SIZES:
            train = january.iloc[:size].sort_values(["pickup", "record_id"])
            train = train.loc[train.pickup.lt(cutoff)]
            validation = february.iloc[:size].sort_values(["pickup", "record_id"])
            fit_ms, predict_ms = benchmark(params, train, validation)
            predict_per_1000 = predict_ms * 1000 / len(validation)
            print(f"{label:<15} {len(train):>9,} {len(validation):>9,} {fit_ms:>11.2f} "
                  f"{predict_ms:>11.2f} {predict_per_1000:>12.2f}", flush=True)


if __name__ == "__main__":
    main()
