"""Report February Ridge validation errors by actual trip duration."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Direct execution starts with for_documentation on the import path.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visualize_predictions import INITIAL_PARAMS, predict_trips


def main():
    actual, predictions = predict_trips("february", "ridge")
    predicted = predictions["ridge"]
    duration_band = pd.cut(
        actual,
        bins=[0, 5, 10, 20, 30, 60, np.inf],
        labels=["Under 5", "5 to <10", "10 to <20", "20 to <30", "30 to <60", "60 to 120"],
        right=False,
    )
    errors = pd.DataFrame({
        "duration_band": duration_band,
        "absolute_error": np.abs(predicted - actual),
        "underestimated": predicted < actual,
    })
    summary = errors.groupby("duration_band", observed=True).agg(
        trips=("absolute_error", "size"),
        mae=("absolute_error", "mean"),
        underestimated=("underestimated", "sum"),
    )
    alpha = INITIAL_PARAMS["ridge"]["regression__alpha"]
    print(f"Ridge (alpha={alpha}) | February 2024 validation | {len(actual):,} trips")
    print(f"Overall MAE = {errors['absolute_error'].mean():.4f} minutes\n")
    print(f"{'Actual duration (min)':<22} {'Trips':>8} {'MAE (min)':>10} {'Underestimated':>16} {'Share':>8}")
    for band, trips, mae, underestimated in summary.itertuples():
        share = underestimated / trips
        print(f"{band:<22} {trips:>8,} {mae:>10.2f} {underestimated:>16,} {share:>8.1%}")


if __name__ == "__main__":
    main()
