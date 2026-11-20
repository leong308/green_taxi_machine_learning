"""Compare the D3 baseline with the selected CatBoost settings on matching trips."""

from hashlib import file_digest
import json
from pathlib import Path

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "lesson_trips.parquet"
SELECTION = ROOT / "lesson_results" / "catboost_selection.json"
BASELINE = {"iterations": 200, "depth": 8, "learning_rate": 0.08, "l2_leaf_reg": 3}
FOLDS = ["2024-02", "2024-03", "2024-04", "2024-05"]
CATS = ["pu_zone", "do_zone", "zone_pair", "pu_borough", "do_borough",
        "borough_pair", "time_band"]
FEATURES = CATS + ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "weekend",
                   "same_zone", "cross_borough", "airport_destination"]
TARGET = "duration_min"
THREADS = 4


def validation_mae(settings, train, validation):
    model = CatBoostRegressor(
        **settings,
        cat_features=CATS,
        loss_function="RMSE",
        random_seed=6001,
        thread_count=THREADS,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(train[FEATURES], train[TARGET])
    predictions = np.maximum(1.0, model.predict(validation[FEATURES]))
    return mean_absolute_error(validation[TARGET], predictions)


def print_row(label, training, validation, baseline, selected):
    reduction = (baseline - selected) / baseline
    print(f"{label:<10} {training:>9} {validation:>10} "
          f"{baseline:>10.4f} {selected:>10.4f} {reduction:>10.2%}", flush=True)


def main():
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    with DATA.open("rb") as stream:
        fingerprint = file_digest(stream, "sha256").hexdigest()
    if selection["data_sha256"] != fingerprint:
        raise ValueError("The selected settings were recorded for a different dataset.")

    winner = selection["winner"]
    expected_scores = [*winner["fold_mae"], selection["may_mae"]]
    trips = pd.read_parquet(DATA, columns=["pickup", *FEATURES, TARGET])
    baseline_scores = []
    selected_scores = []
    print("CatBoost MAE (minutes) | D3 baseline and E3 selection | Same trips")
    print(f"{'Fold':<10} {'Train':>9} {'Validation':>10} "
          f"{'D3 initial':>10} {'Selected':>10} {'Reduction':>10}", flush=True)

    with threadpool_limits(THREADS):
        for month, expected in zip(FOLDS, expected_scores, strict=True):
            boundary = pd.Timestamp(month, tz="America/New_York")
            cutoff = boundary - pd.Timedelta(hours=2)
            end = boundary + pd.offsets.MonthBegin(1)
            train = trips.loc[trips.pickup.lt(cutoff)]
            validation = trips.loc[trips.pickup.ge(boundary) & trips.pickup.lt(end)]
            baseline = validation_mae(BASELINE, train, validation)
            selected = validation_mae(winner["params"], train, validation)
            np.testing.assert_allclose(selected, expected, rtol=0, atol=1e-8)
            baseline_scores.append(baseline)
            selected_scores.append(selected)
            print_row(month, len(train), len(validation), baseline, selected)

    print_row("Feb-Apr", "-", "-", np.mean(baseline_scores[:3]), np.mean(selected_scores[:3]))


if __name__ == "__main__":
    main()
