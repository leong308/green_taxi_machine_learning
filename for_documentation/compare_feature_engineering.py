import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "lesson_trips.parquet"
RESULTS = ROOT / "lesson_results"
TARGET = "duration_min"
NYC = "America/New_York"
SEED = 6001
THREADS = 4

RAW_CATS = ["pu_zone", "do_zone"]
RAW_NUMS = ["pickup_minutes"]
ENGINEERED_CATS = [
    "pu_zone", "do_zone", "zone_pair", "pu_borough", "do_borough",
    "borough_pair", "time_band",
]
ENGINEERED_NUMS = [
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "weekend",
    "same_zone", "cross_borough", "airport_destination",
]


def build_model(key, params, categorical, numerical):
    if key == "catboost":
        return CatBoostRegressor(
            **params, cat_features=categorical, loss_function="RMSE",
            random_seed=SEED, thread_count=THREADS, verbose=False,
            allow_writing_files=False,
        )

    if key == "ridge":
        estimator = Ridge(**params, solver="lsqr")
        numeric_transformer = StandardScaler()
    else:
        estimator = RandomForestRegressor(**params, random_state=SEED, n_jobs=THREADS)
        numeric_transformer = "passthrough"

    encoder = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10)
    preprocess = ColumnTransformer(
        [
            ("categories", encoder, categorical),
            ("numeric", numeric_transformer, numerical),
        ],
        sparse_threshold=1.0,
    )
    return make_pipeline(preprocess, estimator)


def score(key, params, categorical, numerical, train, validation):
    features = categorical + numerical
    model = build_model(key, params, categorical, numerical)
    model.fit(train[features], train[TARGET])
    predictions = np.maximum(1.0, model.predict(validation[features]))
    actual = validation[TARGET]
    return {
        "mae": mean_absolute_error(actual, predictions),
        "rmse": np.sqrt(mean_squared_error(actual, predictions)),
        "r2": r2_score(actual, predictions),
    }


def main():
    # Match the project's May validation split and two-hour gap.
    boundary = pd.Timestamp("2024-05-01", tz=NYC)
    validation_end = boundary + pd.offsets.MonthBegin(1)
    trips = pd.read_parquet(
        DATA,
        columns=["pickup", TARGET] + ENGINEERED_CATS + ENGINEERED_NUMS,
        filters=[("pickup", "<", validation_end)],
    )
    # Give the baseline one numeric timestamp and the two original zone IDs.
    # A nearby fixed origin preserves sub-minute precision for tree models.
    time_origin = pd.Timestamp("2024-01-01", tz=NYC)
    trips["pickup_minutes"] = (trips["pickup"] - time_origin).dt.total_seconds() / 60

    train = trips.loc[trips["pickup"] < boundary - pd.Timedelta(hours=2)]
    validation = trips.loc[trips["pickup"] >= boundary]
    if train.empty or validation.empty:
        raise SystemExit("The prepared data must contain training trips and May validation trips.")

    rows = []
    with threadpool_limits(limits=THREADS):
        for name, key in [("Ridge", "ridge"), ("Random Forest", "forest"), ("CatBoost", "catboost")]:
            # Use each model's existing rolling-validation settings for both fits.
            selection = json.loads((RESULTS / f"{key}_selection.json").read_text(encoding="utf-8"))
            params = selection["winner"]["params"]
            before = score(key, params, RAW_CATS, RAW_NUMS, train, validation)
            after = score(key, params, ENGINEERED_CATS, ENGINEERED_NUMS, train, validation)
            rows.append((name, before, after))

    before_count = len(RAW_CATS) + len(RAW_NUMS)
    after_count = len(ENGINEERED_CATS) + len(ENGINEERED_NUMS)
    print(f"May 2024 validation | Training trips: {len(train):,} | Validation trips: {len(validation):,}")
    print(f"Predictor columns | Before: {before_count} | After: {after_count}")

    print("\nMAE comparison (minutes)")
    print(
        f"{'Model':<15} {'Before feature engineering MAE':>30} "
        f"{'After feature engineering MAE':>29} {'Reduction (min)':>15} {'Reduction (%)':>14}"
    )
    for name, before, after in rows:
        reduction = before["mae"] - after["mae"]
        relative = f'{reduction / before["mae"]:.2%}' if before["mae"] else "n/a"
        print(
            f"{name:<15} {before['mae']:>30.4f} {after['mae']:>29.4f} "
            f"{reduction:>15.4f} {relative:>14}"
        )

    print("\nAdditional validation metrics")
    print(
        f"{'Model':<15} {'Before RMSE (min)':>18} {'After RMSE (min)':>18} "
        f"{'Before R2':>11} {'After R2':>11}"
    )
    for name, before, after in rows:
        print(
            f"{name:<15} {before['rmse']:>18.4f} {after['rmse']:>18.4f} "
            f"{before['r2']:>11.4f} {after['r2']:>11.4f}"
        )


if __name__ == "__main__":
    main()
