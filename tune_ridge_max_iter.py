from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

# 1. Load the same prepared data and keep the other settings fixed.
ROOT = Path(__file__).resolve().parent
trips = pd.read_parquet(ROOT / "data" / "lesson_trips.parquet")
TARGET = "duration_min"
NYC = "America/New_York"
LIMITS = [None, 50, 100, 128, 200, 500, 1000]
MONTHS = ["2024-02", "2024-03", "2024-04"]
CATS = ["pu_zone", "do_zone", "zone_pair", "pu_borough", "do_borough",
        "borough_pair", "time_band"]
NUMS = ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "weekend",
        "same_zone", "cross_borough", "airport_destination"]
FEATURES = CATS + NUMS


def build_model(limit):
    encoder = OneHotEncoder(
        handle_unknown="infrequent_if_exist", min_frequency=10
    )
    preprocess = ColumnTransformer(
        [("categories", encoder, CATS), ("numeric", StandardScaler(), NUMS)],
        sparse_threshold=1.0,
    )
    ridge = Ridge(alpha=1, solver="lsqr", tol=0.0001, max_iter=limit)
    return Pipeline([("preprocess", preprocess), ("regression", ridge)])


# 2. Compare each limit using the same three chronological folds.
scores_table = []
steps_table = []
with threadpool_limits(4):
    for limit in LIMITS:
        label = "Auto (None)" if limit is None else str(limit)
        scores = []
        steps = []
        started = perf_counter()
        for month in MONTHS:
            boundary = pd.Timestamp(month, tz=NYC)
            cutoff = boundary - pd.Timedelta(hours=2)
            end = boundary + pd.offsets.MonthBegin(1)
            train = trips.loc[trips.pickup.lt(cutoff)]
            valid = trips.loc[trips.pickup.ge(boundary) & trips.pickup.lt(end)]
            model = build_model(limit)
            model.fit(train[FEATURES], train[TARGET])
            prediction = np.maximum(1.0, model.predict(valid[FEATURES]))
            scores.append(float(mean_absolute_error(valid[TARGET], prediction)))
            steps.append(int(model.named_steps["regression"].n_iter_[0]))
        scores_table.append({
            "max_iter": label,
            "Feb MAE": scores[0], "Mar MAE": scores[1],
            "Apr MAE": scores[2], "Mean MAE": float(np.mean(scores)),
            "Runtime (s)": perf_counter() - started,
        })
        steps_table.append({
            "max_iter": label,
            "Feb steps": steps[0], "Mar steps": steps[1],
            "Apr steps": steps[2],
        })

# 3. Print scores and the actual number of iterations used.
results = pd.DataFrame(scores_table)
print("\nMAX_ITER COMPARISON: MAE in minutes; lower is better")
print("Fixed: alpha=1, solver=lsqr, tol=0.0001\n")
print(results.to_string(
    index=False, float_format=lambda value: f"{value:.6f}"
))
print("\nACTUAL SOLVER ITERATIONS")
print(pd.DataFrame(steps_table).to_string(index=False))

best_mae = results["Mean MAE"].min()
tied = results.loc[
    np.isclose(results["Mean MAE"], best_mae, rtol=0, atol=1e-10),
    "max_iter",
]
print(f"\nLowest average validation MAE: {best_mae:.6f} minutes")
print("Best or numerically tied settings:", ", ".join(tied))
print("The solver can stop before max_iter when its stopping criteria are met.")
