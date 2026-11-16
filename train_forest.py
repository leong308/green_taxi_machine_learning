#b1 Load shared dataset
from hashlib import file_digest
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from threadpoolctl import threadpool_limits

KEY = "forest"
NYC = "America/New_York"
SEED = 6001
THREADS = 4
TARGET = "duration_min"
DATA = Path("data/lesson_trips.parquet")
MODELS = Path("lesson_models")
RESULTS = Path("lesson_results")

CATS = ["pu_zone", "do_zone", "zone_pair", "pu_borough", "do_borough",
        "borough_pair", "time_band"]
NUMS = ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "weekend",
        "same_zone", "cross_borough", "airport_destination"]
FEATURES = CATS + NUMS

trips = pd.read_parquet(DATA)
assert len(FEATURES) == 15 and trips[FEATURES + [TARGET]].notna().all().all()
print(trips.shape, "| target:", TARGET, "| predictors:", len(FEATURES))
print(trips[["pickup", "pu_zone", "do_zone", TARGET]].head(3))

#b2 Split training and validation
def split_at(month):
    boundary = pd.Timestamp(month, tz=NYC)
    end = boundary + pd.offsets.MonthBegin(1)
    last_training_time = boundary - pd.Timedelta(hours=2)
    history = trips.loc[trips.pickup.lt(last_training_time)]
    evaluation = trips.loc[trips.pickup.ge(boundary) & trips.pickup.lt(end)]
    return history, evaluation

train, valid = split_at("2024-02")
print("Fit rows:", len(train), "| validation rows:", len(valid))
print("Last fitting pickup:", train.pickup.max())
print("First validation pickup:", valid.pickup.min())

#b3 Build Random Forest pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

def build_model(params):
    zone_encoder = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=10
    )

    preprocess = ColumnTransformer(
        [
            ("categories", zone_encoder, CATS),
            ("numeric", "passthrough", NUMS)
        ],
        sparse_threshold=1.0
    )

    estimator = RandomForestRegressor(
        **params,
        random_state=SEED,
        n_jobs=THREADS
    )

    return Pipeline([
        ("preprocess", preprocess),
        ("regression", estimator)
    ])


#b4 Initial training
params = {"n_estimators": 70, "max_depth": 24,
          "min_samples_leaf": 15, "max_features": 0.15}

model = build_model(params)

with threadpool_limits(THREADS):
    model.fit(train[FEATURES], train[TARGET])

prediction = np.maximum(1.0, model.predict(valid[FEATURES]))
february_mae = mean_absolute_error(valid[TARGET], prediction)

print("February MAE:", round(february_mae, 4), "minutes")
print(pd.DataFrame({
    "actual": valid[TARGET].head().to_numpy(),
    "prediction": prediction[:5]
}).round(2))

#b5Hyperparameter tuning

candidates = [
    {"n_estimators": trees, "max_depth": depth,
     "min_samples_leaf": leaf, "max_features": fraction}
    for trees, depth, leaf, fraction in [
        (70, 12, 15, 0.15),
        (70, 18, 10, 0.15),
        (70, 24, 5, 0.15),
        (90, 22, 8, 0.25),
        (90, 28, 4, 0.25),
        (100, 30, 4, 0.35),
    ]
]

print(len(candidates), "configurations")


#b6 Hyperparameter tuning

REUSE_TUNING = True
RESULTS.mkdir(exist_ok=True)
tuning_file = RESULTS / f"{KEY}_tuning.json"

with DATA.open("rb") as file:
    fingerprint = file_digest(file, "sha256").hexdigest()

if REUSE_TUNING:
    cached = json.loads(tuning_file.read_text(encoding="utf-8"))
    if cached["data_sha256"] != fingerprint or cached["candidates"] != candidates:
        raise ValueError("Data or settings changed. Set REUSE_TUNING = False and rerun.")
    search = cached["search"]
else:
    search = []
    with threadpool_limits(THREADS):
        for settings in candidates:
            errors = []
            for month in ["2024-02", "2024-03", "2024-04"]:
                past, future = split_at(month)
                candidate = build_model(settings)
                candidate.fit(past[FEATURES], past[TARGET])
                predicted = np.maximum(1.0, candidate.predict(future[FEATURES]))
                errors.append(float(mean_absolute_error(future[TARGET], predicted)))

            search.append({
                "params": settings,
                "fold_mae": errors,
                "mean_mae": float(np.mean(errors))
            })

    cache = {
        "data_sha256": fingerprint,
        "candidates": candidates,
        "search": search
    }
    tuning_file.write_text(
        json.dumps(cache, indent=2),
        encoding="utf-8"
    )

winner = min(search, key=lambda row: row["mean_mae"])

print(pd.json_normalize(search).round(4))
print("Selected parameters:", winner["params"])
print("Selected validation MAE:", round(winner["mean_mae"], 4))



# B7 - May Evaluation and First Push

past, may = split_at("2024-05")

may_model = build_model(winner["params"])

with threadpool_limits(THREADS):
    may_model.fit(past[FEATURES], past[TARGET])

may_prediction = np.maximum(
    1.0, may_model.predict(may[FEATURES])
)

RESULTS.mkdir(exist_ok=True)

may_mae = float(
    mean_absolute_error(may[TARGET], may_prediction)
)

pd.DataFrame({
    "record_id": may.record_id,
    "prediction": may_prediction
}).to_csv(
    RESULTS / f"{KEY}_may.csv",
    index=False
)

summary = {
    "model": KEY,
    "data_sha256": fingerprint,
    "may_mae": may_mae,
    "winner": winner,
    "search": search
}

(RESULTS / f"{KEY}_selection.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print("May MAE:", round(may_mae, 4), "minutes")

