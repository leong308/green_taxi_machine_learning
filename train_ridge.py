from hashlib import file_digest
from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from threadpoolctl import threadpool_limits
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# load the shared data and define the predictors and target
KEY = "ridge"
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

# split trips by date for training and validation
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

# creation of model for Feb prediction
def build_model(params):
    zone_encoder = OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=10)
    preprocess = ColumnTransformer(
        [("categories", zone_encoder, CATS), ("numeric", StandardScaler(), NUMS)],
        sparse_threshold=1.0,
    )
    estimator = Ridge(alpha=params["alpha"], solver="lsqr")
    return Pipeline([("preprocess", preprocess), ("regression", estimator)])

params = {"alpha": 10}
model = build_model(params)
with threadpool_limits(THREADS):
    model.fit(train[FEATURES], train[TARGET])
prediction = np.maximum(1.0, model.predict(valid[FEATURES]))
february_mae = mean_absolute_error(valid[TARGET], prediction)
print("February MAE:", round(february_mae, 4), "minutes")
print(pd.DataFrame({"actual": valid[TARGET].head().to_numpy(),
                      "prediction": prediction[:5]}).round(2))

# Plot every February validation prediction against the duration of the same trip.
actual = valid[TARGET].to_numpy()
limit = 10 * np.ceil(max(actual.max(), prediction.max()) / 10)
fig, ax = plt.subplots(figsize=(7.2, 6.6), layout="constrained")
ax.scatter(actual, prediction, s=9, color="#2265A0", alpha=0.25, linewidths=0)
ax.plot([0, limit], [0, limit], "--", color="#263645", label="Perfect prediction")
ax.set(
    xlabel="Actual trip duration (minutes)",
    ylabel="Predicted trip duration (minutes)",
    xlim=(0, limit), ylim=(0, limit), aspect="equal",
)
ax.set_title(f"{len(actual):,} trips  |  α = {params['alpha']:g}  |  MAE = {february_mae:.4f} min")
ax.grid(alpha=0.25)
ax.set_axisbelow(True)
ax.legend(loc="upper left")
RESULTS.mkdir(exist_ok=True)
validation_plot = RESULTS / "ridge_february_validation.png"
fig.savefig(validation_plot, dpi=200)
print("Validation graph:", validation_plot)

# added 6 candidates settings
candidates = [{"alpha": value} for value in [1, 0.1, 3, 10, 30, 100]]
print(len(candidates), "configurations")

# fit fresh model for each candidate and month
REUSE_TUNING = True
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
            search.append({"params": settings, "fold_mae": errors,
                           "mean_mae": float(np.mean(errors))})
    cache = {"data_sha256": fingerprint, "candidates": candidates, "search": search}
    tuning_file.write_text(json.dumps(cache, indent=2), encoding="utf-8")
winner = min(search, key=lambda row: row["mean_mae"])
print(pd.json_normalize(search).round(4))
print("Selected parameters:", winner["params"])
print("Selected validation MAE:", round(winner["mean_mae"], 4))

# fit the winner on the eligible trips before May
# record each May prediction with trip ID
past, may = split_at("2024-05")
may_model = build_model(winner["params"])
with threadpool_limits(THREADS):
    may_model.fit(past[FEATURES], past[TARGET])
may_prediction = np.maximum(1.0, may_model.predict(may[FEATURES]))
RESULTS.mkdir(exist_ok=True)
may_mae = float(mean_absolute_error(may[TARGET], may_prediction))
pd.DataFrame({"record_id": may.record_id, "prediction": may_prediction}).to_csv(
    RESULTS / f"{KEY}_may.csv", index=False
)
summary = {"model": KEY, "data_sha256": fingerprint, "may_mae": may_mae,
           "winner": winner, "search": search}
(RESULTS / f"{KEY}_selection.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print("May MAE:", round(may_mae, 4), "minutes")
plt.show()
