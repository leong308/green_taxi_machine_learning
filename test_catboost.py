from hashlib import file_digest
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from threadpoolctl import threadpool_limits
KEY = "catboost"
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

from catboost import CatBoostRegressor
def build_model(params):
    return CatBoostRegressor(
        **params,
        cat_features=CATS,
        loss_function="RMSE",
        random_seed=SEED,
        thread_count=THREADS,
        verbose=False,
        allow_writing_files=False,
    )

settings_file = RESULTS / f"{KEY}_selection.json"
summary = json.loads(settings_file.read_text(encoding="utf-8"))
with DATA.open("rb") as file:
    fingerprint = file_digest(file, "sha256").hexdigest()
if summary["data_sha256"] != fingerprint:
    raise ValueError("Use the same data file as the training run.")
winner = summary["winner"]
past, june = split_at("2024-06")
final_model = build_model(winner["params"])
with threadpool_limits(THREADS):
    final_model.fit(past[FEATURES], past[TARGET])
june_prediction = np.maximum(1.0, final_model.predict(june[FEATURES]))
mse = mean_squared_error(june[TARGET], june_prediction)
print("June MAE:", round(mean_absolute_error(june[TARGET], june_prediction), 4))
print("June RMSE:", round(np.sqrt(mse), 4))
print("June MSE:", round(mse, 4))
print("June R squared:", round(r2_score(june[TARGET], june_prediction), 4))
MODELS.mkdir(exist_ok=True)
artifact = MODELS / f"{KEY}.joblib"
joblib.dump(final_model, artifact, compress=3)
print("Saved", artifact)