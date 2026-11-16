from hashlib import file_digest
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from threadpoolctl import threadpool_limits
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# load the shared data and define the predictors and target
KEY = "combine"
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

# split trips by date for training and validation
def split_at(month):
    boundary = pd.Timestamp(month, tz=NYC)
    end = boundary + pd.offsets.MonthBegin(1)
    last_training_time = boundary - pd.Timedelta(hours=2)
    history = trips.loc[trips.pickup.lt(last_training_time)]
    evaluation = trips.loc[trips.pickup.ge(boundary) & trips.pickup.lt(end)]
    return history, evaluation

keys = ["ridge", "forest", "catboost"]
frozen_file = RESULTS / "frozen_blend.json"
locked = json.loads(frozen_file.read_text(encoding="utf-8"))
with DATA.open("rb") as file:
    fingerprint = file_digest(file, "sha256").hexdigest()
if locked["data_sha256"] != fingerprint:
    raise ValueError("The frozen blend belongs to a different data file.")
first = locked["first"]
second = locked["second"]
weight = locked["first_weight"]
_, june = split_at("2024-06")
predictions = {}
for key in keys:
    model = joblib.load(MODELS / f"{key}.joblib")
    predictions[key] = np.maximum(1.0, model.predict(june[FEATURES]))
predictions["blend"] = weight * predictions[first] + (1 - weight) * predictions[second]
rows = []
for key, pred in predictions.items():
    mse = mean_squared_error(june[TARGET], pred)
    mean_signed_error = np.mean(pred - june[TARGET].to_numpy())
    rows.append({"model": key, "MAE": mean_absolute_error(june[TARGET], pred),
                 "RMSE": np.sqrt(mse), "MSE": mse,
                 "R squared": r2_score(june[TARGET], pred),
                 "Mean signed error": mean_signed_error})
comparison = pd.DataFrame(rows).set_index("model")
print(comparison.round(4).to_string())
comparison.to_csv(RESULTS / "june_comparison.csv")
