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

params = {"iterations": 200, "depth": 6, "learning_rate": 0.08,
          "l2_leaf_reg": 3}
model = build_model(params)
with threadpool_limits(THREADS):
    model.fit(train[FEATURES], train[TARGET])
prediction = np.maximum(1.0, model.predict(valid[FEATURES]))
february_mae = mean_absolute_error(valid[TARGET], prediction)
print("February MAE:", round(february_mae, 4), "minutes")
print(pd.DataFrame({"actual": valid[TARGET].head().to_numpy(),
                      "prediction": prediction[:5]}).round(2))