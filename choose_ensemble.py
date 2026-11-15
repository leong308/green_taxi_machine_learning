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
with DATA.open("rb") as file:
    fingerprint = file_digest(file, "sha256").hexdigest()
_, may = split_at("2024-05")
predictions = {}
may_scores = {}
for key in keys:
    info = json.loads((RESULTS / f"{key}_selection.json").read_text())
    assert info["data_sha256"] == fingerprint, key
    frame = pd.read_csv(RESULTS / f"{key}_may.csv").set_index("record_id")
    assert frame.index.is_unique and set(frame.index) == set(may.record_id), key
    predictions[key] = frame.loc[may.record_id, "prediction"].to_numpy()
    may_scores[key] = mean_absolute_error(may[TARGET], predictions[key])
first, second, third = sorted(keys, key=may_scores.get)

# Print the individual May scores used to select the two models.
ranking_table = pd.DataFrame([
    {"Rank": rank, "Model": key, "May MAE (minutes)": may_scores[key]}
    for rank, key in enumerate([first, second, third], start=1)
])
print("\nIndividual model scores on May")
print(f"Evaluation: {len(may):,} identical May trips")
print(ranking_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
print("Selected models for blending:", first, "and", second)

trials = []
for weight in np.arange(1, 10) / 10:
    blend = weight * predictions[first] + (1 - weight) * predictions[second]
    trials.append((float(weight), mean_absolute_error(may[TARGET], blend)))
weight, blend_mae = min(trials, key=lambda trial: trial[1])

# Display all tested weights without rounding the scores used for selection.
trial_table = pd.DataFrame([
    {f"{first} weight": f"{share:.0%}",
     f"{second} weight": f"{1-share:.0%}",
     "May MAE (minutes)": score,
     "Selected": "Yes" if share == weight else ""}
    for share, score in trials
])
print("\nMay blend weight trials")
print(trial_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
print(f"Lowest blend May MAE: {blend_mae:.4f} minutes")
print()

print("May ranking:", first, second, third)
print(f"Frozen weights: {weight:.0%} {first}, {1-weight:.0%} {second}")
locked = {"first": first, "second": second, "first_weight": weight,
          "data_sha256": fingerprint}
frozen_file = RESULTS / "frozen_blend.json"
frozen_file.write_text(json.dumps(locked, indent=2), encoding="utf-8")
print("Saved", frozen_file)
