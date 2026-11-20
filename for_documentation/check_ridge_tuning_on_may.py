"""Compare the D1 Ridge baseline and selected Ridge settings on the same May trips."""

from hashlib import file_digest
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "lesson_trips.parquet"
SELECTION = ROOT / "lesson_results" / "ridge_selection.json"
MODEL = ROOT / "lesson_models" / "ridge.joblib"
D1_ALPHA = 10


def main():
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    with DATA.open("rb") as stream:
        fingerprint = file_digest(stream, "sha256").hexdigest()
    if selection["data_sha256"] != fingerprint:
        raise ValueError("The selected settings were recorded for a different dataset.")

    trips = pd.read_parquet(DATA)
    boundary = pd.Timestamp("2024-05-01", tz="America/New_York")
    cutoff = boundary - pd.Timedelta(hours=2)
    end = boundary + pd.offsets.MonthBegin(1)
    train = trips.loc[trips.pickup.lt(cutoff)]
    validation = trips.loc[trips.pickup.ge(boundary) & trips.pickup.lt(end)]
    saved_model = joblib.load(MODEL)
    features = list(saved_model.feature_names_in_)
    selected_alpha = selection["winner"]["params"]["alpha"]
    comparisons = [("D1 baseline", D1_ALPHA), ("Selected in E1", selected_alpha)]
    scores = []

    print(f"May 2024 validation | Train: {len(train):,} trips | May: {len(validation):,} trips")
    print(f"{'Setting':<18} {'Alpha':>7} {'MAE (min)':>11} {'LSQR steps':>11}")
    with threadpool_limits(4):
        for label, alpha in comparisons:
            model = clone(saved_model).set_params(
                regression__alpha=alpha, regression__solver="lsqr",
                regression__tol=0.0001, regression__max_iter=None,
            )
            model.fit(train[features], train["duration_min"])
            predictions = np.maximum(1.0, model.predict(validation[features]))
            mae = mean_absolute_error(validation["duration_min"], predictions)
            steps = model.named_steps["regression"].n_iter_[0]
            scores.append(mae)
            print(f"{label:<18} {alpha:>7g} {mae:>11.4f} {steps:>11}")

    np.testing.assert_allclose(scores[1], selection["may_mae"], rtol=0, atol=1e-8)
    mae_reduction = scores[0] - scores[1]
    print(f"Selected MAE reduction = {mae_reduction:.4f} min ({mae_reduction / scores[0]:.2%})")


if __name__ == "__main__":
    main()
