"""Reproduce the initial CatBoost fit and report its learning curve and errors."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "for_documentation" / "figures"
BOUNDARY = pd.Timestamp("2024-02-01", tz="America/New_York")
INITIAL_SETTINGS = {"iterations": 200, "depth": 8, "learning_rate": 0.08, "l2_leaf_reg": 3}
COLOUR = "#3b6c91"


def plot_learning_curve(tree_counts, stage_mae):
    FIGURES.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.axisbelow": True, "grid.alpha": 0.25}):
        fig, ax = plt.subplots(figsize=(6.7, 3.8), layout="constrained")
        ax.plot(tree_counts, stage_mae, color=COLOUR, lw=2)
        ax.scatter(tree_counts[-1], stage_mae[-1], color="#aa4e2b", zorder=3)
        ax.annotate(f"{tree_counts[-1]} trees\nMAE {stage_mae[-1]:.4f} min",
                    (tree_counts[-1], stage_mae[-1]), xytext=(-10, 16),
                    textcoords="offset points", ha="right", va="bottom", color="#79351d")
        ax.set(title="February MAE as trees are added",
               xlabel="Trees used for prediction", ylabel="MAE (minutes)")
        ax.grid()
        fig.savefig(FIGURES / "catboost_february_learning_curve.png", dpi=250)
        plt.close(fig)


def main():
    trips = pd.read_parquet(ROOT / "data" / "lesson_trips.parquet")
    train = trips.loc[trips.pickup.lt(BOUNDARY - pd.Timedelta(hours=2))]
    end = BOUNDARY + pd.offsets.MonthBegin(1)
    validation = trips.loc[trips.pickup.ge(BOUNDARY) & trips.pickup.lt(end)]
    saved = joblib.load(ROOT / "lesson_models" / "catboost.joblib")
    features = saved.feature_names_
    model = clone(saved).set_params(**INITIAL_SETTINGS)
    actual = validation["duration_min"].to_numpy()
    x_validation = validation[features]
    with threadpool_limits(limits=4):
        model.fit(train[features], train["duration_min"])
        predicted = np.maximum(1.0, model.predict(x_validation))
        step = 5
        stage_mae = [
            mean_absolute_error(actual, np.maximum(1.0, values))
            for values in model.staged_predict(x_validation, eval_period=step)
        ]
    tree_counts = np.arange(step, model.tree_count_ + 1, step)
    np.testing.assert_allclose(stage_mae[-1], mean_absolute_error(actual, predicted))
    plot_learning_curve(tree_counts, stage_mae)

    duration_band = pd.cut(
        actual, bins=[0, 5, 10, 20, 30, 60, np.inf],
        labels=["Under 5", "5 to <10", "10 to <20", "20 to <30", "30 to <60", "60 or more"],
        right=False,
    )
    errors = pd.DataFrame({
        "band": duration_band, "absolute_error": np.abs(predicted - actual),
        "underestimated": predicted < actual,
    })
    summary = errors.groupby("band", observed=True).agg(
        trips=("absolute_error", "size"), mae=("absolute_error", "mean"),
        underestimated=("underestimated", "sum"),
    )
    print(f"Initial CatBoost | February 2024 validation | {len(actual):,} trips")
    print(f"Overall MAE = {errors.absolute_error.mean():.4f} minutes\n")
    print(f"{'Actual duration (min)':<22} {'Trips':>8} {'MAE (min)':>10} {'Underestimated':>16} {'Share':>8}")
    for band, trips, mae, underestimated in summary.itertuples():
        print(f"{band:<22} {trips:>8,} {mae:>10.2f} {underestimated:>16,} {underestimated / trips:>8.1%}")
    print("\nLearning curve saved in for_documentation/figures")


if __name__ == "__main__":
    main()
