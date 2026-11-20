"""Plot matched actual and predicted taxi-trip durations for every evaluation trip."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "lesson_trips.parquet"
MODELS = ROOT / "lesson_models"
RESULTS = ROOT / "lesson_results"
NYC = "America/New_York"
TARGET = "duration_min"
NAMES = {"ridge": "Ridge", "forest": "Random Forest", "catboost": "CatBoost",
         "blend": "Frozen blend"}
COLOURS = {"ridge": "#2265a0", "forest": "#b3580a", "catboost": "#914077",
           "blend": "#077d68"}
ACTUAL_COLOUR = "#263645"
MONTHS = {"february": "2024-02", "june": "2024-06"}

# Initial-fit settings from the three train scripts, before rolling validation.
# The saved model artifacts supply the estimator and preprocessing definitions.
INITIAL_PARAMS = {
    "ridge": {"regression__alpha": 10},
    "forest": {"regression__n_estimators": 70, "regression__max_depth": 24,
               "regression__min_samples_leaf": 15, "regression__max_features": 0.15},
    "catboost": {"iterations": 200, "depth": 8, "learning_rate": 0.08,
                 "l2_leaf_reg": 3},
}


def predict_trips(split, model_key):
    """Return evaluation rows and predictions, in identical order for all models."""
    trips = pd.read_parquet(DATA)
    boundary = pd.Timestamp(MONTHS[split], tz=NYC)
    end = boundary + pd.offsets.MonthBegin(1)
    evaluation = trips.loc[trips.pickup.ge(boundary) & trips.pickup.lt(end)]
    if evaluation.empty:
        raise ValueError(f"No evaluation trips for {boundary:%B %Y}.")
    if split == "february":
        # Match the initial January fit and its two-hour cutoff.
        history = trips.loc[trips.pickup.lt(boundary - pd.Timedelta(hours=2))]

    keys = INITIAL_PARAMS if model_key == "all" else (model_key,)
    predictions = {}
    with threadpool_limits(limits=4):
        for key in keys:
            saved_model = joblib.load(MODELS / f"{key}.joblib")
            features = list(saved_model.feature_names_ if key == "catboost"
                            else saved_model.feature_names_in_)
            model = saved_model
            if split == "february":
                model = clone(saved_model).set_params(**INITIAL_PARAMS[key])
                model.fit(history[features], history[TARGET])
            values = np.maximum(1.0, model.predict(evaluation[features]))
            if values.shape != (len(evaluation),) or not np.isfinite(values).all():
                raise ValueError(f"{NAMES[key]} did not return one finite prediction per trip.")
            predictions[key] = values

    if split == "june" and model_key == "all":
        frozen = json.loads((RESULTS / "frozen_blend.json").read_text(encoding="utf-8"))
        with DATA.open("rb") as stream:
            fingerprint = file_digest(stream, "sha256").hexdigest()
        if frozen["data_sha256"] != fingerprint:
            raise ValueError("The frozen blend belongs to a different dataset.")
        weight = frozen["first_weight"]
        predictions["blend"] = (weight * predictions[frozen["first"]]
                                + (1 - weight) * predictions[frozen["second"]])

    return evaluation[TARGET].to_numpy(), predictions


def model_axes(plt, predictions, panel_width, panel_height):
    """Put comparable model panels on the same axes scales."""
    count = len(predictions)
    columns = 2 if count == 4 else count
    rows = 2 if count == 4 else 1
    fig, axes = plt.subplots(rows, columns, squeeze=False, sharex=True, sharey=True,
                             figsize=(panel_width * columns, panel_height * rows + 0.7),
                             layout="constrained")
    return fig, axes.flat


def plot_agreement(plt, actual, predictions, context, limit):
    fig, axes = model_axes(plt, predictions, 4.5, 4.0)
    for ax, (key, predicted) in zip(axes, predictions.items()):
        colour = COLOURS[key]
        ax.scatter(actual, predicted, s=6, color=colour, alpha=0.22, linewidths=0)
        ax.plot([0, limit], [0, limit], "--", color=ACTUAL_COLOUR, lw=1.2)
        mae = np.mean(np.abs(predicted - actual))
        ax.set_title(f"{NAMES[key]}  |  MAE {mae:.2f} min")
        ax.set(xlim=(0, limit), ylim=(0, limit), aspect="equal")
        ax.set_xticks(np.arange(0, limit + 1, 30))
        ax.set_yticks(np.arange(0, limit + 1, 30))
    fig.supxlabel("Actual duration (minutes)")
    fig.supylabel("Predicted duration (minutes)")
    fig.suptitle(f"Actual vs predicted duration | {context}\n"
                 "One dot per trip. Below the dashed line means the trip took longer than predicted.")
    return fig


def plot_trip_order(plt, actual, predictions, context, limit):
    fig, axes = model_axes(plt, predictions, 4.5, 3.5)
    order = np.argsort(actual, kind="stable")
    trip_rank = np.arange(1, len(actual) + 1)
    for ax, (key, predicted) in zip(axes, predictions.items()):
        ax.scatter(trip_rank, predicted[order], s=4, color=COLOURS[key], alpha=0.20,
                   label="Predicted", linewidths=0)
        ax.plot(trip_rank, actual[order], color=ACTUAL_COLOUR, lw=1.5, label="Actual")
        ax.set_title(NAMES[key])
        ax.set(xlim=(1, len(actual)), ylim=(0, limit))
        ticks = np.r_[1, np.arange(5000, len(actual) + 1, 5000)]
        ax.set_xticks(ticks, [f"{tick:,}" for tick in ticks])
        ax.legend(loc="upper left", markerscale=2)
    fig.supxlabel("Trip rank by actual duration (shortest to longest)")
    fig.supylabel("Duration (minutes)")
    fig.suptitle(f"Trip-by-trip comparison | {context}\n"
                 "Each model panel uses the same trip order and the same actual-duration line.")
    return fig


def plot_bias(plt, actual, predictions, context):
    cutoffs = [5, 10, 20, 30, 60]
    groups = np.searchsorted(cutoffs, actual, side="right")
    labels = ["Under 5", "5 to <10", "10 to <20", "20 to <30", "30 to <60", "60 or more"]
    positions = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11.5, 5.5), layout="constrained")
    offsets = (np.arange(len(predictions)) - (len(predictions) - 1) / 2) * 0.15
    for offset, (key, predicted) in zip(offsets, predictions.items()):
        errors = predicted - actual
        quantiles = np.full((3, len(labels)), np.nan)
        for index in positions:
            band = errors[groups == index]
            if band.size:
                quantiles[:, index] = np.percentile(band, [25, 50, 75])
        lower, middle, upper = quantiles
        ax.errorbar(positions + offset, middle, yerr=[middle - lower, upper - middle],
                    fmt="o", color=COLOURS[key], lw=1.5, capsize=3, label=NAMES[key])
    ax.axhline(0, color=ACTUAL_COLOUR, linestyle="--", lw=1)
    ax.set_xticks(positions, [f"{label}\n{np.count_nonzero(groups == index):,} trips"
                             for index, label in enumerate(labels)])
    ax.set(xlabel="Actual trip duration (minutes)", ylabel="Predicted minus actual (minutes)")
    ax.legend(loc="lower left")
    fig.suptitle(f"Where models overpredict or underpredict | {context}\n"
                 "Dots show the median error. Whiskers span the middle 50% of errors.")
    return fig


def plot_error_thresholds(plt, actual, predictions, context):
    thresholds = np.array([5, 10, 15])
    positions = np.arange(len(thresholds))
    width = 0.72 / len(predictions)
    offsets = (np.arange(len(predictions)) - (len(predictions) - 1) / 2) * width
    fig, ax = plt.subplots(figsize=(10.5, 5.2), layout="constrained")
    for offset, (key, predicted) in zip(offsets, predictions.items()):
        absolute_error = np.abs(predicted - actual)
        percentages = 100 * np.mean(absolute_error[:, None] <= thresholds, axis=0)
        bars = ax.bar(positions + offset, percentages, width * 0.93,
                      color=COLOURS[key], label=NAMES[key], zorder=3)
        ax.bar_label(bars, labels=[f"{value:.1f}%" for value in percentages],
                     padding=3, fontsize=9)
    ax.set_xticks(positions, [f"Within {value} minutes" for value in thresholds])
    ax.set(ylim=(0, 105), ylabel="Share of trips (%)")
    ax.set_yticks(np.arange(0, 101, 20))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncols=len(predictions))
    fig.suptitle(f"How close were the predictions? | {context}")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=MONTHS, default="february",
                        help="Initial February validation (default) or the saved June test models.")
    parser.add_argument("--model", choices=("all", *INITIAL_PARAMS), default="all",
                        help="Plot all models (default) or just one model.")
    parser.add_argument("--no-show", action="store_true",
                        help="Save the graphs without opening windows.")
    args = parser.parse_args()
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    actual, predictions = predict_trips(args.split, args.model)
    split_label = "February 2024 validation" if args.split == "february" else "June 2024 test"
    context = f"{split_label} ({len(actual):,} trips)"
    maximum = max(actual.max(), *(values.max() for values in predictions.values()))
    limit = max(120, int(np.ceil(maximum / 30)) * 30)
    folder = RESULTS / "prediction_graphs" / args.split
    if args.model != "all":
        folder = folder / args.model
    folder.mkdir(parents=True, exist_ok=True)

    print(context)
    for key, predicted in predictions.items():
        print(f"{NAMES[key]:<15} MAE {np.mean(np.abs(predicted - actual)):.4f} min")

    with plt.rc_context({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.edgecolor": "#a9b5bf",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#e2e8ee",
        "figure.titlesize": 14,
        "legend.frameon": False,
    }):
        figures = {
            "01_actual_vs_predicted": plot_agreement(plt, actual, predictions, context, limit),
            "02_trip_by_trip": plot_trip_order(plt, actual, predictions, context, limit),
            "03_error_by_duration": plot_bias(plt, actual, predictions, context),
            "04_error_thresholds": plot_error_thresholds(plt, actual, predictions, context),
        }
        for name, fig in figures.items():
            fig.savefig(folder / f"{name}.png", dpi=200)
        print(f"Graphs saved to {folder}")
        if not args.no_show:
            plt.show()
        plt.close("all")


if __name__ == "__main__":
    main()
