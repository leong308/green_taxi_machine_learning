from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "lesson_trips.parquet"
FIGURE = ROOT / "for_documentation" / "figures" / "ridge_regularization_comparison.png"

TARGET = "duration_min"
CATS = [
    "pu_zone", "do_zone", "zone_pair", "pu_borough", "do_borough",
    "borough_pair", "time_band",
]
NUMS = [
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "weekend",
    "same_zone", "cross_borough", "airport_destination",
]
FEATURES = CATS + NUMS
RIDGE_ALPHA = 10.0
SOLVER_TOL = 1e-4


def coefficient_path(x_train, y_train, before, after):
    # Follow the coefficient-path method in the scikit-learn Ridge example:
    # https://scikit-learn.org/1.7/auto_examples/linear_model/plot_ridge_path.html
    alphas = np.unique(np.r_[0.0, RIDGE_ALPHA, np.logspace(-2, 3, 41)])
    fitted = {0.0: before["coefficients"], RIDGE_ALPHA: after["coefficients"]}
    for alpha in alphas:
        if alpha not in fitted:
            model = Ridge(alpha=alpha, solver="lsqr", tol=SOLVER_TOL)
            model.fit(x_train, y_train)
            fitted[alpha] = model.coef_
    return alphas, np.stack([fitted[alpha] for alpha in alphas])


def plot_coefficient_path(alphas, coefficients):
    fig, ax = plt.subplots(figsize=(8.6, 4.8), layout="constrained")
    ax.plot(alphas, coefficients, color="#205C89", alpha=0.22, linewidth=0.8)
    ax.axhline(0, color="#475466", linewidth=1, zorder=2)
    ax.axvline(RIDGE_ALPHA, color="#A94F27", linestyle="--", linewidth=1.4)

    # Symmetric log keeps alpha=0 on the same axis as the positive penalties.
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(0, alphas[-1])
    ticks = sorted({0, 1, 10, 100, 1000, RIDGE_ALPHA})
    labels = {0: "0\nNo regularization", RIDGE_ALPHA: f"{RIDGE_ALPHA:,g}\nRidge"}
    ax.set_xticks(ticks, [labels.get(tick, f"{tick:,g}") for tick in ticks])
    ax.set(xlabel="Regularization strength (alpha)", ylabel="Model weight")
    ax.grid(axis="y", color="#E1E6EA")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=300, bbox_inches="tight", pad_inches=0.12)
    print(f"\nFigure saved to {FIGURE}")
    plt.show()


def main():
    # Match D1's chronological split and the two-hour gap.
    boundary = pd.Timestamp("2024-02-01", tz="America/New_York")
    validation_end = boundary + pd.offsets.MonthBegin(1)
    trips = pd.read_parquet(
        DATA,
        columns=["pickup", TARGET] + FEATURES,
        filters=[("pickup", "<", validation_end)],
    )
    train = trips.loc[trips["pickup"] < boundary - pd.Timedelta(hours=2)]
    validation = trips.loc[trips["pickup"].ge(boundary) & trips["pickup"].lt(validation_end)]
    if train.empty or validation.empty:
        raise SystemExit("The prepared data must contain January and February 2024 trips.")

    # Fit one shared preprocessing transform using the training rows only.
    preprocess = ColumnTransformer(
        [
            ("categories", OneHotEncoder(
                handle_unknown="infrequent_if_exist", min_frequency=10,
            ), CATS),
            ("numeric", StandardScaler(), NUMS),
        ],
        sparse_threshold=1.0,
    )
    x_train = preprocess.fit_transform(train[FEATURES])
    x_validation = preprocess.transform(validation[FEATURES])
    y_train = train[TARGET].to_numpy()
    y_validation = validation[TARGET].to_numpy()

    # On the sparse design, both estimators use LSQR with the same tolerance.
    models = [
        ("Before regularization (OLS)", LinearRegression(tol=SOLVER_TOL)),
        (f"After regularization (Ridge, alpha={RIDGE_ALPHA:g})",
         Ridge(alpha=RIDGE_ALPHA, solver="lsqr", tol=SOLVER_TOL)),
    ]
    results = []
    with threadpool_limits(limits=4):
        for label, model in models:
            model.fit(x_train, y_train)
            # Apply the project's one-minute prediction floor to both models.
            training_prediction = np.maximum(1.0, model.predict(x_train))
            prediction = np.maximum(1.0, model.predict(x_validation))
            results.append({
                "label": label,
                "coefficients": model.coef_,
                "training_mae": mean_absolute_error(y_train, training_prediction),
                "validation_mae": mean_absolute_error(y_validation, prediction),
                "coefficient_norm": np.linalg.norm(model.coef_),
            })

    print(f"January training trips: {len(train):,} | February validation trips: {len(validation):,}")
    print(f"\n{'Model':<42} {'Training MAE (min)':>18} {'February MAE (min)':>18} {'L2 coefficient norm':>20}")
    for row in results:
        print(
            f'{row["label"]:<42} {row["training_mae"]:>18.4f} '
            f'{row["validation_mae"]:>18.4f} {row["coefficient_norm"]:>20.2f}'
        )
    with threadpool_limits(limits=4):
        alphas, coefficients = coefficient_path(x_train, y_train, *results)
    plot_coefficient_path(alphas, coefficients)


if __name__ == "__main__":
    main()
