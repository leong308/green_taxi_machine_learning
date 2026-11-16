"""Additional 5%-step ensemble search: choose on May, compare on June."""

from hashlib import file_digest
from itertools import combinations
from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "lesson_trips.parquet"
RESULTS = ROOT / "lesson_results"
MODELS = ROOT / "lesson_models"
KEYS = ["ridge", "forest", "catboost"]
TARGET = "duration_min"
NYC = "America/New_York"
STEP_PERCENT = 5
PRINT_ALL_TRIALS = False  # True prints every trial instead of the best 10.
CATS = ["pu_zone", "do_zone", "zone_pair", "pu_borough", "do_borough",
        "borough_pair", "time_band"]
NUMS = ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "weekend",
        "same_zone", "cross_borough", "airport_destination"]
FEATURES = CATS + NUMS
WEIGHT_COLUMNS = [f"{key} %" for key in KEYS]


def month_rows(trips, month):
    """Select the same evaluation month used by the original project."""
    start = pd.Timestamp(month, tz=NYC)
    end = start + pd.offsets.MonthBegin(1)
    return trips.loc[trips.pickup.ge(start) & trips.pickup.lt(end)]


def blend_predictions(predictions, weights):
    """Combine matching trip predictions using weights that add to 100%."""
    return sum(weights[key] / 100 * predictions[key] for key in KEYS)


def make_trial(weights, predictions, actual):
    values = blend_predictions(predictions, weights)
    return {
        **{f"{key} %": int(weights[key]) for key in KEYS},
        "Active models": sum(value > 0 for value in weights.values()),
        "May MAE": float(mean_absolute_error(actual, values)),
    }


def search_weights(predictions, actual, step=STEP_PERCENT):
    """Test all pairs, then every nonnegative three-weight combination."""
    if not isinstance(step, int) or step <= 0 or 100 % step:
        raise ValueError("The step must be a positive integer dividing 100.")
    units = 100 // step
    if units < 3:
        raise ValueError("Use a step small enough to give all three models weight.")

    # 1. Every pair gets a full 0%-100% search; no model is excluded first.
    pair_trials = []
    for first, second in combinations(KEYS, 2):
        for first_units in range(units + 1):
            weights = dict.fromkeys(KEYS, 0)
            weights[first] = first_units * step
            weights[second] = 100 - weights[first]
            pair_trials.append({
                "Pair": f"{first} + {second}",
                **make_trial(weights, predictions, actual),
            })

    # 2. Integer steps keep the three weights exactly at a total of 100%.
    three_trials = []
    for ridge_units in range(units + 1):
        for forest_units in range(units - ridge_units + 1):
            catboost_units = units - ridge_units - forest_units
            weights = {
                "ridge": ridge_units * step,
                "forest": forest_units * step,
                "catboost": catboost_units * step,
            }
            three_trials.append(make_trial(weights, predictions, actual))
    return pd.DataFrame(pair_trials), pd.DataFrame(three_trials)


def choose_best(frame, active_models=None):
    """Choose by May MAE; numerical ties prefer fewer active models."""
    candidates = frame
    if active_models is not None:
        candidates = candidates.loc[candidates["Active models"].eq(active_models)]
    if candidates.empty:
        raise ValueError("No candidates meet the requested number of active models.")
    lowest = candidates["May MAE"].min()
    tied = candidates.loc[(candidates["May MAE"] - lowest).abs().le(1e-12)]
    return tied.sort_values(
        ["Active models"] + WEIGHT_COLUMNS, kind="stable"
    ).iloc[0]


def selection_record(row):
    return {
        "weights_percent": {key: int(row[f"{key} %"]) for key in KEYS},
        "active_models": int(row["Active models"]),
        "may_mae": float(row["May MAE"]),
    }


def print_trials(title, frame):
    ordered = frame.sort_values("May MAE", kind="stable")
    shown = ordered if PRINT_ALL_TRIALS else ordered.head(10)
    print(f"\n{title}: {len(frame)} trials")
    if not PRINT_ALL_TRIALS and len(frame) > len(shown):
        print(f"Showing the best {len(shown)}. Set PRINT_ALL_TRIALS=True to see all.")
    print(shown[WEIGHT_COLUMNS + ["May MAE"]].to_string(
        index=False, float_format=lambda value: f"{value:.6f}"
    ))


def make_graphs(pair_trials, three_trials, selected, comparison):
    """Save three figures showing the search and the June comparison."""
    import matplotlib.pyplot as plt

    folder = RESULTS / "additional_ensemble_graphs"
    folder.mkdir(exist_ok=True)
    paths = []
    names = {"ridge": "Ridge", "forest": "Random Forest", "catboost": "CatBoost"}
    colours = {"ridge": "#4477AA", "forest": "#228833", "catboost": "#AA3377"}
    labels = {
        **names,
        "Original frozen blend": "Original blend",
        "Best two-model blend": "Best two-model blend",
        "Best all-three blend": "Best three-model blend",
    }
    row_colours = {
        **colours, "Original frozen blend": "#8D9FA8",
        "Best two-model blend": "#008899", "Best all-three blend": "#DD9933",
    }

    with plt.rc_context({
        "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
        "figure.facecolor": "white", "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#A7B6BE",
        "text.color": "#203541", "axes.labelcolor": "#203541",
    }):
        # Graph 1: show every 5%-step trial for each of the three pairs.
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
        for ax, ((first, second), (_, frame)) in zip(
            axes, zip(combinations(KEYS, 2), pair_trials.groupby("Pair", sort=False))
        ):
            ordered = frame.sort_values(f"{first} %")
            best = choose_best(frame)
            ax.plot(
                ordered[f"{first} %"], ordered["May MAE"],
                marker="o", markersize=4, linewidth=2, color=colours[first],
                label="All tested weights",
            )
            ax.scatter(
                best[f"{first} %"], best["May MAE"], marker="*",
                s=200, color="#EEBB33", edgecolor="#203541", zorder=5,
                label="Lowest May MAE",
            )
            ax.set_title(
                f"{names[first]} + {names[second]}\n"
                f"Best: {best[f'{first} %']:.0f}% / "
                f"{best[f'{second} %']:.0f}% | MAE {best['May MAE']:.4f}",
                pad=12,
            )
            ax.set_xlabel(
                f"{names[first]} weight (%)\n"
                f"{names[second]} receives the remaining weight"
            )
            ax.set_xticks(np.arange(0, 101, 20))
            ax.set_xlim(-2, 102)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=9)
        axes[0].set_ylabel("May MAE (minutes) - lower is better")
        fig.suptitle("Two-model weight searches on May", fontsize=16)
        fig.text(
            0.5, 0.025,
            f"Every point is a tested {STEP_PERCENT}%-step combination. "
            "Stars mark the best weight within each pair.",
            ha="center", fontsize=10,
        )
        fig.subplots_adjust(top=0.76, bottom=0.21, wspace=0.13)
        path = folder / "01_pair_weight_search.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        paths.append(path)

        # Graph 2: map the three weights; CatBoost gets 100 - Ridge - Forest.
        grid = three_trials.pivot(
            index="forest %", columns="ridge %", values="May MAE"
        ).sort_index().sort_index(axis=1)
        fig, ax = plt.subplots(figsize=(9, 8))
        cmap = plt.get_cmap("viridis_r").copy()
        cmap.set_bad("#EDF1F4")
        half_step = STEP_PERCENT / 2
        heatmap = ax.imshow(
            np.ma.masked_invalid(grid.to_numpy()), origin="lower",
            extent=[-half_step, 100 + half_step, -half_step, 100 + half_step],
            cmap=cmap, interpolation="nearest", aspect="equal",
        )
        winner = selected["Overall May winner"]
        winner_weights = winner["weights_percent"]
        ax.scatter(
            winner_weights["ridge"], winner_weights["forest"],
            marker="*", s=250, color="white", edgecolor="black",
            linewidth=1.2, zorder=5, label="Overall May winner",
        )
        pair_weights = selected["Best two-model blend"]["weights_percent"]
        ax.scatter(
            pair_weights["ridge"], pair_weights["forest"],
            s=140, facecolors="none", edgecolors="#0055CC",
            linewidth=2, zorder=6, label="Best two-model blend",
        )
        all_three = selected["Best all-three blend"]
        if all_three["weights_percent"] != winner_weights:
            weights = all_three["weights_percent"]
            ax.scatter(
                weights["ridge"], weights["forest"], marker="^", s=120,
                color="#CC3377", edgecolor="white", zorder=5,
                label="Best blend using all three",
            )
        winner_text = (
            "May winner\n"
            f"Ridge: {winner_weights['ridge']}%\n"
            f"Forest: {winner_weights['forest']}%\n"
            f"CatBoost: {winner_weights['catboost']}%\n"
            f"MAE: {winner['may_mae']:.6f} min"
        )
        ax.text(
            0.97, 0.96, winner_text, transform=ax.transAxes,
            ha="right", va="top", fontsize=11,
            bbox={"facecolor": "white", "edgecolor": "#CDD7DC", "pad": 10},
        )
        ax.legend(loc="upper right", bbox_to_anchor=(0.99, 0.66), fontsize=9)
        ax.set(
            xlabel="Ridge weight (%)", ylabel="Forest weight (%)",
            xticks=np.arange(0, 101, 20), yticks=np.arange(0, 101, 20),
        )
        ax.set_title(
            "Three-model weight search on May\n"
            "CatBoost weight = 100% - Ridge weight - Forest weight", pad=16,
        )
        fig.colorbar(heatmap, ax=ax, shrink=0.8).set_label(
            "May MAE (minutes) - lower is better"
        )
        fig.text(
            0.5, 0.025,
            f"{len(three_trials)} tested combinations; boundary cells include zero weights.\n"
            "Blank cells would require a negative CatBoost weight.",
            ha="center", fontsize=10,
        )
        fig.subplots_adjust(bottom=0.13, top=0.88)
        path = folder / "02_three_model_weight_map.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        paths.append(path)

        # Graph 3: compare errors, selected ratios and error direction on June.
        shown = comparison.loc[comparison["Model"].ne("Overall May winner")].copy()
        blends = shown.loc[shown["Model"].isin([
            "Original frozen blend", "Best two-model blend", "Best all-three blend"
        ])].copy()
        display_names = shown["Model"].map(labels).tolist()
        blend_names = blends["Model"].map(labels).tolist()
        bar_colours = shown["Model"].map(row_colours).tolist()
        blend_colours = blends["Model"].map(row_colours).tolist()
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        ax = axes[0, 0]
        bars = ax.barh(display_names, shown["MAE"], color=bar_colours)
        ax.bar_label(bars, labels=[f"{value:.4f}" for value in shown["MAE"]], padding=4)
        ax.invert_yaxis()
        ax.set_xlim(0, shown["MAE"].max() * 1.17)
        ax.set_title("June MAE: average prediction error")
        ax.set_xlabel("Minutes - lower is better")
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)

        catboost_mae = float(shown.loc[shown["Model"].eq("catboost"), "MAE"].iloc[0])
        gain_seconds = (catboost_mae - blends["MAE"]) * 60
        ax = axes[0, 1]
        bars = ax.barh(blend_names, gain_seconds, color=blend_colours)
        ax.bar_label(
            bars, labels=[f"{value:+.3f} s" for value in gain_seconds], padding=4
        )
        ax.invert_yaxis()
        ax.axvline(0, color="#52636C", linewidth=1)
        ax.set_xlim(min(0, gain_seconds.min()) - 0.25, max(0, gain_seconds.max()) + 0.6)
        ax.set_title("June error reduction compared with CatBoost")
        ax.set_xlabel("Seconds of MAE reduction - positive is better")
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)

        ax = axes[1, 0]
        left = np.zeros(len(blends))
        for key in KEYS:
            values = blends[f"{key} %"].to_numpy()
            ax.barh(blend_names, values, left=left, color=colours[key], label=names[key])
            for y, (start, value) in enumerate(zip(left, values)):
                if value > 0:
                    ax.text(
                        start + value / 2, y, f"{value:.0f}%",
                        ha="center", va="center", color="white",
                        fontsize=9 if value >= 10 else 8, fontweight="bold",
                    )
            left += values
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_title("Model weights chosen using May")
        ax.set_xlabel("Share of the final prediction (%)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=9)

        ax = axes[1, 1]
        bias = shown["Mean signed error"]
        bars = ax.barh(display_names, bias, color=bar_colours)
        ax.bar_label(bars, labels=[f"{value:+.4f}" for value in bias], padding=4)
        ax.invert_yaxis()
        ax.axvline(0, color="#52636C", linewidth=1)
        ax.set_xlim(min(0, bias.min()) - 0.18, max(0, bias.max()) + 0.08)
        ax.set_title("June mean signed error: predicted - actual")
        ax.set_xlabel("Minutes: negative = too short; positive = too long")
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)

        fig.suptitle("June comparison of May-selected weights", fontsize=16)
        fig.text(
            0.5, 0.015,
            "June is an exploratory comparison. "
            "Error reduction is a change in prediction error, not processing speed.",
            ha="center", fontsize=10,
        )
        fig.subplots_adjust(left=0.16, right=0.97, top=0.9, bottom=0.13, wspace=0.75, hspace=0.65)
        path = folder / "03_june_ensemble_comparison.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        paths.append(path)

    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-show", action="store_true",
        help="Save the graphs without opening graph windows.",
    )
    args = parser.parse_args()
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    # 3. Load the existing shared data and align every May prediction by ID.
    trips = pd.read_parquet(DATA)
    with DATA.open("rb") as file:
        fingerprint = file_digest(file, "sha256").hexdigest()
    may = month_rows(trips, "2024-05")
    if may.empty or not may.record_id.is_unique:
        raise ValueError("May needs records with unique record IDs.")
    if trips[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("The prepared predictors and target must be complete.")
    may_predictions = {}
    for key in KEYS:
        info = json.loads(
            (RESULTS / f"{key}_selection.json").read_text(encoding="utf-8")
        )
        if info["data_sha256"] != fingerprint:
            raise ValueError(f"{key}: the prepared dataset does not match.")
        frame = pd.read_csv(RESULTS / f"{key}_may.csv").set_index("record_id")
        if not frame.index.is_unique or set(frame.index) != set(may.record_id):
            raise ValueError(f"{key}: May prediction IDs do not match.")
        values = frame.loc[may.record_id, "prediction"].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{key}: May predictions must be finite.")
        may_predictions[key] = values

    actual_may = may[TARGET].to_numpy()
    pair_trials, three_trials = search_weights(may_predictions, actual_may)
    genuine_three = three_trials.loc[three_trials["Active models"].eq(3)]
    print(f"May selection: {len(may):,} identical trips; {STEP_PERCENT}% steps.")
    print("Each weight vector totals 100%. No model is removed before searching.")

    for pair, frame in pair_trials.groupby("Pair", sort=False):
        print_trials(pair, frame)
    print_trials(
        f"All-three blends (each model receives at least {STEP_PERCENT}%)",
        genuine_three,
    )
    print(f"\nFull three-weight grid: {len(three_trials)} unique combinations,")
    print("including zero weights and single-model endpoints.")

    # 4. Lock these choices using May only, before calculating June scores.
    selected = {
        "Best two-model blend": selection_record(choose_best(three_trials, 2)),
        "Best all-three blend": selection_record(choose_best(three_trials, 3)),
        "Overall May winner": selection_record(choose_best(three_trials)),
    }
    selection_rows = []
    for name, item in selected.items():
        selection_rows.append({
            "Choice": name,
            **{f"{key} %": item["weights_percent"][key] for key in KEYS},
            "May MAE": item["may_mae"],
        })
    print("\nMAY-SELECTED WEIGHTS")
    print(pd.DataFrame(selection_rows).to_string(
        index=False, float_format=lambda value: f"{value:.6f}"
    ))

    # 5. Reload final saved models and compare the fixed choices on June.
    june = month_rows(trips, "2024-06")
    if june.empty:
        raise ValueError("No June evaluation records were found.")
    june_predictions = {}
    with threadpool_limits(4):
        for key in KEYS:
            model = joblib.load(MODELS / f"{key}.joblib")
            values = np.maximum(1.0, model.predict(june[FEATURES]))
            if values.shape != (len(june),) or not np.isfinite(values).all():
                raise ValueError(f"{key}: invalid June predictions.")
            june_predictions[key] = values

    comparisons = {
        key: {name: 100 if name == key else 0 for name in KEYS}
        for key in KEYS
    }
    original = json.loads(
        (RESULTS / "frozen_blend.json").read_text(encoding="utf-8")
    )
    if original["data_sha256"] != fingerprint:
        raise ValueError("The original blend uses a different prepared dataset.")
    original_weights = dict.fromkeys(KEYS, 0.0)
    original_weights[original["first"]] = original["first_weight"] * 100
    original_weights[original["second"]] = (1 - original["first_weight"]) * 100
    comparisons["Original frozen blend"] = original_weights
    comparisons.update({
        name: item["weights_percent"] for name, item in selected.items()
    })
    actual_june = june[TARGET].to_numpy()
    june_rows = []
    for name, weights in comparisons.items():
        values = blend_predictions(june_predictions, weights)
        mse = mean_squared_error(actual_june, values)
        june_rows.append({
            "Model": name,
            **{f"{key} %": float(weights[key]) for key in KEYS},
            "MAE": float(mean_absolute_error(actual_june, values)),
            "RMSE": float(np.sqrt(mse)),
            "MSE": float(mse),
            "R squared": float(r2_score(actual_june, values)),
            "Mean signed error": float(np.mean(values - actual_june)),
        })
    comparison = pd.DataFrame(june_rows)
    print(f"\nJUNE COMPARISON: {len(june):,} identical trips")
    print(comparison.to_string(
        index=False,
        formatters={column: lambda value: f"{value:.0f}" for column in WEIGHT_COLUMNS},
        float_format=lambda value: f"{value:.4f}",
    ))
    print("\nMean signed error = predicted - actual (minutes).")
    print("Positive = overestimation; negative = underestimation, on average.")
    print("June is an exploratory follow-up because its results were already viewed.")

    # 6. Generate graphs from the calculated trials and comparison scores.
    graph_paths = make_graphs(pair_trials, three_trials, selected, comparison)

    # 7. Save this additional experiment separately from the original freeze.
    output = RESULTS / "additional_ensemble_selection.json"
    payload = {
        "step_percent": STEP_PERCENT,
        "selection_month": "2024-05",
        "evaluation_month": "2024-06",
        "june_role": "Exploratory comparison; June was already inspected.",
        "data_sha256": fingerprint,
        "pair_trials_including_endpoints": len(pair_trials),
        "unique_weight_vectors": len(three_trials),
        "genuine_all_three_trials": len(genuine_three),
        "selected": selected,
        "may_trials": [
            selection_record(row) for _, row in three_trials.iterrows()
        ],
        "june_comparison": june_rows,
        "graph_files": [str(path.relative_to(ROOT)) for path in graph_paths],
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nSaved additional results:", output)
    print("\nSaved graphs:")
    for path in graph_paths:
        print(path)

    import matplotlib.pyplot as plt
    if args.no_show:
        plt.close("all")
    else:
        print("\nClose the three graph windows to finish the script.")
        plt.show()


if __name__ == "__main__":
    main()

