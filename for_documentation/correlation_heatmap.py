from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "lesson_raw"
OUTPUT = ROOT / "for_documentation" / "figures"

# Use the 10 original numerical attributes with available values.
# ehail_fee is entirely missing, so its correlation cannot be calculated.
numerical = [
    "passenger_count", "trip_distance", "fare_amount",
    "extra", "mta_tax", "tip_amount", "tolls_amount",
    "improvement_surcharge", "total_amount", "congestion_surcharge",
]

# Reuse Table 5's development trip IDs and duration target.
prepared = pd.read_parquet(
    ROOT / "data" / "lesson_trips.parquet",
    columns=["pickup", "record_id", "duration_min"],
)
months = prepared["pickup"].dt.strftime("%Y-%m")
development = prepared.loc[
    months.isin(["2024-01", "2024-02", "2024-03", "2024-04"])
].set_index("record_id")

# Retrieve the original attributes for the same 60,000 trips.
parts = []

for number in range(1, 5):
    month = f"2024-{number:02d}"
    raw = pd.read_parquet(RAW / f"green_tripdata_{month}.parquet")
    raw["record_id"] = month + "-" + raw.index.astype(str)
    parts.append(raw.set_index("record_id")[numerical])

original = pd.concat(parts).join(
    development[["duration_min"]], how="inner"
)

# Calculate pairwise Spearman correlations using available values.
columns = numerical + ["duration_min"]
correlation = original[columns].corr(method="spearman")

# Print the sample details and correlations with duration.
summary = pd.DataFrame({
    "Valid pairs": original[numerical].notna().sum(),
    "Spearman correlation": correlation.loc[numerical, "duration_min"],
})
summary.index.name = "Original attribute"

print(f"January-April 2024 development records: {len(original):,}")
print("Variables shown: 10 original numerical attributes + trip duration")
print("ehail_fee excluded: all values are missing.\n")
print(summary.to_string(formatters={
    "Valid pairs": "{:,.0f}".format,
    "Spearman correlation": "{:.3f}".format,
}))

# Draw the correlation matrix.
labels = [name.replace("_", " ") for name in columns]
fig, ax = plt.subplots(figsize=(10, 8.5), layout="constrained")
heatmap = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)

ax.set_xticks(range(len(columns)), labels, rotation=45, ha="right")
ax.set_yticks(range(len(columns)), labels)
ax.tick_params(labelsize=10)
ax.set_title("Numerical attributes and trip duration", pad=14)

# Write each correlation value inside its cell.
for row in range(len(columns)):
    for column in range(len(columns)):
        value = correlation.iloc[row, column]

        if pd.isna(value):
            label = "NA"
        else:
            label = "0.00" if abs(value) < 0.005 else f"{value:.2f}"

        ax.text(
            column, row, label,
            ha="center", va="center", fontsize=9,
            color="white" if abs(value) > 0.6 else "#222222",
        )

fig.colorbar(heatmap, ax=ax, shrink=0.85, label="Spearman correlation")

# Save one PNG for the report, then display the graph.
OUTPUT.mkdir(parents=True, exist_ok=True)
image_path = OUTPUT / "correlation_heatmap.png"
fig.savefig(image_path, dpi=300)
print(f"\nFigure saved to: {image_path}")
plt.show()