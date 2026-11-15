from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Find the project folder above for_documentation.
ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "for_documentation" / "figures"

# Use the same January-April development data as Table 2.
trips = pd.read_parquet(ROOT / "data" / "lesson_trips.parquet")
months = trips["pickup"].dt.strftime("%Y-%m")
development = trips.loc[
    months.isin(["2024-01", "2024-02", "2024-03", "2024-04"])
]
duration = development["duration_min"]

# Calculate the median and upper IQR fence.
q1, q3 = duration.quantile([0.25, 0.75])
upper_fence = q3 + 1.5 * (q3 - q1)
median = duration.median()

# Summarise duration at each New York pickup hour.
hourly = development.groupby(development["pickup"].dt.hour)["duration_min"]
hourly_table = pd.DataFrame({
    "Trips": hourly.size(),
    "Q1": hourly.quantile(0.25),
    "Median": hourly.median(),
    "Q3": hourly.quantile(0.75),
})
hourly_table.index.name = "Pickup hour"

# Print the values supporting the figure.
print("\nFIGURE 1 - DURATION DISTRIBUTION AND HOURLY VARIATION")
print(f"January-April 2024 development records: {len(development):,}")
print(f"Median duration: {median:.2f} minutes")
print(f"Upper IQR fence: {upper_fence:.2f} minutes")
print("\nHOURLY DURATION SUMMARY (MINUTES)")
print(hourly_table.round(2).to_string())

# Create the two graphs side by side.
plt.rcParams.update({"font.size": 10, "axes.titlesize": 12})
fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")

# Left: count trips within each duration interval.
axes[0].hist(duration, bins=np.linspace(0, 120, 46), color="#176A78")
axes[0].axvline(median, color="#BE5A30", label="Median")
axes[0].axvline(
    upper_fence, color="#637586", linestyle="--", label="Upper IQR fence"
)
axes[0].set(
    title="Duration distribution",
    xlabel="Duration (minutes)",
    ylabel="Trips",
)
axes[0].legend(fontsize=9)

# Right: show the hourly median and middle 50% of trip durations.
axes[1].fill_between(
    hourly_table.index, hourly_table["Q1"], hourly_table["Q3"],
    color="#BCDCE0", label="Middle 50%",
)
axes[1].plot(
    hourly_table.index, hourly_table["Median"],
    color="#176A78", label="Median",
)
axes[1].set(
    title="Duration by pickup hour",
    xlabel="New York pickup hour",
    ylabel="Duration (minutes)",
    xticks=[0, 4, 8, 12, 16, 20, 23],
)
axes[1].legend(fontsize=9)

# Save the figure for the report, then display it.
OUTPUT.mkdir(parents=True, exist_ok=True)
image_path = OUTPUT / "duration_distribution.png"
fig.savefig(image_path, dpi=300)
print(f"\nFigure saved to: {image_path}")
plt.show()