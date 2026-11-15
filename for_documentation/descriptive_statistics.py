from pathlib import Path
import pandas as pd

# Find the project folder above for_documentation.
ROOT = Path(__file__).resolve().parent.parent

# Read the prepared data and select January-April.
trips = pd.read_parquet(ROOT / "data" / "lesson_trips.parquet")
months = trips["pickup"].dt.strftime("%Y-%m")
development = trips.loc[
    months.isin(["2024-01", "2024-02", "2024-03", "2024-04"])
]
duration = development["duration_min"]

# Calculate the statistics shown in Table 2.
q1, q3 = duration.quantile([0.25, 0.75])
statistics = {
    "Mean": duration.mean(),
    "Median": duration.median(),
    "Standard deviation": duration.std(),
    "Minimum": duration.min(),
    "Maximum": duration.max(),
    "Skewness": duration.skew(),
    "First quartile": q1,
    "Third quartile": q3,
    "Interquartile range": q3 - q1,
    "90th percentile": duration.quantile(0.90),
    "95th percentile": duration.quantile(0.95),
    "99th percentile": duration.quantile(0.99),
}

# Print the same four-column layout as the PDF.
items = list(statistics.items())
print(f"{'Statistic':<21} {'Value':>8}   {'Statistic':<22} {'Value':>8}")
print("-" * 65)

for i in range(6):
    left_name, left_value = items[i]
    right_name, right_value = items[i + 6]
    print(
        f"{left_name:<21} {left_value:>8.2f}   "
        f"{right_name:<22} {right_value:>8.2f}"
    )