import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg" if you’re on macOS
import matplotlib.pyplot as plt
import pandas as pd

# Load your filtered event log
df = pd.read_csv("bpi2017filtered.csv")

# Keep only the event where an application is created
df_app_create = df[df["concept:name"] == "A_Create Application"].copy()

# Convert timestamp to datetime
df_app_create["time:timestamp"] = pd.to_datetime(df_app_create["time:timestamp"], format="mixed")

# Extract weekday and month
df_app_create["weekday"] = df_app_create["time:timestamp"].dt.day_name()
df_app_create["month"] = df_app_create["time:timestamp"].dt.to_period("M")

# Count applications per weekday per month
weekday_month_counts = (
    df_app_create.groupby(["month", "weekday"])
    .size()
    .reset_index(name="num_applications")
)

# Sort weekdays in logical order
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekday_month_counts["weekday"] = pd.Categorical(weekday_month_counts["weekday"], categories=weekday_order, ordered=True)

# Pivot table for visualization
pivot_data = weekday_month_counts.pivot(index="month", columns="weekday", values="num_applications").fillna(0)

# --- Plot heatmap-style trend ---
plt.figure(figsize=(10, 6))
plt.imshow(pivot_data.T, aspect="auto", cmap="Blues", origin="lower")
plt.yticks(range(len(pivot_data.columns)), pivot_data.columns)
plt.xticks(range(len(pivot_data.index)), pivot_data.index.astype(str), rotation=45)
plt.colorbar(label="Number of Applications")
plt.title("📆 Distributional Drift: Applications per Weekday over Time")
plt.xlabel("Month")
plt.ylabel("Weekday")
plt.tight_layout()
plt.savefig("weekday_drift_heatmap.png", dpi=300)
plt.show()

print("✅ Saved plot as weekday_drift_heatmap.png")
