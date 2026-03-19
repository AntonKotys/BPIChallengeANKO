import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

df = pd.read_csv("bpi2017filtered.csv")

# Keep only "A_Create Application" events (application creation)
df_app_create = df[df["concept:name"] == "A_Create Application"].copy()

df_app_create["time:timestamp"] = pd.to_datetime(df_app_create["time:timestamp"], format="mixed")

# weekday and month
df_app_create["weekday"] = df_app_create["time:timestamp"].dt.day_name()
df_app_create["month"] = df_app_create["time:timestamp"].dt.to_period("M")

weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
df_app_create["weekday"] = pd.Categorical(df_app_create["weekday"], categories=weekday_order, ordered=True)

# --- Count number of applications per weekday per month ---
weekday_counts = (
    df_app_create.groupby(["month", "weekday"])
    .size()
    .reset_index(name="num_applications")
)

# single histogram for each month (aka drift)
plt.figure(figsize=(12, 6))
for month, group in weekday_counts.groupby("month"):
    plt.plot(
        group["weekday"],
        group["num_applications"],
        marker="o",
        label=str(month)
    )

plt.title("Distributional Drift: Number of Applications by Weekday (Monthly Trends)")
plt.xlabel("Weekday")
plt.ylabel("Number of Applications")
plt.legend(title="Month", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("weekday_drift_histogram.png", dpi=300)
plt.show()

