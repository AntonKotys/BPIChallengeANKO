import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

CSV_PATH = "bpi2017.csv"
RESOURCE_COL = "org:resource"
TS_COL = "time:timestamp"

OUTPUT_DIR = "resource_shift_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# Load and clean
# ---------------------------
df = pd.read_csv(CSV_PATH, usecols=[RESOURCE_COL, TS_COL])
df = df.dropna(subset=[RESOURCE_COL, TS_COL]).copy()
df[RESOURCE_COL] = df[RESOURCE_COL].astype(str)
df[TS_COL] = pd.to_datetime(df[TS_COL], errors="coerce", utc=True)
df = df.dropna(subset=[TS_COL]).copy()

# Create date and time-of-day helpers
df["date"] = df[TS_COL].dt.date
df["tod_minutes"] = df[TS_COL].dt.hour * 60 + df[TS_COL].dt.minute + df[TS_COL].dt.second / 60.0
df["weekday"] = df[TS_COL].dt.weekday  # Mon=0..Sun=6

# ---------------------------
# Compute "shifts" = first/last activity per day
# ---------------------------
daily = (
    df.groupby([RESOURCE_COL, "date"], as_index=False)[TS_COL]
      .agg(start_time="min", end_time="max", n_events="count")
)

# Convert to matplotlib numeric format for plotting
daily["date_dt"] = pd.to_datetime(daily["date"])
daily["start_tod"] = daily["start_time"].dt.hour + daily["start_time"].dt.minute / 60 + daily["start_time"].dt.second / 3600
daily["end_tod"] = daily["end_time"].dt.hour + daily["end_time"].dt.minute / 60 + daily["end_time"].dt.second / 3600

# ---------------------------
# Helper plotting functions
# ---------------------------
def plot_shift_timeline(resource: str, resource_daily: pd.DataFrame, out_path: str):
    """
    Plot daily shift segments: x = date, y = hour-of-day, segment from start to end.
    """
    resource_daily = resource_daily.sort_values("date_dt").copy()

    fig, ax = plt.subplots(figsize=(12, 4))

    # Each day: draw a vertical segment at that date from start_tod to end_tod
    # Use vlines: x positions are dates, y from start to end
    x = mdates.date2num(resource_daily["date_dt"])
    ax.vlines(x, resource_daily["start_tod"], resource_daily["end_tod"], linewidth=2)

    ax.set_title(f"Estimated daily working window (first→last event): {resource}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Hour of day")
    ax.set_ylim(0, 24)
    ax.yaxis.set_ticks(np.arange(0, 25, 2))

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_time_of_day_hist(resource: str, resource_df: pd.DataFrame, out_path: str, bins: int = 48):
    """
    Histogram of activity timestamps by time-of-day (minutes since midnight).
    bins=48 -> 30-minute bins.
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    data = resource_df["tod_minutes"].values
    ax.hist(data, bins=bins)

    ax.set_title(f"Time-of-day activity density: {resource}")
    ax.set_xlabel("Minutes since midnight")
    ax.set_ylabel("Event count")

    # Nice x ticks at hours
    hour_ticks = np.arange(0, 24 * 60 + 1, 120)  # every 2 hours
    ax.set_xticks(hour_ticks)
    ax.set_xticklabels([f"{int(m//60):02d}:00" for m in hour_ticks], rotation=45, ha="right")

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_weekday_box(resource: str, resource_df: pd.DataFrame, out_path: str):
    """
    Boxplot of time-of-day by weekday.
    Shows if someone works different hours on different weekdays.
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    # Gather per weekday
    groups = []
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for wd in range(7):
        groups.append(resource_df.loc[resource_df["weekday"] == wd, "tod_minutes"].values)

    ax.boxplot(groups, showfliers=False)
    ax.set_title(f"Time-of-day distribution by weekday: {resource}")
    ax.set_xlabel("Weekday")
    ax.set_ylabel("Minutes since midnight")
    ax.set_xticks(np.arange(1, 8))
    ax.set_xticklabels(labels)

    # Nice y ticks at hours
    hour_ticks = np.arange(0, 24 * 60 + 1, 120)
    ax.set_yticks(hour_ticks)
    ax.set_yticklabels([f"{int(m//60):02d}:00" for m in hour_ticks])

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------
# Generate plots for each resource
# ---------------------------
resources = sorted(df[RESOURCE_COL].unique().tolist())

print(f"Found {len(resources)} resources.")
print("Generating plots into:", OUTPUT_DIR)

for r in resources:
    df_r = df[df[RESOURCE_COL] == r].copy()
    daily_r = daily[daily[RESOURCE_COL] == r].copy()

    # Skip extremely tiny resources if desired
    if len(daily_r) < 3:
        continue

    # 1) Shift timeline (daily start→end)
    plot_shift_timeline(
        resource=r,
        resource_daily=daily_r,
        out_path=os.path.join(OUTPUT_DIR, f"{r}_shift_timeline.png")
    )

    # 2) Time-of-day histogram
    plot_time_of_day_hist(
        resource=r,
        resource_df=df_r,
        out_path=os.path.join(OUTPUT_DIR, f"{r}_tod_hist.png"),
        bins=48
    )

    # 3) Weekday boxplot (optional, but useful)
    plot_weekday_box(
        resource=r,
        resource_df=df_r,
        out_path=os.path.join(OUTPUT_DIR, f"{r}_weekday_box.png")
    )

print("✅ Done.")
