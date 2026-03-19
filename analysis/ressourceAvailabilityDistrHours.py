import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "bpi2017.csv"          # adjust path if needed
RESOURCE_COL = "org:resource"
TS_COL = "time:timestamp"

OUT_PNG = "avg_available_hours_per_day_distribution.png"

def main():
    # --- Load ---
    df = pd.read_csv(CSV_PATH, usecols=[RESOURCE_COL, TS_COL])
    df[TS_COL] = pd.to_datetime(df[TS_COL], errors="coerce")
    df = df.dropna(subset=[TS_COL])
    df[RESOURCE_COL] = df[RESOURCE_COL].astype(str)

    # --- Date ---
    df["date"] = df[TS_COL].dt.date

    # --- Daily working hours per resource ---
    daily_hours = (
        df.groupby([RESOURCE_COL, "date"])[TS_COL]
          .agg(lambda s: (s.max() - s.min()).total_seconds() / 3600.0)
          .reset_index(name="hours")
    )

    # (optional) drop zero-length days
    daily_hours = daily_hours[daily_hours["hours"] > 0]

    # --- Average daily hours per resource ---
    avg_hours_per_resource = (
        daily_hours.groupby(RESOURCE_COL)["hours"].mean()
    )

    # --- Stats ---
    avg_overall = float(avg_hours_per_resource.mean())
    q30 = float(avg_hours_per_resource.quantile(0.20))

    print(f"Resources: {len(avg_hours_per_resource)}")
    print(f"Average working hours per day (mean across resources): {avg_overall:.2f}")
    print(f"20% quantile: {q30:.2f}")

    # --- Plot ---
    plt.figure(figsize=(8, 5))
    plt.hist(avg_hours_per_resource.values, bins=100)
    plt.axvline(avg_overall, linestyle="--", linewidth=1,
                label=f"Mean = {avg_overall:.2f} h")
    plt.axvline(q30, linestyle="--", linewidth=1,
                label=f"20% q = {q30:.2f} h")
    plt.title("Distribution of Avg Working Hours per Day (per Resource)")
    plt.xlabel("Average working hours per day")
    plt.ylabel("Number of resources")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    plt.close()

    print(f"Saved plot: {OUT_PNG}")

if __name__ == "__main__":
    main()
