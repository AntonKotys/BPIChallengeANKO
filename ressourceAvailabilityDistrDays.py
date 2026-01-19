import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "bpi2017.csv"          # adjust path if needed
RESOURCE_COL = "org:resource"
TS_COL = "time:timestamp"

OUT_HIST = "avg_available_days_hist.png"
OUT_ECDF = "avg_available_days_ecdf.png"

def main():
    # --- Load data ---
    df = pd.read_csv(CSV_PATH, usecols=[RESOURCE_COL, TS_COL])
    df[TS_COL] = pd.to_datetime(df[TS_COL], errors="coerce")
    df = df.dropna(subset=[TS_COL])
    df[RESOURCE_COL] = df[RESOURCE_COL].astype(str)

    # --- Compute availability ---
    df["date"] = df[TS_COL].dt.date
    df["month"] = df[TS_COL].dt.to_period("M")

    days_per_month = (
        df.groupby([RESOURCE_COL, "month"])["date"]
          .nunique()
          .reset_index(name="days_available")
    )

    avg_days_per_resource = (
        days_per_month.groupby(RESOURCE_COL)["days_available"].mean()
    )

    # --- Stats ---
    overall_avg = float(avg_days_per_resource.mean())
    q30 = float(avg_days_per_resource.quantile(0.20))

    print(f"Resources: {len(avg_days_per_resource)}")
    print(f"Average available days/month: {overall_avg:.2f}")
    print(f"20% quantile: {q30:.2f}")

    # --- Histogram ---
    plt.figure(figsize=(8, 5))
    plt.hist(avg_days_per_resource.values, bins=100)
    plt.axvline(overall_avg, linestyle="--", linewidth=1, label=f"Mean = {overall_avg:.2f}")
    plt.axvline(q30, linestyle="--", linewidth=1, label=f"20% q = {q30:.2f}")
    plt.title("Distribution of Avg Available Days per Month (per Resource)")
    plt.xlabel("Average available days per month")
    plt.ylabel("Number of resources")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_HIST, dpi=150)
    plt.close()

    # --- ECDF ---
    s = avg_days_per_resource.sort_values().reset_index(drop=True)
    y = (s.index + 1) / len(s)

    plt.figure(figsize=(8, 5))
    plt.plot(s.values, y)
    plt.axvline(overall_avg, linestyle="--", linewidth=1, label=f"Mean = {overall_avg:.2f}")
    plt.axvline(q30, linestyle="--", linewidth=1, label=f"30% q = {q30:.2f}")
    plt.title("ECDF of Avg Available Days per Month (per Resource)")
    plt.xlabel("Average available days per month")
    plt.ylabel("Cumulative proportion of resources")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_ECDF, dpi=150)
    plt.close()

    print(f"Saved plots:")
    print(f" - {OUT_HIST}")
    print(f" - {OUT_ECDF}")

if __name__ == "__main__":
    main()
