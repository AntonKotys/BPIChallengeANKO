import os, json
import numpy as np, pandas as pd
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from scipy import stats

DATASET = os.path.join("data", "BPI_Challenge_2017.xes.gz")
OUTDIR = "processing_time_outputs"
MIN_SAMPLES_PER_ACTIVITY = 50

DISTS = {
    "lognorm": stats.lognorm,
    "gamma": stats.gamma,
    "weibull_min": stats.weibull_min,
    "expon": stats.expon,
}

def load_df(path):
    df = pm4py.convert_to_dataframe(xes_importer.apply(path))
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["case:concept:name", "concept:name", "time:timestamp"])
    return df.sort_values(["case:concept:name", "time:timestamp"]).reset_index(drop=True)

def build_durations(df):
    # Fast proxy: time until next event in the same case
    df["next_ts"] = df.groupby("case:concept:name")["time:timestamp"].shift(-1)
    s = df.dropna(subset=["next_ts"]).copy()
    s["duration_sec"] = (s["next_ts"] - s["time:timestamp"]).dt.total_seconds()
    s = s[(s["duration_sec"] > 0) & np.isfinite(s["duration_sec"])]
    return s[["concept:name", "duration_sec"]].rename(columns={"concept:name": "activity"})

def fit_best_distribution(x):
    best = None
    for name, dist in DISTS.items():
        try:
            params = dist.fit(x, floc=0)
            ll = np.sum(dist.logpdf(x, *params))
            aic = 2 * len(params) - 2 * ll
            cand = {"best_dist": name, "params": list(map(float, params)), "aic": float(aic)}
            if best is None or cand["aic"] < best["aic"]:
                best = cand
        except Exception:
            pass
    return best

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"[Load] {DATASET}")
    df = load_df(DATASET)
    print(f"[DF] events={len(df):,}, cases={df['case:concept:name'].nunique():,}")

    s = build_durations(df)
    print(f"[Durations] samples={len(s):,}, activities={s['activity'].nunique():,}")

    out = {}
    for act, g in s.groupby("activity", sort=False):
        x = g["duration_sec"].to_numpy()
        if len(x) < MIN_SAMPLES_PER_ACTIVITY:
            continue
        best = fit_best_distribution(x)
        if best:
            out[str(act)] = {**best, "n": int(len(x)), "unit": "seconds"}

    with open(os.path.join(OUTDIR, "distributions.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"[Saved] {len(out)} activities -> {OUTDIR}/distributions.json")

if __name__ == "__main__":
    main()