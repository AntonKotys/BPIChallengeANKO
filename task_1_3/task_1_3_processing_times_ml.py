import argparse, json, os
import numpy as np, pandas as pd
from scipy import stats
from pm4py.objects.log.importer.xes import importer as xes_importer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor

SEED = 42
DISTS = {
    "lognorm": stats.lognorm,
    "gamma": stats.gamma,
    "weibull_min": stats.weibull_min,
    "expon": stats.expon,
}

# build processing-time samples directly from XES

def extract_samples(xes_path):
    log = xes_importer.apply(xes_path)
    rows = []

    for trace in log:
        events = list(trace)
        for i, ev in enumerate(events):
            if i == 0:
                continue

            t1 = events[i - 1]["time:timestamp"]
            t2 = ev["time:timestamp"]
            dur = (t2 - t1).total_seconds()
            if dur <= 0:
                continue

            rows.append({
                "activity": ev["concept:name"],
                "prev_activity": events[i - 1]["concept:name"],
                "duration_sec": dur,
                "pos": i,
                "case_len": len(events),
                "hour": t2.hour,
                "weekday": t2.weekday(),
                "month": t2.month,
            })

    return pd.DataFrame(rows)

# fit probability distributions per activity

def fit_distributions(df, min_n=50):
    out = {}
    for act, g in df.groupby("activity"):
        x = g["duration_sec"].to_numpy()
        if len(x) < min_n:
            continue

        best = None
        for name, dist in DISTS.items():
            try:
                p = dist.fit(x, floc=0)
                ll = np.sum(dist.logpdf(x, *p))
                aic = 2 * len(p) - 2 * ll
                if best is None or aic < best["aic"]:
                    best = {
                        "best_dist": name,
                        "params": list(map(float, p)),
                        "aic": float(aic),
                        "n": int(len(x)),
                        "unit": "seconds",
                    }
            except Exception:
                pass

        if best:
            out[str(act)] = best

    return out

# train ml point estimation model

def train_ml(df):
    X = df[["activity", "prev_activity", "pos", "case_len", "hour", "weekday", "month"]]
    y = np.log1p(df["duration_sec"].to_numpy())

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        
    pipe = Pipeline([
        ("ohe", ohe),
        ("gb", HistGradientBoostingRegressor(random_state=SEED)),
    ])

    pipe.fit(Xtr, ytr)
    pred = np.expm1(pipe.predict(Xte))
    true = np.expm1(yte)

    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    return pipe, {
        "MAE_seconds": float(mean_absolute_error(true, pred)),
        "RMSE_seconds": rmse,
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xes", default="data/BPI_Challenge_2017.xes.gz")
    ap.add_argument("--out", default="processing_time_outputs")
    ap.add_argument("--mode", choices=["dist", "ml", "both"], default="both")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    np.random.seed(SEED)

    print("[1] Extracting processing time samples...")
    df = extract_samples(args.xes)
    print(f"    Samples: {len(df):,}")

    if args.mode in ("dist", "both"):
        print("[2] Fitting probability distributions...")
        d = fit_distributions(df)
        with open(os.path.join(args.out, "distributions.json"), "w") as f:
            json.dump(d, f, indent=2)
        print(f"    Distributions fitted for {len(d)} activities")

    if args.mode in ("ml", "both"):
        print("[3] Training ML model...")
        model, metrics = train_ml(df)
        with open(os.path.join(args.out, "ml_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print("    ML metrics:", metrics)

if __name__ == "__main__":
    main()