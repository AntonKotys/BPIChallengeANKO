"""
Task 1.4 Benchmark: Compare multiple ML models and feature sets
on the mapped event log for XOR gateway prediction accuracy.

Models: GradientBoosting, RandomForest, XGBoost, LightGBM
Features: standard (temporal only) vs enriched (+ case attributes + aggregated)
k-gram: 2, 3, 4, 5
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation_engine_core_final_version import PROCESS_MODEL, GATEWAYS

ACTIVITY_MAPPING = {
    "A_Create Application": "A_Create Application",
    "A_Submitted": "A_Submitted",
    "W_Handle leads": "W_Handle leads",
    "O_Returned": "O_Returned",
    "O_Sent (mail and online)": "O_Sent (mail and online)",
    "O_Sent (online only)": "O_Sent (mail and online)",
    "A_Accepted": "A_Accepted",
    "A_Incomplete": "A_Incomplete",
    "A_Validating": "A_Validating",
    "W_Complete application": "W_Complete application & A_Concept",
    "A_Concept": "W_Complete application & A_Concept",
    "O_Create Offer": "O_Create Offer & O_Created",
    "O_Created": "O_Create Offer & O_Created",
    "W_Call after offers": "W_Call after offers & A_Complete",
    "A_Complete": "W_Call after offers & A_Complete",
    "W_Validate application": "W_Validate application & A_Validating",
    "A_Cancelled": "A_Cancelled & O_Cancelled",
    "O_Cancelled": "A_Cancelled & O_Cancelled",
    "A_Denied": "END",
    "O_Refused": "END",
    "O_Accepted": "END",
    "W_Call incomplete files": "W_Validate application & A_Validating",
    "A_Pending": "A_Pending",
    "W_Assess potential fraud": "W_Assess potential fraud",
    "W_Personal Loan collection": "W_Personal Loan collection",
    "W_Shortened completion ": "W_Shortened completion",
}


def load_and_map_log():
    df = pd.read_csv("bpi2017.csv")
    rename = {
        "case:concept:name": "case_id",
        "concept:name": "activity",
        "time:timestamp": "timestamp",
    }
    for old, new in rename.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    df = df.sort_values(["case_id", "timestamp"]).reset_index(drop=True)
    df["activity"] = df["activity"].map(lambda a: ACTIVITY_MAPPING.get(a, a))

    prev_case = None
    prev_act = None
    keep = []
    for idx, row in df.iterrows():
        if row["case_id"] != prev_case or row["activity"] != prev_act:
            keep.append(True)
        else:
            keep.append(False)
        prev_case = row["case_id"]
        prev_act = row["activity"]
    return df[keep].reset_index(drop=True)


def get_xor_activities():
    return {act: succs for act, succs in PROCESS_MODEL.items() if len(succs) > 1}


def build_features(df, window_size=5):
    """Build feature matrix for all transitions, vectorized."""
    from sklearn.preprocessing import LabelEncoder

    df = df.copy()
    act_enc = LabelEncoder()
    df["act_idx"] = act_enc.fit_transform(df["activity"].astype(str))
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.dayofweek
    df["case_start"] = df.groupby("case_id")["timestamp"].transform("min")
    df["elapsed"] = (df["timestamp"] - df["case_start"]).dt.total_seconds()

    # Case-level
    if "case:LoanGoal" in df.columns:
        lg_enc = LabelEncoder()
        df["loan_goal_idx"] = lg_enc.fit_transform(df["case:LoanGoal"].fillna("Unknown").astype(str))
    else:
        df["loan_goal_idx"] = 0

    if "case:ApplicationType" in df.columns:
        at_enc = LabelEncoder()
        df["app_type_idx"] = at_enc.fit_transform(df["case:ApplicationType"].fillna("Unknown").astype(str))
    else:
        df["app_type_idx"] = 0

    df["req_amount"] = df["case:RequestedAmount"].fillna(0.0) if "case:RequestedAmount" in df.columns else 0.0

    offer_acts = {"O_Create Offer", "O_Create Offer & O_Created", "O_Created"}
    df["is_offer"] = df["activity"].isin(offer_acts).astype(int)
    df["n_offers_cum"] = df.groupby("case_id")["is_offer"].cumsum()
    df["trace_pos"] = df.groupby("case_id").cumcount()
    df["loop_count"] = df.groupby(["case_id", "activity"]).cumcount()

    if "CreditScore" in df.columns:
        df["credit_filled"] = df["CreditScore"].where(df["CreditScore"] > 0)
        df["last_credit"] = df.groupby("case_id")["credit_filled"].ffill().fillna(0.0)
    else:
        df["last_credit"] = 0.0

    if "OfferedAmount" in df.columns:
        df["offered_filled"] = df["OfferedAmount"].where(df["OfferedAmount"] > 0)
        df["last_offered"] = df.groupby("case_id")["offered_filled"].ffill().fillna(0.0)
    else:
        df["last_offered"] = 0.0

    # Aggregated: mean time between events in this case so far
    df["time_diff"] = df.groupby("case_id")["timestamp"].diff().dt.total_seconds().fillna(0)
    df["mean_time_diff"] = df.groupby("case_id")["time_diff"].transform(
        lambda x: x.expanding().mean()
    )

    # Number of unique resources in this case so far
    if "org:resource" in df.columns:
        df["resource_idx"] = pd.factorize(df["org:resource"])[0]
        df["n_resources_cum"] = df.groupby("case_id")["resource_idx"].transform(
            lambda x: x.expanding().apply(lambda s: s.nunique(), raw=False)
        )
    else:
        df["n_resources_cum"] = 1

    # Target
    df["target_act_idx"] = df.groupby("case_id")["act_idx"].shift(-1)
    df["target_activity"] = df.groupby("case_id")["activity"].shift(-1)
    df_train = df.dropna(subset=["target_act_idx"]).copy()
    df_train["target_act_idx"] = df_train["target_act_idx"].astype(int)

    # Build n-gram matrix
    act_idx_arr = df_train["act_idx"].values
    case_ids = df_train["case_id"].values
    n = len(df_train)

    ngram_matrix = np.full((n, window_size), -1, dtype=int)
    case_boundaries = np.where(case_ids[:-1] != case_ids[1:])[0] + 1
    case_starts = np.concatenate([[0], case_boundaries])
    case_ends = np.concatenate([case_boundaries, [n]])

    for cs, ce in zip(case_starts, case_ends):
        case_acts = act_idx_arr[cs:ce]
        for local_i in range(len(case_acts)):
            global_i = cs + local_i
            start = max(0, local_i - window_size + 1)
            hist = case_acts[start:local_i + 1]
            if len(hist) < window_size:
                ngram_matrix[global_i, window_size - len(hist):] = hist
            else:
                ngram_matrix[global_i, :] = hist

    enriched_cols = np.column_stack([
        df_train["hour"].values,
        df_train["weekday"].values,
        df_train["elapsed"].values,
        df_train["req_amount"].values,
        df_train["loan_goal_idx"].values,
        df_train["app_type_idx"].values,
        df_train["loop_count"].values,
        df_train["trace_pos"].values,
        df_train["n_offers_cum"].values,
        df_train["last_credit"].values,
        df_train["last_offered"].values,
        df_train["mean_time_diff"].values,
        df_train["n_resources_cum"].values,
    ])

    X = np.hstack([ngram_matrix.astype(float), enriched_cols])
    y = df_train["target_act_idx"].values
    groups = df_train["case_id"].values

    meta = df_train[["case_id", "activity", "target_activity"]].reset_index(drop=True)

    return X, y, groups, meta, act_enc


def evaluate_on_xor(model, X_test, meta_test, xor_activities, act_enc):
    """Evaluate model accuracy only at XOR gateways."""
    predictions = model.predict(X_test)
    pred_activities = act_enc.inverse_transform(predictions)

    correct = 0
    total = 0
    per_gw = defaultdict(lambda: {"correct": 0, "total": 0})

    for i in range(len(meta_test)):
        curr = meta_test["activity"].iloc[i]
        true_next = meta_test["target_activity"].iloc[i]

        if curr in xor_activities:
            valid = xor_activities[curr]
            if true_next in valid:
                total += 1
                per_gw[curr]["total"] += 1
                pred = pred_activities[i]
                if pred == true_next:
                    correct += 1
                    per_gw[curr]["correct"] += 1

    acc = correct / total if total > 0 else 0.0
    return acc, total, correct, dict(per_gw)


def run_benchmark():
    print("=" * 80)
    print("TASK 1.4 BENCHMARK — Model & Feature Comparison")
    print("=" * 80)

    print("\n[1] Loading and mapping event log...")
    df = load_and_map_log()
    print(f"    Cases: {df['case_id'].nunique()}, Events: {len(df)}")

    xor_activities = get_xor_activities()

    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier

    results = []

    for window_size in [3, 5, 7]:
        print(f"\n{'=' * 80}")
        print(f"WINDOW SIZE (k-gram context): {window_size}")
        print(f"{'=' * 80}")

        print(f"\n[2] Building features (window={window_size})...")
        t0 = time.time()
        X, y, groups, meta, act_enc = build_features(df, window_size=window_size)
        print(f"    Feature matrix: {X.shape}, built in {time.time()-t0:.1f}s")

        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        meta_test = meta.iloc[test_idx].reset_index(drop=True)

        models = {
            "RandomForest": RandomForestClassifier(
                n_estimators=300, max_depth=15, min_samples_leaf=5,
                random_state=42, n_jobs=-1
            ),
            "HistGradientBoosting": HistGradientBoostingClassifier(
                max_iter=300, max_depth=8, learning_rate=0.1,
                random_state=42
            ),
            "RF_deep": RandomForestClassifier(
                n_estimators=500, max_depth=25, min_samples_leaf=2,
                random_state=42, n_jobs=-1
            ),
            "HistGB_deep": HistGradientBoostingClassifier(
                max_iter=500, max_depth=12, learning_rate=0.05,
                min_samples_leaf=10, random_state=42
            ),
        }

        for model_name, model in models.items():
            print(f"\n  Training {model_name} (window={window_size})...")
            t0 = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - t0

            val_acc = model.score(X_test, y_test)

            acc, total, correct, per_gw = evaluate_on_xor(
                model, X_test, meta_test, xor_activities, act_enc
            )

            print(f"    Val accuracy (all transitions): {val_acc:.3f}")
            print(f"    XOR accuracy: {acc:.1%} ({correct}/{total})")
            print(f"    Train time: {train_time:.1f}s")

            gw_details = {}
            for gw, stats in sorted(per_gw.items()):
                gw_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
                gw_details[gw] = gw_acc
                print(f"      {gw}: {gw_acc:.1%} ({stats['correct']}/{stats['total']})")

            results.append({
                "model": model_name,
                "window": window_size,
                "val_acc": val_acc,
                "xor_acc": acc,
                "xor_total": total,
                "xor_correct": correct,
                "train_time": train_time,
                **{f"gw_{gw}": gw_details.get(gw, 0) for gw in xor_activities},
            })

    # Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"\n{'Model':<22} {'Window':>6} {'Val Acc':>8} {'XOR Acc':>8} {'Time':>6}")
    print("-" * 55)

    results_sorted = sorted(results, key=lambda r: -r["xor_acc"])
    for r in results_sorted:
        print(f"{r['model']:<22} {r['window']:>6} {r['val_acc']:>8.3f} {r['xor_acc']:>7.1%} {r['train_time']:>5.0f}s")

    best = results_sorted[0]
    print(f"\nBEST: {best['model']} (window={best['window']})")
    print(f"  XOR Accuracy: {best['xor_acc']:.1%} ({best['xor_correct']}/{best['xor_total']})")
    print(f"  Val Accuracy: {best['val_acc']:.3f}")


if __name__ == "__main__":
    run_benchmark()
